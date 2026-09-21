"""deepDeband inference helpers (display Rec.709 float [0,1])."""

from __future__ import annotations

from pathlib import Path

import folder_paths
import torch
import torch.nn.functional as F

from .deepdeband_model import DeepDebandModel

_MODEL_CACHE: dict[str, DeepDebandModel] = {}

WEIGHT_URLS = {
    "deepDeband-w": (
        "https://github.com/RaymondLZhou/deepDeband/raw/master/"
        "pytorch-CycleGAN-and-pix2pix/checkpoints/deepDeband-w/latest_net_G.pth"
    ),
    "deepDeband-f": (
        "https://github.com/RaymondLZhou/deepDeband/raw/master/"
        "pytorch-CycleGAN-and-pix2pix/checkpoints/deepDeband-f/latest_net_G.pth"
    ),
}

WEIGHT_FILES = {
    "deepDeband-w": "deepDeband_w.pth",
    "deepDeband-f": "deepDeband_f.pth",
}


def deepdeband_model_dir() -> Path:
    primary = Path(folder_paths.models_dir) / "acescg" / "deepdeband"
    legacy = Path(folder_paths.models_dir) / "bit_depth_enhancement" / "deepdeband"
    if primary.is_dir() or not legacy.is_dir():
        primary.mkdir(parents=True, exist_ok=True)
        return primary
    return legacy


def resolve_weights_path(variant: str) -> Path:
    name = WEIGHT_FILES.get(variant)
    if not name:
        raise ValueError(f"Unknown deepDeband variant: {variant}")
    primary = deepdeband_model_dir() / name
    if primary.is_file():
        return primary
    legacy = Path(folder_paths.models_dir) / "bit_depth_enhancement" / "deepdeband" / name
    if legacy.is_file():
        return legacy
    return primary


def ensure_weights(variant: str, download: bool = False) -> Path:
    path = resolve_weights_path(variant)
    if path.is_file():
        return path
    if not download:
        raise FileNotFoundError(
            f"Missing deepDeband weights: {path}\n"
            f"Download {WEIGHT_URLS[variant]}\n"
            f"and save as {path.name} in {path.parent}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request

    url = WEIGHT_URLS[variant]
    print(f"[AcesCg] Downloading {variant} -> {path}")
    urllib.request.urlretrieve(url, str(path))
    return path


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(variant: str, device: torch.device | None = None, download: bool = False) -> DeepDebandModel:
    device = device or get_device()
    key = f"{variant}:{device}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    path = ensure_weights(variant, download=download)
    model = DeepDebandModel()
    model.load_pretrained(str(path), strict=False)
    model.to(device)
    model.eval()
    _MODEL_CACHE[key] = model
    return model


def _pad_reflections_to_256(image: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """image [H,W,C] -> padded to multiples of 256 via 2x2 reflection tiling."""
    h, w, _ = image.shape
    target_h = ((h + 255) // 256) * 256
    target_w = ((w + 255) // 256) * 256
    if h == target_h and w == target_w:
        return image, h, w

    img_hflip = torch.flip(image, [0])
    img_vflip = torch.flip(image, [1])
    img_both = torch.flip(img_hflip, [1])
    top = torch.cat([image, img_vflip], dim=1)
    bottom = torch.cat([img_hflip, img_both], dim=1)
    block = torch.cat([top, bottom], dim=0)
    bh, bw = block.shape[:2]
    n_h = (target_h + bh - 1) // bh
    n_w = (target_w + bw - 1) // bw
    tiled = block.repeat(n_h, n_w, 1)
    return tiled[:target_h, :target_w, :], h, w


def _run_unet(tile: torch.Tensor, model: DeepDebandModel, device: torch.device) -> torch.Tensor:
    """tile [H,W,C] in [0,1] -> [H,W,C] in [0,1]."""
    x = tile.permute(2, 0, 1).unsqueeze(0).to(device)
    x = x * 2.0 - 1.0
    with torch.no_grad():
        y = model(x)
    y = (y + 1.0) * 0.5
    y = y.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0).cpu()
    return y


def _blend_mask(
    tile_h: int,
    tile_w: int,
    overlap: int,
    is_start_h: bool,
    is_end_h: bool,
    is_start_w: bool,
    is_end_w: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    mask_h = torch.ones(tile_h, dtype=dtype, device=device)
    mask_w = torch.ones(tile_w, dtype=dtype, device=device)
    if not is_start_h and tile_h > overlap:
        mask_h[:overlap] = torch.linspace(0, 1, overlap, dtype=dtype, device=device)
    if not is_end_h and tile_h > overlap:
        mask_h[-overlap:] = torch.linspace(1, 0, overlap, dtype=dtype, device=device)
    if not is_start_w and tile_w > overlap:
        mask_w[:overlap] = torch.linspace(0, 1, overlap, dtype=dtype, device=device)
    if not is_end_w and tile_w > overlap:
        mask_w[-overlap:] = torch.linspace(1, 0, overlap, dtype=dtype, device=device)
    return mask_h.unsqueeze(1) * mask_w.unsqueeze(0)


def deband_hwc(
    image: torch.Tensor,
    model: DeepDebandModel,
    *,
    weighted: bool = True,
    tile_size: int = 256,
    tile_overlap: int = 128,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Deband single image [H,W,C] float [0,1] on CPU tensor in/out."""
    device = device or next(model.parameters()).device
    padded, orig_h, orig_w = _pad_reflections_to_256(image)
    ph, pw, _ = padded.shape
    tile_size = max(128, int(tile_size))
    overlap = max(0, min(int(tile_overlap), tile_size - 1))

    if ph <= tile_size and pw <= tile_size:
        out = _run_unet(padded, model, device)
        return out[:orig_h, :orig_w, :]

    stride = max(1, tile_size - overlap)
    n_h = (ph + stride - 1) // stride
    n_w = (pw + stride - 1) // stride
    acc = torch.zeros(ph, pw, image.shape[2], dtype=image.dtype, device=device)
    weights = torch.zeros(ph, pw, 1, dtype=image.dtype, device=device)

    for i in range(n_h):
        for j in range(n_w):
            y0 = i * stride
            x0 = j * stride
            y1 = min(y0 + tile_size, ph)
            x1 = min(x0 + tile_size, pw)
            tile = padded[y0:y1, x0:x1, :]
            th, tw = tile.shape[:2]
            if th < tile_size or tw < tile_size:
                pad_h = tile_size - th
                pad_w = tile_size - tw
                tile_bchw = tile.permute(2, 0, 1).unsqueeze(0)
                tile_bchw = F.pad(tile_bchw, (0, pad_w, 0, pad_h), mode="replicate")
                tile = tile_bchw.squeeze(0).permute(1, 2, 0)
            processed = _run_unet(tile, model, device)[:th, :tw, :].to(device)
            if weighted:
                w = _blend_mask(
                    th,
                    tw,
                    overlap,
                    i == 0,
                    y1 == ph,
                    j == 0,
                    x1 == pw,
                    image.dtype,
                    device,
                ).unsqueeze(2)
            else:
                w = torch.ones(th, tw, 1, dtype=image.dtype, device=device)
            acc[y0:y1, x0:x1, :] += processed * w
            weights[y0:y1, x0:x1, :] += w

    out = (acc / weights.clamp_min(1e-8)).cpu()
    return out[:orig_h, :orig_w, :]
