"""Deep Deband — AI debanding (deepDeband ICIP 2022).

No colorspace conversion: input gamut = output gamut (caller feeds the right space).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..core.deepdeband_infer import deband_hwc, get_device, load_model

# Real options.
_MODELS = ("deepDeband-w", "deepDeband-f")
_GAMUTS = ("rec709", "AP0", "AP1")
_WEIGHTS = ("disable", "enable")
# Extra strings that may linger after past widget-order remaps (keeps UI valid).
_LEGACY = ("linear", "1.3", "1.5", "2.0")


def _as_choice(value, allowed: tuple[str, ...], default: str) -> str:
    text = str(value).strip()
    if text in allowed:
        return text
    lower = text.lower()
    for item in allowed:
        if item.lower() == lower:
            return item
    return default


class AcesCgDeepDeband:
    """deepDeband U-Net. Same gamut in/out; no OCIO inside this node."""

    @classmethod
    def INPUT_TYPES(cls):
        # Union lists so remapped leftover values do not trip "Invalid input".
        model_ui = list(_MODELS) + list(_GAMUTS) + list(_WEIGHTS) + list(_LEGACY)
        weights_ui = list(_WEIGHTS) + list(_MODELS) + list(_GAMUTS) + list(_LEGACY)
        gamut_ui = list(_GAMUTS) + list(_MODELS) + list(_WEIGHTS) + list(_LEGACY)
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (model_ui, {"default": "deepDeband-w"}),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Blend with original (1 = full deband).",
                    },
                ),
                "tile_size": ("INT", {"default": 256, "min": 256, "max": 512, "step": 256}),
                "tile_overlap": ("INT", {"default": 128, "min": 0, "max": 256, "step": 16}),
                "download_weights": (weights_ui, {"default": "disable"}),
                "input_gamut": (
                    gamut_ui,
                    {
                        "default": "rec709",
                        "tooltip": (
                            "Declare the gamut of the input (output matches). "
                            "No conversion is applied here."
                        ),
                    },
                ),
            },
            "optional": {
                "mask": (
                    "MASK",
                    {
                        "tooltip": "White = apply deepDeband, black = keep original.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "deband"
    CATEGORY = "AcesCg"
    DESCRIPTION = (
        "AI debanding (deepDeband). No gamut conversion: input_gamut is a label only; "
        "output stays in the same space. tile_size must be 256 or 512. "
        "Weights: models/acescg/deepdeband/."
    )

    def deband(
        self,
        image: torch.Tensor,
        model: str,
        strength: float,
        tile_size: int,
        tile_overlap: int,
        download_weights: str = "disable",
        input_gamut: str = "rec709",
        mask: torch.Tensor | None = None,
    ):
        if image.ndim != 4 or image.shape[0] < 1:
            raise ValueError("image must be a non-empty IMAGE batch [N,H,W,C]")
        if float(strength) <= 0.0:
            return (image,)

        model = _as_choice(model, _MODELS, "deepDeband-w")
        download_weights = _as_choice(download_weights, _WEIGHTS, "disable")
        input_gamut = _as_choice(input_gamut, _GAMUTS, "rec709")
        _ = input_gamut

        device = get_device()
        net = load_model(
            model,
            device=device,
            download=(download_weights == "enable"),
        )
        weighted = model == "deepDeband-w"
        height, width = int(image.shape[1]), int(image.shape[2])

        mask_b = None
        if mask is not None:
            mask_b = mask.detach().cpu()
            if mask_b.shape[-2] != height or mask_b.shape[-1] != width:
                mask_b = F.interpolate(
                    mask_b.unsqueeze(1).float(),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            if mask_b.shape[0] == 1 and image.shape[0] > 1:
                mask_b = mask_b.expand(image.shape[0], -1, -1)

        out_frames = []
        for i in range(image.shape[0]):
            frame = image[i].detach().cpu()
            if frame.shape[-1] > 3:
                frame = frame[..., :3]
            elif frame.shape[-1] == 1:
                frame = frame.repeat(1, 1, 3)
            debanded = deband_hwc(
                frame,
                net,
                weighted=weighted,
                tile_size=int(tile_size),
                tile_overlap=int(tile_overlap),
                device=device,
            )
            s = float(strength)
            if s < 1.0:
                debanded = frame * (1.0 - s) + debanded * s
            if mask_b is not None:
                m = mask_b[i].unsqueeze(-1).clamp(0.0, 1.0)
                debanded = frame * (1.0 - m) + debanded * m
            out_frames.append(debanded)

        out = torch.stack(out_frames, dim=0).to(device=image.device, dtype=torch.float32)
        return (out,)
