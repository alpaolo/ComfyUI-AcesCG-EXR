"""Inspect Image — print float IMAGE stats (normalization / stair check)."""

from __future__ import annotations

import numpy as np
import torch


def _report_frame(rgb: np.ndarray, frame_index: int) -> str:
    x = np.ascontiguousarray(rgb[..., :3], dtype=np.float32)
    h, w, _ = x.shape
    flat = x.reshape(-1, 3)
    lines = [
        f"[AcesCg Inspect] frame={frame_index} shape=HxWxC={h}x{w}x3 dtype=float32",
        (
            f"  min={flat.min(axis=0)} max={flat.max(axis=0)} "
            f"mean={flat.mean(axis=0)}"
        ),
        (
            f"  in_[0,1]={bool(flat.min() >= 0.0 and flat.max() <= 1.0)} "
            f"peak={float(flat.max())} below0={int(np.count_nonzero(flat < 0))} "
            f"above1={int(np.count_nonzero(flat > 1.0))}"
        ),
    ]

    # Unique levels on luma (capped sample for speed on huge frames).
    y = (x[..., 0] * 0.2126 + x[..., 1] * 0.7152 + x[..., 2] * 0.0722).ravel()
    if y.size > 2_000_000:
        step = max(1, y.size // 2_000_000)
        y_s = y[::step]
    else:
        y_s = y
    # Quantize to 1e-6 bins so float noise does not explode unique count.
    y_q = np.round(y_s * 1_000_000.0).astype(np.int64)
    n_unique = int(np.unique(y_q).size)
    lines.append(f"  luma_unique_approx={n_unique} (8bit display ~<=256)")

    # Neighbor |diff| on a mid row (stair scale check).
    mid = x[h // 2, :, 1]
    d = np.abs(np.diff(mid))
    d_pos = d[d > 1e-8]
    if d_pos.size:
        lines.append(
            f"  mid_row_G |diff| min={float(d_pos.min()):.8f} "
            f"median={float(np.median(d_pos)):.8f} max={float(d_pos.max()):.8f} "
            f"(1/255={1.0 / 255.0:.8f})"
        )
    else:
        lines.append("  mid_row_G |diff|: no nonzero steps")

    cy, cx = h // 2, w // 2
    y0, x0 = max(0, cy - 2), max(0, cx - 2)
    patch = x[y0 : y0 + 4, x0 : x0 + 4, :]
    lines.append(f"  center_4x4_RGB=\n{np.array2string(patch, precision=6, suppress_small=False)}")
    return "\n".join(lines)


class AcesCgInspectImage:
    """Passthrough IMAGE; prints normalization / stair stats to the Comfy console."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "label": (
                    "STRING",
                    {
                        "default": "inspect",
                        "tooltip": "Tag printed in the console line.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "report")
    FUNCTION = "inspect"
    CATEGORY = "AcesCg"
    DESCRIPTION = (
        "Print float IMAGE stats (shape, min/max, [0,1] check, luma unique count, "
        "neighbor diffs, center 4x4). Full HxWx3 dump would flood the log."
    )
    OUTPUT_NODE = True

    def inspect(self, image: torch.Tensor, label: str = "inspect"):
        if image.ndim != 4 or image.shape[0] < 1:
            raise ValueError("image must be a non-empty IMAGE batch [N,H,W,C]")

        chunks = [f"[AcesCg Inspect] label={label!r} batch={int(image.shape[0])}"]
        frames = image.detach().cpu().float().numpy()
        for i in range(frames.shape[0]):
            rgb = frames[i]
            if rgb.ndim == 2:
                rgb = np.stack([rgb, rgb, rgb], axis=-1)
            elif rgb.shape[-1] == 1:
                rgb = np.repeat(rgb, 3, axis=-1)
            chunks.append(_report_frame(rgb, i))

        report = "\n".join(chunks)
        print(report, flush=True)
        return (image, report)
