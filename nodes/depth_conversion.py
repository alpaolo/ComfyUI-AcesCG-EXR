"""Depth Conversion — only N-bit -> float32 [0,1] (no fill/dither)."""

from __future__ import annotations

import numpy as np
import torch

from ..core.depth_conversion import apply_depth_conversion


def _as_choice(value, allowed: tuple[str, ...], default: str) -> str:
    text = str(value or "").strip().lower()
    for item in allowed:
        if text == item.lower():
            return item
    return default


class AcesCgDepthConversion:
    """Normalize 8/16-bit image or matrix to float32 [0,1]."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "input_bits": (["8", "16"], {"default": "8"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("display_float",)
    FUNCTION = "convert"
    CATEGORY = "Aipermedia-AcesCg"
    DESCRIPTION = (
        "Promote 8/16-bit display codes to float32 [0,1]. "
        "No fill/dither — use Fill Dither (and/or Deband Edge) as separate nodes."
    )

    def convert(self, image: torch.Tensor, input_bits: str):
        bits = int(_as_choice(input_bits, ("8", "16"), "8"))
        frames = image.detach().cpu().numpy()
        out_frames = []
        for i in range(frames.shape[0]):
            rgb, _note = apply_depth_conversion(frames[i], input_bits=bits)
            if rgb.ndim == 2:
                rgb = np.stack([rgb, rgb, rgb], axis=-1)
            elif rgb.shape[-1] == 1:
                rgb = np.repeat(rgb, 3, axis=-1)
            elif rgb.shape[-1] > 3:
                rgb = rgb[..., :3]
            out_frames.append(rgb)

        out = np.stack(out_frames, axis=0)
        return (torch.from_numpy(out).to(device=image.device, dtype=torch.float32),)
