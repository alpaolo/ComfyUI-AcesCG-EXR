"""OCIO Convert — Rec.709 display -> linearized AP0/AP1 (ACES 1.3 / 1.5 / 2.0)."""

from __future__ import annotations

import numpy as np
import torch

from ..core.ocio_convert import ocio_convert_rgb


class AcesCgOCIOConvert:
    """Fixed Rec.709 (sRGB-encoded) input; OCIO linearize + gamut to AP0 or AP1."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "output_gamut": (["AP0", "AP1"], {"default": "AP0"}),
                "aces_version": (["1.3", "1.5", "2.0"], {"default": "2.0"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("scene_linear",)
    FUNCTION = "convert"
    CATEGORY = "AcesCg"
    DESCRIPTION = (
        "OCIO: Rec.709/sRGB-encoded display -> scene-linear AP0 (ACES2065-1) or AP1 (ACEScg). "
        "ACES 1.5 uses the 1.3 CG config (no official 1.5 CG OCIO). No view/ODT."
    )

    def convert(self, image: torch.Tensor, output_gamut: str, aces_version: str):
        if image.ndim != 4 or image.shape[0] < 1:
            raise ValueError("image must be a non-empty IMAGE batch [N,H,W,C]")

        frames = image.detach().cpu().numpy()
        out_frames = []
        for i in range(frames.shape[0]):
            rgb = frames[i]
            if rgb.ndim == 2:
                rgb = np.stack([rgb, rgb, rgb], axis=-1)
            elif rgb.shape[-1] == 1:
                rgb = np.repeat(rgb, 3, axis=-1)
            converted, _route = ocio_convert_rgb(
                rgb,
                aces_version=aces_version,
                output_gamut=output_gamut,
            )
            out_frames.append(converted)

        out = np.stack(out_frames, axis=0)
        return (torch.from_numpy(out).to(device=image.device, dtype=torch.float32),)
