"""Deband Plateau — plateau-fill after OCIO; optional mask to protect texture."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..core.plateau_fill import (
    DEFAULT_EQ_EPS,
    DEFAULT_MIN_RUN,
    DEFAULT_THRESHOLD,
    plateau_fill_rgb,
)


def _resize_mask_to_image(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """MASK [B,H,W] -> [B,H,W] matching image size."""
    if mask.shape[-2] == height and mask.shape[-1] == width:
        return mask
    return F.interpolate(
        mask.unsqueeze(1).float(),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)


class AcesCgDebandPlateau:
    """Plateau-fill deband for scene-linear (or display) float RGB. Mask optional."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": (
                    "FLOAT",
                    {
                        "default": DEFAULT_THRESHOLD,
                        "min": 0.0,
                        "max": 0.25,
                        "step": 0.0001,
                        "tooltip": "Max jump treated as banding (default 1.5/255). Larger = stronger.",
                    },
                ),
                "min_run": (
                    "INT",
                    {
                        "default": DEFAULT_MIN_RUN,
                        "min": 2,
                        "max": 256,
                        "step": 1,
                        "tooltip": "Minimum flat-run length (pixels) before a fill.",
                    },
                ),
                "eq_eps": (
                    "FLOAT",
                    {
                        "default": DEFAULT_EQ_EPS,
                        "min": 0.0,
                        "max": 0.01,
                        "step": 1e-6,
                        "tooltip": "Tolerance for 'same level' inside a plateau.",
                    },
                ),
                "passes": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 8,
                        "step": 1,
                    },
                ),
                "mode": (
                    ["luma", "rgb"],
                    {
                        "default": "luma",
                        "tooltip": "luma = same delta on RGB; rgb = per-channel (can fringe).",
                    },
                ),
            },
            "optional": {
                "mask": (
                    "MASK",
                    {
                        "tooltip": (
                            "White = apply deband, black = keep original. "
                            "Use for sky/gradients; leave sea/detail unmasked."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "deband"
    CATEGORY = "AcesCg"
    DESCRIPTION = (
        "Plateau-fill deband (prefer after OCIO). Optional mask restricts the effect "
        "to flat regions — required for textured areas / video."
    )

    def deband(
        self,
        image: torch.Tensor,
        threshold: float,
        min_run: int,
        eq_eps: float,
        passes: int,
        mode: str,
        mask: torch.Tensor | None = None,
    ):
        if image.ndim != 4 or image.shape[0] < 1:
            raise ValueError("image must be a non-empty IMAGE batch [N,H,W,C]")

        frames = image.detach().cpu().numpy()
        height, width = int(frames.shape[1]), int(frames.shape[2])
        mask_np = None
        if mask is not None:
            mask_b = _resize_mask_to_image(mask.detach().cpu(), height, width)
            if mask_b.shape[0] == 1 and frames.shape[0] > 1:
                mask_b = mask_b.expand(frames.shape[0], -1, -1)
            elif mask_b.shape[0] != frames.shape[0]:
                raise ValueError(
                    f"mask batch {mask_b.shape[0]} != image batch {frames.shape[0]}"
                )
            mask_np = mask_b.numpy().astype(np.float32)

        out_frames = []
        for i in range(frames.shape[0]):
            rgb = frames[i]
            if rgb.ndim == 2:
                rgb = np.stack([rgb, rgb, rgb], axis=-1)
            elif rgb.shape[-1] == 1:
                rgb = np.repeat(rgb, 3, axis=-1)
            elif rgb.shape[-1] > 3:
                rgb = rgb[..., :3]

            debanded = plateau_fill_rgb(
                rgb,
                threshold=float(threshold),
                min_run=int(min_run),
                eq_eps=float(eq_eps),
                passes=int(passes),
                mode=mode,
            )
            if mask_np is not None:
                m = mask_np[i][..., None]
                m = np.clip(m, 0.0, 1.0)
                debanded = rgb.astype(np.float32) * (1.0 - m) + debanded * m
            out_frames.append(debanded.astype(np.float32, copy=False))

        out = np.stack(out_frames, axis=0)
        return (torch.from_numpy(out).to(device=image.device, dtype=torch.float32),)
