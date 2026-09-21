"""Deband Edge — edge-protect + spatial dither/diffuse with optional mask."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..core.edge_deband import edge_deband


def _resize_mask(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if mask.shape[-2] == height and mask.shape[-1] == width:
        return mask
    return F.interpolate(
        mask.unsqueeze(1).float(),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)


class AcesCgDebandEdge:
    """Edge Deband: Edge Threshold, Radius, Post Refine, Display Edge + mask."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "edge_threshold": (
                    "FLOAT",
                    {
                        "default": 2.0 / 255.0,
                        "min": 0.0,
                        "max": 0.25,
                        "step": 1.0 / 255.0,
                        "tooltip": (
                            "Raise until Display Edge keeps only banding flats white. "
                            "Units: float delta (~1/255)."
                        ),
                    },
                ),
                "radius": (
                    "FLOAT",
                    {
                        "default": 12.0,
                        "min": 1.0,
                        "max": 64.0,
                        "step": 1.0,
                        "tooltip": "Diffuse sample distance + dither strength.",
                    },
                ),
                "post_refine": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -16.0,
                        "max": 16.0,
                        "step": 1.0,
                        "tooltip": "Widen (+) or narrow (-) the protected-edge belt.",
                    },
                ),
                "display_edge": (
                    ["disable", "enable"],
                    {
                        "default": "disable",
                        "tooltip": (
                            "Show where Deband acts (white) vs protected detail (black). "
                            "Raise edge_threshold until only banding flats stay white."
                        ),
                    },
                ),
                "dither_amount": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 4.0,
                        "step": 0.05,
                        "tooltip": "Extra noise on diffused pixels (LSB scale).",
                    },
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF}),
            },
            "optional": {
                "mask": (
                    "MASK",
                    {
                        "tooltip": "White = apply Deband, black = keep original.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "deband"
    CATEGORY = "AcesCg"
    DESCRIPTION = (
        "Edge-protect deband with spatial diffuse/dither (haasn-style). "
        "Edge Threshold, Radius, Post Refine, Display Edge. Optional mask."
    )

    def deband(
        self,
        image: torch.Tensor,
        edge_threshold: float,
        radius: float,
        post_refine: float,
        display_edge: str = "disable",
        dither_amount: float = 0.5,
        seed: int = 0,
        mask: torch.Tensor | None = None,
    ):
        if image.ndim != 4 or image.shape[0] < 1:
            raise ValueError("image must be a non-empty IMAGE batch [N,H,W,C]")

        show_edge = str(display_edge).strip().lower() in {"enable", "on", "true", "1"}
        frames = image.detach().cpu().numpy()
        height, width = int(frames.shape[1]), int(frames.shape[2])

        mask_np = None
        if mask is not None and not show_edge:
            mask_b = _resize_mask(mask.detach().cpu(), height, width)
            if mask_b.shape[0] == 1 and frames.shape[0] > 1:
                mask_b = mask_b.expand(frames.shape[0], -1, -1)
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

            debanded = edge_deband(
                rgb,
                edge_threshold=float(edge_threshold),
                radius=float(radius),
                post_refine=float(post_refine),
                display_edge=show_edge,
                dither_amount=float(dither_amount),
                seed=int(seed) + i,
            )
            if mask_np is not None:
                m = np.clip(mask_np[i][..., None], 0.0, 1.0)
                debanded = rgb.astype(np.float32) * (1.0 - m) + debanded * m
            out_frames.append(debanded.astype(np.float32, copy=False))

        out = np.stack(out_frames, axis=0)
        return (torch.from_numpy(out).to(device=image.device, dtype=torch.float32),)
