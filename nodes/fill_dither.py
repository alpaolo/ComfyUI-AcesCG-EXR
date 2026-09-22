"""Fill Dither — stair soften + dither on Rec.709 or scene-linear float32."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..core.fill_dither import (
    STAIR_MAX_STEP_LSBS,
    STAIR_MIN_STEP_LSBS,
    STAIR_RAMP_RADIUS,
    STAIR_SOFTEN_PASSES,
    apply_fill_dither,
    clamp_fill_bits,
)


def _as_int(value, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        number = int(default)
    return max(minimum, min(maximum, number))


def _as_float(value, default: float, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float(default)
    return max(minimum, min(maximum, number))


def _as_choice(value, allowed: tuple[str, ...], default: str) -> str:
    text = str(value or "").strip().lower()
    for item in allowed:
        if text == item.lower():
            return item
    return default


def _resize_mask(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if mask.shape[-2] == height and mask.shape[-1] == width:
        return mask
    return F.interpolate(
        mask.unsqueeze(1).float(),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)


class AcesCgFillDither:
    """Stair fill + dither; place after Depth (Rec.709) or after OCIO (linear)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "working_space": (
                    ["rec709", "linear"],
                    {
                        "default": "rec709",
                        "tooltip": (
                            "rec709 = display float [0,1] (post Depth). "
                            "linear = scene-linear float (post OCIO); no [0,1] clip, no 16-bit snap."
                        ),
                    },
                ),
                "ref_bits": (
                    ["8", "16"],
                    {
                        "default": "8",
                        "tooltip": "LSB reference for step thresholds and dither amount.",
                    },
                ),
                "simulation": (["none", "fill"], {"default": "fill"}),
                "fill": (
                    ["16", "32"],
                    {
                        "default": "32",
                        "tooltip": "rec709 only: 16 snaps to 16-bit grid; 32 = continuous. Ignored snap in linear.",
                    },
                ),
                "soften_mode": (
                    ["luma", "rgb"],
                    {
                        "default": "luma",
                        "tooltip": "luma = same delta on RGB; rgb = per-channel.",
                    },
                ),
                "ramp_radius": (
                    "INT",
                    {
                        "default": STAIR_RAMP_RADIUS,
                        "min": 1,
                        "max": 64,
                        "step": 1,
                    },
                ),
                "min_step_lsbs": (
                    "FLOAT",
                    {
                        "default": STAIR_MIN_STEP_LSBS,
                        "min": 0.0,
                        "max": 8.0,
                        "step": 0.05,
                    },
                ),
                "max_step_lsbs": (
                    "FLOAT",
                    {
                        "default": STAIR_MAX_STEP_LSBS,
                        "min": 0.1,
                        "max": 16.0,
                        "step": 0.05,
                    },
                ),
                "soften_passes": (
                    "INT",
                    {
                        "default": STAIR_SOFTEN_PASSES,
                        "min": 1,
                        "max": 8,
                        "step": 1,
                    },
                ),
                "dither": (["off", "on"], {"default": "off"}),
                "dither_amount": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 4.0,
                        "step": 0.05,
                    },
                ),
            },
            "optional": {
                "mask": (
                    "MASK",
                    {
                        "tooltip": "White = apply fill/dither, black = keep original.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    CATEGORY = "Aipermedia-AcesCg"
    DESCRIPTION = (
        "Stair soften + dither on float32. working_space=rec709 (display) or linear "
        "(scene). Optional mask. Prefer masks on textured areas; ffmpeg deband is "
        "video/8-bit oriented — not used here for float ACES plates."
    )

    def process(
        self,
        image: torch.Tensor,
        working_space: str,
        ref_bits: str,
        simulation: str,
        fill: str,
        soften_mode: str,
        ramp_radius: int,
        min_step_lsbs: float,
        max_step_lsbs: float,
        soften_passes: int,
        dither: str,
        dither_amount: float,
        mask: torch.Tensor | None = None,
    ):
        if image.ndim != 4 or image.shape[0] < 1:
            raise ValueError("image must be a non-empty IMAGE batch [N,H,W,C]")

        working_space = _as_choice(working_space, ("rec709", "linear"), "rec709")
        bits = int(_as_choice(ref_bits, ("8", "16"), "8"))
        simulation = _as_choice(simulation, ("none", "fill"), "fill")
        fill_bits = clamp_fill_bits(int(_as_choice(fill, ("16", "32"), "32")))
        soften_mode = _as_choice(soften_mode, ("luma", "rgb"), "luma")
        dither = _as_choice(dither, ("off", "on"), "off")
        ramp_radius = _as_int(ramp_radius, STAIR_RAMP_RADIUS, 1, 64)
        soften_passes = _as_int(soften_passes, STAIR_SOFTEN_PASSES, 1, 8)
        min_step_lsbs = _as_float(min_step_lsbs, STAIR_MIN_STEP_LSBS, 0.0, 8.0)
        max_step_lsbs = _as_float(max_step_lsbs, STAIR_MAX_STEP_LSBS, 0.1, 16.0)
        dither_amount = _as_float(dither_amount, 0.5, 0.0, 4.0)

        frames = image.detach().cpu().numpy()
        height, width = int(frames.shape[1]), int(frames.shape[2])
        mask_np = None
        if mask is not None:
            mask_b = _resize_mask(mask.detach().cpu(), height, width)
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
            frame_mask = None if mask_np is None else mask_np[i]
            out, _note = apply_fill_dither(
                rgb,
                working_space=working_space,
                ref_bits=bits,
                simulation=simulation,
                fill_bits=fill_bits,
                ramp_radius=ramp_radius,
                min_step_lsbs=min_step_lsbs,
                max_step_lsbs=max_step_lsbs,
                soften_passes=soften_passes,
                soften_mode=soften_mode,
                dither=dither,
                dither_amount=dither_amount,
                mask=frame_mask,
            )
            out_frames.append(out)

        stacked = np.stack(out_frames, axis=0)
        return (torch.from_numpy(stacked).to(device=image.device, dtype=torch.float32),)
