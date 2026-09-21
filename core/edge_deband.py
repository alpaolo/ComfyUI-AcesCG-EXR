"""Edge Deband — edge-protect + spatial dither/diffuse (haasn-style).

Protect strong edges; diffuse flat/banded regions with thresholded neighbor
averages and light dither. Optional display of the active (non-protected) mask.
"""

from __future__ import annotations

import numpy as np


_LUMA_R = 0.2126
_LUMA_G = 0.7152
_LUMA_B = 0.0722


def _luma(rgb: np.ndarray) -> np.ndarray:
    return (
        rgb[..., 0] * _LUMA_R + rgb[..., 1] * _LUMA_G + rgb[..., 2] * _LUMA_B
    ).astype(np.float32, copy=False)


def _edge_strength(rgb: np.ndarray) -> np.ndarray:
    """Per-pixel edge strength: max abs neighbor delta on luma (float [0,..])."""
    y = _luma(rgb)
    dx = np.abs(np.diff(y, axis=1, prepend=y[:, :1]))
    dy = np.abs(np.diff(y, axis=0, prepend=y[:1, :]))
    return np.maximum(dx, dy)


def _morph_binary(mask: np.ndarray, amount: int) -> np.ndarray:
    """Dilate (amount>0) or erode (amount<0) a bool/float mask with box iterations."""
    if amount == 0:
        return mask.astype(np.float32, copy=False)
    out = mask.astype(np.float32)
    steps = abs(int(amount))
    grow = amount > 0
    for _ in range(steps):
        pad = np.pad(out, 1, mode="edge")
        neigh = np.stack(
            [
                pad[:-2, :-2],
                pad[:-2, 1:-1],
                pad[:-2, 2:],
                pad[1:-1, :-2],
                pad[1:-1, 1:-1],
                pad[1:-1, 2:],
                pad[2:, :-2],
                pad[2:, 1:-1],
                pad[2:, 2:],
            ],
            axis=0,
        )
        out = neigh.max(axis=0) if grow else neigh.min(axis=0)
    return out


def edge_deband(
    rgb: np.ndarray,
    *,
    edge_threshold: float = 2.0 / 255.0,
    radius: float = 12.0,
    post_refine: float = 0.0,
    display_edge: bool = False,
    dither_amount: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """HxWx3 float RGB -> debanded HxWx3 (or active-area preview if display_edge)."""
    src = np.ascontiguousarray(rgb[..., :3], dtype=np.float32)
    h, w, _ = src.shape
    edge = _edge_strength(src)

    thr = max(0.0, float(edge_threshold))
    protected = edge > thr

    refine_px = int(round(float(post_refine)))
    if refine_px != 0:
        protected = _morph_binary(protected.astype(np.float32), refine_px) > 0.5

    if display_edge:
        # White = regions that will be dithered/diffused; black = protected detail.
        act = (~protected).astype(np.float32)
        return np.stack([act, act, act], axis=-1)

    r_max = max(1, int(round(float(radius))))
    diffuse_thr = max(thr, 1.0 / 255.0) * 1.0

    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    angles = rng.random((h, w), dtype=np.float32) * (2.0 * np.pi)
    dists = rng.random((h, w), dtype=np.float32) * float(r_max)

    yy, xx = np.meshgrid(np.arange(h, dtype=np.int32), np.arange(w, dtype=np.int32), indexing="ij")

    acc = np.zeros_like(src)
    valid = np.zeros((h, w), dtype=np.float32)
    for k in range(4):
        ang = angles + (0.5 * np.pi * k)
        dx = np.rint(dists * np.cos(ang)).astype(np.int32)
        dy = np.rint(dists * np.sin(ang)).astype(np.int32)
        sx = np.clip(xx + dx, 0, w - 1)
        sy = np.clip(yy + dy, 0, h - 1)
        sample = src[sy, sx]
        delta = np.max(np.abs(sample - src), axis=-1)
        ok = (delta <= diffuse_thr).astype(np.float32)
        acc += sample * ok[..., None]
        valid += ok

    avg = np.where(valid[..., None] > 0.0, acc / np.maximum(valid[..., None], 1.0), src)

    flat = (~protected) & (valid >= 3.0)
    out = src.copy()
    out[flat] = avg[flat]

    amp = float(dither_amount) * (float(radius) / 12.0) * (1.0 / 255.0)
    if amp > 0.0:
        noise = (rng.random(src.shape, dtype=np.float32) - 0.5) * (2.0 * amp)
        out = np.where(flat[..., None], out + noise, out)

    return np.ascontiguousarray(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float32)
