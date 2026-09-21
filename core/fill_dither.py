"""Fill + dither on float32 RGB (Rec.709 display or scene-linear)."""

from __future__ import annotations

from typing import Any

import numpy as np

MAX_FILL_BITS = 32
MIN_FILL_BITS = 8
MAX_QUANTIZE_BITS = 24
STAIR_SOFTEN_PASSES = 2
STAIR_RAMP_RADIUS = 8
STAIR_MIN_STEP_LSBS = 0.35
STAIR_MAX_STEP_LSBS = 2.5
_LUMA_R = 0.2126
_LUMA_G = 0.7152
_LUMA_B = 0.0722


def max_code_for_bits(bits: int) -> int:
    return (1 << int(bits)) - 1


def display_lsb(bits: int) -> float:
    return 1.0 / float(max_code_for_bits(bits))


def clamp_fill_bits(fill_bits: int) -> int:
    bits = int(fill_bits)
    if bits < MIN_FILL_BITS:
        return MIN_FILL_BITS
    if bits > MAX_FILL_BITS:
        return MAX_FILL_BITS
    return bits


def simulation_is_fill(value: Any) -> bool:
    text = str(value or "none").strip().lower()
    return text in {"fill", "on", "true", "1", "yes"}


def dither_enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "off").strip().lower()
    return text in {"on", "true", "1", "yes", "triangular", "uniform"}


def quantize_display_bits(rgb: np.ndarray, bits: int) -> np.ndarray:
    max_code = float(max_code_for_bits(bits))
    return np.round(rgb * max_code) / max_code


def _smoothstep(t: float) -> float:
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return t * t * (3.0 - 2.0 * t)


def _soften_stairs_1d(
    line: np.ndarray,
    lsb: float,
    *,
    radius: int = STAIR_RAMP_RADIUS,
    min_step_lsbs: float = STAIR_MIN_STEP_LSBS,
    max_step_lsbs: float = STAIR_MAX_STEP_LSBS,
) -> np.ndarray:
    n = int(line.shape[0])
    src = np.asarray(line, dtype=np.float32)
    out = src.copy()
    if n < 2:
        return out

    lo = max(0.0, float(min_step_lsbs)) * lsb
    hi = max(lo, float(max_step_lsbs)) * lsb
    diff = np.abs(np.diff(src))
    edges = np.flatnonzero((diff >= lo) & (diff <= hi))
    if edges.size == 0:
        return out

    radius = max(1, int(radius))
    for e in edges:
        left = max(0, int(e) - radius + 1)
        right = min(n - 1, int(e) + radius)
        span = right - left
        if span < 1:
            continue
        v0 = float(src[left])
        v1 = float(src[right])
        if abs(v1 - v0) > hi * 1.25:
            continue
        inv = 1.0 / float(span)
        for k in range(left, right + 1):
            t = _smoothstep(float(k - left) * inv)
            out[k] = v0 + (v1 - v0) * t
    return out


def _soften_stairs_pass(
    channel: np.ndarray,
    lsb: float,
    *,
    radius: int,
    min_step_lsbs: float,
    max_step_lsbs: float,
) -> np.ndarray:
    h, w = channel.shape
    out = np.empty_like(channel, dtype=np.float32)
    kwargs = {
        "radius": radius,
        "min_step_lsbs": min_step_lsbs,
        "max_step_lsbs": max_step_lsbs,
    }
    for y in range(h):
        out[y, :] = _soften_stairs_1d(channel[y, :], lsb, **kwargs)
    for x in range(w):
        out[:, x] = _soften_stairs_1d(out[:, x], lsb, **kwargs)
    return out


def soften_stairs(
    rgb: np.ndarray,
    *,
    ref_bits: int = 8,
    passes: int = STAIR_SOFTEN_PASSES,
    ramp_radius: int = STAIR_RAMP_RADIUS,
    min_step_lsbs: float = STAIR_MIN_STEP_LSBS,
    max_step_lsbs: float = STAIR_MAX_STEP_LSBS,
    soften_mode: str = "luma",
) -> np.ndarray:
    """HxWx3 stair soften. ref_bits sets LSB scale. mode=luma|rgb. No clip."""
    bits = 8 if int(ref_bits) != 16 else 16
    lsb = display_lsb(bits)
    out = np.ascontiguousarray(rgb[..., :3], dtype=np.float32).copy()
    radius = max(1, int(ramp_radius))
    mode = str(soften_mode or "luma").strip().lower()
    kwargs = {
        "radius": radius,
        "min_step_lsbs": float(min_step_lsbs),
        "max_step_lsbs": float(max_step_lsbs),
    }

    if mode == "rgb":
        for _ in range(max(1, int(passes))):
            for c in range(3):
                out[..., c] = _soften_stairs_pass(out[..., c], lsb, **kwargs)
        return out

    for _ in range(max(1, int(passes))):
        y = (
            out[..., 0] * _LUMA_R + out[..., 1] * _LUMA_G + out[..., 2] * _LUMA_B
        ).astype(np.float32, copy=False)
        y_soft = _soften_stairs_pass(y, lsb, **kwargs)
        delta = y_soft - y
        out[..., 0] = out[..., 0] + delta
        out[..., 1] = out[..., 1] + delta
        out[..., 2] = out[..., 2] + delta
    return out


def apply_dither(
    rgb: np.ndarray,
    *,
    lsb: float,
    dither: str = "off",
    dither_amount: float = 0.5,
    rng: np.random.Generator | None = None,
    clip_01: bool = True,
) -> tuple[np.ndarray, str]:
    out = np.ascontiguousarray(rgb[..., :3], dtype=np.float32)
    amount_steps = float(dither_amount)
    if not dither_enabled(dither) or amount_steps <= 0.0:
        return out, "off"

    if rng is None:
        rng = np.random.default_rng()
    mode = str(dither or "on").strip().lower()
    if mode in {"uniform", "rect"}:
        noise = (rng.random(out.shape, dtype=np.float32) - 0.5) * (2.0 * amount_steps * lsb)
        out = out + noise
        note = f"on uniform amount={amount_steps}"
    else:
        u1 = rng.random(out.shape, dtype=np.float32) - 0.5
        u2 = rng.random(out.shape, dtype=np.float32) - 0.5
        out = out + (u1 + u2) * (amount_steps * lsb)
        note = f"on triangular amount={amount_steps}"

    if clip_01:
        out = np.clip(out, 0.0, 1.0)
    return out, note


def apply_fill_dither(
    rgb: np.ndarray,
    *,
    working_space: str = "rec709",
    ref_bits: int = 8,
    simulation: str = "fill",
    fill_bits: int = 16,
    ramp_radius: int = STAIR_RAMP_RADIUS,
    min_step_lsbs: float = STAIR_MIN_STEP_LSBS,
    max_step_lsbs: float = STAIR_MAX_STEP_LSBS,
    soften_passes: int = STAIR_SOFTEN_PASSES,
    soften_mode: str = "luma",
    dither: str = "off",
    dither_amount: float = 0.5,
    mask: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, str]:
    """Fill (stair soften) + optional dither. working_space=rec709|linear."""
    space = str(working_space or "rec709").strip().lower()
    if space in {"linear", "scene", "aces", "ap0", "ap1"}:
        space = "linear"
    else:
        space = "rec709"
    clip_01 = space == "rec709"

    bits = 8 if int(ref_bits) != 16 else 16
    lsb = display_lsb(bits)
    src = np.ascontiguousarray(rgb[..., :3], dtype=np.float32)
    out = src.copy()

    fill_note = "simulation=none"
    if simulation_is_fill(simulation):
        mode = str(soften_mode or "luma").strip().lower()
        if mode not in ("luma", "rgb"):
            mode = "luma"
        n_fill = clamp_fill_bits(fill_bits)
        softened = soften_stairs(
            out,
            ref_bits=bits,
            passes=int(soften_passes),
            ramp_radius=int(ramp_radius),
            min_step_lsbs=float(min_step_lsbs),
            max_step_lsbs=float(max_step_lsbs),
            soften_mode=mode,
        )
        # Grid snap only in display Rec.709; scene-linear keeps continuous ramps.
        if space == "rec709" and n_fill > bits and n_fill <= MAX_QUANTIZE_BITS:
            softened = quantize_display_bits(softened, n_fill)
        if clip_01:
            softened = np.clip(softened, 0.0, 1.0)
        out = softened
        fill_note = (
            f"simulation=fill space={space} bits={n_fill} mode={mode} "
            f"r={int(ramp_radius)} min={float(min_step_lsbs):.2f} "
            f"max={float(max_step_lsbs):.2f}"
        )

    out, dither_note = apply_dither(
        out,
        lsb=lsb,
        dither=dither,
        dither_amount=dither_amount,
        rng=rng,
        clip_01=clip_01,
    )

    if mask is not None:
        m = np.asarray(mask, dtype=np.float32)
        if m.ndim == 2:
            m = m[..., None]
        m = np.clip(m, 0.0, 1.0)
        out = src * (1.0 - m) + out * m

    out = np.ascontiguousarray(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float32)
    note = f"fill_dither {fill_note} dither={dither_note}"
    return out, note
