"""Plateau-fill deband (display or scene-linear). Optional luma-only path."""

from __future__ import annotations

import numpy as np

DEFAULT_EQ_EPS = 1e-5
DEFAULT_MIN_RUN = 2
DEFAULT_THRESHOLD = 1.5 / 255.0
_LUMA_R = 0.2126
_LUMA_G = 0.7152
_LUMA_B = 0.0722


def _fill_plateaus_1d(
    line: np.ndarray,
    *,
    tau: float,
    min_run: int,
    eq_eps: float,
) -> np.ndarray:
    n = int(line.shape[0])
    if n < 2:
        return line
    out = line.copy()
    i = 0
    while i < n:
        level = float(line[i])
        j = i + 1
        while j < n and abs(float(line[j]) - level) <= eq_eps:
            j += 1
        run_len = j - i
        if j < n and run_len >= min_run:
            right = float(line[j])
            delta = abs(right - level)
            if delta < tau and delta > eq_eps:
                left = float(line[i - 1]) if i > 0 else level
                denom = float(run_len + 1)
                for k in range(i, j):
                    t = float(k - i + 1) / denom
                    out[k] = left + (right - left) * t
        i = j
    return out


def _pass_horizontal(channel: np.ndarray, tau: float, min_run: int, eq_eps: float) -> np.ndarray:
    h, w = channel.shape
    out = np.empty_like(channel)
    for y in range(h):
        out[y, :] = _fill_plateaus_1d(channel[y, :], tau=tau, min_run=min_run, eq_eps=eq_eps)
    return out


def _pass_vertical(channel: np.ndarray, tau: float, min_run: int, eq_eps: float) -> np.ndarray:
    h, w = channel.shape
    out = np.empty_like(channel)
    for x in range(w):
        out[:, x] = _fill_plateaus_1d(channel[:, x], tau=tau, min_run=min_run, eq_eps=eq_eps)
    return out


def _soften_channel(
    channel: np.ndarray,
    *,
    tau: float,
    min_run: int,
    eq_eps: float,
    passes: int,
) -> np.ndarray:
    out = np.ascontiguousarray(channel, dtype=np.float32)
    for _ in range(max(1, int(passes))):
        out = _pass_horizontal(out, tau, min_run, eq_eps)
        out = _pass_vertical(out, tau, min_run, eq_eps)
    return out


def plateau_fill_rgb(
    rgb: np.ndarray,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    min_run: int = DEFAULT_MIN_RUN,
    eq_eps: float = DEFAULT_EQ_EPS,
    passes: int = 1,
    mode: str = "luma",
) -> np.ndarray:
    """HxWx3 plateau fill. mode=luma|rgb. No clip."""
    src = np.ascontiguousarray(rgb[..., :3], dtype=np.float32)
    out = src.copy()
    tau = float(threshold)
    min_run = max(2, int(min_run))
    eq_eps = float(eq_eps)
    passes = max(1, int(passes))
    mode = str(mode or "luma").strip().lower()

    if mode == "rgb":
        for c in range(3):
            out[..., c] = _soften_channel(
                out[..., c], tau=tau, min_run=min_run, eq_eps=eq_eps, passes=passes
            )
        return out

    y = (out[..., 0] * _LUMA_R + out[..., 1] * _LUMA_G + out[..., 2] * _LUMA_B).astype(
        np.float32, copy=False
    )
    y_soft = _soften_channel(y, tau=tau, min_run=min_run, eq_eps=eq_eps, passes=passes)
    delta = y_soft - y
    out[..., 0] = out[..., 0] + delta
    out[..., 1] = out[..., 1] + delta
    out[..., 2] = out[..., 2] + delta
    return out
