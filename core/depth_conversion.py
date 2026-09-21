"""Depth conversion: N-bit display codes -> float32 [0,1] only (no fill/dither)."""

from __future__ import annotations

import numpy as np


def max_code_for_bits(bits: int) -> int:
    return (1 << int(bits)) - 1


def display_lsb(bits: int) -> float:
    return 1.0 / float(max_code_for_bits(bits))


def promote_to_float01(rgb: np.ndarray, input_bits: int) -> tuple[np.ndarray, str]:
    """Promote display codes or float buffer to float32 [0,1].

    Integer / code-range (max > 1.5) -> divide by (2^input_bits-1).
    Already-normalized float left as float32.
    """
    src = np.ascontiguousarray(rgb[..., :3])
    bits = int(input_bits)
    if bits not in (8, 16):
        bits = 8
    lsb = display_lsb(bits)
    max_code = float(max_code_for_bits(bits))

    if src.dtype == np.uint8:
        return src.astype(np.float32) * (1.0 / 255.0), "uint8->float01"
    if src.dtype == np.uint16:
        return src.astype(np.float32) * (1.0 / 65535.0), "uint16->float01"
    if np.issubdtype(src.dtype, np.integer):
        return src.astype(np.float32) * lsb, f"{src.dtype.name}->float01/{bits}"

    out = np.ascontiguousarray(src, dtype=np.float32)
    peak = float(np.max(out)) if out.size else 0.0
    if peak > 1.5:
        return out / max_code, f"code_float->float01/{bits}"
    return out, "float01"


def apply_depth_conversion(rgb: np.ndarray, *, input_bits: int = 8) -> tuple[np.ndarray, str]:
    """Return float32 HxWx3 display [0,1] and a short note string."""
    bits = 8 if int(input_bits) != 16 else 16
    out, promote_note = promote_to_float01(rgb, bits)
    out = np.ascontiguousarray(out, dtype=np.float32)
    return out, f"depth {promote_note} -> float32"
