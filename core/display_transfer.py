"""Display transfer helpers for preview (linear vs BT.709 OETF)."""

from __future__ import annotations

import torch

# BT.709 / Rec.709 OETF constants (IEC 61966-2-1 / BT.709).
_BT709_A = 1.099
_BT709_B = 0.099
_BT709_CUT = 0.018
_BT709_SLOPE = 4.5
_BT709_POWER = 0.45


def apply_preview_display(
    images: torch.Tensor,
    *,
    display: str = "linear",
    exposure: float = 1.0,
) -> torch.Tensor:
    """BHWC float -> BHWC float [0,1] for UI encode.

    linear: gain + clamp, no OETF (technical view).
    rec709: gain + BT.709 OETF + clamp (scene-linear -> display codes).
    """
    mode = str(display or "linear").strip().lower()
    x = images.detach().float() * float(exposure)
    if mode in ("rec709", "bt709", "709"):
        x = x.clamp(0.0, 1.0)
        low = _BT709_SLOPE * x
        high = _BT709_A * torch.pow(x.clamp(min=0.0), _BT709_POWER) - _BT709_B
        return torch.where(x < _BT709_CUT, low, high).clamp(0.0, 1.0)
    return x.clamp(0.0, 1.0)
