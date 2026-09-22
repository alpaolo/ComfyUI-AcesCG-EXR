"""Write OpenEXR RGB via PyAV (float32). No view transform / OETF."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import numpy as np


def write_exr_rgb(path: Path, rgb: np.ndarray) -> Path:
    """Write HxWx3 float RGB as OpenEXR float32. Creates parent dirs."""
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"Expected HxWx3+ RGB, got {rgb.shape}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    img = np.ascontiguousarray(rgb[..., :3], dtype=np.float32)

    codec = av.CodecContext.create("exr", "w")
    codec.width = width
    codec.height = height
    codec.pix_fmt = "gbrpf32le"
    codec.time_base = Fraction(1, 1)

    frame = av.VideoFrame.from_ndarray(img, format="gbrpf32le")
    frame.pts = 0
    frame.time_base = codec.time_base
    packets = list(codec.encode(frame)) + list(codec.encode(None))
    path.write_bytes(b"".join(bytes(p) for p in packets))
    return path


def padded_exr_name(prefix: str, frame_index: int, padding: int) -> str:
    """Build name like 'plate.0001.exr'."""
    stem = (prefix or "Aipermedia-AcesCg").strip().replace("\\", "/").split("/")[-1]
    if stem.lower().endswith(".exr"):
        stem = stem[:-4]
    pad = max(1, int(padding))
    return f"{stem}.{int(frame_index):0{pad}d}.exr"
