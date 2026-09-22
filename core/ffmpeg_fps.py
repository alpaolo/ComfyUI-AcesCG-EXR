"""FFmpeg helpers for frame-rate retiming (speed-change, not frame blend).

Rec.709 display mp4/mov in → Rec.709 display out. Timing filters only.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av

# UI labels -> exact rates (NTSC as 1001 fractions).
FPS_RATE_OPTIONS: list[str] = [
    "23.976",
    "24",
    "25",
    "29.97",
    "30",
    "48",
    "50",
    "59.94",
    "60",
    "120",
]

_FPS_RATES: dict[str, Fraction] = {
    "23.976": Fraction(24000, 1001),
    "24": Fraction(24, 1),
    "25": Fraction(25, 1),
    "29.97": Fraction(30000, 1001),
    "30": Fraction(30, 1),
    "48": Fraction(48, 1),
    "50": Fraction(50, 1),
    "59.94": Fraction(60000, 1001),
    "60": Fraction(60, 1),
    "120": Fraction(120, 1),
}

# Color tags copied onto the encode (no pixel gamut conversion).
@dataclass(frozen=True)
class VideoColorTags:
    primaries: str | None = None
    transfer: str | None = None
    colorspace: str | None = None
    range: str | None = None
    pix_fmt: str | None = None


def parse_fps_rate(label: str | float | int | Fraction) -> Fraction:
    if isinstance(label, Fraction):
        if label <= 0:
            raise ValueError(f"Invalid fps rate: {label!r}")
        return label
    text = str(label).strip()
    if text in _FPS_RATES:
        return _FPS_RATES[text]
    aliases = {
        "23.98": "23.976",
        "29.970": "29.97",
        "29.997": "29.97",
        "59.940": "59.94",
    }
    if text in aliases:
        return _FPS_RATES[aliases[text]]
    value = Fraction(text).limit_denominator(1001)
    if value <= 0:
        raise ValueError(f"Invalid fps rate: {label!r}")
    return value


def find_ffmpeg() -> str:
    which = shutil.which("ffmpeg")
    if which:
        return which
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError(
            "ffmpeg not found (PATH or imageio_ffmpeg). Install ffmpeg or imageio-ffmpeg."
        ) from exc


def has_audio_stream(path: str | Path) -> bool:
    with av.open(str(path), mode="r") as container:
        return len(container.streams.audio) > 0


def _enum_name(value) -> str | None:
    if value is None:
        return None
    name = getattr(value, "name", None)
    if isinstance(name, str) and name and name.upper() not in {"UNSPECIFIED", "RESERVED", "NB"}:
        return name.lower()
    text = str(value).strip().lower()
    if not text or text in {"unspecified", "unknown", "none"}:
        return None
    return text


def probe_video_color(path: str | Path) -> VideoColorTags:
    with av.open(str(path), mode="r") as container:
        if not container.streams.video:
            return VideoColorTags()
        stream = container.streams.video[0]
        pix = None
        try:
            pix = stream.format.name if stream.format is not None else None
        except Exception:
            pix = None
        return VideoColorTags(
            primaries=_enum_name(getattr(stream, "color_primaries", None)),
            transfer=_enum_name(getattr(stream, "color_trc", None)),
            colorspace=_enum_name(getattr(stream, "colorspace", None)),
            range=_enum_name(getattr(stream, "color_range", None)),
            pix_fmt=pix,
        )


def resolve_color_tags(path: str | Path) -> VideoColorTags:
    """Rec.709 in / Rec.709 out tags. Timing-only; no pixel gamut conversion."""
    probed = probe_video_color(path)
    return VideoColorTags(
        primaries="bt709",
        transfer="bt709",
        colorspace="bt709",
        range=probed.range or "tv",
        pix_fmt=probed.pix_fmt,
    )


def color_ffmpeg_args(tags: VideoColorTags) -> list[str]:
    args: list[str] = []
    if tags.primaries:
        args += ["-color_primaries", tags.primaries]
    if tags.transfer:
        args += ["-color_trc", tags.transfer]
    if tags.colorspace:
        args += ["-colorspace", tags.colorspace]
    if tags.range:
        rng = tags.range
        if rng in {"tv", "mpeg", "limited"}:
            args += ["-color_range", "tv"]
        elif rng in {"pc", "jpeg", "full"}:
            args += ["-color_range", "pc"]
    return args


def atempo_chain(factor: float) -> str:
    """Build chained atempo filters; each stage must stay in [0.5, 2.0]."""
    if factor <= 0.0:
        raise ValueError(f"atempo factor must be > 0, got {factor}")
    stages: list[float] = []
    remaining = float(factor)
    while remaining > 2.0 + 1e-9:
        stages.append(2.0)
        remaining /= 2.0
    while remaining < 0.5 - 1e-9:
        stages.append(0.5)
        remaining /= 0.5
    stages.append(remaining)
    return ",".join(f"atempo={s:.10g}" for s in stages)


def convert_fps_speed(
    input_path: str | Path,
    output_path: str | Path,
    *,
    src_fps: float | Fraction | str = 24.0,
    dst_fps: float | Fraction | str = 25.0,
    crf: int = 18,
    preset: str = "slow",
    audio_bitrate: str = "192k",
) -> Path:
    """Retiming: setpts=src/dst*PTS,fps=dst + atempo=dst/src.

    Rec.709 display in → Rec.709 display out. Timing only (no gamut conversion).
    """
    src = parse_fps_rate(src_fps)
    dst = parse_fps_rate(dst_fps)

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = find_ffmpeg()
    ratio = src / dst
    pts_expr = f"{ratio.numerator}/{ratio.denominator}*PTS"
    dst_fps_str = f"{float(dst):.6f}".rstrip("0").rstrip(".")
    audio_filt = atempo_chain(float(dst / src))
    color_args = color_ffmpeg_args(resolve_color_tags(input_path))

    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(input_path)]
    if has_audio_stream(input_path):
        filt = (
            f"[0:v]setpts={pts_expr},fps={dst_fps_str}[v];"
            f"[0:a]{audio_filt}[a]"
        )
        cmd += [
            "-filter_complex",
            filt,
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-c:v",
            "libx264",
            "-crf",
            str(int(crf)),
            "-preset",
            str(preset),
            *color_args,
            "-c:a",
            "aac",
            "-b:a",
            str(audio_bitrate),
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    else:
        filt = f"[0:v]setpts={pts_expr},fps={dst_fps_str}[v]"
        cmd += [
            "-filter_complex",
            filt,
            "-map",
            "[v]",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            str(int(crf)),
            "-preset",
            str(preset),
            *color_args,
            "-movflags",
            "+faststart",
            str(output_path),
        ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {err}")
    if not output_path.is_file() or output_path.stat().st_size < 1:
        raise RuntimeError(f"ffmpeg produced no output: {output_path}")
    return output_path
