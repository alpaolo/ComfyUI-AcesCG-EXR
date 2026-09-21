"""OCIO Rec.709 (sRGB-encoded) -> linearized AP0/AP1 via ACES CG configs."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
OCIO_DIR = PACKAGE_ROOT / "ocio"
# Fallback to the ACESCG reference tree if package configs are missing.
REF_OCIO_DIR = Path(r"I:\VFX\ACESCG\ocio")

# Fixed display input for this node (Rec.709 / sRGB-encoded). Output is always linear.
TRANSFER = "linear"


@dataclass(frozen=True)
class OcioRoute:
    aces_version: str
    output_gamut: str
    config_path: Path
    input_colorspace: str
    output_colorspace: str
    label: str


# ACES 1.5 has no official CG OCIO config; routes through the 1.3 CG config.
_REGISTRY: dict[str, dict] = {
    "1.3": {
        "label": "ACES 1.3 CG",
        "ocio_file": "cg-config-v1.0.0_aces-v1.3_ocio-v2.1.ocio",
        "input": "sRGB - Texture",
        "AP0": "ACES2065-1",
        "AP1": "ACEScg",
    },
    "1.5": {
        "label": "ACES 1.5 (via 1.3 CG config)",
        "ocio_file": "cg-config-v1.0.0_aces-v1.3_ocio-v2.1.ocio",
        "input": "sRGB - Texture",
        "AP0": "ACES2065-1",
        "AP1": "ACEScg",
    },
    "2.0": {
        "label": "ACES 2.0 CG",
        "ocio_file": "cg-config-v4.0.0_aces-v2.0_ocio-v2.5.ocio",
        "fallback_ocio_file": "cg-config-v3.0.0_aces-v2.0_ocio-v2.4.ocio",
        "input": "sRGB Encoded Rec.709 (sRGB)",
        "AP0": "ACES2065-1",
        "AP1": "ACEScg",
    },
}


def _find_config(filename: str) -> Path | None:
    for folder in (OCIO_DIR, REF_OCIO_DIR):
        path = folder / filename
        if path.is_file():
            return path.resolve()
    return None


def resolve_route(aces_version: str, output_gamut: str) -> OcioRoute:
    version = str(aces_version).strip()
    gamut = str(output_gamut).strip().upper()
    if version not in _REGISTRY:
        raise ValueError(f"aces_version={version!r} unsupported. Use 1.3, 1.5, or 2.0.")
    if gamut not in ("AP0", "AP1"):
        raise ValueError(f"output_gamut={output_gamut!r} unsupported. Use AP0 or AP1.")

    entry = _REGISTRY[version]
    path = _find_config(entry["ocio_file"])
    if path is None and entry.get("fallback_ocio_file"):
        path = _find_config(entry["fallback_ocio_file"])
    if path is None:
        raise FileNotFoundError(
            f"OCIO config missing for ACES {version}: {entry['ocio_file']} "
            f"(looked in {OCIO_DIR} and {REF_OCIO_DIR})"
        )

    return OcioRoute(
        aces_version=version,
        output_gamut=gamut,
        config_path=path,
        input_colorspace=str(entry["input"]),
        output_colorspace=str(entry[gamut]),
        label=str(entry["label"]),
    )


@lru_cache(maxsize=8)
def _cpu_processor(config_path: str, input_cs: str, output_cs: str):
    import PyOpenColorIO as ocio

    cfg = ocio.Config.CreateFromFile(config_path)
    return cfg.getProcessor(input_cs, output_cs).getDefaultCPUProcessor()


def ocio_convert_rgb(
    rgb: np.ndarray,
    *,
    aces_version: str = "2.0",
    output_gamut: str = "AP0",
) -> tuple[np.ndarray, OcioRoute]:
    """HxWx3 display Rec.709 float [0,1] -> HxWx3 scene-linear AP0/AP1."""
    if rgb.ndim != 3 or rgb.shape[-1] < 3:
        raise ValueError(f"Expected HxWx3+, got {rgb.shape}")

    route = resolve_route(aces_version, output_gamut)
    proc = _cpu_processor(
        str(route.config_path),
        route.input_colorspace,
        route.output_colorspace,
    )

    src = np.ascontiguousarray(rgb[..., :3], dtype=np.float32)
    flat = src.reshape(-1, 3).copy()
    proc.applyRGB(flat)
    out = np.nan_to_num(flat.reshape(src.shape), nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(out, dtype=np.float32), route
