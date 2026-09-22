"""Save EXR — write float IMAGE batch to a folder with numeric frame padding."""

from __future__ import annotations

from pathlib import Path

import folder_paths
import numpy as np
import torch

from ..core.exr_write import padded_exr_name, write_exr_rgb


def _resolve_folder(folder: str) -> Path:
    """Absolute path as-is; relative path under Comfy output; empty -> output/Aipermedia-AcesCg."""
    output_root = Path(folder_paths.get_output_directory()).resolve()
    text = (folder or "").strip()
    if not text:
        return output_root / "Aipermedia-AcesCg"
    path = Path(text)
    if not path.is_absolute():
        path = (output_root / path).resolve()
        try:
            path.relative_to(output_root)
        except ValueError as exc:
            raise ValueError(
                f"Relative folder escapes Comfy output: {folder!r} -> {path}"
            ) from exc
        return path
    return path.resolve()


class AcesCgSaveEXR:
    """Export scene/display float RGB to OpenEXR (no gamma bake). Batch = padded sequence."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "folder": (
                    "STRING",
                    {
                        "default": "Aipermedia-AcesCg",
                        "tooltip": (
                            "Output folder. Relative paths are under Comfy output/; "
                            "absolute paths (e.g. I:/VFX/out) are used as-is."
                        ),
                    },
                ),
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "Aipermedia-AcesCg",
                        "tooltip": "Base name before the frame number: prefix.0001.exr",
                    },
                ),
                "padding": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 8,
                        "step": 1,
                        "tooltip": "Zero-pad width for the frame index (4 -> 0001).",
                    },
                ),
                "start_frame": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 99999999,
                        "step": 1,
                        "tooltip": "Frame number for batch index 0; then +1 per image.",
                    },
                ),
                "overwrite": (["enable", "disable"], {"default": "enable"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("paths",)
    FUNCTION = "save"
    CATEGORY = "Aipermedia-AcesCg"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Write float RGB to OpenEXR float32 (linear, no OETF). "
        "Batch writes prefix.NNNN.exr with configurable padding."
    )

    def save(
        self,
        image: torch.Tensor,
        folder: str,
        filename_prefix: str,
        padding: int,
        start_frame: int,
        overwrite: str,
    ):
        if image.ndim != 4 or image.shape[0] < 1:
            raise ValueError("image must be a non-empty IMAGE batch [N,H,W,C]")

        out_dir = _resolve_folder(folder)
        out_dir.mkdir(parents=True, exist_ok=True)
        allow_overwrite = str(overwrite).lower() == "enable"

        frames = image.detach().cpu().numpy()
        written: list[str] = []
        for i in range(frames.shape[0]):
            frame_index = int(start_frame) + i
            name = padded_exr_name(filename_prefix, frame_index, padding)
            path = out_dir / name
            if path.exists() and not allow_overwrite:
                raise FileExistsError(f"EXR exists (overwrite=disable): {path}")
            rgb = frames[i]
            if rgb.ndim == 2:
                rgb = np.stack([rgb, rgb, rgb], axis=-1)
            elif rgb.shape[-1] == 1:
                rgb = np.repeat(rgb, 3, axis=-1)
            write_exr_rgb(path, rgb)
            written.append(str(path))

        # Newline-joined list for batch; single path if N=1
        paths_text = "\n".join(written)
        return {"ui": {"text": [paths_text]}, "result": (paths_text,)}
