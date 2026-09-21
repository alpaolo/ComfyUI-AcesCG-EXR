"""Preview Linear — temp UI preview with no gamma / OETF / view transform.

Maps float RGB to 8-bit PNG by linear scale + clamp only (x * exposure * 255).
Does not apply sRGB, Rec.709, or ACES Output. External grading remains the color QC reference.
"""

from __future__ import annotations

import os
import random

import folder_paths
import numpy as np
import torch
from PIL import Image


class AcesCgPreviewLinear:
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_acescg_lin_" + "".join(
            random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5)
        )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "exposure": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 64.0,
                        "step": 0.01,
                        "tooltip": (
                            "Linear gain before clamp (stops-ish via multiplier). "
                            "Does NOT apply gamma."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "preview"
    CATEGORY = "AcesCg"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Preview float RGB without gamma/OETF. UI PNG is linear*exposure clamped to 8-bit; "
        "passthrough IMAGE is unchanged."
    )

    def preview(self, image: torch.Tensor, exposure: float = 1.0):
        if image.ndim != 4 or image.shape[0] < 1:
            raise ValueError("image must be a non-empty IMAGE batch [N,H,W,C]")

        os.makedirs(self.output_dir, exist_ok=True)
        gain = float(exposure)
        results = []

        for batch_number, frame in enumerate(image):
            # Linear only: gain + clamp. No pow(), no sRGB encode.
            rgb = frame.detach().cpu().float()
            if rgb.shape[-1] == 1:
                rgb = rgb.repeat(1, 1, 3)
            elif rgb.shape[-1] > 3:
                rgb = rgb[..., :3]
            display = (rgb * gain).clamp(0.0, 1.0).numpy()
            img8 = np.clip(display * 255.0, 0, 255).astype(np.uint8)
            pil = Image.fromarray(img8, mode="RGB")

            file = f"acescg_linear{self.prefix_append}_{batch_number:05d}.png"
            pil.save(os.path.join(self.output_dir, file), compress_level=1)
            results.append({"filename": file, "subfolder": "", "type": self.type})

        return {"ui": {"images": results}, "result": (image,)}
