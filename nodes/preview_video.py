"""Preview Video — UI video preview with linear or Rec.709 display; VIDEO passthrough."""

from __future__ import annotations

import os
from fractions import Fraction

import folder_paths
from comfy_api.latest import ComfyExtension, Input, InputImpl, Types, io, ui
from typing_extensions import override

from ..core.display_transfer import apply_preview_display


class AcesCgPreviewVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AcesCgPreviewVideo",
            display_name="Preview Video",
            category="Aipermedia-AcesCg",
            description=(
                "Preview VIDEO in the UI. display=linear: gain+clamp, no OETF. "
                "display=rec709: BT.709 OETF (for scene-linear). "
                "Passthrough VIDEO is unchanged; only the UI encode is remapped."
            ),
            inputs=[
                io.Video.Input("video"),
                io.Combo.Input(
                    "display",
                    options=["linear", "rec709"],
                    default="linear",
                    tooltip=(
                        "linear = technical view (no gamma). "
                        "rec709 = apply BT.709 OETF before 8-bit preview encode."
                    ),
                ),
                io.Float.Input(
                    "exposure",
                    default=1.0,
                    min=0.0,
                    max=64.0,
                    step=0.01,
                    tooltip="Linear gain before clamp / OETF.",
                ),
            ],
            outputs=[io.Video.Output("video", tooltip="Passthrough input video.")],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        video: Input.Video,
        display: str = "linear",
        exposure: float = 1.0,
    ) -> io.NodeOutput:
        components = video.get_components()
        frames = apply_preview_display(
            components.images,
            display=display,
            exposure=float(exposure),
        )
        fps = components.frame_rate
        if not isinstance(fps, Fraction):
            fps = Fraction(fps).limit_denominator(1001)

        preview_video = InputImpl.VideoFromComponents(
            Types.VideoComponents(
                images=frames,
                audio=components.audio,
                frame_rate=fps,
            )
        )

        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            "Aipermedia-AcesCg/preview_video",
            folder_paths.get_temp_directory(),
            int(frames.shape[2]),
            int(frames.shape[1]),
        )
        out_name = f"{filename}_{counter:05}_.mp4"
        out_path = os.path.join(full_output_folder, out_name)
        preview_video.save_to(
            out_path,
            format=Types.VideoContainer.MP4,
            codec="auto",
            preset="ultrafast",
        )
        preview = ui.PreviewVideo(
            [ui.SavedResult(out_name, subfolder, io.FolderType.temp)]
        )
        return io.NodeOutput(video, ui=preview)


class AcesCgPreviewVideoExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [AcesCgPreviewVideo]
