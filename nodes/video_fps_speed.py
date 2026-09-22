"""Fps Converter — load Rec.709 mp4/mov + speed-retime; Rec.709 VIDEO out + preview."""

from __future__ import annotations

import os
from pathlib import Path
from typing_extensions import override

import folder_paths
from comfy_api.latest import ComfyExtension, InputImpl, io, ui

from ..core.ffmpeg_fps import FPS_RATE_OPTIONS, convert_fps_speed, parse_fps_rate

_PRESETS = [
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
]

_CONTAINER_EXTS = {".mp4", ".mov"}


def _list_mp4_mov_files() -> list[str]:
    input_dir = folder_paths.get_input_directory()
    names: list[str] = []
    for name in os.listdir(input_dir):
        path = os.path.join(input_dir, name)
        if os.path.isfile(path) and Path(name).suffix.lower() in _CONTAINER_EXTS:
            names.append(name)
    return sorted(names)


def _validate_container(path: str | Path) -> None:
    suffix = Path(path).suffix.lower()
    if suffix not in _CONTAINER_EXTS:
        raise ValueError(f"Fps Converter accepts only mp4/mov, got {suffix or '(none)'}")


class AcesCgVideoFpsSpeed(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AcesCgVideoFpsSpeed",
            display_name="Fps Converter",
            category="Aipermedia-AcesCg",
            description=(
                "Load Rec.709 mp4/mov, speed-retime with ffmpeg (setpts+fps+atempo). "
                "Output is Rec.709 display. Default 24→25. Timing only — no gamut convert."
            ),
            inputs=[
                io.Combo.Input(
                    "file",
                    options=_list_mp4_mov_files(),
                    upload=io.UploadType.video,
                    tooltip="Rec.709 mp4/mov from ComfyUI/input (upload allowed).",
                ),
                io.Combo.Input(
                    "from_fps",
                    options=FPS_RATE_OPTIONS,
                    default="24",
                    tooltip="Source rate of the clip (what it was authored at).",
                ),
                io.Combo.Input(
                    "to_fps",
                    options=FPS_RATE_OPTIONS,
                    default="25",
                    tooltip="Target rate after speed conversion.",
                ),
                io.Int.Input("crf", default=18, min=0, max=51, step=1),
                io.Combo.Input("preset", options=_PRESETS, default="slow"),
                io.String.Input(
                    "filename_prefix",
                    default="Aipermedia-AcesCg/fps_converter",
                    tooltip="Prefix under ComfyUI output/.",
                ),
            ],
            outputs=[io.Video.Output("video")],
            is_output_node=True,
        )

    @classmethod
    def validate_inputs(cls, file, **_kwargs):
        if not folder_paths.exists_annotated_filepath(file):
            return f"Invalid video file: {file}"
        path = folder_paths.get_annotated_filepath(file)
        suffix = Path(path).suffix.lower()
        if suffix not in _CONTAINER_EXTS:
            return f"Only mp4/mov allowed, got {suffix}"
        return True

    @classmethod
    def fingerprint_inputs(cls, file, **_kwargs):
        path = folder_paths.get_annotated_filepath(file)
        return os.path.getmtime(path)

    @classmethod
    def execute(
        cls,
        file: str,
        from_fps: str = "24",
        to_fps: str = "25",
        crf: int = 18,
        preset: str = "slow",
        filename_prefix: str = "Aipermedia-AcesCg/fps_converter",
    ) -> io.NodeOutput:
        src_path = Path(folder_paths.get_annotated_filepath(file))
        _validate_container(src_path)

        src = parse_fps_rate(from_fps)
        dst = parse_fps_rate(to_fps)
        probe = InputImpl.VideoFromFile(str(src_path))
        width, height = probe.get_dimensions()

        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix,
            folder_paths.get_output_directory(),
            width,
            height,
        )
        out_name = f"{filename}_{counter:05}_.mp4"
        out_path = Path(full_output_folder) / out_name

        convert_fps_speed(
            src_path,
            out_path,
            src_fps=src,
            dst_fps=dst,
            crf=int(crf),
            preset=str(preset),
        )

        out_video = InputImpl.VideoFromFile(str(out_path))
        preview = ui.PreviewVideo(
            [ui.SavedResult(out_name, subfolder, io.FolderType.output)]
        )
        return io.NodeOutput(out_video, ui=preview)


class AcesCgVideoExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [AcesCgVideoFpsSpeed]
