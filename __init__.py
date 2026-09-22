"""Aipermedia-AcesCg ComfyUI package — separate nodes, shared CATEGORY."""

from .nodes.colorspace_ocio import AcesCgOCIOConvert
from .nodes.deband_edge import AcesCgDebandEdge
from .nodes.depth_conversion import AcesCgDepthConversion
from .nodes.fill_dither import AcesCgFillDither
from .nodes.inspect_image import AcesCgInspectImage
from .nodes.preview_linear import AcesCgPreviewLinear
from .nodes.preview_video import AcesCgPreviewVideo
from .nodes.save_exr import AcesCgSaveEXR
from .nodes.video_fps_speed import AcesCgVideoFpsSpeed

NODE_CLASS_MAPPINGS = {
    "AcesCgDepthConversion": AcesCgDepthConversion,
    "AcesCgFillDither": AcesCgFillDither,
    "AcesCgDebandEdge": AcesCgDebandEdge,
    "AcesCgOCIOConvert": AcesCgOCIOConvert,
    "AcesCgPreviewLinear": AcesCgPreviewLinear,
    "AcesCgPreviewVideo": AcesCgPreviewVideo,
    "AcesCgSaveEXR": AcesCgSaveEXR,
    "AcesCgInspectImage": AcesCgInspectImage,
    "AcesCgVideoFpsSpeed": AcesCgVideoFpsSpeed,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AcesCgDepthConversion": "Depth Conversion",
    "AcesCgFillDither": "Fill Dither",
    "AcesCgDebandEdge": "Deband Edge",
    "AcesCgOCIOConvert": "OCIO Convert",
    "AcesCgPreviewLinear": "Preview Linear",
    "AcesCgPreviewVideo": "Preview Video",
    "AcesCgSaveEXR": "Save EXR",
    "AcesCgInspectImage": "Inspect Image",
    "AcesCgVideoFpsSpeed": "Fps Converter",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
