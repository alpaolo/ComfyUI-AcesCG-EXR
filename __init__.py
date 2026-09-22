"""AcesCg ComfyUI package — separate nodes, shared CATEGORY AcesCg."""

from .nodes.colorspace_ocio import AcesCgOCIOConvert
from .nodes.deband_edge import AcesCgDebandEdge
from .nodes.depth_conversion import AcesCgDepthConversion
from .nodes.fill_dither import AcesCgFillDither
from .nodes.inspect_image import AcesCgInspectImage
from .nodes.preview_linear import AcesCgPreviewLinear
from .nodes.save_exr import AcesCgSaveEXR

NODE_CLASS_MAPPINGS = {
    "AcesCgDepthConversion": AcesCgDepthConversion,
    "AcesCgFillDither": AcesCgFillDither,
    "AcesCgDebandEdge": AcesCgDebandEdge,
    "AcesCgOCIOConvert": AcesCgOCIOConvert,
    "AcesCgPreviewLinear": AcesCgPreviewLinear,
    "AcesCgSaveEXR": AcesCgSaveEXR,
    "AcesCgInspectImage": AcesCgInspectImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AcesCgDepthConversion": "Depth Conversion",
    "AcesCgFillDither": "Fill Dither",
    "AcesCgDebandEdge": "Deband Edge",
    "AcesCgOCIOConvert": "OCIO Convert",
    "AcesCgPreviewLinear": "Preview Linear",
    "AcesCgSaveEXR": "Save EXR",
    "AcesCgInspectImage": "Inspect Image",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
