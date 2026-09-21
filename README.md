# ComfyUI-AcesCG-EXR

ComfyUI custom nodes for ACES / ACEScg display→scene pipelines and float EXR export.

## Install

Copy or clone into `ComfyUI/custom_nodes/ComfyUI-ACESCG` (or this repo name), then restart ComfyUI / reload custom nodes.

```text
pip install -r requirements.txt
```

OCIO CG configs ship under `ocio/`. deepDeband weights go in `ComfyUI/models/acescg/deepdeband/`.

## Nodes (`CATEGORY = AcesCg`)

| Node | Role |
|------|------|
| Depth Conversion | 8/16-bit → float32 `[0,1]` |
| Fill Dither | Stair soften + dither (rec709 / linear) |
| Deband Plateau | Plateau-fill deband + mask |
| Deband Edge | Edge-protect diffuse/dither + mask |
| Deep Deband | deepDeband U-Net |
| OCIO Convert | Rec.709 display → AP0/AP1 scene-linear |
| Preview Linear | Linear preview (no OETF) |
| Save EXR | Float32 OpenEXR sequence |
| Inspect Image | Console float IMAGE stats |

## Contributors

- **Paolo Alberti** — architecture & design

## License

Use at your own risk alongside ComfyUI.
