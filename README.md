# ComfyUI-AcesCG-EXR

ComfyUI custom nodes for ACES / ACEScg display→scene pipelines and float EXR export.

## Install

Copy or clone into `ComfyUI/custom_nodes/Aipermedia-AcesCg`, then restart ComfyUI / reload custom nodes.

```text
pip install -r requirements.txt
```

OCIO CG configs ship under `ocio/`.

## Nodes (`CATEGORY = Aipermedia-AcesCg`)

| Node | Role |
|------|------|
| Depth Conversion | 8/16-bit → float32 `[0,1]` |
| Fill Dither | Stair soften + dither (rec709 / linear) |
| Deband Edge | Edge-protect diffuse/dither + mask |
| Fps Converter | Rec.709 mp4/mov speed-retime → Rec.709 (default 24→25) |
| OCIO Convert | Rec.709 display → AP0/AP1 scene-linear |
| Preview Linear | Linear preview (no OETF) |
| Preview Video | Video UI preview (linear / Rec.709) |
| Save EXR | Float32 OpenEXR sequence |
| Inspect Image | Console float IMAGE stats |

## Contributors

- **Paolo Alberti** — architecture & design

## Notes

© 2026 [Aipermedia.com](https://aipermedia.com)

## License

Use at your own risk alongside ComfyUI.
