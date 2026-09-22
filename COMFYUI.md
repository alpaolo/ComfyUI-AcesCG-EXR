# ACESCG → ComfyUI — struttura base

Obiettivo: **sviluppare custom node nella ComfyUI**.
Workspace nodi: `H:\ComfyUI_windows_portable\ComfyUI\custom_nodes`.
Workspace dove prendere codice già sviluppato:  `I:\VFX\ACESCG`.
QC colore: grading esterno su **EXR linear** (no bake).

Reference CLI : `H:\ComfyUI_windows_portable\ComfyUI`.

---

## Regola d’oro (non negoziabile)

**Un processo = un nodo Comfy = una scatola sul canvas.**

- **Stesso gruppo menu:** tutti i nodi custom sotto la categoria Comfy **`Aipermedia-AcesCg`** (un solo ramo nel Add Node), così li trovi insieme.
- **Nodi separati:** ogni processo è una classe / box diverso. **NON** un mega-nodo `AcesCg Pipeline` / `Run All`.
- Il **grafo** collega i box; sostituisce `workflow.json` come orchestratore.
- `Aipermedia-AcesCg` = pacchetto su disco; **`Aipermedia-AcesCg`** = etichetta di gruppo in UI.

In ogni classe nodo:

```python
CATEGORY = "Aipermedia-AcesCg"   # stesso gruppo per tutti
```

| Processo | Nodo (titolo UI) | `CATEGORY` | File tipico |
|----------|------------------|------------|-------------|
| Decode 8-bit | Decode U8 | `AcesCg` | `nodes/decode_u8.py` |
| Depth 8→float (+ fill) | Depth Conversion | `AcesCg` | `nodes/depth_conversion.py` |
| Color OCIO | OCIO Convert | `AcesCg` | `nodes/colorspace_ocio.py` |
| Deband edge | Deband Edge | `AcesCg` | `nodes/deband_edge.py` |
| Cast float16 (opz.) | Cast Float16 | `AcesCg` | `nodes/bit_depth_cast.py` |
| Preview linear | Preview Linear | `AcesCg` | `nodes/preview_linear.py` |
| Save EXR | Save EXR | `AcesCg` | `nodes/save_exr.py` |

Menu Comfy ≈:

```text
Add Node
└── AcesCg                    ← un solo gruppo
    ├── Decode U8            ← nodi separati
    ├── Depth Conversion
    ├── OCIO Convert
    ├── Deband Edge
    ├── Cast Float16
    ├── Preview Linear
    └── Save EXR
```

Ognuno ha **INPUT → OUTPUT** propri; salti uno step togliendo quel box (es. niente Deband).

Nodi **default** Comfy (Load Image/Video, Resize, …) fuori da `AcesCg`, dove non serve logica ACES.

---

## Principi

1. **Un gruppo `AcesCg`, tanti nodi separati** (vedi tabella). Vietato fondere step diversi in un solo nodo.
2. Decode = **solo uint8** (nessun `/255`).
3. Float nasce **solo** in `ACESCG Depth Conversion`.
4. Deband **solo dopo** OCIO (scene-linear), e solo se quel nodo è nel grafo.
5. Preview linear = **no gamma**; EXR linear = riferimento QC.

---

## Package (contenitore di molti nodi)

```
H:\ComfyUI_windows_portable\ComfyUI\custom_nodes\
└── Aipermedia-AcesCg\              ← PACKAGE (non “un nodo”)
    ├── __init__.py              ← registra N classi in NODE_CLASS_MAPPINGS
    ├── nodes\
    │   ├── decode_u8.py         ← 1 nodo: Decode U8
    │   ├── depth_conversion.py  ← 1 nodo: Depth Conversion
    │   ├── colorspace_ocio.py   ← 1 nodo: OCIO Convert
    │   ├── deband_edge.py       ← 1 nodo: Deband Edge
    │   ├── preview_linear.py    ← 1 nodo: Preview Linear
    │   ├── save_exr.py          ← 1 nodo: Save EXR
    │   └── bit_depth_cast.py    ← 1 nodo: Cast Float16 (opz.)
    ├── ocio\                    ← config o path OCIO
    └── README.md
```

`__init__.py` espone **più** entry, esempio concettuale:

```text
NODE_CLASS_MAPPINGS = {
  "ACESCG_DecodeU8": ...,
  "ACESCG_DepthConversion": ...,
  "ACESCG_OCIOConvert": ...,
  "AcesCgDebandEdge": ...,
  "ACESCG_PreviewLinear": ...,
  "ACESCG_SaveEXR": ...,
}
```

Sei (o più) nodi in menu Comfy, **non uno**.

---

## Grafo tipico (ogni box = un nodo distinto)

```
[Decode U8]                     ← AcesCg / processo 1
        │  U8_RGB
        ▼
[Depth Conversion]              ← AcesCg / processo 2  (unico /255 + fill 16)
        │  DISPLAY_FLOAT
        ▼
[OCIO Convert]                  ← AcesCg / processo 3
        │  SCENE_LINEAR
        ▼
[Deband Edge]                   ← AcesCg / processo 4 (opzionale)
        │  SCENE_LINEAR
        ├──────────────────────► [Preview Linear]   ← AcesCg
        ▼
[Save EXR]                      ← AcesCg → EXR out
```

Tutti questi box: **CATEGORY = `"Aipermedia-AcesCg"`**, classi distinte.  
Se togli Deband Edge, colleghi OCIO → Save/Preview.  
Se togli Depth, il grafo è sbagliato (float non deve nascere altrove).

---

## Contratti tra nodi (socket)

| Semantica | Contenuto | Prodotto da (solo quel nodo) |
|-----------|-----------|------------------------------|
| `U8_RGB` | codici 8-bit | Decode U8 |
| `DISPLAY_FLOAT` | [0,1] display | Depth Conversion |
| `SCENE_LINEAR` | AP0/AP1 (o log) | OCIO Convert (poi Deband se presente) |
| file `.exr` | half/float linear | Save EXR |

Preview Linear e Save EXR: **nessuna** OETF/sRGB/ACES Output di default.

---

## Depth (solo nel nodo Depth Conversion)

- `simulation`: `none` \| `fill`
- `fill`: 8..16 (max 16)
- τ ≈ `1.01/255`, poi quantize a `(2^fill-1)` livelli
- Dither: `on`/`off` (stesso nodo Depth, non un altro processo fondato con OCIO)

---

## Config CLI → widget per nodo

| JSON CLI | Dove finisce |
|----------|----------------|
| `workflow.json` lista step | **il grafo** (non un nodo “run workflow”) |
| `depth-conversion` block | widget del **solo** nodo Depth |
| `presets/conversions.json` | widget del **solo** nodo OCIO |
| `debanding.json` | widget del **solo** nodo Deband |

---

## Ordine implementazione (un nodo alla volta)

1. Package + `__init__.py` (mappings vuoti / stub)  
2. Nodo **Decode U8**  
3. Nodo **Depth Conversion**  
4. Nodo **OCIO Convert**  
5. Nodo **Save EXR**  
6. Nodo **Deband Edge**  
7. Nodo **Preview Linear**  
8. Example workflow Comfy (JSON grafo con **tutti** i box collegati)

---

## MCP ComfyUI

Serve a testare il grafo a nodi già installati. Non autorizza a collassare i processi in un unico nodo.
