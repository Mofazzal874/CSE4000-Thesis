# Gemini image-generation prompt — C3k2Mamba block diagram

Paste everything in the fenced block below into Gemini (image generation).
The `fig_c3k2mamba.drawio` file is the editable master; this prompt is only to
get a polished raster render. **After Gemini generates it, check every text
label against the "Exact labels" list — image models frequently misspell or
invent text.**

---

```
Create a clean, publication-quality technical block diagram for a computer-vision
thesis. Horizontal left-to-right dataflow. White background. Flat vector style
(no photographic texture, no 3-D perspective, no drop shadows, no skew). All text
in a serif font similar to Times New Roman, crisp and fully legible. Thin black
orthogonal (right-angled) connector arrows with small solid arrowheads. Wide aspect
ratio, roughly 1200 x 560.

TITLE (top-left, bold, ~14pt):
"C3k2Mamba: bidirectional local-window selective scan in place of the C3k2 bottleneck (explored variant)"

The diagram shows one neural-network block. Draw these elements in order, left to right,
connected by arrows:

1. INPUT FEATURE — a stack of 4 thin vertical plates (light blue fill #dae8fc, blue
   outline #6c8ebf), slightly offset up-and-to-the-right to look like a short stack of
   sheets. Label "C x H x W" centered above it.
2. Green rounded rectangle (fill #d5e8d4, green outline #82b366): "Window partition"
3. Green rounded rectangle: "LayerNorm + in_proj -> x, z"
4. From box 3 the path SPLITS into two amber rounded rectangles (fill #ffe6cc, orange
   outline #d79b00), one above the other:
      upper: "Forward scan (SSM, d_state = 4)"
      lower: "Reverse scan (flip -> scan -> flip)"
   Both arrows leaving box 3 are labelled "x".
5. The two amber boxes both feed a small white circle containing a "+" (element-wise sum).
6. The "+" circle feeds a small white circle containing a "x" (element-wise product / gate).
7. Below the gate, a green rounded rectangle "SiLU(z) gate" feeds UP into the "x" circle.
   A dashed arrow runs from box 3 down and across to this "SiLU(z) gate" box, labelled "z".
8. The "x" gate circle feeds a green rounded rectangle "out_proj".
9. "out_proj" feeds a white circle "+" (residual add).
10. A dashed arrow labelled "residual" runs along the TOP of the diagram from the
    "Window partition" box across to this residual "+" circle.
11. residual "+" feeds a green rounded rectangle "out_norm".
12. "out_norm" feeds a green rounded rectangle "Window reverse".
13. OUTPUT FEATURE — a stack of 4 thin vertical plates IDENTICAL to the input stack
    (same light blue, same size), labelled "C x H x W" above it. (Input and output are
    the same shape — this block is shape-preserving.)

GROUPING BOX: draw a dashed, rounded rectangle with NO fill and a thin grey-purple
outline (#9673a6) that encloses BOTH scan boxes, the "+" sum circle, the "x" gate circle,
and the "SiLU(z) gate" box. Label it in small italic serif at its top-left corner:
"Bidirectional selective scan (SSM core)". The arrows carrying "x" enter this box from the
left and the gate's output arrow leaves it on the right.

LEGEND (bottom, inside a light dashed grey box titled "Legend"), one row of swatches:
   - a small blue plate-stack  =  "Feature map (input = output shape, C x H x W)"
   - a green rounded box        =  "Operation (partition / proj / norm)"
   - an amber rounded box       =  "Selective scan (SSM)"
   - a white "+" circle          =  "Element-wise sum"
   - a white "x" circle          =  "Element-wise product (gate)"

CAPTION LINE (small, below the main row, above the legend):
"Shape-preserving drop-in for the C3k2 bottleneck; the surrounding Conv-bottleneck-Conv
wrapper is unchanged. Injected at neck layers 13, 16, 19, 22, 25, 28; window size 6-8."

Keep arrows straight and non-crossing. Keep colours exactly as specified. Do not add any
elements, icons, or text that are not listed here.
```

---

## Exact labels (verify these in the output — nothing else should appear as text)

- Title: `C3k2Mamba: bidirectional local-window selective scan in place of the C3k2 bottleneck (explored variant)`
- `C x H x W` (input, appears twice — input and output)
- `Window partition`
- `LayerNorm + in_proj -> x, z`
- `Forward scan (SSM, d_state = 4)`
- `Reverse scan (flip -> scan -> flip)`
- `SiLU(z) gate`
- `out_proj`
- `out_norm`
- `Window reverse`
- `Bidirectional selective scan (SSM core)` (dashed group label)
- edge labels: `x` (twice), `z`, `residual`
- glyphs: `+` (sum, twice), `x` (product gate)
- legend + caption text as written above

## Colour key (hex)
| Element | Fill | Outline |
|---|---|---|
| Feature-map plates | `#dae8fc` | `#6c8ebf` |
| Operations (green) | `#d5e8d4` | `#82b366` |
| Selective scan (amber) | `#ffe6cc` | `#d79b00` |
| Sum / product circles | `#ffffff` | `#000000` |

## Notes
- The `.drawio` file stays the editable source of truth for the report; use the Gemini
  image only if it looks cleaner, and keep the two visually consistent.
- If Gemini garbles the internal math (`in_proj`, `d_state`, `SiLU`), fix those labels
  by hand or fall back to exporting the `.drawio` to PNG.
