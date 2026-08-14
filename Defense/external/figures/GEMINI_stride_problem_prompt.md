# Gemini image-generation prompt — stride problem ("why P2")

`fig_stride_problem.drawio` is the editable master; this is only for a polished raster.
This figure uses NO photos — it is a pure grid-geometry schematic. **Verify the grid
counts (1, 2x2, 4x4, 8x8) and that the target box is identical in all four panels —
image models routinely get grids and repeats wrong.**

```
Create a clean, publication-quality schematic diagram for a computer-vision thesis.
White background. Flat vector style (no 3-D, no shadows, no photographic texture).
Serif font similar to Times New Roman, crisp and legible. Wide aspect ratio ~800 x 390.

TITLE (top-left, bold): "Why a P2 (stride-4) head: the same ~12 px human under each scale's grid"
Subtitle (below, small italic): "Each panel is the same 32x32 px image patch (one P5 cell)
at 640-px input; the target box is identical in all four."

Draw FOUR equal square panels in a horizontal row, each the same size, with a thin dark-grey
border. Each panel represents the SAME 32x32 pixel image patch, divided into a grid of
equal cells with light-grey lines:
  Panel 1 (left):  NO internal grid lines -- a single cell (this is one P5 cell).
  Panel 2:         a 2 x 2 grid.
  Panel 3:         a 4 x 4 grid.
  Panel 4 (right): an 8 x 8 grid; give THIS panel a RED border (it is the added scale).

In EVERY panel, draw the SAME small red rectangle (thin red outline, no fill, no colour
inside) in the SAME position -- a narrow upright person-sized box, about 12 px tall in
patch units. Because the box is fixed size, it is a small fraction of Panel 1 but spans
several cells of Panel 4. Put a small red label "target" under the box in Panel 1 only.

Under each panel, a three-line centred caption:
  Panel 1: "P5 - stride 32 / cell = 32 px / target ~ 0.4 cell"
  Panel 2: "P4 - stride 16 / cell = 16 px / target ~ 0.75 cell"
  Panel 3: "P3 - stride 8 / cell = 8 px / target ~ 1.5 cells"
  Panel 4 (dark-red, bold): "P2 - stride 4 (added) / cell = 4 px / target ~ 3 cells"

Caption line (bottom, small): "A detection cell spans 'stride' pixels. A ~12 px human is a
fraction of one P5 or P4 cell, but spans about three P2 cells, enough to be localized
precisely. This is why the added stride-4 (P2) head resolves targets the standard P3-P5
scales miss."

Keep the grids exact (1, 2x2, 4x4, 8x8). Keep the target box identical and in the same
place in all four panels. Do not add elements, colours, or text not listed here.
```

## Exact labels to verify
- Title + italic subtitle as above
- Per-panel captions: `P5 · stride 32 / cell = 32 px / target ~ 0.4 cell`; `P4 · stride 16 / … 0.75 cell`; `P3 · stride 8 / … 1.5 cells`; `P2 · stride 4 (added) / … 3 cells`
- `target` label under Panel 1 only
- bottom caption sentence

## Colour key
| Element | Colour |
|---|---|
| Panel borders (P5/P4/P3) | dark grey `#333333` |
| Panel border (P2, added) | red `#FF0000` |
| Grid lines | light grey `#CCCCCC` |
| Target box + P2 caption text | red / dark-red `#FF0000` / `#C00000` |

## Notes
- No images are pasted into this figure — it is schematic. Its real-data companion is the
  detection-grid figure (`detgrid_c2a_s8` / `detgrid_c2a_s4`), already in Chapter IV.
- The single hardest thing for the model is the exact grid counts and the identical
  repeated target box; if it garbles them, export the `.drawio` to PNG instead.
