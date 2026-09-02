# Gemini image-generation prompt — CBAM effect (clutter suppression)

`fig_cbam_effect.drawio` is the editable master. In the draw.io the right-hand panel is a
PASTE slot for the real `cbam_overlay.png`; the Gemini raster below draws a stylized
stand-in for it — replace with the real overlay if you use the draw.io version.

```
Create a clean, publication-quality conceptual diagram for a computer-vision thesis.
White background. Flat vector style (no 3-D, no shadows, no skew). Serif font similar to
Times New Roman. Thin black orthogonal arrows. Aspect ratio ~1000 x 430.

TITLE (top-left, bold): "What CBAM does: background clutter suppressed, faint target amplified"

Left to right:
1. A light-grey square tile labelled below "Feature before CBAM (illustrative)". Inside it:
   three soft ORANGE blobs (fill #FFCC99) scattered around = background clutter, and one
   small faint GREY vertical ellipse (fill #CFCFCF) = a weak human response.
2. An arrow to a green rounded box (fill #D5E8D4, outline #82B366): "CBAM / channel + spatial
   attention", with a small italic caption under it "Mc: which channels · Ms: where".
3. An arrow to a second light-grey square tile labelled below "Feature after CBAM". Inside it:
   the same three blobs now FADED to near-white (fill #F4F4F4), and the small vertical ellipse
   now BRIGHT RED (fill #FF6666) = amplified target response.
4. A dashed arrow labelled "real example" to a square panel labelled below "Real attention
   (example)". Inside this panel: a stylized top-down aerial disaster scene with a warm
   red/orange heat overlay concentrated on small human figures (a jet-colormap attention map).

Small legend (bottom-left): an orange dot "background clutter", a red dot "target response".

Caption line (bottom): "The tiles illustrate the mechanism; the right panel is the model's
real spatial-attention map. CBAM down-weights background clutter and amplifies faint human
responses at near-zero parameter cost."

Keep colours exactly as specified. Do not add elements or text not listed here.
```

## Exact labels to verify
`What CBAM does: background clutter suppressed, faint target amplified`,
`Feature before CBAM (illustrative)`, `CBAM / channel + spatial attention`,
`Mc: which channels · Ms: where`, `Feature after CBAM`, `real example`,
`Real attention (example)`, legend `background clutter` / `target response`, caption.

## Colour key
| Element | Fill | Outline |
|---|---|---|
| Feature tile | `#EEEEEE` | `#999999` |
| Clutter blob (before) | `#FFCC99` | `#E69138` |
| Faint target (before) | `#CFCFCF` | `#AAAAAA` |
| CBAM box | `#D5E8D4` | `#82B366` |
| Amplified target (after) | `#FF6666` | `#CC0000` |

## Note
- The real panel: in the draw.io, paste `cbam_overlay.png` (your actual layer-10 attention
  map). Gemini can only draw a stylized stand-in.
