# Gemini image-generation prompt — ablation waterfall

`fig_ablation_waterfall.drawio` is the editable master; this is only for a polished
raster. **Verify every number against the list below — image models routinely alter digits.**

```
Create a clean, publication-quality waterfall bar chart for a computer-vision thesis.
White background. Flat vector style (no 3-D, no shadows, no skew). Serif font similar to
Times New Roman, crisp and legible. Aspect ratio ~700 x 460.

TITLE (top-left, bold): "Additive ablation on C2A: what each component contributes to AP50"
Top-right small italic note: "y-axis zoomed to 0.838-0.856"

Y-axis label (vertical, left): "AP50 (COCO, C2A test)". Y-axis ticks at 0.840, 0.845,
0.850, 0.855. The axis is ZOOMED: bottom = 0.838, top = 0.856.

Four columns left to right, each 90 px wide, on the same baseline:

1. "Baseline" - a solid light-blue bar (fill #dae8fc, outline #6c8ebf) rising from the axis
   bottom to 0.8432. Value label above it: "0.8432". Under the column: "Baseline / 20.0 M / 13.7 ms".
2. "+CBAM" - a green floating bar (fill #d5e8d4, outline #82b366) from 0.8432 up to 0.8473.
   Inside it, green text "+0.0041"; above it "0.8473". Under: "+CBAM / 19.1 M / 13.5 ms".
3. "+CBAM+P2" - a green floating bar OUTLINED IN RED (fill #d5e8d4, outline #FF0000, thick)
   from 0.8473 up to 0.8533. Inside, bold green "+0.0060"; above, bold "0.8533".
   Under, in green: "+CBAM+P2 (recommended) / 19.6 M / 14.6 ms".
4. "+Mamba" - a short RED bar (fill #f8cecc, outline #b85450) hanging DOWNWARD from 0.8533
   to 0.8521 (a decrease). Red label beside it "-0.0012"; below the bar "0.8521".
   Under, in dark red: "+Mamba (negative) / 22.0 M / 41.1 ms (2.8x)".

Thin dashed grey step-connectors link the top of each column to the base of the next
(0.8432 -> +CBAM, 0.8473 -> +CBAM+P2, 0.8533 -> +Mamba).

Small legend (bottom-left): a green swatch "gain", a red swatch "no gain / loss", and
italic "(bar height = change in AP50 from the previous model)".

Caption (bottom): "P2 is the driver (+0.0060 AP50, and the largest very-tiny-recall gain).
CBAM is near-free: fewer parameters than the baseline and the lowest latency. Mamba shows
no meaningful accuracy change (-0.0012 AP50) at 2.8x latency, so it is excluded from the
deployed model."

Keep colours exactly as specified. Do not add elements or numbers not listed here.
```

## Exact numbers to verify in the output
- AP50 levels: **0.8432 → 0.8473 → 0.8533 → 0.8521**
- deltas: **+0.0041, +0.0060, −0.0012**
- params: **20.0 / 19.1 / 19.6 / 22.0 M**
- latency: **13.7 / 13.5 / 14.6 / 41.1 ms** (Mamba 2.8×)
- y-axis: **0.838–0.856**, ticks 0.840/0.845/0.850/0.855
- +CBAM+P2 outlined red (recommended); +Mamba bar is red and points **down**

## Colour key
| Element | Fill | Outline |
|---|---|---|
| Baseline bar | `#dae8fc` | `#6c8ebf` |
| Gain bar | `#d5e8d4` | `#82b366` (P2: `#FF0000`) |
| Loss bar (Mamba) | `#f8cecc` | `#b85450` |
