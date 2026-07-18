# Gemini image-generation prompts — SAHI and TTA inference pipelines

Two paper-ready pipeline diagrams. Paste each fenced block into Gemini (image
generation). **The `.drawio` files (`fig_sahi.drawio`, `fig_tta.drawio`) remain the
editable masters** — use the Gemini raster only if it comes out cleaner, and verify
every label against the "Exact labels" list (image models misspell technical text).

Design note: the figures show only the **method/config**, no accuracy numbers — those
go in the caption after the CBAM+P2 re-run, so the figure never needs re-editing.

---

## 1. SAHI — slicing-aided hyper inference

```
Create a clean, publication-quality technical pipeline diagram for a computer-vision
thesis. Horizontal left-to-right dataflow. White background. Flat vector style (no
photographic texture, no 3-D perspective, no drop shadows, no skew). Serif font similar
to Times New Roman, crisp and legible. Thin black orthogonal arrows with small solid
arrowheads. Wide aspect ratio, roughly 1040 x 460.

TITLE (top-left, bold, ~13pt):
"SAHI: slice the image into overlapping tiles, detect on each, then merge (inference-time only)"

Draw these stages left to right, connected by arrows:

1. INPUT: a rectangle depicting a simple top-down aerial disaster scene in muted grey-green
   tones, with a few tiny red dots scattered on it (people). Label below: "full aerial image".
2. Blue rounded rectangle (fill #dae8fc, outline #6c8ebf): "Slicer: overlapping tiles + full-image pass".
3. TILE GRID (centrepiece): the same aerial scene overlaid with a 2x2 grid of four
   semi-transparent green tiles (fill #d5e8d4, outline #82b366) that visibly OVERLAP each
   other at the seams. Small italic caption under it: "overlapping tiles".
4. Purple rounded rectangle (fill #e1d5e7, outline #9673a6, bold): "CBAM + P2 detector
   (per tile, conf 0.15)".
5. Blue rounded rectangle (fill #dae8fc, outline #6c8ebf): "Merge: GREEDYNMM (IOS, thr 0.5)".
6. OUTPUT: the same aerial scene, now with small red bounding boxes around each person dot.
   Label below: "merged detections".

CONFIG CAPTION (small text, below the row): "Configurations swept: slice 256 / 320 / 512 / 640 px,
overlap 25-30%, tile confidence 0.15-0.20, full-image pass on."

Keep arrows straight and non-crossing. Keep colours exactly as specified. Do not add any
element, icon, or text not listed here. Do not print any accuracy or mAP numbers.
```

**Exact labels (verify — nothing else should appear as text):**
`SAHI: slice the image into overlapping tiles, detect on each, then merge (inference-time only)`,
`full aerial image`, `Slicer: overlapping tiles + full-image pass`, `overlapping tiles`,
`CBAM + P2 detector (per tile, conf 0.15)`, `Merge: GREEDYNMM (IOS, thr 0.5)`,
`merged detections`, and the config caption. **No mAP / F1 / recall numbers in the image.**

---

## 2. TTA — test-time augmentation

```
Create a clean, publication-quality technical pipeline diagram for a computer-vision
thesis. Horizontal left-to-right dataflow, but with three parallel branches that fan out
then merge. White background. Flat vector style (no photographic texture, no 3-D
perspective, no drop shadows, no skew). Serif font similar to Times New Roman, crisp and
legible. Thin black orthogonal arrows with small solid arrowheads. Aspect ratio ~1000 x 470.

TITLE (top-left, bold, ~13pt):
"TTA: run the same image at multiple scales (and horizontal flip), then merge (inference-time only)"

Draw:

1. INPUT (left): a rectangle depicting a simple top-down aerial disaster scene in muted
   grey-green tones with a few tiny red dots (people). Label below: "input image".
2. The input FANS OUT with three arrows into three stacked blue rounded rectangles
   (fill #dae8fc, outline #6c8ebf), one above the other:
      "scale 1.0",  "scale 0.83",  "scale 0.67".
   Small italic caption spanning the three: "+ horizontal flip".
3. All three branches feed one purple rounded rectangle (fill #e1d5e7, outline #9673a6,
   bold): "CBAM + P2 detector (imgsz 1280)".
4. That feeds a blue rounded rectangle (fill #dae8fc, outline #6c8ebf): "Merge (NMS)".
5. OUTPUT (right): the same aerial scene with small red bounding boxes around each person
   dot. Label below: "detections".

CONFIG CAPTION (small text, below the row): "Scales [1.0, 0.83, 0.67] each with {original,
horizontal flip}; model trained at 640 px, TTA run at imgsz 1280."

Keep the three branches parallel and non-crossing. Keep colours exactly as specified. Do
not add any element, icon, or text not listed here. Do not print any accuracy or mAP numbers.
```

**Exact labels (verify):**
`TTA: run the same image at multiple scales (and horizontal flip), then merge (inference-time only)`,
`input image`, `scale 1.0`, `scale 0.83`, `scale 0.67`, `+ horizontal flip`,
`CBAM + P2 detector (imgsz 1280)`, `Merge (NMS)`, `detections`, and the config caption.
**No mAP / miss-rate numbers in the image.**

---

## Shared colour key (hex)
| Element | Fill | Outline |
|---|---|---|
| Process / merge box | `#dae8fc` | `#6c8ebf` |
| Tiles / scale branches accent | `#d5e8d4` | `#82b366` |
| Detector (CBAM + P2) | `#e1d5e7` | `#9673a6` |
| Image nodes | muted grey-green with red dots/boxes | — |

## Notes
- These are **inference-time** methods — say so in the title (both figures do) so an
  examiner does not think they change the trained model.
- Numbers stay in the caption, updated after the CBAM+P2 SAHI/TTA re-run (FINDINGS §3):
  SAHI-512 F1 / VT-recall, SAHI-256 VT-recall, TTA@1280 mAP50 gain + miss-rate — cite each
  to the new run's `summary.json`, not the old Mamba reference.
- If Gemini's rendered detector/merge text garbles, fix by hand or export the `.drawio`.
