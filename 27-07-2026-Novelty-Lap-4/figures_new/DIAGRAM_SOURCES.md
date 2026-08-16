# Diagram sources for the report Part B figures (2026-08-06)

Your workflow: open the `.drawio` file (diagrams.net) to see exact boxes/arrows/labels, optionally
generate a reference image from the Gemini prompt, then redraw by hand in PowerPoint with
**Times New Roman**, matching the existing report diagrams. Export targets follow
`Defense/STYLE_GUARD.md` Part B. Save final PNGs into `Defense/draft2_06_8_26/figures/`
with these exact names (all NEW files, nothing overwritten).

---
(Blend symbol is GAMMA everywhere: alpha is taken by the CIoU loss in Ch. III.)

---
## 1. `fig_assign_mechanism.png`  (module diagram; reserve 6.5 cm; export ~1.8:1)
Source: `fig_assign_mechanism.drawio`. Report ref: Figure~\ref{fig:assignmech} (Ch. III).

**Content contract (must match the text):** inputs "Ground-truth box g" and "Candidate box b" each
feed BOTH a "CIoU(g,b)" box (annotated: collapses for boxes < 16 px) and an "NWD similarity
S = exp(−W₂(g,b)/C), C = 12.8" box (annotated: stays informative for tiny boxes). The two feed a
highlighted blend box "u = (1−γ)·CIoU + γ·S" (arrow labels "1−γ" and "γ"). The blend plus a
"Classification score s" box feed "Alignment score t = s^p · u^q", which feeds "Top-k selection →
positive training samples". NO footnote line inside the image: that explanation lives in the body prose (July caption convention).

**Gemini prompt:**
"Draw a clean academic block diagram, white background, black thin rectangular boxes, Times New
Roman text, left-to-right flow. Left column: two boxes 'Ground-truth box g' and 'Candidate box b'.
Both connect to two parallel middle boxes: 'CIoU(g,b)' on top and 'NWD similarity S = exp(−W2/C)'
below. Both middle boxes connect into one highlighted box (light yellow fill, bold border):
'u = (1−γ)·CIoU + γ·S', with the top arrow labelled '1−γ' and the bottom arrow labelled 'γ'. A
separate box 'Classification score s' above joins the highlighted box's output into a box
'Alignment score t = s^p · u^q', which flows into a final box 'Top-k selection → positive training
samples'. Flat 2D, no shadows, no colors except the one yellow highlight."

---
## 2. `fig_partB_chain.png`  (wide chain; reserve 4.5 cm; export ~3.2:1)
Source: `fig_partB_chain.drawio`. Report ref: Figure~\ref{fig:partbchain} (Ch. IV §partB intro).
Style must echo the existing `fig_ablation_chain.png` so the two read as siblings.

**Content contract:** five boxes left→right: (1) "CBAM + P2 (recommended model, official split
AP50 0.853)"; (2) "Scene-disjoint re-split (98% scene leakage audited; −1.6 AP50)"; (3) highlighted
"Tiny-aware assignment u = (1−γ)·CIoU + γ·NWD, γ = 0.5 — kept: +2.1 pts recall < 8 px"; (4)
"Real-data fine-tune: C2A + drone + campus + disaster (248 labelled frames, ×20 oversample)"; (5)
"One model, five held-out test domains". One dashed box BELOW box 2, connected by a dashed arrow
labelled "tried and dropped": "Frequency-gated evidence branch — no gain (ΔAP_S −0.6, +0.55 M
params); reported as negative".

**Gemini prompt:**
"Draw a wide, flat academic flow diagram (aspect about 3:1), white background, Times New Roman,
five thin-bordered rectangles connected left to right by black arrows. Box texts: 'CBAM + P2
(recommended model)'; 'Scene-disjoint re-split (−1.6 AP50)'; 'Tiny-aware assignment γ = 0.5
(+2.1 pts recall < 8 px)' with light-yellow fill and bold border; 'Real-data fine-tune (drone +
campus + disaster, ×20 oversample)'; 'One model, five test domains'. Below the second box, a
dashed-border rectangle 'Frequency-gated branch — no gain, rejected' connected upward by a dashed
arrow labelled 'tried and dropped'. No shadows, no 3D, single accent color."

---
## 3. `fig_real_samples.png`  (image grid; reserve 8.5 cm; export ~1.4:1) — assembled in PPT, no drawio
Report ref: Figure~\ref{fig:realsamples} (Ch. IV §Adaptation). 2 rows × 3 cells, thin sub-labels
(a)–(f) under each cell, 11 pt Times New Roman, no borders around the grid.

| cell | image source (pick a representative frame WITH GT boxes) |
|---|---|
| (a) drone 10 m | `Drone Shoot/extracted_v1/annotations/_qc/` overlays or re-render from test_v1 |
| (b) drone 30 m | same source, mid-altitude frame |
| (c) drone 50 m | same source, high-altitude frame (people smallest) |
| (d) campus occlusion | campus_eval_v1 frame with partial occlusion, boxes on |
| (e) disaster train (flood, Chennai) | disaster-train-v1 boxed render |
| (f) unseen event (Israel) | scratchpad xevent_boxed renders (session 2026-08-06) or re-render |

Caption in the tex already covers (a)–(f); keep the cell order above so the caption matches.
Note: AP watermark visible in (f) is fine (source attributed in Ch. V); do not crop it out.

---
## Also copy into `Defense/draft2_06_8_26/figures/` when ready (already rendered, in this folder):
- `fig_gamma_dose.png` (reserve 7.0 cm) — Figure~\ref{fig:gammadose}
- `fig_sim2real_bars.png` (reserve 6.5 cm) — Figure~\ref{fig:sim2real}
(Regenerate any time with `python make_partB_figs.py` in this folder.)
