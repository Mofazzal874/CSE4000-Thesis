# How real detection papers write (style guide) — read before drafting (2026-08-06)

**Purpose:** the persistent "mechanism" so we don't drift into AI-slop when writing. Built from reading
two full open-access papers in the exact subfield (JSTARS gates full text to scrapers, so these stand in —
their Results conventions ARE the JSTARS/TGRS norm). One is a careful published paper; one is a hype-y
preprint. **Copy the published one, avoid the preprint's tics.**

## Sources actually read (full text)
- **GS-YOLO** — "A lightweight high-accuracy model for small target detection in drone aerial images",
  PLOS ONE 21(6), 2025. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0350840
  → the model to imitate (near-identical topic to ours: UAV small-target detection, VisDrone/UAVDT).
- **PLUSNet** — "Purifying, Labeling, and Utilizing: A High-Quality Pipeline for Small Object Detection",
  arXiv 2504.20602 → cautionary: good structure, but preprint hype vocabulary to avoid.
- (Search surfaced but IEEE-gated: αS-YOLO, JSTARS 2025 — same structure, couldn't read full text.)

## The 6 conventions both papers follow (= what "human style" looks like here)
1. **Prose, not bullets.** Results/Experiments are flowing paragraphs. Zero bullet lists. Tables carry
   the data; sentences carry the argument.
2. **Every claim has a number attached.** GS-YOLO: *"reduces parameters by 73.3% ... while improving
   mAP50 by 0.9%."* Never a bare "significantly improves".
3. **Each table gets a one-sentence lead-in before it.** *"The quantitative results are presented in
   Table 8."* / *"As shown in Tab. I, query-based and anchor-free methods significantly lag behind
   anchor-based methods."* Introduce → then the table → then 1-3 sentences reading the key rows.
4. **Comparisons use comparative framing + the trade-off**, not a bare "we win": *"Larger YOLOv8
   variants achieve higher mAP50 but require significantly more parameters and suffer reduced FPS,
   making them unsuitable."* State what you give up.
5. **Ablations narrated component-by-component, each with the delta, then combined:** *"When only the
   C2FGhostLight module is added, the model ... mAP50 increases to 32.9% ... When the two components
   work together ... mAP50 further increases to 33.6%."* Walk the reader up the table row by row.
6. **Conclusion = headline numbers + ONE explicit limitation.** GS-YOLO ends: *"GS-YOLO has an obvious
   practical limitation in extreme occlusion scenarios: visual analysis shows significant detection
   failures for densely overlapping pedestrians and targets occluded by trees."* Honest, specific, dated
   to evidence. **This is exactly our disaster cross-event / <8px story — write it the same way.**

## GOOD vs SLOP (published GS-YOLO vs preprint PLUSNet — verbatim)
| | GOOD (imitate) | SLOP (delete on sight) |
|---|---|---|
| stating a result | "GS-YOLO achieves 33.6% mAP50 with only 0.84M parameters" | "these compelling results strongly emphasize the superior performance and **untapped potential** of our method" |
| framing a finding | "This demonstrates that its lightweight design minimizes redundancy without sacrificing detection performance" | "We **encouragingly find** that our approach consistently delivers significant improvements" |
| verbs | achieves, reduces, improves, outperforms, lags behind, balances | conclusively demonstrate, strongly emphasize, remarkably, encouragingly |
**Rule:** the number makes the claim; the adjective just adds slop and invites a reviewer to push back.
Banned words: *compelling, remarkable(ly), encouraging(ly), conclusively, untapped, novel (unless truly),
comprehensive, robust, leverage, a wide range of, it is worth noting, in order to.*

## Reusable sentence frames (fill the brackets — keeps you in-register)
- Table lead-in: **"The [comparison/ablation] results are reported in Table N."**
- Effect size: **"[Method] raises [metric] from X to Y (+Z pp; 95% CI [a,b], p<0.001)."**
- Comparison + trade-off: **"Relative to [baseline], [method] improves [metric] by Z pp while [cost]."**
- Ablation increment: **"Adding [component] alone changes [metric] from X to Y."**
- Honest limitation: **"[Method] has a clear limitation: [specific failure], as [evidence] shows."**
- Non-significant result (our cross-event): **"The gain is not statistically significant (Δ=+1.7 pp,
  95% CI [−3.2, 5.7], p=0.47); target-domain fine-tuning does not generalize to unseen events."**

## IEEE journal Experiments-section skeleton (JSTARS/TGRS standard)
Section IV Experiments → A. Datasets · B. Evaluation Metrics · C. Implementation Details ·
D. Comparison with State-of-the-Art (the big table + prose reading) · E. Ablation Study (component table +
component-by-component prose) · F. Discussion / Limitations. Length: results+discussion ≈ 3-5 pages,
mostly prose around 3-6 tables and 3-5 figures. One idea per paragraph, topic sentence first.

## For OUR paper specifically
- Headline the D2 assignment (with the p<0.001 CI) and the drone sim-to-real (+46pp). Baseline = CBAM+P2.
- Do NOT claim SOTA (we're below the C2A YOLOv9-e bar) — claim protocol + targeted lever + honest sim-to-real.
- Write the cross-event / <8px / void-FP results as GS-YOLO wrote its occlusion limitation: specific,
  numeric, un-hedged. That honesty is the paper's credibility, not its weakness.
- First pass over-produces; the second pass must be SHORTER than the first. Give the drafting request a
  word budget + audience ("JSTARS reviewers who know detection") every time.
