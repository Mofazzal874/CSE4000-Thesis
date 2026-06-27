# Few-shot reality curve — RESULT (Phase 1 complete, 2026-06-16)

All 10 cells done (CBAM+P2 & Mamba × N∈{0,20,50,100,200}); leak-safe sampler confirmed
**0 train/test leakage** (SARD Roboflow export is source-disjoint; 1387 distinct train source
photos available). Eval on SARD test (570 imgs, 732 person boxes), same protocol as the C2A chain.

## Headline table — SARD test COCO AP@[.5:.95] (×100)

| N real | CBAM+P2 | Mamba | Lee et al. no-synth | Lee et al. Archangel rendered-synth |
|---|---|---|---|---|
| 0  | 0.4  | 0.03 | —    | —    |
| 20 | 18.5 | 16.9 | 2.1  | 11.5 |
| 50 | 28.2 | 23.7 | —    | —    |
| 100| 32.4 | 31.1 | —    | —    |
| 200| **39.9** | 35.6 | 7.0 | 17.7 |

mAP50 @ N=200: CBAM+P2 **0.763**, Mamba 0.743. AP_small @200: CBAM+P2 0.204.
(very-tiny recall = 0 everywhere is a SARD artifact — SARD has only 8 boxes <16px; it is a
medium-scale 32–96px dataset, not a tiny-object one.)

## Findings
1. **Zero-shot collapses (~0), few-shot recovers fast.** 20 real images → ~18 AP; 200 → ~40 AP.
   Smooth monotonic curve = the headline figure.
2. **C2A-pretrained ≫ published synthetic-pretraining anchors** (Lee et al. arXiv:2405.15203,
   SARD, AP@[.5:.95]): 39.9 vs 17.7 (Archangel rendered-synth) and vs 7.0 (no synthetic) at
   N=200; beats them at every N. Supports: "composited-REAL (C2A) is a stronger SAR pretraining
   corpus than rendered-synthetic."
3. **CBAM+P2 ≥ Mamba at every N** (39.9 vs 35.6 @200) — the Mamba-doesn't-help finding extends
   to the transfer/few-shot regime. CBAM+P2 is the model to carry forward.

## HONESTY CAVEAT (state in the thesis)
The Lee et al. comparison is CONTEXTUAL, not a controlled head-to-head: different base detector
(YOLO11m vs theirs), different SARD version/split, different "N real images" definition. So:
- PRIMARY claim = our own zero-shot→few-shot curve (fully controlled, our data/protocol).
- The anchor comparison is indicative ("our C2A-pretrained numbers substantially exceed published
  synthetic-pretraining anchors on SARD"), NOT "we beat method X in a controlled setup."
A controlled head-to-head (re-run their Archangel pipeline) is out of scope / future work.

## Files
- `cross_dataset_SARD/ablation_master/fewshot_curve.csv` + `.png`
- per-cell: `cross_dataset_SARD/runs_fewshot/<model>_N<k>/metrics/fewshot_summary.json`

## Next
Phase 2 = the deployable model: `joint_c2a_sard_train.py` (joint C2A+SARD CBAM+P2, eval on both
test sets). Then write.
