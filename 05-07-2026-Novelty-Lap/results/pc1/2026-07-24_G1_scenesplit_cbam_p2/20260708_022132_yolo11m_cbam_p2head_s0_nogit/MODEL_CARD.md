# Model Card -- yolo11m_cbam_p2head (run 20260708_022132_yolo11m_cbam_p2head_s0_nogit)

**Status:** CBAM+P2 ablation (`yolo11m_cbam_p2head`) per system_spec_thesis.md
Section 3 (row 4) -- backbone C2PSA replaced with CBAM (reduction=16,
kernel=7) AND a 4th detection scale P2/4 (stride 4) added ->
Detect(P2,P3,P4,P5). Effective batch held at 16 (physical 8
+ grad-accum) to match the baseline/CBAM runs. Compared against `yolo11m_baseline`,
`yolo11m_cbam`, and `yolo11m_p2head` (same data/seed/protocol).

## Intended Use
Aerial / SAR human detection on C2A imagery. Single class: `person`.

## Training data
C2A dataset (Nihal et al., ICPR 2024). Split-md5: `{'train': '6cd79d405f4b4515395117f8824e3d0f', 'val': 'abe020e99163763e36d55aab60cffa32', 'test': '83c91bea9503f1eaa681520d6b0c4bb5'}`. Image format: PNG.

## Evaluation
- Val mAP50: None
- Test mAP50: None
- Test mAP50-95: None
- Test AP_small: 0.6131730404063783
- Test F1 (per-image mean): 0.8431810372549019
- Test F2 (per-image mean): 0.8313594083333332
- Latency p50 / p95: 6.4957000031427015 / 9.429449999151984 ms

## Limitations
- THESIS scope: generated from SEEDS=[0] (1 seed for ablation-chain
  models per system_spec_thesis.md Section 3.1). Variance / paired-
  significance testing is deferred to the paper (run yolov11m_paper.py
  with 5 seeds for that). The reported numbers are single-seed point
  estimates, adequate for the thesis report's comparison table.
- Architecture-specific metrics in Section 11.6 (attention maps, per-stride
  AP, SSM state norms, dilation contributions) are N/A for vanilla YOLO11m
  and are logged as [SKIPPED] in logs/skipped_metrics.txt.

## Early-stopping configuration
- Ultralytics fitness patience = 50 (raised from spec's 30; matches HIT-UAV
  Sci Reports 2024 convention and the historic Ultralytics default).
- Custom F2 patience = 40 (raised from spec's 20). Stops training if F2
  has not improved for 40 consecutive epochs. Belt-and-suspenders on top
  of the built-in stopper.
- See docs/2026-05-29_yolo11m_final_month_writeup.md for the literature
  check that motivated these values.

## Ethical considerations
SAR / humanitarian use case. Not validated for surveillance.
