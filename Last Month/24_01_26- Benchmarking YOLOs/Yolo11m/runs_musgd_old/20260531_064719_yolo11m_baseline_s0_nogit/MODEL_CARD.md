# Model Card -- yolo11m_baseline (run 20260531_064719_yolo11m_baseline_s0_nogit)

**Status:** Phase A row 1 of the final-month ablation matrix
(`yolo11m_baseline`, P0) per system_spec.md Section 9.6 / Section 25.
This is the reference baseline against which CBAM / P2Head / Mamba
variants are compared.

## Intended Use
Aerial / SAR human detection on C2A imagery. Single class: `person`.

## Training data
C2A dataset (Nihal et al., ICPR 2024). Split-md5: `{'train': 'e03db54f95acd15a4a83a5bb28765024', 'val': '84793a339a7eadb47727c3e3bcd150d7', 'test': 'c750aaf4476ff926294cce7034281eab'}`. Image format: PNG.

## Evaluation
- Val mAP50: None
- Test mAP50: None
- Test mAP50-95: None
- Test AP_small: 0.644477858318374
- Test F1 (per-image mean): 0.8514535604503182
- Test F2 (per-image mean): 0.8458976157611355
- Latency p50 / p95: 5.079999988083728 / 5.359490000410005 ms

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
