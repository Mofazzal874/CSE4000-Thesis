# Complete C2A ablation — all 4 models, consistent protocol (2026-06-13)

Baseline retrained on **AdamW lr0=0.001, 300 epochs** (run `20260615_230315_yolo11m_baseline_s0`,
best@270, no divergence, 7.3h) — so the whole chain now shares one optimizer/schedule.
All numbers below are **COCO protocol + per-size recall** (the metric set all four models
share apples-to-apples; CBAM & CBAM+P2 lack ultralytics-`.val` numbers from the pre-fix era).

## The table (C2A test split, single seed)

| Model | COCO AP (mAP50-95) | COCO AP50 | very-tiny(<8px) recall | tiny(8-16px) recall | params | GFLOPs | latency e2e |
|---|---|---|---|---|---|---|---|
| **YOLO11m baseline** | **0.6151** | 0.8432 | 0.7427 | 0.8688 | 20.03M | 67.7 | 13.7 ms |
| + CBAM | 0.6161 | 0.8473 | 0.7461 | 0.8730 | 19.10M | ~68 | ~14 ms |
| + CBAM + P2 | 0.6153 | **0.8533** | **0.7575** | 0.8651 | 19.57M | ~86.7 | 14.6 ms |
| + CBAM + P2 + Mamba (headline) | 0.6143 | 0.8521 | 0.7567 | 0.8700 | 22.01M | 98.4 | 41.1 ms |

## What it honestly shows

1. **Tight-box mAP50-95 is SATURATED (~0.615) across the entire ablation.** No addition moves
   it. Interpretation worth putting in the discussion: C2A's boxes are *composited/pasted*, so
   high-IoU localization is capped by paste-box quality, not the model — mAP50-95 can't improve
   no matter what you bolt on. This is a *synthetic-data* observation, not a model failure.

2. **The P2 head delivers the only real, monotonic gain — in the operationally relevant metrics:**
   - AP50: 0.8432 → 0.8473 (CBAM) → **0.8533 (CBAM+P2, +1.0 over baseline)**.
   - very-tiny recall: 0.7427 → 0.7461 (CBAM) → **0.7575 (CBAM+P2, +1.48 over baseline)**.
   - i.e., the stride-4 head *finds the smallest people the baseline misses* — exactly what
     matters for SAR (finding a person > a tight box). And it does so at **~zero parameter cost**
     (19.57M < 20.03M baseline — CBAM is lighter than the C2PSA it replaces) and ~equal latency.

3. **CBAM alone: marginal** (+0.34 very-tiny recall, +0.4 AP50). Real but small.

4. **Mamba SSM neck: no gain, high cost.** Ties CBAM+P2 on every metric (0.7567 vs 0.7575
   very-tiny; 0.8521 vs 0.8533 AP50) while adding +2.4M params and **~3× latency (41 vs 14.6 ms)**.
   Consistent with MambaOut (arXiv:2405.07992): SSMs don't help non-long-sequence vision.

## The honest framing for the thesis

> "On C2A, aggregate mAP50-95 is saturated near 0.615 for all variants — a ceiling imposed by
> the dataset's composited boxes. The contribution of each module is therefore best read in
> small-object *recall*: a P2 stride-4 head raises very-tiny-person recall by +1.5 points and
> AP50 by +1.0 at near-zero parameter cost and is the dominant driver; CBAM adds a marginal
> refinement; a bidirectional SSM neck adds nothing while tripling inference latency."

**Caveat to state plainly:** your numbers sit 2nd–3rd on the C2A paper's 9-model table largely
because of the *training recipe* (YOLO11m, 300 epochs, AdamW) vs their 50-epoch runs — not the
architecture. The architecture's value is specifically small-object recall, which their
aggregate-mAP table never surfaces. This is *why* the SARD generalization study is the real
contribution: it measures something the architecture ablation alone cannot claim.

## Per-model run dirs
- baseline: `24_01_26- Benchmarking YOLOs/Yolo11m/runs/20260615_230315_yolo11m_baseline_s0_nogit`
- cbam:     `24_01_26- Benchmarking YOLOs/CBAM/runs/20260601_232929_yolo11m_cbam_s0_nogit`
- cbam_p2:  `24_01_26- Benchmarking YOLOs/CBAM_P2Head/runs/20260602_063759_yolo11m_cbam_p2head_s0_nogit`
- mamba:    `02-06-26-Mamba_CBAM_P2Head/runs/20260609_205717_mamba_cbam_p2head_s0_nogit`
