# Cross-validation & hyperparameter tuning — explained for the thesis (2026-06-28)

## 0. Correction: which model is "highest performing"?
From `2026-06-13_complete_ablation_table.md` (C2A test, single seed):

| Model | mAP50-95 | AP50 | very-tiny recall | latency |
|---|---|---|---|---|
| baseline | 0.6151 | 0.8432 | 0.7427 | 13.7 ms |
| +CBAM | 0.6161 | 0.8473 | 0.7461 | ~14 ms |
| **+CBAM+P2** | 0.6153 | **0.8533** | **0.7575** | **14.6 ms** |
| +CBAM+P2+Mamba | **0.6143 (lowest)** | 0.8521 | 0.7567 | **41.1 ms (3x)** |

Mamba is the **lowest mAP50-95 and 3x slower** — a confirmed null result. The best model on the
operationally relevant metrics (AP50, tiny-recall), and the deployment + joint-training model, is
**CBAM+P2**. => Run any cross-validation / robustness study on **CBAM+P2**, not Mamba.

## 1. What cross-validation is
Estimating generalization without trusting one train/test split. **k-fold:** shuffle data, split
into k folds; for each fold i, train on the other k-1 folds and test on fold i; average the k scores
=> mean ± std. Common k=5. Gives an error bar and reduces split luck.

## 2. Reality for object detection / this project
- k-fold = **training the model k times.** ~5-6 h/run for CBAM+P2 => 5-fold ≈ **25-30 h** on the
  shared A6000 (Mamba even slower). Big commitment on a crash-prone box.
- True k-fold CV is **uncommon in DL detection papers**; standard = fixed train/val/test split +
  **multiple random seeds** (mean ± std). The spec already plans **SEEDS=[0..4]** — that IS the
  DL-standard robustness measure.
- **Leakage trap:** C2A is semi-synthetic (pasted humans on shared backgrounds) and SARD has
  augmented duplicates (`.rf.<hash>`). Naive random k-fold scatters near-identical images across
  folds => falsely optimistic/invalid. Must use **group k-fold by source scene/photo id** (the
  source-disjoint logic already built for the few-shot SARD study).

**Recommendation:** multi-seed (3-5 seeds) on CBAM+P2 -> report mean ± std. Only do full 5-fold
group-CV on CBAM+P2 if ~30 h spare compute and you want a dedicated generalization section.

## 3. Hyperparameter tuning
High-impact knobs (tiny-aerial-human detection):
- **imgsz (640 -> 960/1280)** — biggest lever for small objects (more VRAM/latency). Great ablation.
- **augmentation** (mosaic, copy-paste, mixup, scale, hsv) — helps small/rare objects.
- optimizer/lr0/lrf/warmup/weight_decay — pinned AdamW lr0=0.001 (MuSGD diverged on P2).
- loss gains (box/cls/dfl) — secondary.
- inference: conf, iou(NMS), max_det — tune on val only.
- **SAHI** tile size/overlap — inference-only; big for tiny objects on hi-res frames.

How (by cost):
1. Manual/grid sweep on 1-3 knobs, compare on the **validation** set — transparent, thesis-fit. ✅
2. Ultralytics `model.tune()` — genetic HPO, runs ~hundreds of trainings — **infeasible** here. ❌
3. Ray Tune — same cost problem. ❌

**Hard rule:** tune on **validation**, report final on **test**. Tuning on test = leakage = invalid.

## 4. Prioritization (final month, shared/crashy GPU) — value per hour
1. **SAHI evaluation** (inference-only, cheap) — strengthens the tiny-object claim. Do it.
2. **Multi-seed CBAM+P2** (3 seeds) — error bars on the headline. Standard.
3. **One focused imgsz sweep** (640 vs 960) on CBAM+P2 — real small-object insight.
4. (optional) 5-fold group-CV on CBAM+P2 — rigorous but ~30 h.
5. Skip: full automated HPO, and CV-on-Mamba — low value, high cost.
