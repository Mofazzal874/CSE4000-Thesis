# Metric contract reference for lap-3 runs (2026-07-12)

**Source of truth (do not duplicate their tables — follow them):**
- `..\..\Last Month\system_spec.md` — **§11 Complete Metrics Catalog** (11.1 detection quality,
  11.2 training dynamics, 11.3 efficiency, 11.4 calibration, 11.5 qualitative grids + failure
  taxonomy, 11.6 architecture-specific, 11.7 env.json, 11.8 energy) + §12 significance,
  §13 every-metric-gets-a-plot, §16 run-ID/naming.
- `..\..\Last Month\system_spec_thesis.md` — **§6 thesis MUST-have set** (the trimmed contract:
  F2 primary; per-size recall bins very-tiny <8² / tiny 8–16² / small 16–32² / medium / large;
  opt-threshold F1/F2; efficiency measured on the 4070 Ti S; §6.1 defers 5-seed significance /
  energy / ONNX to paper phase), §9 infra rules (smoke-before-run, OOM ladder, power-failure
  resume, dual early stop 50/40, GPU util ≥85%), §10 pre-run checklist.
- Implementation: lap-1 **script 04** is the eval harness that realizes this contract
  (`..\..\05-07-2026-Novelty-Lap\scripts\04_eval_fusion_ablation.py`). `[SKIPPED] <metric> —
  <reason>` lines to `skipped_metrics.txt` whenever something is N/A — never silently drop.

## What each lap-3 stage must log
| Stage | Contract level |
|---|---|
| S0 smoke (2 ep) | EXEMPT from the contract — only losses + `[fccg-verify]` gate health each epoch |
| S1/S2 paired 50-ep pilots | LIGHT set: mAP50, mAP50-95, AP_small, per-size recall (incl. VT<8²), F2, params/GFLOPs + the FCCG-specific set below; read as DELTA vs the paired control, seed 0 only |
| S3/S4 full 300-ep protocol runs + anchors | FULL `system_spec_thesis.md` §6 must-have set via script 04 + env.json (§11.7 fields) + run-ID naming (§16) + per-stride AP |
| S5 paper finals | §6 set + the §6.1-deferred items that the target venue needs (bootstrap CI on mAP50/mAP50-95/F1/F2/AR_small/VT-recall per §11.1; significance per §12; energy §11.8 only if venue asks) |

## FCCG-specific metrics (new §11.6-style subset — the paper's MECHANISM evidence)
Same spirit as the existing CBAM/P2/Mamba subsets in `system_spec.md` §11.6; each FCCG run
(S1 onward) must additionally emit:
1. `gate_map_examples.png` — gate g overlays on 6 representative test images, one row per seam
   (P3 seam = YAML layer 16, P2 seam = layer 20).
2. `gate_stats.csv` — per seam over the val set: `gate_mean`, `gate_std`, `gate_entropy`
   (saturation watch; complements the training-time FCCGActiveCheck guard).
3. `gate_gt_contrast` — mean g INSIDE GT person boxes vs OUTSIDE (per seam, per split).
   This is the headline mechanism metric: the gate is only doing its job if g_in ≫ g_out.
4. `gate_off_delta` — eval the SAME weights once with `force_zero_gate=True`
   (module hook, no retrain): Δ mAP50 / Δ AP_small / Δ VT-recall = causal evidence the
   context gate matters. Cheap (one extra val pass) — required in every ablation table.
5. `evidence_contribution_norm` — mean ‖γ·HFE(ev)·g‖ / ‖ev‖ per seam (how much the evidence
   branch actually modifies features; near-zero ⇒ dead branch).
6. `fflup_kernel_divergence` — mean KL(predicted 3×3 kernel ‖ uniform) per FFLUp layer
   (how far the adaptive upsampler moved from its box-blur init).
7. Inherited from the P2 contract (continuity with all prior slides): `per-stride_AP` and
   `tiny_obj_recall_by_head` — which head catches the <8 px people.

## Standing notes
- Latency numbers for any table: measured on PC-1's 4070 Ti SUPER only (never mix hardware —
  `system_spec_thesis.md` §5 rule).
- Same frozen splits + md5 checks as always; per-image scores saved so significance can be run
  later without re-evaluating.
- Every CSV metric gets a PNG; keep `PLOTS_INDEX.md` per §7/§13.
