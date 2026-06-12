# Mamba+CBAM+P2 — first VALID run complete (verdict)

**Date:** 2026-06-10
**Run:** `runs/20260609_205717_mamba_cbam_p2head_s0_nogit` (150 ep, AdamW lr0=0.001,
grad-clip 1.0, seed 0). Survived a power cut + resume.

## Validity — CONFIRMED genuine Mamba (the injection fix worked)
- module_table: 36 Linear + 12 Conv1d + 12 LayerNorm = the 6 LocalWindowSSM blocks
  (at neck layers 13,16,19,22,25,28). params **22.01 M** / GFLOPs **98.4** / 183 layers
  (vs the stripped CBAM+P2: 19.57 M / 86.7 / 153).
- mamba_ssm.json populated: fwd-vs-reverse scan cosine distance **0.836** (directions
  learned independently — not degenerate), SSM state A≈2.5 bounded, state_norm_ok=true.
- Trajectory smooth, NO divergence (ep87 — where the old stripped 300-run blew up — is
  clean at mAP50 0.850). Best epoch 149, ran full 150, F2 stopper never fired.

## Result — Mamba does NOT beat CBAM+P2 (apples-to-apples, COCO eval)
| Model | mAP50-95 | AP50 | AP_small | very-tiny recall | params | latency e2e |
|---|---|---|---|---|---|---|
| CBAM            | 0.6161 | 0.847 | 0.616  | —      | 19.1 M  | — |
| CBAM+P2 (300ep) | 0.6153 | 0.853 | 0.6156 | 0.7575 | 19.57 M | 14.6 ms |
| Mamba+CBAM+P2 (150ep) | 0.6143 | 0.852 | 0.6146 | 0.7567 | 22.0 M | 41.1 ms |

**Conclusion:** all three within 0.002 mAP50-95 (tied). The SSM neck adds **+2.4 M params
and ~2.8× latency** for **no accuracy gain**. P2 is the real driver (lifts AP50
0.847→0.853). This is a credible NULL result (architecture metrics prove the SSM was
active), consistent with the pivot-doc finding.

### Caveat
CBAM+P2 ran 300 ep (stopped 218); Mamba ran 150 ep (best 149). Different schedules.
But Mamba plateaued by ~ep120 (mAP50-95 0.612→0.616 over ep117→150), so a longer Mamba
run is very unlikely to overturn the tie — not worth ~30 h to chase ~0.001.

## Recommended next steps
1. **Baseline retrain on AdamW lr0=0.001** — the only remaining optimizer inconsistency
   (old baseline = MuSGD). Closes a clean 4-row ablation table. ~6 h.
2. **Cross-dataset SARD zero-shot** on CBAM+P2 (and Mamba) — generalization evidence.
3. **SAHI / augmentation (Phase C)** — where real small-object gains come from for aerial.
4. **Framing:** present CBAM+P2 as the recommended model; Mamba as an explored SSM variant
   that did not improve (honest negative result). Decide with advisor.

## Housekeeping
- Delete empty crashed shells: `runs/20260609_112100_*` (on_pretrain callback crash) and
  `runs/20260609_205637_*` (false start) — both have no ultra/ or results.csv.
