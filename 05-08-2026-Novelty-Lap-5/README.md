# Novelty-Lap-5 — YOLO26m backbone study (started 2026-08-05)

**Branch:** `novelty-lap-5` (create from `novelty-lap-4`: `git checkout -b novelty-lap-5`).
**Folder:** this one. Lap-4 stays a clean **yolo11m** story; YOLO26m lives here.

## The question
Does a newer backbone (**YOLO26m**) + our lap-1..4 stack (CBAM → +P2 head → +tiny-aware
assignment α → +sim-to-real) beat the yolo11m results — and are our contributions **complementary**
(stack on the newer backbone → new best) or **subsumed** (the backbone alone already captures the
gain)? Either outcome is publishable: complementary = "port the mechanism forward"; subsumed = an
honest "newer backbone closes the gap our mechanism targeted" finding + the mechanism insight holds.

## Starting baseline — plain YOLO26m (PARKED from lap-4, do NOT re-run)
Off-the-shelf `yolo26m.pt` fine-tuned on C2A scene-split (runner `21_yolo26_anchor.py`, AdamW seed 0,
236 ep best@186). Scene-split TEST (n=2040), eval = `18_s1_eval.py`:

| model | AP50 | F2 | AP_small | VT-recall(<8) | tiny | small | ECE |
|---|---|---|---|---|---|---|---|
| **YOLO26m (base)** | 0.8552 | 0.8364 | 0.6229 | 0.742 | 0.836 | 0.858 | 0.018 |
| yolo11m+assign (lap-4 D2) | 0.8441 | 0.8284 | 0.6151 | **0.7508** | 0.823 | 0.841 | 0.014 |
| yolo11m control (lap-4) | 0.8449 | 0.8257 | 0.5951 | 0.7299 | 0.834 | 0.850 | 0.019 |

Read: YOLO26m base > vanilla yolo11m overall, BUT loses to yolo11m+assignment on **very-tiny (<8px)**
recall — so the tiny-aware lever still has something the bigger backbone doesn't. That gap is exactly
what lap-5 tests: does YOLO26m+assignment win both?

**Where the parked artifacts live (preserved, not deleted):**
- eval json: `05-07-2026-Novelty-Lap\results\pc1\runs_s1\s1_eval_yolo26m.json`
- model (PC-1): `...\10-07-2026-Novelty-Lap-3\scripts\runs_anchor\yolo26m\weights\best.pt`
- local weights (PC-1): `...\scripts\yolo26m.pt` (43MB, downloaded on laptop, copied over — loads offline)
- runner: `10-07-2026-Novelty-Lap-3\scripts\21_yolo26_anchor.py`

## Planned runs (matched to the lap-4 protocol: scene-split, AdamW, seed 0, F2 primary)
1. YOLO26m + CBAM
2. YOLO26m + CBAM + P2 head
3. YOLO26m + CBAM + P2 + tiny-aware assignment (α=0.5) — the full detector
4. #3 + sim-to-real joint fine-tune (C2A + drone + campus + disaster) — the full pipeline
Eval every run with `18_s1_eval.py` on C2A scene-split TEST (+ the real held-out sets for #4).

## ⚠️ Feasibility flags — VERIFY on PC-1 BEFORE building (web-verify + grep the installed source)
YOLO26 is not YOLO11 internally; our patches may not transplant. Check FIRST:
- **Assignment patch (`20_assign_patch.py`)** monkeypatches `TaskAlignedAssigner.iou_calculation`.
  Confirm YOLO26m's loss still uses `TaskAlignedAssigner` (grep the installed `ultralytics/utils/tal.py`
  + `.../loss.py`). If YOLO26 changed the assigner/loss (it is marketed end-to-end / NMS-free), the α
  lever needs re-targeting — this is the #1 unknown.
- **CBAM + P2 head insertion:** our modules attach at YOLO11m yaml layer indices. Diff `yolo26m.yaml`
  vs `yolo11m.yaml` (neck/head differ) and find the correct P2 tap + CBAM slots. Re-verify param/GFLOP
  budget after.
- **Eval assumptions:** if YOLO26m is NMS-free/end-to-end, confirm `18_s1_eval.py` (conf sweep,
  per-size recall, pycocotools) still applies unchanged; check how predictions/NMS are exposed.
- **Efficiency row:** record YOLO26m params/GFLOPs/latency on the 4070 Ti S (it is likely heavier than
  yolo11m's 19.6M — the honest cost side of any win).

## Status
- [ ] create branch `novelty-lap-5`
- [ ] verify the 4 feasibility flags on PC-1
- [ ] runs 1–4 + evals
- [ ] compare vs lap-4 yolo11m table → complementary or subsumed?
