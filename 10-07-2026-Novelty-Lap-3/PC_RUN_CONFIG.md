# PC & Run Config — Lap 3 (snapshot 2026-07-10; source of truth = repo CLAUDE.md, update both)

All PCs remote via AnyDesk. Laptop = analysis/labeling/git only (ultralytics 8.0.196 CANNOT load
YOLO11 checkpoints — never eval on laptop).

| PC | Path | GPU | venv | Role in lap 3 |
|---|---|---|---|---|
| PC-1 | `E:\Thesis_mofazzal_2007074` | RTX 4070 Ti S 16 GB | `mofazzal1` | **PROTOCOL MACHINE** — all comparable 300-ep retrains (currently: G1 baseline retrain RUNNING — do not disturb; queue S3/S4 anchors after) |
| PC-2 | `D:\student_2k20\2007074` | 2×A6000 (GPU1 = ours, NEVER grab busy GPU0) | `2007074` | Heavy prototypes / 50-ep pilots (batch 28 + mem-fraction cap; WDDM spillover past 48 GB is silent) |
| PC-3 | `D:\2007074` (lab spare) | — | `2007074` | Overflow pilots |
| PC-4 | `D:\thesis_2007074` | RTX 4070 12 GB **fp32-only → `--no-amp` mandatory** | `2007074` | Smokes, 50-ep pilots (b4), fine-tune/aug experiments; has epoch125.pt + datasets + drone videos |

## Non-negotiable run rules
1. Activate venv FIRST — `(env)` must show in prompt; if `(base)` shows, `conda deactivate` first.
   PC-1 needs the PowerShell exec-policy line before activation.
2. All commands PowerShell syntax. All training kill-safe/resume; **re-set `$env:C2A_ROOT` before
   resuming PC-1 runs**.
3. Before FIRST training on any PC with new code: run lap-1 script 02 `--check-load`.
4. Every result copies back to `..\05-07-2026-Novelty-Lap\results\<pc>\<date>_<gate>_<name>\` and
   gets a `MANIFEST.md` entry — single inbox across laps, no exceptions.
5. Ablation defaults: PATIENCE=50, F2_PATIENCE=40, SEEDS=[0..4] (user override of spec 30/20/1);
   optimizer pinned AdamW lr0=0.001 (MuSGD falsified locally).
6. Metric contract: **`Last Month\system_spec_thesis.md` §6 (thesis MUST-have set, F2 primary,
   per-size recall bins) + `Last Month\system_spec.md` §11 (full catalog: dynamics, efficiency,
   calibration, arch-specific, env.json, §16 run-IDs)** via lap-1 script 04 (`--save-preds` =
   resumable; fusion is CPU-heavy at conf 0.001). Lap-3 stage→contract mapping + the NEW
   FCCG-specific metric subset (gate maps, gate-GT contrast, gate-off delta):
   `docs\2026-07-12_metric_contract_reference.md`.
7. GPU flush only with lap-1 script 07, dry-run first (shared boxes).

## Lap-3 specific placement (planned, amend as gates fire)
- S0 module selftests + 2-ep smoke → laptop code, PC-4 smoke.
- S1/S2 paired 50-ep pilots (scene-split, seed 0) → PC-4 (fp32 b4) or PC-2 GPU1.
- S3 full 300-ep protocol runs + S4 anchors (YOLOv9-e repro, YOLO26m-AdamW baseline) → PC-1 queue,
  AFTER the running G1 baseline finishes.
- Possible DA pillar probe (GRL domain-classifier on frozen CBAM+P2 features; C2A-vs-own-drone
  frames) → PC-2 GPU1 or PC-4; unlabeled frames extracted with lap-1 script 05 (test-frame guard:
  script 06 logic — the 60 frozen test frames NEVER enter any train pool).
