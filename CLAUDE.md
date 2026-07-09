# CLAUDE.md — thesis project map (read this first, every session)

**Project:** Undergrad thesis → paper. Tiny/occluded HUMAN detection in aerial DISASTER imagery
(drone/UAV). Base model YOLO11m (Ultralytics). Datasets: C2A (synthetic composite), SARD (real),
own 3-altitude drone footage (10/30/50 m, 4K).

**CURRENT GOAL (lap 2, branch `novelty-lap-2`, since 2026-07-08):** design a NOVEL ARCHITECTURE
(composite, not a single plug-and-play swap) on the YOLO11m base that credibly outranks 2025/26
published results for tiny/occluded aerial humans. Protocol/eval work from lap 1 (scene split,
fusion, NWD) is SUPPORTING material, not the headline.

## Branch map
- `main` — defense-report era state.
- `novelty-direction` — lap 1: leakage audit, scene-disjoint split, C-WBF fusion, NWD pilots.
- `novelty-lap-2` — lap 2 (CURRENT): architecture novelty. Folder `08-07-2026-Novelty-Lap-2/`.

## Folder map (top level of `d:\Academics\thesis folder\`)
- `08-07-2026-Novelty-Lap-2/` — CURRENT lap: `ARCHITECTURE_PROPOSAL.md` (FCCG-YOLO: high-freq
  evidence branch × large-kernel context gate + FreqFusion-lite + tiny/occlusion loss stack; gated
  plan S0-S5) + `docs/2026-07-08_A_image_needs_analysis.md` (8 measured needs N1-N8) +
  `docs/2026-07-08_B_sota_research.md` (verified 2025/26 SOTA + mechanism catalog; SOTA bar on C2A
  = YOLOv9-e 0.8927 mAP50 / 0.6883 mAP @57M, official leaky split; NOTE 0.6883 contradicts our
  "~0.615 ceiling" — ceiling NOT absolute, reproduce YOLOv9-e under our protocol).
- `05-07-2026-Novelty-Lap/` — lap 1: `scripts/` (00-07 tools, see below), `evidence/` (leakage
  proof + scene_assignment.csv), `results/` (**RESULTS INBOX** — every remote-PC result lands in
  `results\<pc>\<date>_<gate>_<name>\` and MUST get an entry in `results\MANIFEST.md`),
  `RUNNING_GUIDE.md` (gates G0-G4 decision tree), `PC_CHECKLIST.md` (per-PC commands, PowerShell),
  `PC1_RUN_STATUS.md` (live G1 run record).
- `c2a/C2A_Dataset/new_dataset3/` — official C2A (LEAKY: 98% of test scenes appear in train).
  `new_dataset3_scenesplit_v1/` — scene-disjoint re-split (clean benchmark; built by script 01).
- `Drone Shoot/` — own videos + `extracted_v1/` (60 frozen TEST frames + manifest.json; labeling
  in Roboflow → COCO JSON; test frames are NEVER trained on).
- `Last Month/` — the completed thesis-era work: ablation runs (`24_01_26- Benchmarking YOLOs/`),
  SAHI+TTA reports, cross_dataset_SARD (zero-shot collapse ~99%), deployable_model (epoch125.pt =
  joint C2A+SARD model, C2A 0.878/SARD 0.917), system_spec*.md (METRIC CONTRACT — follow it),
  docs/ (dated findings).
- `01-02-2026- ablation study/`, `01-03-2026-Onward Model trying/`, `31-03-26(Mamba-ViT-CNN)/` —
  history: early ablations, Mamba attempts (injection bug, AtrousSSM failure), SAHI/TTA evals.
- `Defense/` — the defense report (XeLaTeX). DO NOT touch without explicit request.
- `docs/` — dated research/decision docs. Key: `2026-07-04_novelty_direction_research.md`,
  `2026-07-05_publication_roadmap_need_analysis.md`, `Novelty_chat_with_fable5.md`.

## Lap-1 scripts (in `05-07-2026-Novelty-Lap/scripts/`, all selftested)
00 leakage evidence · 01 scene-split builder (deterministic via evidence\scene_assignment.csv) ·
02 NWD loss patch + CBAM-compat loader + `--check-load` (ALWAYS run before training on a new PC) ·
03 C-WBF fusion lib · 04 eval runner (FULL spec §6/§11 metric contract; `--save-preds` = resumable;
fusion is CPU-heavy at conf 0.001) · 05 drone frame extractor · 06 self-train builder (test-frame
abort guard) · 07 GPU flusher (shared boxes; dry-run first).

## PC map (all remote via AnyDesk; laptop = analysis/labeling/git only — its ultralytics 8.0.196
CANNOT load YOLO11 ckpts)
- PC-1 `E:\Thesis_mofazzal_2007074`, RTX 4070 Ti S 16GB, venv mofazzal1 (PowerShell + exec-policy
  line). THE PROTOCOL MACHINE — all comparable retrains run here.
- PC-2 `D:\student_2k20\2007074`, 2×A6000 SHARED (GPU1=ours, never grab busy GPU0), venv 2007074.
- PC-3 `D:\2007074` (lab spare), venv 2007074.
- PC-4 `D:\thesis_2007074`, RTX 4070 12GB fp32-only (**--no-amp mandatory**), venv 2007074;
  has epoch125.pt + datasets + drone videos.
- Rules: activate venv first (`(env)` in prompt, `conda deactivate` if `(base)` shows), commands
  are PowerShell, kill-safe/resume everywhere (re-set `$env:C2A_ROOT` before resuming PC-1 runs).

## Key numbers (verified; full details in memory + results\MANIFEST.md)
- CBAM+P2 official C2A test: mAP50 .8533 / AP .6153 / AP_small .6156 / VT-recall .7575 (19.6M, 87G,
  15.7ms). Scene-split (clean): mAP50 .8372 / AP .6107 / AP_small .6132 → leakage ≈ −1.6 AP50 pt
  (MODEST — supporting finding, not headline).
- C2A data: 63% boxes <16px, 34.5% <8px, mAP50-95 ceiling ~.615 (paste-label noise), 0 empty images.
- Gate G2 CLOSED: NWD α-sweep (0/.5/.7) → +0.73 VT-recall at α=.5, does not scale, AP_small flat.
- Mamba CLOSED (2 genuine negative runs). ECA null. Copy-paste aug negative.
- SARD: median 53px (NOT a tiny-object set), test 96% Roboflow near-dupes (197 unique bases).
- Gate G1 half-done: scene-split CBAM+P2 ✅ (above); baseline retrain RUNNING on PC-1.
- Gate G3 (C-WBF vs NMS): dev pass incomplete; redo cheaply on clean model later (--conf-floor).

## Standing user rules
- Verify feasibility before proposing; no plug-and-play-only novelty; save detailed outputs to
  dated md files in docs/ (or the current lap folder); PowerShell syntax; check files exist on the
  target PC before giving run commands; every figure/number traceable; don't inflate findings —
  state effect sizes honestly.
