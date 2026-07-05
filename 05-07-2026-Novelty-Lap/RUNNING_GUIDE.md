# NOVELTY LAP — RUNNING GUIDE (decision-tree, gate-driven)
**Branch:** `novelty-direction` · **Created:** 2026-07-05 · **You read this top-to-bottom and run the commands. Every phase ends in a GATE with explicit PASS/FAIL arrows. Every FAIL arrow still lands on something publishable.**

---

## 0. What is already DONE (verified on the laptop, 2026-07-05)

| Artifact | Status | Where |
|---|---|---|
| Leakage evidence pack | ✅ RAN: 98.28% of test scenes in train, 96.97% of test images share a train background, 0 unseen-scene test images; 6 visual proof montages | `evidence/leakage_stats.json`, `evidence/sibling_pairs/` |
| SARD-test duplicate audit | ✅ RAN: 570 test images = only **197 unique base photos** (174 bases duplicated) | `evidence/leakage_stats.json → sard_test_audit` |
| Scene-disjoint split `c2a_scenesplit_v1` | ✅ BUILT + validated: 2,039 perceptual background clusters, 6135/2040/2040 (60/20/20), size/category distributions matched, zero cluster crossing, COCO GT regenerated | `c2a/C2A_Dataset/new_dataset3_scenesplit_v1/` + canonical `evidence/scene_assignment.csv` |
| C-WBF fusion module | ✅ SELFTEST 6/6 (exact-value fusion, permutation, localization>NMS, calibration reduces ECE) | `scripts/03_cwbf_fusion.py` |
| Fusion ablation runner | ✅ SYNTHETIC end-to-end PASSED (wbf ≥ nms on AP50/F1/meanIoU after max-score fix) | `scripts/04_eval_fusion_ablation.py` |
| NWD loss patch | ✅ SELFTEST 5/5 incl. live patch/unpatch; train-time abort-if-inactive callback (mamba-bug insurance) | `scripts/02_nwd_loss_patch.py` |
| Drone frame extractor | ✅ SMOKE-RAN on the real 10/30/50 m videos (2027/1816/2238 frames @60 fps) | `scripts/05_extract_drone_frames.py` |
| Self-train dataset builder | ✅ SELFTEST (conf/size filters, negatives, **abort if a test frame enters the pool**, SAHI-guided preds-json path) | `scripts/06_selftrain_build.py` |
| Paper-claim verification | ✅ C2A paper + official GitHub state **no split methodology, no leakage acknowledgment**; Table 5 = 0.259/0.660 confirmed | `docs/2026-07-04_*` + this file §1 |

**Nothing about the finding requires redoing your existing work.** Official-split results stay in the paper (they're the comparable-to-prior-art numbers). The scene split ADDS a second evaluation protocol.

---

## 1. Machine map

| Machine | GPU | Role in this plan |
|---|---|---|
| Laptop (D:) | none | analysis, split generation (done), labeling prep |
| **PC-1** | RTX 4070 Ti SUPER 16 GB (`E:\Thesis_mofazzal_2007074`) | Phase 1 scene-split retrains (the protocol machine) |
| **PC-2** | 2× RTX A6000 (`D:\student_2k20\2007074`) | Phase 3 fusion evals (fast), spare trainer, NWD full run |
| **PC-3** | lab (`D:\2007074`) | second arm of the NWD α-sweep / overflow |
| **PC-4** | RTX 4070 12 GB, **fp32 only** (`D:\thesis_2007074`, has `epoch125.pt`) | Phase 2 NWD pilots, Phase 4 fine-tunes (`--no-amp`, batch 4) |

Copy `05-07-2026-Novelty-Lap/` (scripts + evidence) to every PC you use. Python needs: numpy, opencv-python (already everywhere), ultralytics+pycocotools (already on PCs).

---

## 2. THE DECISION TREE

```
P0 Verify & port (laptop+PC, 0.5 day)
 └─ G0: split reproduced on PC? SARD cross-split audit clean?
     ├─ YES ────────────────────────────────► P1
     └─ NO  → fix paths / use --assignment CSV (deterministic); SARD dirty →
              dedupe SARD test (report both numbers)          ► P1
P1 Quantify leakage (PC-1, ~3 GPU-days, runs UNATTENDED)
 └─ G1: scene-split test drop vs official split?
     ├─ ≥2 pts AP50 (or ≥1.5 COCO AP)  → HEADLINE = "benchmark leakage" ► P2∥P3∥P4
     ├─ 0.5–2 pts                      → still a finding, softer wording ► P2∥P3∥P4
     └─ <0.5 pts                       → HEADLINE becomes "split audit + protocol
                                          + real-world benchmark" (also novel,
                                          also publishable)              ► P2∥P3∥P4
P2 NWD loss (PC-4 pilot ~12 h, then full)          [runs PARALLEL to P3/P4]
 └─ G2: pilot α=0.5 vs α=0 control on val: ΔAP_small or ΔVT-recall?
     ├─ ≥ +1.0 pt → full scene-split run (PC-2, ~1.5 d) + α∈{0.3,0.7} sweep (PC-3)
     ├─ +0.5–1.0  → try α=0.7 once; if still flat → record as ablation row
     └─ < +0.5    → honest NULL row in the paper; paper stands on P1+P3+P4 ► P5
P3 C-WBF fusion (PC-2, ~0.5 day, ZERO training)    [runs PARALLEL]
 └─ G3: vs NMS-merge on C2A test (large-res subset) + SARD test:
     ├─ recovers ≥50% of sliced-F1 loss OR +0.5 AP50/AP_small → CONTRIBUTION
     ├─ smaller but positive → ablation row (still first WBF-in-SAHI-setting data)
     └─ negative → negative-result paragraph; TTA-ensemble numbers still stand ► P5
P4 Real-world (PC-4 + ~3 h of YOUR labeling)       [START LABELING DAY 1]
 ├─ 4a: extract + label 60 frames (3 altitudes) → frozen real test set
 ├─ 4b: eval ladder {epoch125, +enriched, +NWD, ±C-WBF} on it  ► altitude table
 ├─ 4c: SAHI-guided self-training on the unlabeled pool → fine-tune
 └─ G4: adapted model vs best ladder entry on real test:
     ├─ ΔmAP50 ≥ +2 pts → deployment-adaptation claim
     └─ flat/negative   → report as finding (pseudo-labeling limits at 60 fps
                          same-site footage); the ladder table is the contribution
P5 Assemble paper (3–4 days, no GPU) — §8
```

**Failure-floor guarantee:** even if G2, G3 AND G4 all fail, the paper still contains: (i) the leakage audit + scene-disjoint protocol (P1, novel, already secured), (ii) the SARD duplication audit, (iii) the measured C2A↔SARD scale-regime analysis, (iv) the 3-altitude real-footage benchmark (nobody has one for C2A models), (v) three instrumented negative results (Mamba, copy-paste, pseudo-labeling). That is a coherent dataset-audit + deployment-study paper for *Drones*/*Remote Sensing*. The gates only decide how much METHOD novelty sits on top.

---

## 3. Phase 0 — Verify & port (laptop + one PC, half a day)

**3.1 (laptop, optional 10 min) Hash-threshold sensitivity** — confirms cluster stability for the paper appendix:
```powershell
cd "d:\Academics\thesis folder\05-07-2026-Novelty-Lap\scripts"
python 01_make_scene_split.py --no-materialize --hash-thr 10 --out "C:\Users\Dell\AppData\Local\Temp\claude\scenesplit_thr10_check"
# EXPECT: cross-prefix merges within ~4±10; cluster count ~2039±20. Record both numbers.
```

**3.2 (each training PC) Reproduce the split bit-for-bit** — uses the canonical CSV, no hashing:
```powershell
cd <...>\05-07-2026-Novelty-Lap\scripts
python 01_make_scene_split.py --root "<PC's new_dataset3 path>" --assignment ..\evidence\scene_assignment.csv
# EXPECT: "ALL VALIDATION CHECKS PASSED", 6135/2040/2040.
```

**3.3 (PC-4) SARD cross-split leakage audit** — one command, 1 min. First find the SARD root (`Get-ChildItem D:\thesis_2007074\common`), then:
```powershell
python -c "from pathlib import Path; import collections; b=lambda n:n.split('_jpg.rf.')[0].split('.rf.')[0]; S={sp:{b(p.stem) for p in Path(r'D:\thesis_2007074\common\sard').glob(sp+'/images/*')} for sp in ('train','valid','test')}; print({k:len(v) for k,v in S.items()}); print('train-vs-test shared bases:', len(S['train']&S['test']), '| val-vs-test:', len(S['valid']&S['test']))"
```
- shared bases = 0 → SARD numbers are clean, say so in the paper.
- shared bases > 0 → **SARD also leaks**; build a base-deduped SARD test (keep first copy per base) and report both numbers. (This strengthens the audit story; it does not sink anything.)

**3.4 (laptop) git-freeze the evidence** — commit `05-07-2026-Novelty-Lap/` (scripts + evidence + this guide) on `novelty-direction`.

---

## 4. Phase 1 — Quantify the leakage (PC-1, ~3 GPU-days unattended)

Two retrains on the scene split, same protocol as the June ablation (AdamW lr0=0.001, 300 ep, patience 50, seed 0, batch 8/16 per model — the runbook `docs/2026-05-28_anydesk_pc_env_setup.md` applies unchanged).

```powershell
# one-time per model dir: fresh copy so splits.md5 freezes on the NEW split
robocopy "E:\Thesis_mofazzal_2007074\Benchmarking YOLOs\CBAM_P2Head" "E:\Thesis_mofazzal_2007074\Benchmarking YOLOs\CBAM_P2Head_SceneSplit" /E /XD runs smoke __pycache__
robocopy "E:\Thesis_mofazzal_2007074\Benchmarking YOLOs\Yolo11m"     "E:\Thesis_mofazzal_2007074\Benchmarking YOLOs\Yolo11m_SceneSplit"     /E /XD runs smoke __pycache__

# point BOTH at the scene split and launch exactly like the June runs:
$env:C2A_ROOT = "E:\Thesis_mofazzal_2007074\common\c2a\C2A_Dataset\new_dataset3_scenesplit_v1"
cd "E:\Thesis_mofazzal_2007074\Benchmarking YOLOs\CBAM_P2Head_SceneSplit"
python yolo11m_cbam_p2head_thesis.py     # then the same for Yolo11m_SceneSplit
```
- Canary (from the runbook): `runs/<id>/ultra/results.csv` gains a row every ~2-2.5 min; if the GPU parks at P8 with no rows → the known dataloader deadlock → `NUM_WORKERS=4`, `CACHE='disk'`, relaunch.
- The script's own `[splits] FROZEN new split md5` line must appear on first launch (fresh folder ⇒ fresh freeze). If it says MISMATCH you launched in the OLD folder — stop, use the copy.

**GATE G1** — fill this table from each run's `metrics/summary.json`:

| Model | Official-split AP50 / COCO AP (June runs) | Scene-split AP50 / COCO AP | Δ |
|---|---|---|---|
| YOLO11m baseline | 0.8432 / 0.6151 | ? | ? |
| CBAM+P2 | 0.8533 / 0.6153 | ? | ? |

Branch per §2. Whatever happens, also record per-size recall — leakage may hit tiny objects hardest (memorized background → sprite pops out).

---

## 5. Phase 2 — NWD pilot then full (PC-4 pilot; PC-2/PC-3 full+sweep)

Pilot = two identical 25-epoch fine-tunes from `epoch125.pt` on the **official** split (isolates the loss change from the split change), differing ONLY in α:

```powershell
cd <...>\05-07-2026-Novelty-Lap\scripts
# control (α=0 ⇒ pure CIoU through the SAME wrapper — apples to apples)
python 02_nwd_loss_patch.py --train --weights D:\thesis_2007074\epoch125.pt --data "D:\thesis_2007074\common\c2a\C2A_Dataset\new_dataset3\..." --alpha 0.0 --epochs 25 --lr0 0.0003 --batch 4 --no-amp --project runs_nwd --name pilot_a00
# treatment
python 02_nwd_loss_patch.py --train --weights D:\thesis_2007074\epoch125.pt --data <same> --alpha 0.5 --epochs 25 --lr0 0.0003 --batch 4 --no-amp --project runs_nwd --name pilot_a05
```
(`--data` = a yaml whose train/val point at the official split — reuse the pattern of the enriched run's `ft_data.yaml`, or point at `new_dataset3` root with a 5-line yaml like the one 01 writes.)

- MUST-SEE line in the log within the first minute: `[nwd-verify] OK — NWD blend active in loss path`. If instead it aborts with `[nwd-verify] FATAL`, ultralytics internals drifted — do NOT trust any run; ping me with the traceback.
- fp16 NaNs on PC-4 are known → `--no-amp` is mandatory there.

Evaluate both `best.pt` with the fusion runner (whole-frame only, honest single-model numbers):
```powershell
python 04_eval_fusion_ablation.py --weights runs_nwd\pilot_a05\weights\best.pt --images-dir <val>\images --gt-json <val>\val_annotations.json --slices 256 --tta-imgsz 0 --save-preds preds_val_a05.json
```
**GATE G2** on val: ΔAP_small / Δrecall_very_tiny (a05 − a00) → branch per §2. If PASS: full 300-ep scene-split run with `02 --train` on PC-2 (batch ≤28 per the VRAM note, or batch 16 to stay safe), α sweep {0.3, 0.7} on PC-3, and 3 seeds for the final table only.

---

## 6. Phase 3 — C-WBF fusion ablation (PC-2, half a day, no training)

Uses the deployable `epoch125.pt` (and later the NWD model). Three commands per dataset:

```powershell
cd <...>\05-07-2026-Novelty-Lap\scripts
# 1) VAL predictions (model runs once; fusion is replayed offline)
python 04_eval_fusion_ablation.py --weights <epoch125.pt> --images-dir <scenesplit-or-official val>\images --gt-json <val json> --slices 256 --overlap 0.30 --tta-imgsz 1280 --save-preds preds_val.json
# 2) fit calibration ON VAL (never on test)
python 04_eval_fusion_ablation.py --load-preds preds_val.json --gt-json <val json> --fit-cal cal.json
# 3) TEST ablation, all merge modes, calibrated cwbf
python 04_eval_fusion_ablation.py --weights <epoch125.pt> --images-dir <test>\images --gt-json <test json> --slices 256 --overlap 0.30 --tta-imgsz 1280 --save-preds preds_test.json
python 04_eval_fusion_ablation.py --load-preds preds_test.json --gt-json <test json> --cal cal.json
```
Repeat step 3 for SARD test (SARD is 640×640: tiles are skipped automatically; the whole+TTA ensemble is what's being fused there). Output: `preds_test_ablation.csv` with one row per mode carrying the FULL spec §6/§11 contract — AP50_allpoint, COCO 12-stat block (AP/AP50/AP75/AP_small/medium/large + AR_1/10/100 + AR by size), OptThr_F1/Best_F1, OptThr_F2/Best_F2, precision/recall/F1/F2 at conf=0.25, TP/FP/FN at both thresholds, 5-bin per-size recall (+n per bin), ECE/MCE/Brier of the fused output, meanIoU_TP, latency mean/p50/p95 + FPS, optional bootstrap CI95 (`--bootstrap 1000`), plus per-mode curve CSVs/PNGs and per-image TP/FP/FN CSVs in `*_artifacts/`; anything uncomputable lands in `skipped_metrics.txt`. Add `--bootstrap 1000` on final paper tables. See PC_CHECKLIST.md for the exact per-machine command sequence.

**GATE G3** vs the `nms` row (≈ your current merge) and vs the June SAHI report (F1 0.829 / VT-recall 0.8292 at slice256): branch per §2.
Bonus row for the paper: `--tta-imgsz 1280` alone vs fused — shows what fusion adds beyond TTA.

---

## 7. Phase 4 — Real-world benchmark + adaptation (PC-4 + your hands; START LABELING FIRST)

**7a. Extract + label (day 1):**
```powershell
python 05_extract_drone_frames.py            # -> Drone Shoot\extracted_v1\{test_frames,selftrain_frames}
```
Label ALL of `test_frames/` (60 frames: 20 per altitude) in Roboflow — box every person, class `person`, **export as COCO JSON** (that's what 04 consumes) and also YOLO txt (for any Ultralytics val). ~2–3 h. Commit `manifest.json`.

**7b. The ladder (each model × each altitude):**
```powershell
python 04_eval_fusion_ablation.py --weights <MODEL> --images-dir extracted_v1\test_frames\10m --gt-json <roboflow-coco-10m.json> --slices 640 --overlap 0.25 --tta-imgsz 1280 --save-preds preds_10m_<model>.json
python 04_eval_fusion_ablation.py --load-preds preds_10m_<model>.json --gt-json <...> --cal cal.json
```
MODELS: `epoch125.pt` → `finetune_enriched best.pt` (this finally completes the pending drone-FP validation of Angle B) → NWD model (if G2 passed). 4K frames: slice 640 is right; expect sliced ≫ whole for 30 m/50 m.

**7c. SAHI-guided self-training:**
```powershell
# pseudo-label the UNLABELED pool with the best ladder model (sliced+fused = SAHI-guided)
python 04_eval_fusion_ablation.py --weights <best-ladder-model> --images-dir extracted_v1\selftrain_frames\10m --slices 640 --overlap 0.25 --tta-imgsz 1280 --save-preds preds_st_10m.json    # repeat 30m/50m, then merge jsons or run per-altitude
python 06_selftrain_build.py --frames-dir extracted_v1\selftrain_frames --preds-json preds_st_10m.json --cal cal.json --manifest extracted_v1\manifest.json --out D:\thesis_2007074\selftrain_v1 --conf-thr 0.6 --keep-empty
# fine-tune exactly like the enriched run: C2A-train + SARD-train + selftrain_v1\train_list.txt
#   (ft yaml pattern = the one in runs_joint\20260702_144117_finetune_enriched\ft_data.yaml)
python 02_nwd_loss_patch.py --train --weights <best-ladder-model> --data ft_selftrain.yaml --alpha <0.5 if G2 passed else 0.0> --epochs 25 --lr0 0.0002 --batch 4 --no-amp --project runs_selftrain --name st_v1
```
Guards you get for free: 06 ABORTS if any test frame sneaks into the pool; sweep `--conf-thr` {0.5, 0.6, 0.7} if the first result is flat (cheap — no re-inference needed, 06 just re-filters).

**GATE G4**: re-run 7b with the adapted model → Δ vs its init on the SAME frozen test set; also re-check C2A/SARD test for no-regression (protocol from the enriched run: stay within ~0.005 of init). Branch per §2.

---

## 8. Phase 5 — Assembly (what goes in the paper)

**Master table** (rows = models, columns = protocols): {baseline, CBAM+P2, (+NWD), (+Mamba null)} × {C2A official test, C2A scene-split test, SARD test (and deduped SARD if 3.3 fired), drone 10/30/50 m} × {single 640, +TTA, +sliced-NMS, +C-WBF}. Most cells already exist or come from one 04 command.

**Claims wording** (keep these exact senses):
1. "We identify and quantify background leakage in the C2A benchmark (98.3% of test scenes appear in training) and release a scene-disjoint protocol" — supported by `evidence/` regardless of gates.
2. "Calibrated weighted box fusion for sliced inference" — claim only at the level G3 supports.
3. "Tiny-object-aware loss" — only if G2 passed; else one honest null sentence.
4. "A 3-altitude real-flight benchmark + same-site adaptation study" — supported by 7b even if G4 fails.
5. Negatives stated plainly: SSM neck (2 genuine runs), copy-paste, enrichment SARD-dip, (maybe) pseudo-labeling.

**Venues in order:** Remote Sensing (MDPI) → Drones (MDPI) → ISPRS JPRS (only if G1 drop is dramatic AND G3 lands) → IEEE Access. Conference fallback: ICPR/ICIP short.

**Reproducibility pack to commit:** `evidence/`, `scene_assignment.csv`, `manifest.json`, all `*_ablation.csv`, seeds, ultralytics/torch versions (already in each run's `env.json`).

---

## 9. Troubleshooting quick refs
- **Shell:** all commands are PowerShell. `dir : positional parameter ... 'X'` means you pasted a cmd.exe line — use `Get-ChildItem -Recurse -Filter X` instead.
- **Env:** prompt shows `(base)` → `conda deactivate` once, then `where.exe python` (first hit must be the venv). `Activate.ps1 ... disabled` → `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first.
- `Can't get attribute 'CBAM'` on load → that PC's `04`/`02` is an OLD copy (pre-fix); re-copy the `scripts\` folder. The fix registers the checkpoint's custom CBAM classes before `YOLO()` loads — both 02 and 04 now do this.
- `Can't get attribute 'C3k2'` → that machine's ultralytics is too old for YOLO11 (the LAPTOP is 8.0.196 → cannot load these weights at all; use PC-1/2/3/4 for anything that loads a model).
- `04` "taking forever" → EXPECTED, not a hang: per image it runs whole@640 + TTA@1280 (augment ≈ 3–4 high-res passes) + 256px tiles, batch=1, ×2040. `[infer] N/2040` climbing = fine. Fast pass: `--limit 200`. Biggest cut: drop `--tta-imgsz 1280` (but then val has no TTA source to calibrate). Kill-safe: resumes from `--save-preds`.
- `[splits] Split md5 mismatch` → you launched in an OLD model folder; use the `_SceneSplit` copy (§4).
- `[nwd-verify] FATAL` → ultralytics loss internals changed; STOP, report traceback. Do not trust the run.
- PC-4 NaN in epoch 0 → you forgot `--no-amp`.
- GPU parked at P8, no results.csv rows → dataloader deadlock (runbook §1.1): workers 4, cache disk.
- GStreamer warnings from OpenCV on video read → cosmetic (Anaconda), ignore.
- Scene-split retrain looks *better* than official → possible (leakage can also depress training); report it, it's still the finding.
