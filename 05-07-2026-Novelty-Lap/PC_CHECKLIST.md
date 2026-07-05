# PC-BY-PC CHECKLIST — PowerShell edition (self-sufficient)
(Every block is complete: terminal, environment activation, `cd`, exact commands. Gates/branches live in RUNNING_GUIDE.md §2.
**All commands below are PowerShell** — that is what you are using. Do NOT paste cmd.exe `::`/`echo`/`dir /s /b`/`activate.bat` lines; those were the old version and caused the `dir /s /b` error.
Legend: ⏱ time · 👀 must appear · ⛔ stop & report.)

---

## 0. Universal rules (read once)

1. **Activate the PC's env, and ONLY that env.** The prompt must read `(venv-name) PS ...` — with **no `(base)`** in front. If you see `(base)`, run `conda deactivate` once (removes Anaconda from PATH; your venv stays). To stop base auto-activating in every future window: `conda config --set auto_activate_base false`.
2. **Confirm the right Python** after activating: `where.exe python` → the FIRST line must be the venv's `python.exe`. If it isn't, you're about to run the wrong Python (no CUDA).
3. **Always `cd` into `...\05-07-2026-Novelty-Lap\scripts`** before running scripts — they use relative paths (`..\evidence\...`) and drop outputs into the current directory.
4. **Kill-safe / resume:** every long command can be stopped anytime (slot ends, power cut). Re-run the **same command**: `01` re-links instantly, `04 --save-preds` prints `[resume] N images already predicted`, `02 --train --resume` continues from `last.pt`. Nothing corrupts.
5. **Shared GPU (PC-2/PC-3):** scripts auto-pick a free GPU (free = <1 GB & <10% util; prefer GPU 1; never grab a busy GPU). 👀 first line: `[gpu] usage [...] -> picking GPU 1`. Manual override: `$env:CUDA_VISIBLE_DEVICES="1"` before launching. Keep `--device 0` in commands — after pinning, "0" IS the picked physical GPU.
6. **Set-ExecutionPolicy first if Activate.ps1 is blocked:** `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` (per-session, harmless).
7. **When a run finishes**, copy `*_ablation.csv`, `*_artifacts\`, `cal.json`, `preds_*.json`, `metrics\summary.json` back to the laptop under `05-07-2026-Novelty-Lap\results\<pc-name>\` and `git commit` on branch `novelty-direction`.
8. **The laptop cannot load YOLO11 weights** (its ultralytics 8.0.196 predates `C3k2`) — the laptop is ONLY for frame extraction + Roboflow labeling + git. All model runs happen on PC-1/2/3/4.

---

## LAPTOP (this machine, D:) — Day 1, ~30 min of your time
Terminal: PowerShell. No venv needed (Anaconda python is fine here — extraction is CPU-only).
```powershell
cd "d:\Academics\thesis folder\05-07-2026-Novelty-Lap\scripts"
python 05_extract_drone_frames.py
```
👀 `Drone Shoot\extracted_v1\` with `test_frames\{10m,30m,50m}` (20 each), `selftrain_frames` (~80 each), `manifest.json`.
Then:
1. **START LABELING** `test_frames\` in Roboflow — all 60 frames, class `person`, box every person. Export **COCO JSON**. This is the only long manual task; start it today.
2. `git add "05-07-2026-Novelty-Lap" "Drone Shoot/extracted_v1/manifest.json" docs; git commit -m "novelty lap: pipeline + evidence + frozen drone test frames"`
3. Copy the whole `05-07-2026-Novelty-Lap\` folder to each PC (paths in each section).

---

## PC-1 — RTX 4070 Ti SUPER (protocol machine) — Phase 1, launch Day 1
Terminal: PowerShell. Venv: **mofazzal1**. Copy folder to: `E:\Thesis_mofazzal_2007074\05-07-2026-Novelty-Lap`.
```powershell
# 0. activate (only this env; no base)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
E:\Thesis_mofazzal_2007074\mofazzal1\Scripts\Activate.ps1
where.exe python        # first hit must be ...\mofazzal1\Scripts\python.exe

# 1. rebuild the scene split bit-for-bit (~2 min)
cd "E:\Thesis_mofazzal_2007074\05-07-2026-Novelty-Lap\scripts"
python 01_make_scene_split.py --root "E:\Thesis_mofazzal_2007074\common\c2a\C2A_Dataset\new_dataset3" --assignment ..\evidence\scene_assignment.csv
# 👀 ALL VALIDATION CHECKS PASSED + 6135/2040/2040   ⛔ anything else

# 2. fresh script folders (so splits.md5 freezes on the NEW split)
robocopy "E:\Thesis_mofazzal_2007074\Benchmarking YOLOs\CBAM_P2Head" "E:\Thesis_mofazzal_2007074\Benchmarking YOLOs\CBAM_P2Head_SceneSplit" /E /XD runs smoke __pycache__
robocopy "E:\Thesis_mofazzal_2007074\Benchmarking YOLOs\Yolo11m" "E:\Thesis_mofazzal_2007074\Benchmarking YOLOs\Yolo11m_SceneSplit" /E /XD runs smoke __pycache__

# 3. RUN 1 of 2 — CBAM+P2 on the scene split (⏱ ~1.5 days unattended)
$env:C2A_ROOT = "E:\Thesis_mofazzal_2007074\common\c2a\C2A_Dataset\new_dataset3_scenesplit_v1"
cd "E:\Thesis_mofazzal_2007074\Benchmarking YOLOs\CBAM_P2Head_SceneSplit"
python yolo11m_cbam_p2head_thesis.py *>&1 | Tee-Object -FilePath .\scenesplit_run.log

# 4. RUN 2 of 2 — AFTER run 1 finishes (single-GPU rule):
$env:C2A_ROOT = "E:\Thesis_mofazzal_2007074\common\c2a\C2A_Dataset\new_dataset3_scenesplit_v1"
cd "E:\Thesis_mofazzal_2007074\Benchmarking YOLOs\Yolo11m_SceneSplit"
python yolo11m_thesis.py *>&1 | Tee-Object -FilePath .\scenesplit_run.log
```
👀 within 15 min of each: smoke PASS → `[splits] FROZEN new split md5` → `runs\<id>\ultra\results.csv` gaining a row every ~2–2.5 min.
⛔ `Split md5 mismatch` = launched in OLD folder (use `_SceneSplit`) · GPU parked at P8, no new rows = dataloader deadlock (runbook §1.1: NUM_WORKERS=4, CACHE='disk', relaunch — resumes).
Power cut: re-run the same command — thesis scripts resume from `last.pt`.
→ **GATE G1**: each run's `metrics\summary.json` vs June official-split numbers (guide §4).

---

## PC-2 — 2× RTX A6000, SHARED (GPU 1 = yours) — Phase 3, launch Day 1 (no training)
Terminal: PowerShell. Venv: **2007074**. Folder at: `D:\student_2k20\2007074\05-07-2026-Novelty-Lap`.
`epoch125.pt` is NOT at the root here — it lives inside the joint run folder (path below, confirmed 157,740,736 bytes).
```powershell
# 0. activate ONLY the venv (drop base if present)
cd "D:\student_2k20\2007074"
.\2007074\Scripts\Activate.ps1
conda deactivate 2>$null      # only if the prompt shows (base); harmless otherwise
where.exe python              # first hit must be D:\student_2k20\2007074\2007074\Scripts\python.exe

# weights path as a variable (reuse everywhere)
$W = "D:\student_2k20\2007074\A6000 run\runs_joint\20260627_162506_cbam_p2head_joint_c2a_sard\ultra\weights\epoch125.pt"

cd "D:\student_2k20\2007074\05-07-2026-Novelty-Lap\scripts"

# 1. prove it loads (~1 min)  👀 "[check-load] OK ... CBAM_present=True"   ⛔ "Can't get attribute 'CBAM'" = old 04/02, re-copy scripts
python 02_nwd_loss_patch.py --check-load --weights $W

# 2. rebuild the split (~2 min)  👀 ALL VALIDATION CHECKS PASSED
python 01_make_scene_split.py --root "D:\student_2k20\2007074\common\c2a\C2A_Dataset\new_dataset3" --assignment ..\evidence\scene_assignment.csv

# 3. FIRST a fast sanity pass (200 imgs, ~few min) to confirm the whole chain, THEN the full run:
python 04_eval_fusion_ablation.py --weights $W --limit 200 --images-dir "D:\student_2k20\2007074\common\c2a\C2A_Dataset\new_dataset3_scenesplit_v1\val\images" --gt-json "D:\student_2k20\2007074\common\c2a\C2A_Dataset\new_dataset3_scenesplit_v1\val\val_annotations.json" --slices 256 --overlap 0.30 --tta-imgsz 1280 --modes nms --save-preds preds_val_sanity.json

# 4. FULL VAL predictions (⏱ SLOW — see note; kill-safe, resumes)
python 04_eval_fusion_ablation.py --weights $W --images-dir "D:\student_2k20\2007074\common\c2a\C2A_Dataset\new_dataset3_scenesplit_v1\val\images" --gt-json "D:\student_2k20\2007074\common\c2a\C2A_Dataset\new_dataset3_scenesplit_v1\val\val_annotations.json" --slices 256 --overlap 0.30 --tta-imgsz 1280 --modes nms --save-preds preds_val.json

# 5. fit calibration on VAL (seconds, offline, CPU ok)
python 04_eval_fusion_ablation.py --load-preds preds_val.json --gt-json "D:\student_2k20\2007074\common\c2a\C2A_Dataset\new_dataset3_scenesplit_v1\val\val_annotations.json" --fit-cal cal.json

# 6. FULL TEST predictions (⏱ slow, kill-safe)
python 04_eval_fusion_ablation.py --weights $W --images-dir "D:\student_2k20\2007074\common\c2a\C2A_Dataset\new_dataset3_scenesplit_v1\test\images" --gt-json "D:\student_2k20\2007074\common\c2a\C2A_Dataset\new_dataset3_scenesplit_v1\test\test_annotations.json" --slices 256 --overlap 0.30 --tta-imgsz 1280 --save-preds preds_test.json

# 7. TEST ablation with calibration + CIs (minutes, offline) -> the G3 table
python 04_eval_fusion_ablation.py --load-preds preds_test.json --gt-json "D:\student_2k20\2007074\common\c2a\C2A_Dataset\new_dataset3_scenesplit_v1\test\test_annotations.json" --cal cal.json --bootstrap 1000

# 8. SARD test (copy the GT json next to the scripts first; tiles auto-skip at 640x640):
copy "D:\student_2k20\2007074\A6000 run\..\cross_dataset_SARD\sard_test_coco_gt.json" .    # adjust if your SARD path differs; find it: Get-ChildItem D:\student_2k20\2007074 -Recurse -Filter sard_test_coco_gt.json
python 04_eval_fusion_ablation.py --weights $W --images-dir "D:\student_2k20\2007074\common\sard\search-and-rescue\test\images" --gt-json ".\sard_test_coco_gt.json" --slices 0 --tta-imgsz 1280 --save-preds preds_sard.json
python 04_eval_fusion_ablation.py --load-preds preds_sard.json --gt-json ".\sard_test_coco_gt.json" --cal cal.json --bootstrap 1000
```
**⏱ WHY 04 IS SLOW (expected, not a hang):** per image it runs whole@640 **+ TTA@1280 (augment=True ≈ 3–4 high-res passes) + 256px tiles**, batch=1, ×2040. TTA@1280 dominates. `[infer] N/2040` counting up = working.
Speed levers if time-boxed: `--limit 200` (quick pass, banks a CSV) · drop `--tta-imgsz 1280` (biggest cut, but then val has no TTA source to calibrate — only for a rushed first look). For the real numbers, let the full run go and resume tomorrow.
👀 outputs: `preds_test_ablation.csv` (3 rows nms/wbf/cwbf, full metric columns) + `preds_test_artifacts\` → **GATE G3** (guide §6).
Slot ends? Ctrl+C anytime; tomorrow: activate venv → `cd scripts` → re-run the same command (it resumes).

---

## PC-3 — lab spare (`D:\2007074`) — idle until G2 passes, then the α-sweep
Terminal: PowerShell. Venv: **2007074**. Copy folder to: `D:\2007074\05-07-2026-Novelty-Lap`. Weights: `D:\2007074\epoch125.pt` (per SETUP_PC3 Part E; if missing, find with the Get-ChildItem below).
```powershell
cd "D:\2007074"
.\2007074\Scripts\Activate.ps1
conda deactivate 2>$null
where.exe python
cd "D:\2007074\05-07-2026-Novelty-Lap\scripts"

# confirm weights exist (else copy epoch125.pt to D:\2007074\ first)
Get-ChildItem "D:\2007074" -Recurse -Filter epoch125.pt | Select-Object FullName, Length

python 01_make_scene_split.py --root "D:\2007074\common\c2a\C2A_Dataset\new_dataset3" --assignment ..\evidence\scene_assignment.csv
python 02_nwd_loss_patch.py --check-load --weights D:\2007074\epoch125.pt

# one-time: official-split data yaml (pilot isolates the loss change)
@"
path: D:/2007074/common/c2a/C2A_Dataset/new_dataset3
train: train/images
val: val/images
test: test/images
nc: 1
names:
  0: person
"@ | Set-Content -Encoding ascii official_pc3.yaml

# ONLY IF GATE G2 PASSED (guide §5) — α-sweep arms, sequential:
python 02_nwd_loss_patch.py --train --resume --weights D:\2007074\epoch125.pt --data official_pc3.yaml --alpha 0.3 --epochs 25 --lr0 0.0003 --batch 8 --project runs_nwd --name pilot_a03
python 02_nwd_loss_patch.py --train --resume --weights D:\2007074\epoch125.pt --data official_pc3.yaml --alpha 0.7 --epochs 25 --lr0 0.0003 --batch 8 --project runs_nwd --name pilot_a07
```

---

## PC-4 — RTX 4070 12 GB, fp32 ONLY (`D:\thesis_2007074`) — Phase 2 pilot, launch Day 1
Terminal: PowerShell. Venv: under `D:\thesis_2007074` (find it below). Copy folder to: `D:\thesis_2007074\05-07-2026-Novelty-Lap`. Weights: `D:\thesis_2007074\epoch125.pt` (confirmed — the enriched fine-tune used exactly this).
```powershell
# 0. find + activate the venv (the folder under D:\thesis_2007074 with Scripts\Activate.ps1)
Get-ChildItem "D:\thesis_2007074" -Recurse -Filter Activate.ps1 -ErrorAction SilentlyContinue | Select-Object -First 1 FullName
# activate the one it prints, e.g.:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
D:\thesis_2007074\<venv>\Scripts\Activate.ps1
conda deactivate 2>$null
python -c "import torch; print('CUDA', torch.cuda.is_available())"   # must be True

# 1. SARD cross-split leakage audit (1 min — any result is fine, we just record it)
python -c "from pathlib import Path; b=lambda n:n.split('_jpg.rf.')[0].split('.rf.')[0]; S={sp:{b(p.stem) for p in Path(r'D:\thesis_2007074\common\sard\search-and-rescue').glob(sp+'/images/*')} for sp in ('train','valid','test')}; print({k:len(v) for k,v in S.items()}); print('train-test shared bases:', len(S['train']&S['test']), '| valid-test:', len(S['valid']&S['test']))"

# 2. rebuild the split (~2 min)
cd "D:\thesis_2007074\05-07-2026-Novelty-Lap\scripts"
python 01_make_scene_split.py --root "D:\thesis_2007074\common\c2a\C2A_Dataset\new_dataset3" --assignment ..\evidence\scene_assignment.csv

# 3. de-risk the checkpoint load (1 min)  👀 "[check-load] OK ... CBAM_present=True"   ⛔ any traceback
python 02_nwd_loss_patch.py --check-load --weights D:\thesis_2007074\epoch125.pt

# 4. one-time official-split data yaml
@"
path: D:/thesis_2007074/common/c2a/C2A_Dataset/new_dataset3
train: train/images
val: val/images
test: test/images
nc: 1
names:
  0: person
"@ | Set-Content -Encoding ascii official_pc4.yaml

# 5. NWD pilot — CONTROL first, then TREATMENT (⏱ ~6 h each, sequential; --no-amp MANDATORY on this box)
python 02_nwd_loss_patch.py --train --resume --weights D:\thesis_2007074\epoch125.pt --data official_pc4.yaml --alpha 0.0 --epochs 25 --lr0 0.0003 --batch 4 --no-amp --project runs_nwd --name pilot_a00
python 02_nwd_loss_patch.py --train --resume --weights D:\thesis_2007074\epoch125.pt --data official_pc4.yaml --alpha 0.5 --epochs 25 --lr0 0.0003 --batch 4 --no-amp --project runs_nwd --name pilot_a05
# 👀 within the FIRST minute of each: "[nwd-patch] ACTIVE" AND "[nwd-verify] OK"   ⛔ "[nwd-verify] FATAL" or NaN losses (you forgot --no-amp)

# 6. evaluate BOTH pilots on official VAL (⏱ ~30 min each; whole-frame only = fast, no TTA/tiles) -> GATE G2
python 04_eval_fusion_ablation.py --weights runs_nwd\pilot_a00\weights\best.pt --images-dir "D:\thesis_2007074\common\c2a\C2A_Dataset\new_dataset3\val\images" --gt-json "D:\thesis_2007074\common\c2a\C2A_Dataset\new_dataset3\val\val_annotations.json" --slices 0 --tta-imgsz 0 --modes nms --save-preds preds_val_a00.json
python 04_eval_fusion_ablation.py --weights runs_nwd\pilot_a05\weights\best.pt --images-dir "D:\thesis_2007074\common\c2a\C2A_Dataset\new_dataset3\val\images" --gt-json "D:\thesis_2007074\common\c2a\C2A_Dataset\new_dataset3\val\val_annotations.json" --slices 0 --tta-imgsz 0 --modes nms --save-preds preds_val_a05.json
# compare coco_AP_small + per_size_recall.very_tiny/tiny between a05 and a00 -> GATE G2
```
LATER (once your 60 drone frames are labeled): Phase 4 ladder + self-training on this PC — commands in RUNNING_GUIDE.md §7.

---

## Dependency picture
```
Day 1 parallel:  LAPTOP extract+label | PC-1 scene retrain #1 | PC-4 audit->check-load->a00 | PC-2 check-load->split->val preds
Day 2-3:         labeling continues   | PC-1 retrain #2       | PC-4 a05 + val evals        | PC-2 test ablation + SARD
Day ~4:          G1 + G2 + G3 decided -> RUNNING_GUIDE §2 tree; PC-3 joins if G2 passed
Day 4+:          Phase 4 ladder + self-training on PC-4 (needs YOUR labels)
Anytime after G1: start writing the leakage/motivation section (already fully supported by evidence/)
```

## Quick fixes for things that already bit us
- `dir : positional parameter ... 'X'` → you pasted a cmd.exe command in PowerShell. Use `Get-ChildItem -Recurse -Filter X`.
- prompt shows `(base)` → `conda deactivate` (once), then re-check `where.exe python`.
- `Can't get attribute 'CBAM'` when loading weights → your `04`/`02` on that PC is an OLD copy; re-copy the scripts folder.
- `Can't get attribute 'C3k2'` → that machine's ultralytics is too old for YOLO11 (the laptop is 8.0.196) — don't load weights there; use PC-1/2/3/4.
- `04` "taking forever" → expected (TTA@1280 + tiles, batch=1). Use `--limit 200` for a quick pass; it's kill-safe and resumes.
- `Activate.ps1 cannot be loaded ... disabled` → run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` then activate.
