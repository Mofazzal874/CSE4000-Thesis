# PC-1 leakage retrain — LIVE RUN RECORD (started 2026-07-08 ~02:17)
Keep this file until both G1 runs finish. This is the paper's HEADLINE experiment (scene-disjoint re-split proving C2A background leakage).

## Machine
- **PC-1**, root `E:\Thesis_mofazzal_2007074`, **RTX 4070 Ti SUPER 16 GB**, venv **mofazzal1**, PowerShell.

## ✅ Run 1 DONE (2026-07-24 confirmed; ran 262 ep, F2-early-stop best epoch 152, 12.1h) — CBAM+P2 on the scene-disjoint split
**HEADLINE NUMBERS (test / val), run `20260708_022132_yolo11m_cbam_p2head_s0_nogit`:**
- TEST: AP50 **0.8372** · AP **0.6107** · AP_small **0.6132** · VT-recall **0.7454** · tiny 0.8459 · small 0.8591
- VAL:  AP50 0.8565 · AP 0.6311 · AP_small 0.6336
- Effic: 19.57M params · 86.7 GFLOPs · 15.67ms e2e · ECE 0.026 · CO2 2.0kg
- LEAKAGE vs official split (0.8533/0.6153/0.6156/0.7575): AP50 −1.6 · AP_small −0.2 · VT −1.2 → MODEST (supporting finding).
- **= the number FCCG-YOLO must beat at full protocol.** Copy run folder → `results\pc1\2026-07-24_G1_scenesplit_cbam_p2\`.

## Run 1 (ORIGINAL launch record) — CBAM+P2 on the scene-disjoint split
- Folder: `E:\Thesis_mofazzal_2007074\Benchmarking YOLOs\CBAM_P2Head_SceneSplit`
- Dataset: `E:\Thesis_mofazzal_2007074\common\c2a\C2A_Dataset\new_dataset3_scenesplit_v1` (6135/2040/2040, frozen md5 train=6cd79d40… val=abe020e9… test=83c91bea…)
- Verified correct at launch: DATASET_ROOT = scenesplit_v1, val=2040 (scene split, not official 2043), `[splits] FROZEN new split md5`, CBAM@10, 19,592,246 params, 764/764 transferred, AMP passed.
- Config (from script): SEEDS=[0], batch=8/nbs=16, PATIENCE=50, F2_PATIENCE=40, 300-ep cap, AdamW, cache=ram, deterministic. ~1.5 days.
- Launch command:
  ```powershell
  $env:C2A_ROOT = "E:\Thesis_mofazzal_2007074\common\c2a\C2A_Dataset\new_dataset3_scenesplit_v1"
  cd "E:\Thesis_mofazzal_2007074\Benchmarking YOLOs\CBAM_P2Head_SceneSplit"
  python yolo11m_cbam_p2head_thesis.py *>&1 | Tee-Object -FilePath .\scenesplit_run.log
  ```

## RESUME (failproof — the run saves last.pt every epoch, auto-resumes)
Any interruption (Ctrl+C / power cut / slot end) → come back and run EXACTLY:
```powershell
E:\Thesis_mofazzal_2007074\mofazzal1\Scripts\Activate.ps1
$env:C2A_ROOT = "E:\Thesis_mofazzal_2007074\common\c2a\C2A_Dataset\new_dataset3_scenesplit_v1"
cd "E:\Thesis_mofazzal_2007074\Benchmarking YOLOs\CBAM_P2Head_SceneSplit"
python yolo11m_cbam_p2head_thesis.py *>&1 | Tee-Object -FilePath .\scenesplit_run.log -Append
```
→ prints `[resume] ... enabling resume=True` + `[RESUMING]`, continues from last epoch.
**3 rules:** (1) ALWAYS re-set `$env:C2A_ROOT` first (env vanishes per session; if you forget, the split-check STOPS you with a mismatch — it won't train wrong data). (2) run from the SAME `_SceneSplit` folder. (3) never delete `runs\` while incomplete.
Progress check: `Get-ChildItem "...\CBAM_P2Head_SceneSplit\runs" -Recurse -Filter results.csv | %{ (Get-Content $_.FullName | Measure-Object -Line).Lines - 1 }`

## Run 2 (AFTER Run 1 finishes — single GPU, sequential) — baseline on scene split
```powershell
robocopy "E:\Thesis_mofazzal_2007074\Benchmarking YOLOs\Yolo11m" "E:\Thesis_mofazzal_2007074\Benchmarking YOLOs\Yolo11m_SceneSplit" /E /XD runs smoke __pycache__
Remove-Item "E:\Thesis_mofazzal_2007074\Benchmarking YOLOs\Yolo11m_SceneSplit\common\splits" -Recurse -Force -ErrorAction SilentlyContinue
$env:C2A_ROOT = "E:\Thesis_mofazzal_2007074\common\c2a\C2A_Dataset\new_dataset3_scenesplit_v1"
cd "E:\Thesis_mofazzal_2007074\Benchmarking YOLOs\Yolo11m_SceneSplit"
python yolo11m_thesis.py *>&1 | Tee-Object -FilePath .\scenesplit_run.log
```

## GATE G1 — BOTH RUNS DONE (CBAM+P2 07-08, baseline yolo11m 07-09) — COMPLETE
Scene-split TEST vs official (COCO AP50 / AP_small):
| Model | Official | Scene-split (clean) | Leakage Δ |
|---|---|---|---|
| CBAM+P2 | 0.8533 / 0.6156 | **0.8372 / 0.6132** (VT-recall 0.7454) | −1.6 AP50 / −0.2 AP_s |
| baseline yolo11m | 0.843 / 0.615 | **0.8193 / 0.5960** (AR_small 0.6684) | ~−1.5 AP50 / ~−1.9 AP_s |
- Leakage MODEST both models (~−1.6 AP50) → supporting finding, not headline (lap-2/3 verdict holds).
- **On the CLEAN split, CBAM+P2 beats baseline +1.8 AP50 / +1.7 AP_small** → confirms CBAM+P2 is the right base for FCCG.
- FCCG-YOLO's bar to beat (scene-split test): 0.8372 / 0.6132 / VT-recall 0.7454.
- Both run folders → copy to `results\pc1\` when convenient. PC-1 now FREE (nothing pending).
