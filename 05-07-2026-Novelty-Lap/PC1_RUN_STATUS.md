# PC-1 leakage retrain — LIVE RUN RECORD (started 2026-07-08 ~02:17)
Keep this file until both G1 runs finish. This is the paper's HEADLINE experiment (scene-disjoint re-split proving C2A background leakage).

## Machine
- **PC-1**, root `E:\Thesis_mofazzal_2007074`, **RTX 4070 Ti SUPER 16 GB**, venv **mofazzal1**, PowerShell.

## Run 1 (IN PROGRESS) — CBAM+P2 on the scene-disjoint split
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

## GATE G1 (the headline result)
When each run finishes, read `runs\<id>\metrics\summary.json`. Compare scene-split vs official-split:
| Model | Official AP50 / COCO-AP_small | Scene-split (fill in) |
|---|---|---|
| CBAM+P2 | 0.853 / 0.616 | ? |
| baseline yolo11m | 0.843 / 0.615 | ? |
A meaningful DROP on the scene split = leakage confirmed = paper headline. (Also record per-size recall — leakage may hit tiny objects hardest.)
