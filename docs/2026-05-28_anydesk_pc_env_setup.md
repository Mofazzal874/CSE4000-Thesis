# AnyDesk PC — Environment Setup Runbook (v2, post-first-run)

**Date:** 2026-05-28 (v2 written same night after the first successful training launch)
**Target machine:** Remote AnyDesk PC — i7-14700K / 128 GB RAM / RTX 4070 Ti SUPER 16 GB / Windows 11 Home
**Working folder:** `E:\Thesis_mofazzal_2007074`
**venv name:** `mofazzal1` → `E:\Thesis_mofazzal_2007074\mofazzal1\`
**Script:** `E:\Thesis_mofazzal_2007074\Benchmarking YOLOs\Yolo9s\yolov9s_final_month.py` *(note the space — always quote the path)*
**Dataset:** `E:\Thesis_mofazzal_2007074\common\c2a\C2A_Dataset\new_dataset3` *(deeper than the spec's example — there's an extra `C2A_Dataset\` level)*
**Shell:** PowerShell only (not cmd.exe)

---

## 0. Verified environment (snapshot taken during first successful run, 2026-05-28)

| Component | Value |
|---|---|
| OS | Windows 11 Home, build 10.0.26200 |
| Python (default) | 3.11.9 at `C:\Program Files\Python311\python.exe` |
| Other Python installed | 3.13 (ignored — we use 3.11 explicitly via `py -3.11`) |
| pip | 24.0 |
| git | 2.47.1.windows.1 |
| NVIDIA driver | **591.86** (supports up to CUDA 13.1) |
| GPU | RTX 4070 Ti SUPER, 16 GB, capability 8.9 |
| CUDA toolkit (`nvcc`) | 12.6.85 — installed but not strictly required (PyTorch wheel ships its own runtime) |
| Free space on E: | ~199 GB at setup time |
| Dataset splits | train **6,129** / val **2,043** / test **2,043** = **10,215** total (60/20/20) |
| Inside venv: torch | **2.12.0+cu126** |
| Inside venv: ultralytics | 8.4.56 |
| Inside venv: codecarbon | latest (uses `OfflineEmissionsTracker`, see §3 fix) |
| Outputs land at | `E:\Thesis_mofazzal_2007074\Benchmarking YOLOs\Yolo9s\runs\<run_id>\` (next to the script, per spec §5.2) |

Frozen pip snapshot: `E:\Thesis_mofazzal_2007074\pip_freeze_mofazzal1_2026-05-28.txt`

---

## 1. Important Windows-specific traps we hit (so the next session doesn't waste hours)

Read this section before doing anything. Each item below cost real time on the first attempt.

### 1.1 `WinError 1455 — paging file too small` with `num_workers=8` + `cache='ram'` → 2.5-hour silent hang

**Symptom:** Training enters epoch 1 normally. At some point during val 1 (or any later batch), output freezes mid-progress-bar. GPU drops to **0 % util / 9 W / P8 power state**, holds ~5.6 GB VRAM (model still loaded), but does nothing. Python process is alive but `Get-Process python | Select CPU` doesn't increase between samples. The Python `logging` log (`runs/<run_id>/logs/train.log`) stops at the initial `run_id=...` line. No `results.csv` ever appears.

**Root cause:** PyTorch on Windows can't `fork()`. It uses `spawn()` + Win32 named file mappings for DataLoader worker → main shared-memory tensor passing. Those mappings consume **pagefile commit space**, not physical RAM. With `cache='ram'` holding ~6 GB of image tensors and 8 workers each opening shared-mem channels, Windows runs out of commit and refuses the next mapping. The earlier traceback that *looked* recoverable —

```
RuntimeError: Couldn't open shared file mapping: <torch_24988_...>, error code: <1455>
```

— is a worker crashing silently. The main process then deadlocks waiting on dead workers. **This does not happen on Linux/Kaggle** because Linux uses POSIX shm via `/dev/shm`, not pagefile commit.

**Fix applied (no reboot needed):** `NUM_WORKERS = 4` in the script's CONFIG block. Halving worker count halves the shared-mem channel count, putting total commit well under the Windows limit. `cache='ram'` was kept and works fine at 4 workers.

**Fallback if 4 workers still hangs:** set `CACHE = "disk"`. Disk cache uses OS file mapping (not PyTorch shared-mem) so it bypasses the issue entirely. Costs ~5-10 % wall-clock vs RAM cache. Bonus: more deterministic across runs (Ultralytics itself warns `cache='ram' may produce non-deterministic training results`).

**Long-term fix (requires reboot — skipped here because reboot would lose AnyDesk access):** Raise the Windows pagefile to 32 GB initial / 64 GB max on E: via `sysdm.cpl → Advanced → Performance Settings → Advanced → Virtual memory`.

**Canary check after restart:** within ~15 min of launch, `runs/<run_id>/ultra/results.csv` should exist with ≥ 2 rows. If GPU drops to P8 power state and `results.csv` doesn't appear within 5 min of epoch 1's val-bar reaching 100 % → the dataloaders deadlocked again; kill + fall back to `cache='disk'`.

### 1.2 PowerShell execution policy blocks venv activation

Symptom on first try: `Activate.ps1 cannot be loaded because running scripts is disabled on this system.`

Fix (one-time, current user only):
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### 1.3 Console codepage CP-437 mangles Ultralytics progress bars into mojibake

Progress bar characters (`█`, `▏`, `─`, `╸`) render as `ΓöÇΓöÇ`, `Γò╕`, `Γöü`. Purely cosmetic — training is fine, the numbers and text are still readable. Fix per session:

```powershell
chcp 65001
```

Permanent — paste into `notepad $PROFILE`:
```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
```

### 1.4 `Tee-Object` + Ultralytics progress bars = "frozen window" illusion

Ultralytics uses carriage-return-only progress bars. PowerShell's `Tee-Object` buffers until newline, so for long stretches it looks like the window is hung even though training is fine. **Don't kill the training based on the visible window — verify via `nvidia-smi` and the `results.csv` file from a second window** (see §5).

### 1.5 `accumulate` removed from Ultralytics 8.4.x train API

Old OOM-retry chains that set `kwargs["accumulate"] = ...` will fail immediately with `SyntaxError: 'accumulate' is not a valid YOLO argument`. Use `kwargs["nbs"] = BATCH_SIZE` instead — Ultralytics computes `accumulate = round(nbs / batch)` internally, so the effective batch behavior is identical.

### 1.6 CodeCarbon: `country_iso_code` only on `OfflineEmissionsTracker`

The online `EmissionsTracker` no longer accepts `country_iso_code` (raises `__init__() got an unexpected keyword argument`). Switch to:
```python
from codecarbon import OfflineEmissionsTracker
tracker = OfflineEmissionsTracker(country_iso_code=COUNTRY_ISO_CODE, ...)
```
Offline uses static grid intensity per country — fine for our reporting needs and doesn't ping a server.

### 1.7 `pynvml` is deprecated → `nvidia-ml-py`

Newer torch emits `FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead.` (Possibly dozens of times — once per worker.) The import name (`import pynvml`) is identical, just the PyPI package name changed. Fix:
```powershell
pip uninstall -y pynvml
pip install nvidia-ml-py
```

### 1.8 Env vars are per-session; persist them if you want them to survive new PowerShell windows

```powershell
[System.Environment]::SetEnvironmentVariable("C2A_ROOT", "E:\Thesis_mofazzal_2007074\common\c2a\C2A_Dataset\new_dataset3", "User")
```
Effect appears in **new** PowerShell windows only.

---

## 2. Daily routine (after first-time setup is complete)

Every new PowerShell session:

```powershell
cd "E:\Thesis_mofazzal_2007074\Benchmarking YOLOs\Yolo9s"
E:\Thesis_mofazzal_2007074\mofazzal1\Scripts\Activate.ps1
$env:C2A_ROOT = "E:\Thesis_mofazzal_2007074\common\c2a\C2A_Dataset\new_dataset3"
$env:CUDA_VISIBLE_DEVICES = "0"

# OPTIONAL but recommended: clean console encoding
chcp 65001 > $null

# Launch
python ".\yolov9s_final_month.py" 2>&1 | Tee-Object -FilePath ".\full_run_$(Get-Date -Format yyyyMMdd_HHmmss).log"
```

If `C2A_ROOT` is already persisted at User scope (per §1.8), the `$env:C2A_ROOT = ...` line is redundant but harmless.

---

## 3. One-shot first-time setup commands (in order)

**Pre-check:** `nvidia-smi` — the GPU must be idle (< 1 GB VRAM, util 0 %). If a previous training run is still alive, wait for it to finish. Steps 3.1–3.3 below are GPU-safe; only §4 (smoke/full run) needs the GPU free.

### 3.1 Create + activate the venv (`mofazzal1`)

```powershell
cd E:\Thesis_mofazzal_2007074
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned   # one-time, if not already done
py -3.11 -m venv mofazzal1
.\mofazzal1\Scripts\Activate.ps1
python --version            # Python 3.11.9
where.exe python            # first hit MUST be mofazzal1\Scripts\python.exe
```

### 3.2 Install PyTorch FIRST (cu126 wheel)

```powershell
python -m pip install --upgrade pip wheel setuptools
pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision

# Verify
python -c "import torch; print('torch=', torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('cuda_version=', torch.version.cuda); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); p=torch.cuda.get_device_properties(0); print(f'vram={p.total_memory/1024**3:.1f} GB cap={p.major}.{p.minor}')"
```

Required output: `cuda_available= True`, device `NVIDIA GeForce RTX 4070 Ti SUPER`, `cap=8.9`. If `cuda_available= False`, the CPU wheel was installed by mistake — `pip uninstall -y torch torchvision` and re-run with the `--index-url`.

### 3.3 Install everything else

```powershell
pip install `
    "ultralytics>=8.3.0" "timm>=1.0.0" "sahi>=0.11.0" `
    thop openpyxl "pandas<3.0" "matplotlib<3.10" `
    scikit-learn scipy statsmodels codecarbon nvidia-ml-py psutil `
    torchinfo tqdm opencv-python PyYAML seaborn tabulate `
    pycocotools onnx onnxruntime

pip freeze | Out-File -Encoding utf8 E:\Thesis_mofazzal_2007074\pip_freeze_mofazzal1_2026-05-28.txt
```

(Note: `nvidia-ml-py` not `pynvml` — see §1.7.)

Quick import sanity check:
```powershell
python -c @'
import torch, ultralytics, numpy, pandas, cv2, yaml, matplotlib, sklearn, scipy, statsmodels
import codecarbon, pynvml, psutil, torchinfo, tqdm, seaborn, tabulate, sahi, timm, thop, openpyxl
from pycocotools.coco import COCO
print('ALL IMPORTS OK')
'@
```

### 3.4 Persist `C2A_ROOT` so you don't have to set it every session

```powershell
[System.Environment]::SetEnvironmentVariable("C2A_ROOT", "E:\Thesis_mofazzal_2007074\common\c2a\C2A_Dataset\new_dataset3", "User")
```

---

## 4. Running the script

### 4.1 Sanity check the GPU + dataset

```powershell
cd "E:\Thesis_mofazzal_2007074\Benchmarking YOLOs\Yolo9s"
E:\Thesis_mofazzal_2007074\mofazzal1\Scripts\Activate.ps1
$env:C2A_ROOT = "E:\Thesis_mofazzal_2007074\common\c2a\C2A_Dataset\new_dataset3"  # if not persisted
$env:CUDA_VISIBLE_DEVICES = "0"

Test-Path "$env:C2A_ROOT\train\images"       # True
nvidia-smi                                    # idle: < 1 GB used, util 0 %, P8
```

### 4.2 Smoke (mandatory per spec §17, runs automatically if no recent marker)

The script auto-runs smoke if no `_PASSED_<model_tag>.txt` marker exists in `smoke/` within the last 24 h. To force-run smoke only, set `SMOKE_TEST = True` in the script's CONFIG block. Expect < 5 min.

### 4.3 Full run

```powershell
python ".\yolov9s_final_month.py" 2>&1 | Tee-Object -FilePath ".\full_run_$(Get-Date -Format yyyyMMdd_HHmmss).log"
```

---

## 5. Monitoring from a second PowerShell window (do NOT touch the training window)

The training window is misleading because of the Tee-Object + progress-bar interaction (§1.4). Always monitor from a separate window.

### 5.1 Set up live tail of per-epoch metrics

After the script prints the new run_id (right after `[pre-flight] OK`), copy it, then in a second window:

```powershell
$run = "<paste run_id here>"
$csv = "E:\Thesis_mofazzal_2007074\Benchmarking YOLOs\Yolo9s\runs\$run\ultra\results.csv"

# Wait for the file to appear, then live-tail it
while (-not (Test-Path $csv)) { Start-Sleep -Seconds 5 }
Write-Host "results.csv exists -- tailing..." -ForegroundColor Green
Get-Content $csv -Wait -Tail 50
```

One new row per completed epoch (~every 2-2.5 min on this GPU at batch=32).

### 5.2 Quick health snapshot (re-run whenever)

```powershell
$run = "<run_id>"
$base = "E:\Thesis_mofazzal_2007074\Benchmarking YOLOs\Yolo9s\runs\$run"

Write-Host "`n=== GPU ===" -ForegroundColor Cyan
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv

Write-Host "`n=== Epochs done (results.csv) ===" -ForegroundColor Cyan
if (Test-Path "$base\ultra\results.csv") {
    $r = Import-Csv "$base\ultra\results.csv"
    Write-Host ("Completed: {0}" -f $r.Count)
    $r | Select-Object -Last 3 | Format-Table epoch, "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)" -AutoSize
} else {
    Write-Host "results.csv not yet written -- epoch 1 still in progress"
}

Write-Host "`n=== Latest python processes ===" -ForegroundColor Cyan
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id, @{n="CPU_s";e={[math]::Round($_.CPU,1)}}, @{n="WS_MB";e={[math]::Round($_.WorkingSet/1MB,0)}}
```

### 5.3 Hang detection (when to kill)

A training run is **hung** if all of these hold:
- GPU at 0 % util **and** ~9 W **and** P8 power state, for > 60 seconds
- Main python PID's `CPU_s` does not change between two samples taken 30 s apart
- `results.csv`'s `LastWriteTime` is older than ~5 min and we're past epoch 1
- No new line in `runs/<run_id>/logs/train.log` for > 5 min

If hung:
```powershell
Stop-Process -Name python -Force
# Then edit script: NUM_WORKERS=4 (already done) AND CACHE='disk', and relaunch.
```

### 5.4 Beep when the full run finishes (optional)

```powershell
$mf = "E:\Thesis_mofazzal_2007074\Benchmarking YOLOs\Yolo9s\runs\<run_id>\manifest.json"
while (-not (Test-Path $mf)) { Start-Sleep -Seconds 60 }
[console]::Beep(880, 300); [console]::Beep(660, 300); [console]::Beep(880, 600)
Write-Host "`n*** RUN FINISHED ***" -ForegroundColor Green
```

---

## 6. Troubleshooting cheatsheet (problems already encountered)

| Symptom | Cause | Fix |
|---|---|---|
| Training output frozen mid-progress, GPU at 0 % / 9 W / P8, no new `results.csv` row | DataLoader workers deadlocked via WinError 1455 (§1.1) | Kill, drop `NUM_WORKERS = 4`, optionally `CACHE = 'disk'`, relaunch |
| `Activate.ps1 cannot be loaded ... execution policy` | Default PS execution policy is `Restricted` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `'accumulate' is not a valid YOLO argument` | Ultralytics 8.4.x removed the arg | Replace with `kwargs["nbs"] = BATCH_SIZE` (§1.5) |
| `EmissionsTracker.__init__() got an unexpected keyword argument 'country_iso_code'` | New codecarbon API | Use `OfflineEmissionsTracker` (§1.6) |
| Console shows `ΓöÇΓöÇΓöÇ` instead of progress bar | CP-437 codepage | `chcp 65001` (§1.3) |
| Training window looks frozen but GPU is at 80 % util | Tee-Object + progress-bar buffering — NOT a hang | Ignore the window; watch `results.csv` (§5) |
| `cuda_available= False` after install | CPU-only torch wheel | `pip uninstall -y torch torchvision`, reinstall with `--index-url https://download.pytorch.org/whl/cu126` |
| Dataset not found at script start | `C2A_ROOT` env var not set in this session | `$env:C2A_ROOT = "...\C2A_Dataset\new_dataset3"` (or persist per §1.8) |
| `Get-Content -Wait` errors with "Cannot find path" | File not yet created | Use the `while (-not Test-Path)` loop in §5.1 |
| `pycocotools` install fails with a compiler error | Needs Visual C++ Build Tools | Install "Build Tools for Visual Studio 2022" → Desktop dev with C++ |
| `cv2` DLL load failed | Missing VC++ runtime | Install "Microsoft Visual C++ 2015-2022 Redistributable (x64)" |
| GPU util stuck at 0 % during what should be training | dataloader bottleneck | Confirm `num_workers >= 4`, `cache='ram'` or `'disk'`, `pin_memory=True` |
| `pynvml` deprecation warnings (many copies) | Package renamed | `pip install nvidia-ml-py` (§1.7) — purely cosmetic but spammy |

---

## 7. Quick reference card

```
venv:               E:\Thesis_mofazzal_2007074\mofazzal1
activate:           E:\Thesis_mofazzal_2007074\mofazzal1\Scripts\Activate.ps1
script:             E:\Thesis_mofazzal_2007074\Benchmarking YOLOs\Yolo9s\yolov9s_final_month.py
dataset:            E:\Thesis_mofazzal_2007074\common\c2a\C2A_Dataset\new_dataset3
outputs:            E:\Thesis_mofazzal_2007074\Benchmarking YOLOs\Yolo9s\runs\<run_id>\

env vars:           $env:C2A_ROOT, $env:CUDA_VISIBLE_DEVICES=0
torch wheel index:  https://download.pytorch.org/whl/cu126
python:             3.11.9
torch:              2.12.0+cu126
ultralytics:        8.4.56
driver:             591.86

mandatory script tweaks vs the original CONFIG block:
  NUM_WORKERS  = 4                          # was 8 -- WinError 1455 with cache='ram' (§1.1)
  CACHE        = "ram"                      # kept; switch to "disk" if 4 workers still hangs
  + use kwargs["nbs"] = BATCH_SIZE  in OOM retry, NOT kwargs["accumulate"]  (§1.5)
  + use OfflineEmissionsTracker(...)  NOT EmissionsTracker(...)             (§1.6)
```
