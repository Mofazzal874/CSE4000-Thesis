# AnyDesk PC — Environment Setup Runbook (FINAL)

**Date:** 2026-05-28
**Target machine:** Remote AnyDesk PC (i7-14700K / 128 GB RAM / RTX 4070 Ti SUPER 16 GB, Windows 11)
**Working folder:** `E:\Thesis_mofazzal_2007074`
**venv name:** `mofazzal1` → `E:\Thesis_mofazzal_2007074\mofazzal1\`
**Script:** `E:\Thesis_mofazzal_2007074\Benchmarking YOLOs\Yolo9s\yolov9s_final_month.py`
**Dataset:** `E:\Thesis_mofazzal_2007074\common\c2a\C2A_Dataset\new_dataset3`
**Shell:** PowerShell (not cmd.exe)

## Confirmed on this PC (diagnostic results, 2026-05-28)

| Component | Value | Status |
|---|---|---|
| Python (default) | 3.11.9 at `C:\Program Files\Python311\python.exe` | ✅ |
| Python 3.13 | Also installed — ignore, we use 3.11 | ✅ |
| pip | 24.0 | ✅ |
| venv module | OK | ✅ |
| git | 2.47.1.windows.1 | ✅ |
| NVIDIA driver | 591.86 (supports up to CUDA 13.1) | ✅ |
| GPU | RTX 4070 Ti SUPER, 16 GB | ✅ |
| CUDA toolkit (nvcc) | 12.6.85 — bonus, not required | ✅ |
| Free space on E: | ~199 GB | ✅ |
| Dataset splits | train 6,129 / val 2,043 / test 2,043 = 10,215 | ✅ |

**Reminder:** Outputs (runs/, plots/, weights/) land **next to the script** per spec §5.2 — i.e. in `E:\Thesis_mofazzal_2007074\Benchmarking YOLOs\Yolo9s\` — NOT at the repo root.

---

## ⚠ Before starting — wait for the GPU to be idle

Confirm no other training is running:

```powershell
nvidia-smi
```

Memory usage should be under ~1 GB and util ~0 %. If a previous run is still going, **wait for it to finish** before Step 5 (smoke) or Step 6 (full run). Steps 1–4 (venv + pip installs) are GPU-safe and can run in parallel with anything.

---

## Step 1 — Create and activate the venv `mofazzal1`

```powershell
cd E:\Thesis_mofazzal_2007074
py -3.11 -m venv mofazzal1
.\mofazzal1\Scripts\Activate.ps1
```

Prompt should change to:
```
(mofazzal1) PS E:\Thesis_mofazzal_2007074>
```

If activation is blocked with an "execution policy" error, run **once**:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
then re-run the activation line.

Confirm:
```powershell
python --version                                          # Python 3.11.9
where.exe python                                          # first hit must be E:\Thesis_mofazzal_2007074\mofazzal1\Scripts\python.exe
python -c "import sys; print(sys.prefix)"                 # E:\Thesis_mofazzal_2007074\mofazzal1
```

---

## Step 2 — Install PyTorch FIRST (CUDA 12.6 wheel)

PyTorch must be installed from the official CUDA wheel index **before** anything else, otherwise pip pulls a CPU-only build.

```powershell
python -m pip install --upgrade pip wheel setuptools
pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision
```

Verify before continuing:
```powershell
python -c "import torch; print('torch=', torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('cuda_version=', torch.version.cuda); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); p=torch.cuda.get_device_properties(0); print(f'vram={p.total_memory/1024**3:.1f} GB cap={p.major}.{p.minor}')"
```

Required output (bold parts MUST match):
```
torch= 2.x.x+cu126
cuda_available= True
cuda_version= 12.6
device= NVIDIA GeForce RTX 4070 Ti SUPER
vram=16.0 GB cap=8.9
```

If `cuda_available= False`, STOP. Wrong wheel installed — uninstall and re-run with the `--index-url`:
```powershell
pip uninstall -y torch torchvision
pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision
```

---

## Step 3 — Install all remaining packages

```powershell
pip install `
    "ultralytics>=8.3.0" `
    "timm>=1.0.0" `
    "sahi>=0.11.0" `
    thop `
    openpyxl `
    "pandas<3.0" `
    "matplotlib<3.10" `
    scikit-learn `
    scipy `
    statsmodels `
    codecarbon `
    pynvml `
    psutil `
    torchinfo `
    tqdm `
    opencv-python `
    PyYAML `
    seaborn `
    tabulate `
    pycocotools `
    onnx `
    onnxruntime
```

Freeze for reproducibility (spec §11.7):
```powershell
pip freeze | Out-File -Encoding utf8 E:\Thesis_mofazzal_2007074\pip_freeze_mofazzal1_2026-05-28.txt
```

Import-only sanity check (no model loaded yet):
```powershell
python -c @'
import torch, ultralytics, numpy, pandas, cv2, yaml, matplotlib, sklearn, scipy, statsmodels
import codecarbon, pynvml, psutil, torchinfo, tqdm, seaborn, tabulate, sahi, timm, thop, openpyxl
from pycocotools.coco import COCO
print('torch        ', torch.__version__)
print('ultralytics  ', ultralytics.__version__)
print('cuda?        ', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
print('ALL IMPORTS OK')
'@
```

Last line MUST be `ALL IMPORTS OK`. If any import errors, install just that package and rerun.

---

## Step 4 — Set environment variables (every PowerShell session)

```powershell
$env:C2A_ROOT = "E:\Thesis_mofazzal_2007074\common\c2a\C2A_Dataset\new_dataset3"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

# Confirm
Test-Path "$env:C2A_ROOT\train\images"       # True
Test-Path "$env:C2A_ROOT\val\images"         # True
Test-Path "$env:C2A_ROOT\test\images"        # True
echo "C2A_ROOT     = $env:C2A_ROOT"
echo "CUDA_VISIBLE = $env:CUDA_VISIBLE_DEVICES"
```

Optional — persist `C2A_ROOT` across all future sessions:
```powershell
[System.Environment]::SetEnvironmentVariable("C2A_ROOT", "E:\Thesis_mofazzal_2007074\common\c2a\C2A_Dataset\new_dataset3", "User")
```
You'd need a fresh PowerShell window for that to take effect.

---

## Step 5 — Pre-flight sanity (spec §8 + §24)

```powershell
cd "E:\Thesis_mofazzal_2007074\Benchmarking YOLOs\Yolo9s"

python -c @'
import torch, platform
print("Python   :", platform.python_version())
print("Torch    :", torch.__version__)
print("CUDA?    :", torch.cuda.is_available())
print("CUDA ver :", torch.version.cuda)
print("Device   :", torch.cuda.get_device_name(0))
p = torch.cuda.get_device_properties(0)
print(f"VRAM     : {p.total_memory/1024**3:.1f} GB")
print(f"SMs      : {p.multi_processor_count}")
print(f"Cap      : {p.major}.{p.minor}")
assert p.major == 8 and p.minor == 9, "Wrong GPU selected"
assert abs(p.total_memory/1024**3 - 16.0) < 0.5, "VRAM is not 16 GB"
print("SANITY OK")
'@

Get-PSDrive E | Select-Object @{n="Free_GB";e={[math]::Round($_.Free/1GB,1)}}
nvidia-smi
```

Must show `SANITY OK`, free space ≥ 20 GB, and GPU idle.

---

## Step 6 — Smoke test (mandatory — spec §17)

Open `E:\Thesis_mofazzal_2007074\Benchmarking YOLOs\Yolo9s\yolov9s_final_month.py` in a text editor and set:
```python
SMOKE_TEST = True
```

Run the smoke:
```powershell
cd "E:\Thesis_mofazzal_2007074\Benchmarking YOLOs\Yolo9s"
python ".\yolov9s_final_month.py" 2>&1 | Tee-Object -FilePath .\smoke_run.log
```

Expected: completes in **< 5 minutes**. If it fails, the `smoke/` directory is kept — read the error, fix it, re-run smoke. **Do not flip to the full run on a failed smoke.**

---

## Step 7 — Full run

Flip `SMOKE_TEST = False` back in the script, then:

```powershell
cd "E:\Thesis_mofazzal_2007074\Benchmarking YOLOs\Yolo9s"
python ".\yolov9s_final_month.py" 2>&1 | Tee-Object -FilePath ".\full_run_$(Get-Date -Format yyyyMMdd_HHmmss).log"
```

In a **second PowerShell window**, monitor:
```powershell
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv -l 2
```

Target (spec §19.3):
- GPU util avg ≥ 85 %
- VRAM peak 12–14.5 GB
- Power draw ≥ 220 W

If util drops < 80 % after epoch 1, check `logs/nvidia_smi_loop.csv`, then verify `NUM_WORKERS=8` and `cache='ram'` are taking effect.

---

## Daily routine (after first-time setup is done)

Every new PowerShell session:

```powershell
cd "E:\Thesis_mofazzal_2007074\Benchmarking YOLOs\Yolo9s"
E:\Thesis_mofazzal_2007074\mofazzal1\Scripts\Activate.ps1
$env:C2A_ROOT = "E:\Thesis_mofazzal_2007074\common\c2a\C2A_Dataset\new_dataset3"
$env:CUDA_VISIBLE_DEVICES = "0"
python ".\yolov9s_final_month.py"
```

---

## Troubleshooting cheatsheet

| Symptom | Cause | Fix |
|---|---|---|
| `cuda_available= False` after Step 2 | CPU-only torch wheel installed | `pip uninstall -y torch torchvision`, re-run Step 2 with `--index-url` |
| `Activate.ps1 cannot be loaded ... execution policy` | Restricted policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `ImportError: DLL load failed` on `cv2` | Missing VC++ runtime | Install "Microsoft Visual C++ 2015-2022 Redistributable (x64)" |
| `OSError: [WinError 1455] paging file too small` | Win pagefile too small for RAM cache | `sysdm.cpl` → Advanced → Performance → Virtual memory → set ≥ 32 GB on E: |
| `FileNotFoundError: dataset` | `C2A_ROOT` not set in this session | Re-run Step 4 |
| `pycocotools` install fails | Needs VC++ Build Tools | Install "Build Tools for Visual Studio 2022" → "Desktop development with C++" |
| GPU util stuck at 0 % | Dataloader bottleneck | Confirm `NUM_WORKERS=8`, `cache='ram'`, `pin_memory=True` |
| `RuntimeError: CUDA out of memory` | Batch too large or other process on GPU | Wait for other GPU process to clear, then re-run; script has OOM retry chain (spec §20.1) |

---

## Quick reference card

```
venv:         E:\Thesis_mofazzal_2007074\mofazzal1
activate:     E:\Thesis_mofazzal_2007074\mofazzal1\Scripts\Activate.ps1
script:       E:\Thesis_mofazzal_2007074\Benchmarking YOLOs\Yolo9s\yolov9s_final_month.py
dataset:      E:\Thesis_mofazzal_2007074\common\c2a\C2A_Dataset\new_dataset3
outputs land: E:\Thesis_mofazzal_2007074\Benchmarking YOLOs\Yolo9s\runs\<run_id>\

env vars:     $env:C2A_ROOT, $env:CUDA_VISIBLE_DEVICES=0
torch wheel:  https://download.pytorch.org/whl/cu126
python:       3.11.9
```
