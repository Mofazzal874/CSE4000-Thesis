# PC-2 (A6000) clean setup — environment + run

Root cause of the earlier failures: the only `python` on PATH was **MSYS2** (`C:\msys64\...`),
which makes Unix-style venvs (no `Scripts\activate.bat`) and can't use CUDA PyTorch. Fix = install
**native Windows Python (python.org)** and build the venv with the `py` launcher. CUDA itself is
fine (nvidia-smi: 2× A6000, driver 560.76, CUDA 12.6).

Run everything in **cmd.exe**. Do the parts in order.

---

## PART A — install native Windows Python (one time, PER-USER only)
This PC is shared. The install below is **per-user** — it goes into YOUR profile
(`C:\Users\<you>\AppData\Local\Programs\Python\`), modifies only YOUR PATH, needs no admin,
and does NOT affect any other user. (`winget` is not present on this box, so we install
directly from python.org.)
```cmd
powershell -Command "[Net.ServicePointManager]::SecurityProtocol='Tls12'; Invoke-WebRequest https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe -OutFile %TEMP%\py311.exe"
%TEMP%\py311.exe /quiet InstallAllUsers=0 InstallLauncherAllUsers=0 PrependPath=1 Include_launcher=1
```
- `InstallAllUsers=0` + `InstallLauncherAllUsers=0` = everything per-user (nothing system-wide).
- Wait ~1–2 min (silent install, no window). Then **close this cmd and open a NEW one** (PATH refresh).

Verify:
```cmd
py --version
py -0p
```
- `py --version` must print `Python 3.11.9`.
- `py -0p` must list a path like `C:\Users\<you>\AppData\Local\Programs\Python\Python311\` (NOT msys64).
- If `py` is still "not recognized": open a browser → https://www.python.org/downloads/release/python-3119/
  → download **Windows installer (64-bit)** → run it, tick **"Add python.exe to PATH"** +
  **"py launcher"**, choose **"Install for me only"** if asked, new cmd, retry.

## PART B — create + activate the venv (use `py`, not `python`)
```cmd
cd /d D:\student_2k20\2007074
rmdir /s /q 2007074
py -m venv 2007074
dir 2007074\Scripts
2007074\Scripts\activate.bat
```
- `dir 2007074\Scripts` must show **`activate.bat`, `python.exe`, `pip.exe`**.
- After activate, the prompt becomes `(2007074) D:\student_2k20\2007074>`.
- **Confirm you're in the RIGHT python (not MSYS2):**
  ```cmd
  python -c "import sys; print(sys.executable)"
  ```
  Must print `D:\student_2k20\2007074\2007074\Scripts\python.exe` (NOT msys64).

## PART C — install packages + verify CUDA (inside the activated venv)
```cmd
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install ultralytics pandas pyyaml matplotlib opencv-python pycocotools openpyxl thop
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```
- Last line must print `... True 2`. If it prints `False` → CPU torch got installed; redo the
  torch line (you must be in the `(2007074)` venv).

## PART D — put code + data in place
| What | Where on PC-2 | How |
|---|---|---|
| This code folder | `D:\student_2k20\2007074\A6000_run\` | transfer the `A6000 run` folder (rename to `A6000_run`, no space) |
| C2A dataset | `D:\student_2k20\2007074\common\c2a\C2A_Dataset\new_dataset3\` | **download** Kaggle `rgbnihal/c2a-dataset` |
| SARD dataset | `D:\student_2k20\2007074\common\sard\search-and-rescue\` | transfer your folder (or re-download same Roboflow version) |
| C2A init checkpoint | `D:\student_2k20\2007074\c2a_cbam_p2head_best.pt` | transfer the 38 MB `best.pt` |

(The script auto-finds the datasets; the checkpoint path matches `EXPLICIT_C2A_BEST_PT`.)

## PART E — run
```cmd
cd /d D:\student_2k20\2007074\A6000_run
python joint_c2a_sard_train.py
```
- `SMOKE_TEST=True` by default → 2-epoch dry run. Look for:
  `[gpu] ... NVIDIA RTX A6000 | 47.x GB` and a `DEPLOYABLE MODEL RESULT` block, no errors.
- Then open `joint_c2a_sard_train.py`, set **`SMOKE_TEST = False`**, save, and run again — the real
  training (~3–6 h, resume-safe; just re-run if power cuts).
- Monitor in another cmd: `nvidia-smi -i 1 -l 2` → GPU 1 util >90%, memory ~44–47 GB.

## Quick fail map
- `activate.bat` missing → you used MSYS2 python; rebuild with `py -m venv` (Part B).
- `torch.cuda.is_available()` False → not in venv, or CPU torch — reinstall cu126 inside `(2007074)`.
- DataLoader hangs (WinError 1455) → set `NUM_WORKERS = 8` in the script.
- OOM → the script's ladder drops batch automatically (40→32→24→16).
