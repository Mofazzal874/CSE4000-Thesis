# PC-3 setup — env "2007074" + folders + transfers + runs (root: D:\2007074)

Same recipe that worked on PC-2, but the project root is **D:\2007074** (no `student_2k20`).
Run everything in **cmd.exe**. Do the parts in order.

---

## PART A — install native Windows Python 3.11.9 (per-user, one time)
(If PC-3 already has a real python.org Python with the `py` launcher, skip to PART B and verify `py --version`.)
```cmd
powershell -Command "[Net.ServicePointManager]::SecurityProtocol='Tls12'; Invoke-WebRequest https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe -OutFile %TEMP%\py311.exe"
%TEMP%\py311.exe /quiet InstallAllUsers=0 InstallLauncherAllUsers=0 PrependPath=1 Include_launcher=1
```
Wait ~1-2 min (silent). **Close cmd, open a NEW cmd**, verify:
```cmd
py --version          ::  must print  Python 3.11.9
```
If `py` is "not recognized": download Python 3.11.9 from python.org, run it, tick **Add to PATH** + **py launcher** (install "for me only"), new cmd, retry.

## PART B — create + activate the venv "2007074"
```cmd
mkdir D:\2007074
cd /d D:\2007074
py -m venv 2007074
2007074\Scripts\activate.bat
```
Prompt must change to `(2007074) D:\2007074>`. Confirm it's the venv python:
```cmd
python -c "import sys; print(sys.executable)"    ::  must be  D:\2007074\2007074\Scripts\python.exe
```

## PART C — install packages + verify CUDA
```cmd
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install ultralytics pandas pyyaml matplotlib opencv-python pycocotools openpyxl thop
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```
Last line must print `... True N` (N = number of GPUs). If `False` -> CPU torch slipped in; redo the
torch line inside the activated venv. (`pycocotools` build failing is non-fatal — COCO AP is skipped.)

## PART D — create the folder structure (under D:\2007074)
```cmd
mkdir "D:\2007074\A6000 run"
mkdir "D:\2007074\common\c2a\C2A_Dataset\new_dataset3"
mkdir "D:\2007074\common\sard\search-and-rescue"
mkdir "D:\2007074\Drone Shoot"
mkdir "D:\2007074\drone_negatives"
```

## PART E — transfer these onto PC-3
| What | Destination on PC-3 |
|---|---|
| All 5 scripts (joint_c2a_sard_train.py, eval_checkpoint.py, run_on_drone_footage.py, enrich_c2a_for_robustness.py, make_negative_tiles.py) | `D:\2007074\A6000 run\` |
| C2A dataset (train/val/test) | `D:\2007074\common\c2a\C2A_Dataset\new_dataset3\` |
| SARD dataset (train/valid/test) | `D:\2007074\common\sard\search-and-rescue\` |
| C2A init checkpoint `c2a_cbam_p2head_best.pt` | `D:\2007074\c2a_cbam_p2head_best.pt` |
| Deployable model `epoch125.pt` (for eval/inference/fine-tune init) | `D:\2007074\epoch125.pt` |
| Drone videos (10m/30m/50m .MP4) | `D:\2007074\Drone Shoot\` |

Only transfer what the run you intend needs (e.g. data enrichment doesn't need the GPU/model; inference
needs `epoch125.pt` + videos; training needs the datasets + `c2a_cbam_p2head_best.pt`).

## PART F — fix the paths in the scripts (one find-replace)
The scripts contain PC-2 paths `D:\student_2k20\2007074\...`. On PC-3, replace the root:
- In **VS Code**: open the `A6000 run` folder → **Ctrl+Shift+H** (Replace in Files) →
  Find `D:\student_2k20\2007074`  →  Replace `D:\2007074`  →  Replace All.
- This fixes `EXPLICIT_C2A_BEST_PT`, the inference `MODEL`/`VIDEOS_DIR`, the enrich/negatives paths, and
  `eval_checkpoint.py`'s `CKPT`. (Dataset auto-discovery already includes `D:\2007074`, so data is found
  even without this — but the explicit checkpoint/video paths still need it.)
- Then set `eval_checkpoint.py`'s `CKPT` to the actual epoch125 path, e.g. `D:\2007074\epoch125.pt`.

## PART G — verify the scripts load
```cmd
cd /d "D:\2007074\A6000 run"
python -m py_compile joint_c2a_sard_train.py eval_checkpoint.py run_on_drone_footage.py enrich_c2a_for_robustness.py make_negative_tiles.py
echo if no error above, all good
```

## PART H — what to run (pick by what PC-3 is for)
All scripts auto-pick a free GPU (prefer the assigned fallback) and need the `(2007074)` venv active.

- **Data enrichment (angle B — no GPU needed):**
  ```cmd
  python make_negative_tiles.py          :: -> D:\2007074\drone_negatives  (then delete person tiles)
  python enrich_c2a_for_robustness.py    :: VERIFY_ONLY=True first; check _verify\ ; then set False
  ```
- **Evaluate a checkpoint on C2A+SARD test:**
  ```cmd
  python eval_checkpoint.py              :: set CKPT first
  ```
- **Run detection on drone footage (SAHI):**
  ```cmd
  python run_on_drone_footage.py         :: needs epoch125.pt + Drone Shoot videos
  ```
- **Train / fine-tune the joint model:**
  ```cmd
  python joint_c2a_sard_train.py         :: SMOKE_TEST=False; resumes if interrupted
  ```

## Quick fail map
- `(2007074)` not in prompt → venv not active; re-run `2007074\Scripts\activate.bat`.
- `torch.cuda.is_available()` False → CPU torch; reinstall cu126 inside the venv.
- `ModuleNotFoundError: torch` → venv not active.
- script can't find C2A/best.pt/videos → you skipped a transfer or the find-replace (PART E/F).
- DataLoader hangs (WinError 1455) → lower `NUM_WORKERS` in joint_c2a_sard_train.py to 4.
