# GPU PC Run Guide (Phase 1) — do this tonight

You are on the AnyDesk PC (RTX 4070 Ti SUPER), demo folder copied to
`E:\Thesis_mofazzal_2007074\demo\` (without `data\c2a_test` — the script
auto-falls-back to the dataset already on E:).

## Step 0 — activate the training environment (cmd: Win+R → cmd)
```
cd /d E:\Thesis_mofazzal_2007074
dir /AD /B
```
If a venv folder exists (venv/.venv/env/...): `E:\Thesis_mofazzal_2007074\<name>\Scripts\activate.bat`
If none exists, training used system Python — skip activation.

VERIFY (the real gate — expected: `2.12.0+cu126 True 8.4.56`):
```
python -c "import torch, ultralytics; print(torch.__version__, torch.cuda.is_available(), ultralytics.__version__)"
```
Wrong versions or `False` → wrong Python; find the right venv, install nothing.

Then:
```
cd /d E:\Thesis_mofazzal_2007074\demo\app
```

## Step 1 — check SAHI (likely already installed — the July 7 ablation used it)
```
python -c "import sahi; print(sahi.__version__)"
```
If that errors: `pip install sahi`

Also stop the PC sleeping mid-run: `powercfg /change standby-timeout-ac 0`

## Step 2 — SMOKE TEST (~1 min). Run this FIRST, every time.
```
python precompute.py --smoke
```
PASS looks like: it prints the resolved paths (c2a test exists: True), the GPU
name, then processes 2 C2A images x 3 configs and 6 drone frames x 4 configs
with box counts, and writes files into `..\results\`.
FAIL cases:
- "missing model" → the models folder didn't copy fully
- CBAM/pickle error → you're not running from inside `app\` (cbam_modules must
  be importable) — make sure you `cd` into the app folder first
- c2a exists: False → dataset path differs; tell Claude the real path

## Step 3 — full precompute (~15–35 min)
```
python precompute.py
```
250 C2A test images x 3 configs + 75 drone frames (25/video) x 4 configs.
SAHI+TTA on 4K drone frames is the slow part (a few seconds per frame).
Outputs land in `E:\Thesis_mofazzal_2007074\demo\results\`:
- predictions_c2a.json, predictions_drone.json
- drone_frames\*.jpg (75 frames — these MUST come back to the laptop)
- latency_summary.json, curation_hints.csv

## Step 4 — annotated drone video(s) (~20–40 min each at full length)
Start with the 50m video (tiny humans = where SAHI shines):
```
python annotate_video.py --video 50m.MP4
```
(defaults: enriched model + SAHI+TTA, conf 0.30, every 3rd frame)

Optional extras if time allows, in order of demo value:
```
python annotate_video.py --video 10m.MP4
python annotate_video.py --video 50m.MP4 --model cbam_p2
python annotate_video.py --video 30m.MP4
```
Pressed for time? Render a 30 s highlight instead of the full video:
```
python annotate_video.py --video 50m.MP4 --start 20 --duration 30
```
Outputs land in `..\results\annotated_videos\`.

## Step 5 — copy results back
Copy the ENTIRE `E:\Thesis_mofazzal_2007074\demo\results\` folder back to the
laptop at `D:\Academics\thesis folder\Defense\demo\results\`.
(JSONs + frames ≈ 150–250 MB; annotated videos add a few hundred MB.)

## If something breaks
Copy the full error text and paste it to Claude. Do not debug alone at 1 AM.
