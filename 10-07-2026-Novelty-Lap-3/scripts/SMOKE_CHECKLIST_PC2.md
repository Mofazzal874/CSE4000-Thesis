# S0 smoke — PC-2 EXACT commands (2×A6000 SHARED, GPU1 = ours) · PowerShell · kill-safe
*(backup path created 2026-07-12 because PC-4 shut down; S0 is hardware-agnostic — a PC-2 pass
counts exactly the same. Latency numbers for the paper come from PC-1 later regardless.)*

**BEFORE ANYTHING (AnyDesk, laptop → PC-2):** copy `d:\Academics\thesis folder\10-07-2026-Novelty-Lap-3\scripts\`
→ `D:\student_2k20\2007074\10-07-2026-Novelty-Lap-3\scripts\`. **Copy from the LAPTOP, not from
PC-4's disk — the laptop has the fixed `10_fccg_modules.py` (embedded CBAM); PC-4's copy is stale.**

```powershell
# ===== 0) existence checks — all True except maybe scenesplit (then 3b)
Test-Path D:\student_2k20\2007074\10-07-2026-Novelty-Lap-3\scripts\10_fccg_modules.py
Test-Path D:\student_2k20\2007074\10-07-2026-Novelty-Lap-3\scripts\11_fccg_smoke.py
Test-Path D:\student_2k20\2007074\common\c2a\C2A_Dataset\new_dataset3
Test-Path D:\student_2k20\2007074\common\c2a\C2A_Dataset\new_dataset3_scenesplit_v1
Test-Path D:\student_2k20\2007074\05-07-2026-Novelty-Lap\scripts\01_make_scene_split.py

# ===== 1) venv (exact PC-2 dance from the lap-1 checklist)
cd "D:\student_2k20\2007074"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\2007074\Scripts\Activate.ps1
if (Get-Command conda -ErrorAction SilentlyContinue) { conda deactivate }
where.exe python     # FIRST line must be D:\student_2k20\2007074\2007074\Scripts\python.exe
python -c "import torch; print('CUDA', torch.cuda.is_available())"
python -c "import ultralytics; print('ultralytics', ultralytics.__version__)"

# ===== 2) GPU etiquette — NEVER take a busy GPU0. Look, then pin OUR GPU1:
nvidia-smi
$env:CUDA_VISIBLE_DEVICES = "1"     # after pinning, device "0" IS physical GPU1

# ===== 3) selftest + build check (~3 min)
cd "D:\student_2k20\2007074\10-07-2026-Novelty-Lap-3\scripts"
python .\10_fccg_modules.py --selftest       # 👀 16/16 incl. "t10 CBAM lazy build + pickle"
python .\10_fccg_modules.py --check-load     # 👀 "~20.1M params" · "FCCG ACTIVE {model.16/model.20 ~0.5}" · "CHECK-LOAD OK"
# ⛔ KeyError 'CBAM' here = you copied the stale file; re-copy from the laptop.

# ===== 3b) ONLY IF scenesplit Test-Path printed False — deterministic rebuild (~2 min)
cd "D:\student_2k20\2007074\05-07-2026-Novelty-Lap\scripts"
python 01_make_scene_split.py --root "D:\student_2k20\2007074\common\c2a\C2A_Dataset\new_dataset3" --assignment ..\evidence\scene_assignment.csv
cd "D:\student_2k20\2007074\10-07-2026-Novelty-Lap-3\scripts"

# ===== 4) smoke data yaml (PC-2-local paths)
@"
path: D:/student_2k20/2007074/common/c2a/C2A_Dataset/new_dataset3_scenesplit_v1
train: train/images
val: val/images
test: test/images
nc: 1
names:
  0: person
"@ | Set-Content -Encoding ascii scenesplit_pc2.yaml

# ===== 5) the 2-epoch smoke (⏱ ~10-20 min at batch 16; AMP is FINE on the A6000)
python .\11_fccg_smoke.py --data .\scenesplit_pc2.yaml --batch 16
# 👀 epoch 1: losses decreasing, no NaN
# 👀 each epoch end: "[fccg-verify] OK gates={ L16:0.xx, L20:0.xx }"  (0.02<g<0.98)
# 👀 end: "[smoke] S0 SMOKE COMPLETE"
# ⛔ if GPU0 was the only free one: STOP, do not use it — wait for GPU1 (house rule).
# (do NOT raise batch past ~28 on this box — WDDM silently spills past 48 GB; 16 is plenty)

# ===== 6) results home (AnyDesk copy back to laptop)
# copy D:\student_2k20\2007074\10-07-2026-Novelty-Lap-3\scripts\runs_smoke\fccg_s0
#   -> d:\Academics\thesis folder\05-07-2026-Novelty-Lap\results\pc2\2026-07-12_S0_fccg_smoke\
# + one line in results\MANIFEST.md
```

**S0 pass =** selftest 16/16 · CHECK-LOAD OK (≤22.5M params) · 2 epochs no error · `[fccg-verify]`
healthy each epoch · losses decreasing. Pass ⇒ S1 paired pilots (can also run on PC-2 GPU1 while
PC-4 is down — batch 16, same protocol, note the hardware in MANIFEST).
**Metrics note:** smoke exempt from the metric contract; S1+ logs per
`..\docs\2026-07-12_metric_contract_reference.md`.
