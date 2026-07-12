# S0 smoke — PC-4 EXACT commands (RTX 4070 12 GB, fp32-only) · PowerShell · kill-safe

**BEFORE ANYTHING (AnyDesk file transfer, laptop → PC-4):** copy the whole folder
`d:\Academics\thesis folder\10-07-2026-Novelty-Lap-3\scripts\` → `D:\thesis_2007074\10-07-2026-Novelty-Lap-3\scripts\`
(keep filenames — checkpoints will remember the module name `10_fccg_modules`).

```powershell
# ===== 0) existence checks (house rule) — ALL must print True except the scenesplit one (see 3b)
Test-Path D:\thesis_2007074\10-07-2026-Novelty-Lap-3\scripts\10_fccg_modules.py
Test-Path D:\thesis_2007074\10-07-2026-Novelty-Lap-3\scripts\11_fccg_smoke.py
Test-Path D:\thesis_2007074\common\c2a\C2A_Dataset\new_dataset3
Test-Path D:\thesis_2007074\common\c2a\C2A_Dataset\new_dataset3_scenesplit_v1
Test-Path D:\thesis_2007074\05-07-2026-Novelty-Lap\scripts\01_make_scene_split.py

# ===== 1) venv (same dance as the G2 run)
Get-ChildItem "D:\thesis_2007074" -Recurse -Filter Activate.ps1 -ErrorAction SilentlyContinue | Select-Object -First 1 FullName
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# now run the Activate.ps1 path it printed, e.g.:  D:\thesis_2007074\<venv>\Scripts\Activate.ps1
if (Get-Command conda -ErrorAction SilentlyContinue) { conda deactivate }   # PC-4 has no conda — skips cleanly
python -c "import torch; print('CUDA', torch.cuda.is_available())"    # MUST print True
python -c "import ultralytics; print('ultralytics', ultralytics.__version__)"   # record it

# ===== 2) module selftests (CPU-independent; ~1 min)   👀 expect 15/15 PASS + "SELFTEST OK"
cd "D:\thesis_2007074\10-07-2026-Novelty-Lap-3\scripts"
python .\10_fccg_modules.py --selftest

# ===== 3) YAML build + forward + checkpoint roundtrip (~2 min)
python .\10_fccg_modules.py --check-load
# 👀 expect: "built OK — ~20.1M params" · "FCCG ACTIVE {model.16: ~0.5, model.20: ~0.5}" · "CHECK-LOAD OK"
# ⛔ parse traceback ⇒ STOP, record the ultralytics version + full error into a txt, report back.
# (2026-07-12 fix: KeyError 'CBAM' on 8.4.83 — CBAM lineage classes are now EMBEDDED and
#  registered by 10_fccg_modules itself; if you see that error you're running a stale copy.)

# ===== 3b) ONLY IF the scenesplit Test-Path in step 0 printed False — rebuild it (~2 min, deterministic)
cd "D:\thesis_2007074\05-07-2026-Novelty-Lap\scripts"
python 01_make_scene_split.py --root "D:\thesis_2007074\common\c2a\C2A_Dataset\new_dataset3" --assignment ..\evidence\scene_assignment.csv
cd "D:\thesis_2007074\10-07-2026-Novelty-Lap-3\scripts"

# ===== 4) write the smoke data yaml (scene split, PC-4-local paths)
@"
path: D:/thesis_2007074/common/c2a/C2A_Dataset/new_dataset3_scenesplit_v1
train: train/images
val: val/images
test: test/images
nc: 1
names:
  0: person
"@ | Set-Content -Encoding ascii scenesplit_pc4.yaml

# ===== 5) the 2-epoch smoke (⏱ roughly 20-40 min on the 4070; --no-amp MANDATORY on this box)
python .\11_fccg_smoke.py --data .\scenesplit_pc4.yaml --batch 4 --no-amp
# 👀 within epoch 1: normal box/cls/dfl losses (no NaN — NaN means amp slipped on)
# 👀 at each epoch end: "[fccg-verify] OK gates={ L16:0.xx, L20:0.xx }"  (0.02<g<0.98 = healthy)
# 👀 at the end: "[smoke] S0 SMOKE COMPLETE"
# ⛔ CUDA OOM at batch 4 ⇒ rerun with --batch 2 (record it)

# ===== 6) results home (house rule; AnyDesk copy back to laptop)
# copy D:\thesis_2007074\10-07-2026-Novelty-Lap-3\scripts\runs_smoke\fccg_s0
#   -> d:\Academics\thesis folder\05-07-2026-Novelty-Lap\results\pc4\2026-07-12_S0_fccg_smoke\
#      (results.csv, args.yaml, weights\best.pt + last.pt, the console log if saved)
# + one line in results\MANIFEST.md
```

**S0 pass =** selftest 15/15 · CHECK-LOAD OK (params ≤22.5M) · 2 epochs no error · `[fccg-verify]`
healthy each epoch · losses decreasing. Pass ⇒ S1 paired 50-ep pilots (ranking doc §5).
**Fallback:** if the scene-split rebuild is blocked for any reason, the smoke MAY run on the
official split instead (S0 only tests trainability): reuse the existing `official_pc4.yaml` from
the G2 folder or write the same yaml with `new_dataset3` as path. S1 pilots MUST use the scene split.
**Metrics note:** smoke is EXEMPT from the metric contract; every run from S1 onward logs per
`..\docs\2026-07-12_metric_contract_reference.md` (= `Last Month\system_spec_thesis.md` §6 +
`system_spec.md` §11 + the FCCG-specific gate metrics).
