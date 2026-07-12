# S0 smoke checklist — PC-4 (or PC-2 GPU1) · PowerShell · kill-safe

Copy the whole `10-07-2026-Novelty-Lap-3\scripts\` folder to the PC first (keep filenames —
checkpoints will remember the module name `10_fccg_modules`).

```powershell
# 0) existence checks (house rule: verify before running)
Test-Path .\10_fccg_modules.py; Test-Path .\11_fccg_smoke.py
Test-Path D:\thesis_2007074            # PC-4 root (adjust if PC-2: D:\student_2k20\2007074)

# 1) venv — prompt must show (env); if (base) shows: conda deactivate
#    (PC-1 only would need the exec-policy line first)

# 2) module selftests on this PC's torch (expect: SELFTEST OK, 15/15)
python .\10_fccg_modules.py --selftest

# 3) YAML build + forward + checkpoint roundtrip (needs the PC's modern ultralytics)
python .\10_fccg_modules.py --check-load
# expect: "~20.1M params", "FCCG ACTIVE {...gate means ~0.5}", "CHECK-LOAD OK"

# 4) 2-epoch smoke on the scene split (PC-4: --no-amp MANDATORY, batch 4)
#    data.yaml = this PC's copy of new_dataset3_scenesplit_v1\data.yaml (paths inside
#    must point at the LOCAL dataset copy — same file used by earlier lap runs)
python .\11_fccg_smoke.py --data <LOCAL_SCENESPLIT>\data.yaml --batch 4 --no-amp
# PC-2 instead:  python .\11_fccg_smoke.py --data <...> --batch 16 --device 1

# 5) results home (house rule)
# copy runs_smoke\fccg_s0 -> d:\...\05-07-2026-Novelty-Lap\results\<pc>\2026-07-XX_S0_fccg_smoke\
# + one line in results\MANIFEST.md
```

**S0 pass =** selftest 15/15 · check-load OK (params ≤22.5M) · 2 epochs train w/o error ·
`[fccg-verify]` healthy each epoch · losses decreasing. Then S1 paired pilots per ranking doc §5.
**If check-load fails on parse:** the ultralytics version on that PC differs from the lineage —
record exact version + error in the results folder; do NOT hand-edit ultralytics sources.
