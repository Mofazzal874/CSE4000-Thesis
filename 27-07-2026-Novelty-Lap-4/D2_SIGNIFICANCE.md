# D2 significance — seeds + bootstrap (started 2026-08-06)

Goal: turn the D2 tiny-aware assignment claim (+2.0pp AP_small / +2.09pp VT-recall, seed 0) from a
single-seed result into a **reviewer-proof** one. Two independent kinds of evidence:

| evidence | question it answers | cost | script |
|---|---|---|---|
| **multi-seed** | does the gain survive RE-TRAINING with new randomness? | ~12h/run, GPU | `20_assign_patch.py` |
| **bootstrap** | does the gain survive a different TEST sample? | ~mins, no GPU | `28_bootstrap_sig.py` |

Only the D2 pair needs this. Drone (+46pp) and disaster (honest limit) don't — the effect is either
overwhelming or already reported as marginal.

## Folder / naming (all under PC-1 `...\scripts\runs_s1\`)
| model | seed 0 (DONE) | seed 1 | seed 2 |
|---|---|---|---|
| control (α=0) | `s3_control_full` | `s3_control_full_s1` | `s3_control_full_s2` |
| assign (α=0.5) | `s3_assign_full`  | `s3_assign_full_s1`  | `s3_assign_full_s2`  |
- eval json per run: `s1_eval_<name>.json`  ·  bootstrap json: `bootstrap_<A>_vs_<B>.json`
- archive to laptop `05-07-2026-Novelty-Lap\results\pc1\runs_s1\` (small jsons only; weights stay on PC-1/CD)

## Step 0 — read seed-0's exact config (so seeds match)  [PC-1]
```powershell
Get-Content runs_s1\s3_assign_full\args.yaml | Select-String "data:|batch:|imgsz:|epochs:|patience:|lr0:|optimizer:|cache:|pretrained:|seed:"
```
Set `$DATA` to the `data:` path it prints. Expected config: pretrained yolo11m.pt, 300ep, patience 50,
batch 12, imgsz 640, AdamW lr0 0.001, cache ram, seed 0.

## Step 1 — SMOKE (2 epochs, ~6 min) before the 48h commitment  [PC-1]
```powershell
python 20_assign_patch.py --train --data $DATA --alpha 0.5 --pretrained yolo11m.pt `
  --epochs 2 --patience 0 --name s3_smoke_sig --seed 1 --device 0 --reserve-gb 1
```
PASS = you see `[assign-patch] ACTIVE`, `[assign-verify] OK` (patch fired), losses fall, checkpoints
write. Then delete it: `Remove-Item runs_s1\s3_smoke_sig -Recurse -Force`.

## Step 2 — the 4 seed runs, UNATTENDED via the chain script  [PC-1]
**Preferred (hands-off):** `run_d2_seeds.ps1` chains all 4 + auto-evals each, resume-safe (skip-if-done
markers + --resume). Launch once, walk away; after a power cut just launch it again.
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
cd "E:\Thesis_mofazzal_2007074\10-07-2026-Novelty-Lap-3\scripts"
.\run_d2_seeds.ps1
```
Manual fallback (if you'd rather launch each yourself): the 4 commands below. Each ~12h; `--reserve-gb 1`
protects the slot; `--save-period 25` + `--resume` = power-cut safe. Assign first, then control; seed 1 then 2.
```powershell
# (1) assign seed 1
python 20_assign_patch.py --train --data $DATA --alpha 0.5 --pretrained yolo11m.pt --epochs 300 --patience 50 `
  --name s3_assign_full_s1 --seed 1 --batch 12 --device 0 --reserve-gb 1 --save-period 25 --resume
# (2) control seed 1
python 20_assign_patch.py --train --data $DATA --alpha 0.0 --pretrained yolo11m.pt --epochs 300 --patience 50 `
  --name s3_control_full_s1 --seed 1 --batch 12 --device 0 --reserve-gb 1 --save-period 25 --resume
# (3) assign seed 2
python 20_assign_patch.py --train --data $DATA --alpha 0.5 --pretrained yolo11m.pt --epochs 300 --patience 50 `
  --name s3_assign_full_s2 --seed 2 --batch 12 --device 0 --reserve-gb 1 --save-period 25 --resume
# (4) control seed 2
python 20_assign_patch.py --train --data $DATA --alpha 0.0 --pretrained yolo11m.pt --epochs 300 --patience 50 `
  --name s3_control_full_s2 --seed 2 --batch 12 --device 0 --reserve-gb 1 --save-period 25 --resume
```
After (1)+(2) you already have a **2-seed** result; (3)+(4) make it the standard **3-seed**.

## Step 3 — eval each finished run  [PC-1]
```powershell
$T="E:\Thesis_mofazzal_2007074\common\c2a\C2A_Dataset\new_dataset3_scenesplit_v1\test"
python 18_s1_eval.py --variant control --tag <run_name> --weights runs_s1\<run_name>\weights\best.pt `
  --images-dir "$T\images" --gt-json "$T\test_annotations.json" --device 0
```
Report each seed's AP_small + VT-recall + F2; aggregate = mean ± std over the 3 seeds (see table below).

## Step 4 — bootstrap (NO GPU, run anytime, incl. now on seed 0 + real sets)  [PC-1]
```powershell
$T="E:\Thesis_mofazzal_2007074\common\c2a\C2A_Dataset\new_dataset3_scenesplit_v1\test"
python 28_bootstrap_sig.py --tag-a s3_control_full --tag-b s3_assign_full --gt-json "$T\test_annotations.json"
# cross-event honest-limit check:
$D="E:\Thesis_mofazzal_2007074\drone_data\annotations"
python 28_bootstrap_sig.py --tag-a s3_xevent --tag-b d3all_xevent --gt-json "$D\disaster-xevent-v1\test\_annotations_sc.coco.json"
```

## Results log (fill as runs land)
| metric | s0 Δ | s1 Δ | s2 Δ | mean±std | bootstrap (seed0, n=2040, 1000 boots) |
|---|---|---|---|---|---|
| VT-recall<8 | +2.09 | +1.66 | +1.11 | +1.62 +- 0.49 | **+2.08pp [1.77, 2.38] p<0.001 SIG** |
| F2 | +0.27 | -0.11 | -0.04 | +0.04 +- 0.20 | +0.27pp [0.03, 0.50] p=0.024 SIG |
| AP50 | -0.08 | +0.09 | -0.03 | -0.01 +- 0.09 | -0.08pp [-0.32, 0.17] p=0.54 n.s. |
| tiny 8-16 | -1.15 | | | | -1.15pp [-1.48, -0.84] p<0.001 (sig cost) |
| small 16-32 | -0.95 | | | | -0.95pp [-1.4, -0.5] p<0.001 (sig cost) |
| medium 32-96 | -1.3?? | | | | -13.2pp [-17.9, -9.3] p<0.001 (sig cost, n=410) |
- AP_small (COCO) not bootstrapped (pycocotools per-resample too slow); VT-recall<8 IS the direct, clean tiny claim.
- BOOTSTRAP VERDICT: the +2.08pp <8px gain is significant vs test-set noise (p<0.001) = a genuine tiny-specialist
  reallocation. Seeds (below) add the TRAINING-variance half; bootstrap alone is already JSTARS-strength.
- CROSS-EVENT bootstrap (s3_xevent vs d3all_xevent, n=23): AP50 +1.69pp [-3.19,5.69] p=0.47 n.s.; F2 +1.62 p=0.41 n.s.;
  precision@25 -13.1pp p=0.018 (sig WORSE) -> disaster fine-tune does NOT significantly generalize cross-event. Honest limit, confirmed.

## Status
- [x] scripts ready + selftested (20 seed/save-period, 28 bootstrap)
- [x] Step 0 config read (data=scenesplit_pc1.yaml, 300ep/pat50/b12/AdamW lr0.001)
- [x] **BOOTSTRAP DONE** — D2 VT-recall significant (p<0.001); cross-event n.s. (honest limit confirmed)
- [x] all 4 seed runs + evals DONE (2026-08-16). VT delta +2.09/+1.66/+1.11 = +1.62+-0.49 (replicates);
  AP_small delta +2.00/-1.29/+0.53 = noise -> claim dropped. Report updated (draft2).
