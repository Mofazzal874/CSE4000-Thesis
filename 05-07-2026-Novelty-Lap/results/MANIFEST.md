# RESULTS INBOX — manifest + copying convention
This folder is the **single home for every result copied from the remote PCs**. When paper-writing
starts, everything needed lives here and is findable through this file. **Every transfer gets one
entry below — no exceptions.** (If an entry is missing, the data effectively doesn't exist.)

## The convention (how to copy results from any PC)
1. **Destination pattern:** `results\<pc>\<YYYY-MM-DD>_<gate>_<shortname>\`
   - `<pc>` ∈ pc1 | pc2 | pc3 | pc4 · `<gate>` ∈ G1 (leakage) | G2 (NWD) | G3 (fusion) | G4 (real-world) | misc
   - Example: `results\pc1\2026-07-09_G1_scenesplit_cbam_p2\`
2. **What to COPY:** `metrics\` (summary.json etc.), `results.csv`, `args.yaml`, `env.json`,
   `*_ablation.csv/.json`, `preds_*.json`, `cal.json`, `*_artifacts\`, logs (`*.log`), plots, and
   `weights\best.pt` **only for models we may reuse** (pilots' best.pt OK, they're 40 MB).
3. **What to SKIP:** `epoch*.pt` intermediate checkpoints (hundreds of MB each, regenerable),
   dataset caches, `train_batch*.jpg` mosaics (optional).
4. **After copying:** add a manifest entry (template below), then `git add results\<...>\*.csv *.json *.md`
   (small text files only — git-ignore the big preds jsons if the repo gets heavy).
5. **Naming truth:** on the REMOTE PC the run stays where it ran (that's the working copy);
   the laptop copy here is the archival/paper copy.

### Entry template
```
### <date> — <pc> — <gate> — <one-line what>
- Path: results\<pc>\<folder>\
- Source (remote): <full path on the PC>
- Key numbers: <the 2-4 numbers that matter>
- Status/next: <what this decided / what's still owed>
```

---

## Entries

### 2026-07-07 — pc4 — G2 — NWD pilot trio (α=0.0 / 0.5 / 0.7), 25-ep fine-tunes from epoch125 — ✅ COMPLETE, GATE CLOSED
- Path: `results\pc4\2026-07-07_G2_nwd_pilots\`  (verified complete 2026-07-08: all 16 checklist items OK)
- Source (remote): `D:\thesis_2007074\05-07-2026-Novelty-Lap\scripts\` (runs\detect\runs_nwd + preds_val_a0* + artifacts + official_pc4.yaml)
- Contents: pilot_a00/a05/a07 full run dirs (results.csv 25/25 each, best.pt+last.pt, args.yaml, curves) + all three 04-evals (preds json + ablation csv/json + artifacts).
- Key numbers (04-eval, official val, whole-frame) — **THE PAPER'S NWD ABLATION ROW**:
  | α | AP_small | very_tiny R | tiny R | Best_F1 |
  |---|---|---|---|---|
  | 0.0 | 0.6292 | 0.7464 | 0.8641 | 0.8592 |
  | 0.5 | 0.6291 | **0.7537** | 0.8670 | 0.8597 |
  | 0.7 | 0.6292 | 0.7514 | 0.8668 | 0.8601 |
  Config identical across pilots (epoch125 init, official split, batch 4, AdamW, seed 0, amp off).
- **GATE G2 VERDICT: CLOSED, marginal-positive.** Effect peaks at α=0.5 (+0.73pt very-tiny recall, zero cost), does not scale with α, AP_small capped by label ceiling. No full 300-ep run. NWD = supporting ablation row with a complete 3-point dose-response.

### (pending) — pc2 — G3 — fusion val preds
- Remote only so far: `D:\student_2k20\2007074\05-07-2026-Novelty-Lap\scripts\preds_val.json` (scene-split val, nms row AP50_allpoint 0.8926 / coco_AP 0.6911). Copy after the full G3 table exists (cal.json + preds_test + ablation csv).

### (pending) — pc1 — G1 — CBAM+P2 scene-split retrain
- Running since 2026-07-08 ~02:17 on PC-1. When done copy `runs\<id>\{metrics,logs}\`, `results.csv`, `args.yaml`, `env.json`, `scenesplit_run.log`, `weights\best.pt` → `results\pc1\<date>_G1_scenesplit_cbam_p2\`. See `..\PC1_RUN_STATUS.md`.

### 2026-07-24 - pc2 - S0 - FCCG smoke (2-epoch trainability gate) - PASS
- Location: results\pc2\2026-07-24_S0_fccg_smoke\fccg_s0\ (best.pt, last.pt, results.csv, args.yaml)
- Model: YOLO11m+CBAM+P2+FCCG (context-gated evidence), 20.12M params, 98.9 GFLOPs; modules parsed active (CBAM@10, FFLUp@11/14/18, FCCGFuse@16/20).
- 2 epochs from scratch, C2A scene-split (6135/2040), A6000, batch16, AMP, ~15 min.
- Losses fell (box 2.73->1.90, cls 2.05->1.36, dfl 1.90->1.32); val mAP50 0.531->0.626 (2-ep scratch, not a real number).
- KEY: gates ACTIVE+LEARNING: L16 0.779->0.698, L20 0.661->0.587 (off 0.5 init, unsaturated). Ckpt roundtrip OK.
- Verdict: S0 CLEARED. Next = S1 paired 50-ep pilots (control vs +FCCG).

### 2026-07-24 - pc1 - G1 - CBAM+P2 scene-split baseline (300-ep protocol) - DONE
- Run: 20260708_022132_yolo11m_cbam_p2head_s0_nogit (262 ep, F2 early-stop @152, 12.1h, seed 0).
- TEST: AP50 0.8372 / AP 0.6107 / AP_small 0.6132 / VT-recall 0.7454 / tiny 0.8459 / small 0.8591.
- VAL:  AP50 0.8565 / AP 0.6311 / AP_small 0.6336. Effic 19.57M / 86.7 GFLOPs / 15.67ms / ECE 0.026.
- Leakage vs official (0.8533/0.6153/0.6156/0.7575): -1.6 AP50 / -0.2 AP_small / -1.2 VT = MODEST.
- THIS is the authoritative scene-split baseline FCCG-YOLO must beat at full protocol.
- Copy folder -> results\pc1\2026-07-24_G1_scenesplit_cbam_p2\ (metrics, results.csv, summary.json, args, best.pt).

### 2026-07-25 - pc1 - misc - S1 paired 50-ep pilots (control CBAM+P2 vs +FCCG), scene-split - DONE (FCCG NULL)
- Path: `results\pc1\runs_s1\` (s1_control\, s1_fccg\ : results.csv, args.yaml, best.pt, s1_metrics.json + s1_eval_<v>.json). Big `*_preds.json` gitignored.
- Source (remote): `E:\Thesis_mofazzal_2007074\10-07-2026-Novelty-Lap-3\scripts\runs_s1\`
- Protocol: 50-ep from scratch, AdamW lr0=0.001, seed 0, 640px, scene-split (6135/2040/2040). Eval = script 18_s1_eval (AP_small via pycocotools + per-size recall), scene-split TEST.
- Key numbers (TEST):
  | run | AP_small | VT-recall(<8) | tiny(8-16) | small(16-32) | AP50 |
  |---|---|---|---|---|---|
  | control | 0.5596 | 0.6931 | 0.8120 | 0.8304 | 0.8208 |
  | +FCCG | 0.5535 | 0.6907 | 0.8172 | 0.8308 | 0.8185 |
- VERDICT: FCCG = clean NULL (AP_small -0.61pp, VT-recall -0.24pp; within noise; reproduced on PC-2 too). Demoted to a documented NEGATIVE ablation row. Pass bar (+1.5 AP_small OR +2.0 VT-recall) missed on every bin.

### 2026-07-25 - pc1 - misc - S2 tiny-aware assignment (NWD-in-TAL, alpha=0.5), scene-split - DONE (BORDERLINE +)
- Path: `results\pc1\runs_s1\s2_assign\` + `s1_eval_s2_assign.json`. Runner = 20_assign_patch.py.
- Source (remote): `E:\Thesis_mofazzal_2007074\...\scripts\runs_s1\s2_assign\`
- Mechanism: scale-robust NWD blended (alpha=0.5) into TaskAlignedAssigner.iou_calculation (assignment-side; distinct insertion point from the failed G2 loss-side NWD). Same 50-ep protocol as s1_control.
- Key numbers (TEST) vs s1_control:
  | metric | control | s2_assign | delta |
  |---|---|---|---|
  | VT-recall (<8) | 0.6931 | 0.7120 | **+1.89pp** |
  | AP_small | 0.5596 | 0.5526 | -0.70pp |
  | small (16-32) | 0.8304 | 0.8199 | -1.05pp |
  | AP50 | 0.8208 | 0.8232 | +0.24pp |
- VERDICT: BORDERLINE POSITIVE. First lever to move <8px recall (+1.89pp, ~8x FCCG's delta) - just under the +2.0 bar, a targeted trade (helps <8px, costs 16-32px + mAP50-95 -0.96). Assignment-over-loss direction VALIDATED. Next: alpha=1.0 sweep (s2_assign_a10).

### 2026-07-25 - pc1 - misc - P1 seam-reliance probe (C2A, CBAM+P2 s1_control) - DONE [json still on remote]
- Remote: `E:\Thesis_mofazzal_2007074\...\scripts\runs_probe\seam_probe_c2a.json` (copy to results\pc1\ when convenient). Runner = 19_seam_probe.py.
- Probe: low-pass blur + JPEG re-compression of scene-split TEST, degradation eval.
- Key numbers (drop from original): blur sigma=2 -> VT-recall -65.7pp / AP50 -53.7pp; JPEG q=90 (near-lossless!) -> AP50 -65.0pp / VT -59.7pp / AP_small -49.2pp.
- Read: SEVERE high-frequency reliance (tiny-object signal lives in the band JPEG discards) -> motivates D3 harmonization. CAVEAT: needs SARD (real) control to separate synthetic-seam reliance from generic tiny-object fragility.
