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
