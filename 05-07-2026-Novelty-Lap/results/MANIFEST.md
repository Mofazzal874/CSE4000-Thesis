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

### 2026-07-25 - pc1 - misc - S2 tiny-aware assignment (NWD-in-TAL) alpha-sweep {0,0.5,1.0}, scene-split - DONE (KEEPER)
- Paths: `results\pc1\runs_s1\s2_assign\` (a=0.5) + `s2_assign_a10\` (a=1.0) + s1_eval_s2_assign{,_a10}.json. Runner = 20_assign_patch.py.
- Mechanism: scale-robust NWD blended (weight `alpha`) into TaskAlignedAssigner.iou_calculation (assignment-side; distinct from the failed G2 loss-side NWD). 50-ep-from-scratch, same protocol as s1_control.
- Dose-response (TEST):
  | metric | control | a=0.5 | a=1.0 |
  |---|---|---|---|
  | VT-recall (<8) | 0.6931 | 0.7120 (+1.89) | 0.7293 (+3.62) |
  | tiny (8-16) | 0.8120 | 0.8094 | 0.7947 |
  | small (16-32) | 0.8304 | 0.8199 | 0.7974 |
  | AP_small | 0.5596 | 0.5526 (-0.70) | 0.5053 (-5.43) |
  | AP50 | 0.8208 | 0.8232 | 0.8078 |
- VERDICT: **KEEPER** (architecture contribution; replaces FCCG). `alpha` = a clean MONOTONIC trade dial: higher alpha -> more <8px recall, less small/AP_small. Sweet spot **alpha=0.5** (+1.89 VT for -0.70 AP_small, ratio ~2.7); alpha=1.0 clears +2.0 VT (+3.62) but wrecks AP_small (-5.43, ratio ~0.67). First real tiny-object lever; assignment>loss VALIDATED. NEXT: confirm at full protocol (S3, 300-ep pretrained) + pivot to D3 sim-to-real.

### 2026-07-26 - pc1 - S3 - assignment (alpha=0.5) FULL protocol (pretrained, 300-ep) - DONE
- Path: results\pc1\runs_s1\s3_assign_full\ + s1_eval_s3_assign_full.json. Runner 20_assign_patch (--pretrained yolo11m --epochs 300 --patience 50). Early-stopped 291/300, best@241.
- TEST (scene-split): AP50 0.8441 / AP_small 0.6151 (= C2A ~0.615 label ceiling) / VT-recall 0.7508 / tiny 0.8228 / small 0.841.
- vs G1 CBAM+P2 (300-ep, thesis script): +0.69 AP50 / +0.19 AP_small / +0.54 VT-recall; tiny/small traded down. NOTE: not a clean pair (different runner) -> clean pair = s3_control_full (alpha=0, running on PC-2). Full-protocol assignment gain is SMALLER than the 50-ep pilot; PC-2 control settles it.

### 2026-07-26 - pc1 - D3 - sim-to-real fine-tune (C2A+assignment -> +labeled drone) - STRONG HEADLINE
- Path: results\pc1\runs_d3\d3_joint_A\ + runs_s1\s1_eval_{s3_drone_before,d3_drone_after,d3_c2a}.json. Runner 24_joint_finetune (fine-tune s3_assign_full on C2A + 20x oversampled labeled drone@1280px, AdamW 5e-4, 80-ep best@30, cache=disk).
- SIM-TO-REAL GAP CLOSED (drone test, 60 frozen frames, BEFORE=C2A zero-shot vs AFTER=fine-tuned):
  | metric | BEFORE | AFTER | delta |
  |---|---|---|---|
  | AP50 | 0.324 | 0.725 | +40.1pp (2.2x) |
  | Best_F2 (PRIMARY) | 0.376 | 0.755 | +37.9pp |
  | recall@0.25 | 0.337 | 0.828 | +49.2pp |
  | mAP50-95 | 0.175 | 0.421 | +24.6pp |
- C2A NO REGRESSION (fine-tuned on C2A test vs S3): AP50 0.8441->0.8366 (-0.75pp), AP_small 0.6151->0.6026 (-1.25pp), VT-recall 0.7508->0.7503 (-0.05pp).
- VERDICT: sim-to-real pillar = STRONG. +40pp AP50 / recall 0.34->0.83 on the real drone benchmark at ~0 C2A cost. CAVEATS: drone test medium-dominated (very_tiny n=0, tiny n=7) = DOMAIN-transfer result not tiny-object; precision 0.575->0.518 @0.25 (more FPs) -> measure void-FP next. NEXT: eval d3 on R-set (real disaster) + void-FP before/after + S5 tables.

### 2026-07-26 - pc1 - D3 - R-set (real DISASTER) eval: drone-tune does NOT transfer - KEY FINDING
- Path: runs_s1\s1_eval_{s3_rset_before,d3_rset_after}.json. R-set = 25 real flood/rubble frames, 17px median (tiny+cluttered).
- BEFORE (C2A zero-shot) vs AFTER (drone-tuned): AP50 0.134->0.125, F2 0.202->0.187, recall@0.25 0.127->0.122 -- FLAT / slightly worse.
- FINDING: domain adaptation is domain-SPECIFIC. Drone-shoot (sports fields, medium people) tuning gave +40pp on drone test but 0 on disaster R-set. Both models ~0.13 AP50 on R-set (tiny + disaster clutter = hard). To get a real-DISASTER result, need DISASTER training frames (mine from Footage From News(Real), label ~50-100; also fixes void-FP). Honest limitation + the Tier-1 hard-negative next step.

### 2026-07-27 - pc1 - CLEAN D2 pair: control (a=0) vs assignment (a=0.5) FULL protocol - CONFIRMED +
- Path: runs_s1\s3_control_full\ (best@242, power-cut corrupted last.pt but best.pt intact) + s1_eval_s3_control_full.json vs s1_eval_s3_assign_full.json. Same runner (20_assign_patch), same protocol (pretrained, 300-cap, patience 50), same GPU (PC-1) -> ONLY alpha differs = the clean architecture pair.
- Delta (assignment a=0.5 - control a=0), C2A scene-split TEST:
  | metric | control | assign | delta |
  |---|---|---|---|
  | Best_F2 (PRIMARY) | 0.8257 | 0.8284 | +0.27pp |
  | AP_small | 0.5951 | 0.6151 | +2.00pp |
  | very-tiny recall (<8) | 0.7299 | 0.7508 | +2.09pp |
  | tiny (8-16) | 0.8343 | 0.8228 | -1.15pp |
  | small (16-32) | 0.8504 | 0.841 | -0.94pp |
  | AP50 | 0.8449 | 0.8441 | -0.08pp |
- VERDICT: D2 = tiny-object SPECIALIST, cleanly confirmed. Concentrated gain on the SAR-critical regime (+2.00 AP_small, +2.09 <8px recall) at NEUTRAL overall F2 (+0.27) / AP50 (-0.08) and a -1pp cost on 8-32px. Clears both pass bars. Clean matched control shows the lever is STRONGER than the G1 comparison (+0.54 VT) implied -- G1's different runner undersold it. Paper framing: "reallocates capacity to the very-tiniest people; +2pp AP_small/<8px recall without hurting overall F2."

### 2026-08-06 - pc1 - LAP4 - disaster retrain (d3_all: C2A+drone+campus+disaster) - STRONG: R-set TRIPLED
- Path: runs_d3\d3_all\ (best@66, 80ep, run_config.json + args.yaml frozen; CBAM+P2 backbone, 19.57M/86.7GFLOPs)
  + runs_s1\s1_eval_d3all_{rset,drone,campus}.json. Runner 24 --preset d3_all. 248 real train frames
  (drone 101 + disaster 60 + campus 87), oversample x20, on C2A base; eval on FROZEN held-out sets.
- R-SET (real disaster, 25 frames) -- the headline. Both prior models ~0.13; disaster labels transfer where drone didn't:
  | stage | AP50 | F2 | recall@.25 | AP_small |
  |---|---|---|---|---|
  | zero-shot (C2A) | 0.134 | 0.202 | 0.127 | - |
  | drone-tuned (D3) | 0.125 | 0.187 | 0.122 | - |
  | **disaster-tuned (d3_all)** | **0.411** | **0.472** | **0.451** | 0.127 |
  => +27.7pp AP50 / +27pp F2 / +32pp recall vs zero-shot. CLEAN CONTROL: drone-tune 0.134->0.125 (nothing),
  disaster-tune 0.134->0.411 (3x) on the SAME R-set = target-domain labels necessary; cross-domain (drone) don't.
- DRONE test (60 frozen), AP50_allpoint (authoritative saved jsons): zero-shot 0.335 -> drone-only 0.783 -> all-real 0.795
  (F2 0.771, recall 0.837). Adding disaster+campus = +1.25pp on drone = NO regression (earlier "+7pp" was a stale figure).
- CAMPUS eval (25, occlusion, new): AP50 0.689 / F2 0.666 / recall 0.718 / AP_small 0.262.
- C2A regression (d3all_c2a vs s3_assign_full): AP50 0.844->0.837 (-0.71pp), F2 -0.45, AP_small 0.615->0.602 (-1.34),
  VT-recall -0.41; ECE 0.0142->0.0111 (better). = NEGLIGIBLE, no forgetting. Efficiency: 19.57M/86.7GFLOPs/39.7MB/8.7ms/115FPS.
- CAVEATS (must state honestly): (1) disaster-train + R-set share the SAME 2 news clips (deduped frames) =>
  SAME-DOMAIN/within-video adaptation, NOT cross-video generalization; the drone-vs-disaster contrast is valid
  (same R-set) but absolute claim = "small target-domain budget -> big same-domain gain". (2) very_tiny <8px R-set
  recall = 0 (n=54) still missed. (3) R-set = 25 frames, wide CI. (4) VOID-FP FIX NOT DEMONSTRATED: R-set precision@.25
  0.521(zero-shot)->0.513(disaster) = FLAT; the win is RECALL (+32pp) at neutral precision, NOT a proven void-FP drop
  (needs a dedicated void-region FP check). VERDICT: disaster pillar = STRONG (R-set 3x); all 4 held-out gates pass
  (C2A held, drone held, R-set tripled, campus solid) -> ONE unified real-domain model. Flips the earlier
  "domain-specific limitation" into a demonstrated FIX (target-domain labels necessary; drone labels don't transfer).

### 2026-08-06 - pc1 - LAP4 - CROSS-EVENT test (unseen AP Israel disaster clip, 23 frames) - HONEST: R-set gain was target-specific
- Path: runs_s1\s1_eval_{d3all_xevent,s3_xevent}.json (+ pngs/env/scorecorr). Set = disaster-xevent-v1 (23 frames,
  1 AP news clip, DISJOINT from R-set+disaster-train via dHash guard). Small/medium people (median 28px, 0 very-tiny).
- PURPOSE: kill the "R-set shares videos with disaster-train" attack by testing on a genuinely unseen event.
- RESULT (does disaster fine-tuning generalize cross-event?): compare s3_assign_full (NO disaster) vs d3_all on the SAME unseen set:
  | on unseen Israel event | AP50 | F2 | AP_small | small-recall | precision@25 |
  |---|---|---|---|---|---|
  | s3 (C2A+assign, no disaster) | 0.396 | 0.435 | 0.256 | 0.502 | 0.732 |
  | d3_all (disaster-trained)   | 0.413 | 0.451 | 0.235 | 0.461 | 0.601 |
  | delta from disaster training | +1.7 | +1.6 | -2.1 | -4.2 | -13.1 |
- HONEST VERDICT: disaster fine-tuning adds only +1.7pp AP50 (WITHIN NOISE, 23 frames) on an unseen event, at a -13pp
  precision cost. The R-set +27.7pp was INFLATED by (a) within-video overlap with training clips + (b) R-set being the
  tiny regime (C2A weak there, 0.13). On same-size unseen data the gain vanishes. => target-VIDEO adaptation, NOT
  disaster-domain generalization. Silver lining: the C2A+assignment BASE model transfers to unseen real disaster
  small/medium people ON ITS OWN (0.40 AP50); it fails specifically on <16px real people. Cross-event test = defensibility
  WIN (measured + reported honestly; kills the same-video attack). Paper: demote disaster to an honest case study
  (within-video strong, cross-event marginal); headline = D2 assignment + DRONE sim-to-real (+46pp). Future work: broader disaster data.

### 2026-08-06 - pc1 - LAP4 - BOOTSTRAP significance (paired image-level, 28_bootstrap_sig, 1000 boots) - D2 gain SIGNIFICANT
- Path: runs_s1\bootstrap_s3_control_full_vs_s3_assign_full.json + bootstrap_s3_xevent_vs_d3all_xevent.json.
- Method: resample test images w/ replacement (paired A/B), recompute metric, delta=B-A -> 95% CI + two-sided p. No GPU/retrain.
- D2 pair (s3_control_full vs s3_assign_full, n=2040): the tiny-object claim is significant vs test-set noise:
  | metric | delta | 95% CI (pp) | p |
  |---|---|---|---|
  | VT-recall<8 | +2.08 | [1.77, 2.38] | <0.001 SIG |
  | F2 | +0.27 | [0.03, 0.50] | 0.024 SIG |
  | AP50 | -0.08 | [-0.32, 0.17] | 0.54 n.s. |
  | tiny 8-16 | -1.15 | [-1.48,-0.84] | <0.001 (sig cost) |
  | small 16-32 | -0.95 | [-1.4,-0.5] | <0.001 (sig cost) |
  | medium 32-96 | -13.2 | [-17.9,-9.3] | <0.001 (sig cost, n=410) |
  => genuine tiny-SPECIALIST reallocation, statistically validated. (AP_small not bootstrapped-pycocotools too slow; VT-recall = the clean tiny claim.)
- Cross-event (s3_xevent vs d3all_xevent, n=23): AP50 +1.69 [-3.19,5.69] p=0.47 n.s.; F2 +1.62 p=0.41 n.s.;
  precision@25 -13.1 [-25.3,-2.3] p=0.018 (sig WORSE). => disaster fine-tune does NOT significantly generalize
  cross-event; honest limitation CONFIRMED with a p-value. NEXT: 3-seed D2 runs add the training-variance half
  (bootstrap = test-variance only). Bootstrap alone already JSTARS-strength.

### 2026-08-05 - YOLO26m PARKED to lap-5 (removed from the lap-4 yolo11m story)
- Decision: lap-4 = a clean yolo11m story. YOLO26m gets its OWN lap (novelty-lap-5): base + CBAM + P2 +
  assignment + sim-to-real, to test whether the newer backbone subsumes or complements our contributions.
- The plain-YOLO26m anchor (AP50 0.8552 / F2 0.8364 / AP_small 0.6229 / VT 0.742) is the lap-5 STARTING
  baseline, recorded in `27-07-2026-Novelty-Lap-4\..\05-08-2026-Novelty-Lap-5\README.md`. Data preserved,
  not deleted: eval json `runs_s1\s1_eval_yolo26m.json`; model on PC-1 `runs_anchor\yolo26m\weights\best.pt`;
  runner `21_yolo26_anchor.py`. NOT part of any lap-4 paper table.

### 2026-07-25 - pc1 - misc - P1 seam-reliance probe (C2A, CBAM+P2 s1_control) - DONE [json still on remote]
- Remote: `E:\Thesis_mofazzal_2007074\...\scripts\runs_probe\seam_probe_c2a.json` (copy to results\pc1\ when convenient). Runner = 19_seam_probe.py.
- Probe: low-pass blur + JPEG re-compression of scene-split TEST, degradation eval.
- Key numbers (drop from original): blur sigma=2 -> VT-recall -65.7pp / AP50 -53.7pp; JPEG q=90 (near-lossless!) -> AP50 -65.0pp / VT -59.7pp / AP_small -49.2pp.
- Read: SEVERE high-frequency reliance (tiny-object signal lives in the band JPEG discards) -> motivates D3 harmonization. CAVEAT: needs SARD (real) control to separate synthetic-seam reliance from generic tiny-object fragility.
