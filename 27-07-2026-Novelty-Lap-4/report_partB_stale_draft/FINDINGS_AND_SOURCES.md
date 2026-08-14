# Findings & Sources — verified numbers for the report

> Every figure/number in the report must trace to a row here. All values below are read directly
> from run `metrics/summary.json` (COCO eval) unless noted. **Do not use the old `Mofa_thesis_13.4.26`
> deck's numbers** — see the ⚠️ supersession note.

## ⚠️ Supersession — the old deck (13.4.26) is OUTDATED
The old progress deck (`Last Month/Mofa_thesis_13.4.26.txt`) tells the *pre-final* story where
**Mamba+CBAM+P2 is "best" (mAP50 0.877, mAP50-95 0.654) at "zero parameter cost" (19.59 M)**.
Both claims are **wrong for the report** — they came from the *buggy* run where Mamba was silently
stripped (so it had CBAM+P2's params and CBAM+P2's scores). The **final-month AdamW re-runs** (below)
are canonical:
- Mamba is a **NULL result** (ties CBAM+P2 on accuracy), and
- the valid Mamba run **adds +2.4 M params (22.0 M) and 98.4 GFLOPs** — *not* zero-cost.

The deck is still useful for: narrative structure, the CBAM-vs-ECA comparison, SAHI/TTA numbers,
the Nihal SOTA table, and references — all captured below with caveats.

---

## 1. PRIMARY ABLATION — canonical (AdamW, C2A, 4070 Ti SUPER, 1 seed, COCO test eval)

| Model | params (M) | GFLOPs | layers | COCO AP | AP50 | AP75 | AP_small | AR_100 | Best_F1 | Best_F2 | latency e2e mean (ms) | ECE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLO11m baseline | 20.03 | 67.65 | 126 | 0.6151 | 0.8432 | 0.6547 | 0.6147 | 0.6913 | 0.8497 | 0.8399 | 13.68 | — |
| +CBAM | 19.08 | 66.86 | 123 | 0.6161 | 0.8473 | 0.6563 | 0.6160 | 0.6923 | 0.8499 | 0.8406 | 13.48 | 0.0206 |
| +CBAM+P2 **(recommended)** | 19.57 | 86.66 | 153 | 0.6153 | **0.8533** | 0.6603 | 0.6156 | **0.7030** | 0.8479 | **0.8442** | 14.55 | 0.0214 |
| +Mamba+CBAM+P2 **(negative)** | 22.01 | 98.44 | 183 | 0.6143 | 0.8521 | 0.6622 | 0.6146 | 0.7044 | 0.8458 | 0.8438 | 41.06 | 0.0197 |

Source runs:
- baseline: `Last Month/24_01_26- Benchmarking YOLOs/Yolo11m/runs/20260615_230315_yolo11m_baseline_s0_nogit/`
- cbam: `.../CBAM/runs/20260601_232929_yolo11m_cbam_s0_nogit/`
- cbam+p2: `.../CBAM_P2Head/runs/20260602_063759_yolo11m_cbam_p2head_s0_nogit/`
- mamba: `Last Month/02-06-26-Mamba_CBAM_P2Head/runs/20260609_205717_mamba_cbam_p2head_s0_nogit/`
- verdict: `.../02-06-26-Mamba_CBAM_P2Head/docs/2026-06-10_mamba_run_complete_verdict.md`

**Story:** aggregate COCO AP is saturated (~0.615, all tied). **P2 is the real driver** — it lifts
AP50 (0.843→0.853), AR_100 (0.691→0.703), Best_F2 (0.840→0.844), and very-tiny recall, at a small
param cost. CBAM is marginal but cheap (fewer params than baseline, similar/better F-scores, lowest
latency). **Mamba adds params + 2.8× latency for nothing → excluded from the deployed model.**

**Per-size recall (P2 models, from `summary.json` → p2_detection_scales.per_size_recall):**
Mamba+CBAM+P2 = very-tiny 0.7567 / tiny 0.870 / small 0.887 / medium 0.811. Baseline & CBAM per-size
in each run's `metrics/per_size.csv` (read when writing Ch IV).
Mamba SSM diagnostics (proves it was genuinely active): fwd-vs-reverse scan cosine-distance 0.836,
6 blocks at neck layers 13/16/19/22/25/28, d_state 4, state_norm_ok.

## 2. ATTENTION SELECTION — "why CBAM over ECA" (cite as-is; earlier Kaggle run, config differs)
From `Mofa_thesis_13.4.26.txt` + `01-02-2026- ablation study/`. **Caveat:** older protocol (T4,
different epochs/optimizer) — present as a preliminary attention-module comparison, NOT in the Table-1
AdamW column. Re-run under AdamW deferred to the paper.
- Test (30-img operational set): baseline F2 0.8441 / small-obj recall 0.8807; **+ECA** F2 0.8435 / 0.8835;
  **+CBAM** F2 **0.8491** / small-obj recall **0.8920** (+1.29%). CBAM: best recall/F1/F2, lowest latency,
  fewer params than C2PSA. ECA: faster but no consistent recall/mAP gain.
- **→ CBAM chosen** for operational superiority at lower cost.

## 3. INFERENCE-TIME ENHANCEMENT — SAHI / TTA  ⚠️ MODEL-MISMATCH OPEN ITEM
All SAHI/TTA in the deck were run on the **old Mamba+CBAM+P2** model (March, `31-03-26(Mamba-ViT-CNN)/SAHI+TTA/`),
which is *not* the recommended model. Numbers (per-image box matching @IoU 0.5; mAP not comparable to ultralytics):
- **SAHI-512** (ov0.25): P 0.867 / R 0.865 / F1 0.866 / F2 0.865 / VT-recall 0.808 / 187 ms — best balanced.
- **SAHI-256** (ov0.30): P 0.790 / R 0.872 / F1 0.829 / F2 0.854 / VT-recall **0.829** (+6.24%) / 534 ms — best VT recall.
- **TTA@1280:** mAP50 0.890 (+1.64%), mAP50-95 0.698 (+5.39%), miss-rate 14%→10%. TTA@1920 degrades.
- SAHI+TTA combined = no extra gain over SAHI.
- Copy-paste augmentation = **negative result** (medium recall collapsed −13%).
- **DECIDED (2026-07-02):** user will **re-run SAHI on CBAM+P2** (recommended model). Report's
  inference-time section gets a slot; numbers above are the old-Mamba reference only, pending the new run.

## 4. SOTA COMPARISON — Nihal et al. (ICPR 2024, C2A, same test split, no SAHI/TTA)
| Model | mAP50 | mAP50-95 | Params | Source |
|---|---|---|---|---|
| Faster R-CNN | 0.634 | 0.366 | ~41M | Nihal 2024 |
| RetinaNet | 0.693 | 0.383 | ~37M | Nihal 2024 |
| RTMDet | 0.708 | 0.442 | — | Nihal 2024 |
| Cascade R-CNN | 0.735 | 0.486 | ~69M | Nihal 2024 |
| DINO | 0.789 | 0.471 | ~47M | Nihal 2024 |
| YOLOv5 | 0.808 | 0.492 | — | Nihal 2024 |
| YOLOv9-e | 0.893 | 0.688 | 57.3M | Nihal 2024 |
| **YOLO11m baseline (ours)** | 0.843 | 0.615 | 20.1M | ours |
| **CBAM+P2 (ours)** | 0.853 | 0.615 | 19.6M | ours |
> Note: our numbers here are the **final-month COCO AP/AP50** (0.853/0.615), NOT the old deck's
> 0.856/0.642. Headline framing: we match/beat all baselines except YOLOv9-e at **~3× fewer params**;
> vs YOLOv9-e we trade a little mAP for a far smaller, deployable model — the lightweight variant sharpens this.

## 5. C2A DATASET FACTS (Nihal et al.)
10,215 images; ~360,000 labeled human instances; sizes 150×150 → 3400×3400 px; ~20–40 instances/image;
~47% of people <10 px, ~52% 10–50 px, ~1% >50 px. Categories/scenes: collapsed buildings, fire, flood,
traffic accident. C2A is **composited** (AIDER disaster backgrounds + U²-Net-segmented human poses from
LSP/MPII-MPHB, random scale+position, horizontal bbox) → semi-synthetic. Split counts (frozen, md5):
train 6129 / val 2043 / test 2043 (from `system_spec` smoke log). Single class `person` (C2A category
name is a placeholder "0" → mapped to `person`).

## 6. HARDWARE / SOFTWARE (Experimental Setup table)
RTX 4070 Ti SUPER 16 GB (Ada, cap 8.9), i7-14700K, 128 GB RAM, Win 11, single GPU (no DDP). imgsz 640,
batch 16 (OOM ladder 16/8/4), AMP, AdamW lr0=0.001, cos LR lrf=0.01, up to 300 ep, patience 50 / F2 40,
close_mosaic 10, default aug. Exact CUDA/torch/ultralytics versions → each run's `env.json` (read when writing).

## 7. REFERENCE CANDIDATES (→ build references.bib, IEEE)
- **C2A / Nihal et al.** — "UAV-Enhanced Combination to Application: … Human Detection Dataset for Disaster Scenarios," ICPR 2024 (arXiv 2408.04922).
- **YOLO11 / Ultralytics** — Jocher et al., Ultralytics YOLO11.
- **CBAM** — Woo et al., "CBAM: Convolutional Block Attention Module," ECCV 2018.
- **ECA-Net** — Wang et al., "ECA-Net: Efficient Channel Attention…," CVPR 2020.
- **Mamba** — Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces," 2023 (arXiv 2312.00752).
- **Mamba-YOLO** — Wang et al., "Mamba YOLO: SSMs-based Object Detection," AAAI 2025 (context only; COCO not C2A).
- **RT-DETR** — Zhao et al., "DETRs Beat YOLOs on Real-time Object Detection," CVPR 2024.
- **SAHI** — Akyon et al., "Slicing Aided Hyper Inference…," ICIP 2022.
- Aerial-SAR detection: "Aerial Person Detection for Search and Rescue: Survey and Benchmarking"; "Real-Time Human Detection for Aerial Captured Video Sequences via Deep Models"; "Deep Learning for Human Detection from Aerial Images in Search and Rescue Missions."
- (future work only) **SARD** dataset; HERIDAL; VisDrone; Okutama-Action; AIDER.

---

## 8. PART B — EXTENDED STUDY (added 2026-08, laps 1–4). Source of truth: `../05-07-2026-Novelty-Lap/results/MANIFEST.md` + eval jsons in `results/pc1/runs_s1/`
| Report claim | Value | Source |
|---|---|---|
| Scene leakage: test scenes in train | 98% | lap-1 script 00 evidence; repo CLAUDE.md folder map |
| Scene-disjoint split counts | 6,135 / 2,040 / 2,040 | MANIFEST S1 protocol line |
| Leakage cost (CBAM+P2) | AP50 0.853→0.837 (−1.6), AP_S −0.2, VT −1.2 | MANIFEST 2026-07-24 G1 entry |
| Scene-split test instance counts | 72,847 total; VT 24,806 / tiny 20,893 / small 26,738 / medium 410 | s1_eval_*.json per_size_n |
| α sweep (50-ep) | VT .693/.712/.729; AP_S .560/.553/.505 (α=0/.5/1) | MANIFEST 2026-07-25 S2 entry |
| D2 matched pair (300-ep) | see tab:d2pair; VT +2.08 [1.77,2.38] p<.001; F2 +0.27 [.03,.50] p=.024; AP50 −0.08 n.s.; tiny −1.15; small −0.95; medium −13.2 | s1_eval_s3_{control,assign}_full.json + bootstrap_s3_control_full_vs_s3_assign_full.json |
| Control vs re-baseline gap | 0.8449 vs 0.8372 (launcher/early-stop diff) | MANIFEST 2026-07-26 S3 note |
| FCCG null (50-ep pair) | VT .691 vs .693; AP_S .554 vs .560; +0.55M params, +12 GFLOPs (19.57→20.12M, 86.7→98.9G) | MANIFEST 2026-07-25 S1 + 2026-07-24 S0 entries |
| Seam probe (JPEG q90) | AP50 −65 points | MANIFEST 2026-07-25 P1 probe entry |
| Own-data table | drone 101/6,313 + 14/973 + 60/3,584 (44px); campus 87/4,510 (37px) + 25/663 (37px); disaster 60/1,341 (23px); R-set 25/1,070 (17px); x-event 23/318 (28px) | counted from _annotations_sc.coco.json 2026-08-06 (session log) + MANIFEST |
| Zero-shot / adapted drone | 0.335 → 0.783 (drone-only) → 0.795 (all-real); recall@.25 0.337→0.837 | s1_eval_{s3_drone_before,d3_drone_after,d3all_drone}.json |
| C2A after adaptation | 0.844→0.837 (−0.71); VT −0.41; ECE .0142→.0111 | s1_eval_d3all_c2a.json vs s1_eval_s3_assign_full.json |
| R-set within-video | 0.133→0.411 (drone-tune 0.124); recall@.25 .127→.451; small-band .230→.569; VT 0/54 all models | s1_eval_{s3_rset_before,d3_rset_after,d3all_rset}.json |
| Cross-event (unseen, 23 fr) | 0.396→0.413 (+1.7 [−3.2,5.7] p=.47 n.s.); precision@.25 −13.1 p=.018 | s1_eval_{s3,d3all}_xevent.json + bootstrap_s3_xevent_vs_d3all_xevent.json |
| Multi-domain table | C2A .837/.824/.821; drone .795/.771/.837; R-set .411/.472/.451; x-event .413/.451/.374; campus .689/.666/.718 | the five d3all eval jsons |
| Bootstrap method | paired image-level, 1,000 resamples, percentile 95% CI, two-sided p | scripts/28_bootstrap_sig.py |
| Extended GPU cost | ≈120 GPU-h ≈ 20 kWh (estimate) | summed run durations, MANIFEST |
| PENDING | 3-seed mean±std for D2 pair (chain running on PC-1) | 27-07-2026-Novelty-Lap-4/D2_SIGNIFICANCE.md |
