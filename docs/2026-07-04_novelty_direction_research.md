# Novelty direction research — Mamba final verdict + the new plan (2026-07-04)

Branch: `novelty-direction` (created today from `main`). Compiled from (1) a full local audit of
`Last Month/`, `31-03-26(Mamba-ViT-CNN)/`, `01-03-2026-Onward Model trying/`, and (2) web research with
per-paper verification fetches. The web-research agent hit the session usage limit mid-synthesis
(resets 18:30 Asia/Dhaka); its verified per-paper fetches were salvaged and are the basis of every
citation below. Anything NOT verified is flagged. Builds on — does not repeat —
`Last Month/docs/2026-06-29_novelty_research_findings.md`.

---

## 1. Where you actually stand (from the local audit — all numbers from run JSONs)

| Asset | Number | Source |
|---|---|---|
| CBAM+P2 on C2A test (s0) | COCO AP 0.6153 / AP50 0.8533 / AP_small 0.6156 / VT-recall 0.7575 | `Last Month/docs/2026-06-13_complete_ablation_table.md` |
| mAP50-95 ceiling on C2A | **~0.615 for EVERY architecture** (baseline/CBAM/CBAM+P2/Mamba all within 0.002) | same |
| SAHI slice256_ov30 | very-tiny recall 0.7668 → **0.8292** (+6.2 pt), but F1 drops 0.850 → 0.829 | `31-03-26(Mamba-ViT-CNN)/SAHI+TTA/benchmark_reports/MASTER_REPORT_SAHI_TTA_NOAUG.txt` |
| TTA@1280 | mAP50 0.890 / mAP50-95 0.698 (best aggregate) | same |
| Best deployable (epoch125.pt, joint C2A+SARD) | C2A 0.878 / SARD 0.917 mAP50 | `deployable_model/` README + finetune_summary |
| Enriched fine-tune (2026-07-02, PC-4) | C2A 0.874 (held) / SARD 0.898 (−0.019 dip); drone-FP re-test PENDING | `runs_joint/20260702_144117_finetune_enriched/metrics/finetune_summary.json` |
| Zero-shot C2A→SARD | collapses ~99% (mAP50 0.0042–0.0147 across all variants) | `Last Month/cross_dataset_SARD/ablation_master/sard_generalization.csv` |
| Drone footage | 3 unlabeled 4K videos (10 m / 30 m / 50 m); only 10 m has been inference-processed | `Drone Shoot/`, `drone_inference_out/10m/` |

**Never tried anywhere in the repo** (grep-verified): WBF, Soft-NMS/any NMS variant, NWD/Wasserstein
loss, RFLA, Wise/Shape/Inner-IoU, super-resolution, adaptive slicing. Tile merging in
`run_on_drone_footage.py` is plain `torchvision.ops.nms`.

---

## 2. Mamba — the final, cited verdict: **CLOSED. Do not reopen.**

Four independent lines of evidence now converge:

1. **Your own two genuine runs are negative.**
   - LocalWindowSSM neck (6 blocks, run `20260609_205717`): COCO AP 0.6143 vs CBAM+P2 0.6153 — an exact
     tie — at +2.4 M params and **2.8× latency** (41.1 vs 14.6 ms). A credible null.
   - AtrousSSM (Mar 2026, genuine 24.16 M run): mAP50 0.823 vs 0.874 — **−10%**, 2× latency, worse
     tiny-object recall. A genuine failure.
2. **Independent aerial benchmark: Mamba loses to plain YOLO.** On SeaDronesSee (UAV small objects):
   MambaYOLO-L 25.5 < YOLOv8-m 26.1 < YOLO11-m 26.3 AP; the authors state SSM "optimizations do not
   translate to multi-scale small object detection" (PLOS One, peer-reviewed —
   pmc.ncbi.nlm.nih.gov/articles/PMC12212484). LocalMamba shows AP_small 26.0 vs Vim 26.1 — zero gain
   on the one axis you need.
3. **The literature's own flagship results don't support a neck-injection gain.**
   - Mamba-YOLO (arXiv 2406.05835) — **still an unaccepted preprint** as of today (verified: no venue
     found; v2 Dec 2024). Its headline "+7.5% mAP" is for the *Tiny* scale vs smaller baselines, on
     COCO, with no AP_small claim and no YOLO11 comparison.
   - MambaOut (**CVPR 2025, peer-reviewed** — arXiv 2405.07992) argues SSM *may* help detection — but
     only as a *full long-sequence backbone*, which is exactly what you cannot build (below). Your
     null result must be framed via latency + neck-vs-backbone distinction, NOT "MambaOut says SSM
     fails detection" (see June-29 doc §5 for the corrected framing).
4. **Engineering blocker unchanged:** `mamba-ssm`/`causal-conv1d`/`selective_scan` CUDA kernels have
   no Windows wheels (Linux-only; github.com/state-spaces/mamba/issues/662). Your training boxes are
   all Windows.

**Answer to "will Mamba improve accuracy over CBAM+P2+SAHI+TTA?": No.** And C2A's own label ceiling
(mAP50-95 saturated at ~0.615 for every architecture — attributed to composited paste-box geometry)
means *no architecture change of any kind* can move the headline metric there. The thesis already owns
a well-executed, correctly-instrumented SSM null result — that IS the Mamba contribution. Spend the
remaining weeks where the metric can actually move: **the loss, the fusion step, and the
synthetic→real gap.**

---

## 3. What the web research verified (salvaged evidence, per-paper fetches)

### 3a. Tiny-object losses — pluggable into Ultralytics, Windows-safe, cheap
| Technique | Evidence (verified) | Status |
|---|---|---|
| **NWD** — Normalized Gaussian Wasserstein Distance (arXiv 2110.13389) | +6.7 AP over standard baseline on AI-TOD (tiny-object benchmark); expanded version **accepted in ISPRS J. P&RS (Q1)** | Verified from abstract |
| NWD in YOLOv8-family (implementations exist) | RMH-YOLO (Sensors 2025, doi 10.3390/s25227088): NWD+InnerCIoU on YOLOv8n → +9.2 mAP50 / +6.4 mAP50-95 on VisDrone. GS-YOLO (ACM CAICE 2025): **~+1.0 mAP from the NWD term alone**. LACF-YOLO: +3.5 mAP on VisDrone | Verified via search snippets; module-level attribution only clean in GS-YOLO |
| **RFLA** — Gaussian receptive-field label assignment (**ECCV 2022**, arXiv 2208.08738) | +4.0 AP over SOTA on AI-TOD; explicitly designed for both anchor-based AND point/anchor-free priors | Verified from abstract; integration into YOLO11's TAL assigner is real work (1+ week) |
| Wise-IoU (arXiv preprint, not accepted) | YOLOv7 COCO AP75 53.03→54.50 | Verified; generic, not tiny-specific — low priority |

Why NWD fits *your* dataset precisely: C2A has **47% of instances under 10 px** (verified against the
C2A paper's stats) — the exact regime where IoU-based regression/assignment degenerates and NWD is
designed to fix. Honest expectation on C2A: this will NOT break the 0.615 ceiling (that's label noise),
but it plausibly moves AP_small / very-tiny recall by 1–3 pt, and it directly targets the SARD
AP_small weakness (0.294) of the deployable model.

### 3b. Box fusion — a real implementation gap, zero training cost
- **WBF** (Solovyev et al., arXiv 1910.13302) — **peer-reviewed: Image and Vision Computing 2021**.
  Averages clustered boxes weighted by confidence instead of discarding them (NMS/NMM).
- **Verified gap: SAHI does NOT implement WBF.** Its `postprocess/combine.py` contains only
  NMS / NMM / GreedyNMM / LSNMS classes (fetched and confirmed). Nobody has published
  calibration-aware WBF as the SAHI tile-merge step.
- Your own data says the merge step is the bottleneck: SAHI slice256 buys +6.2 pt very-tiny recall but
  **loses F1 (0.850→0.829)** — i.e. the greedy merge is creating precision damage that a weighted
  fusion can plausibly recover.
- Supporting precedent: ASAHI (Remote Sensing 2023, doi 10.3390/rs15051249) replaced SAHI's NMS with
  Cluster-DIoU-NMS + adaptive slice-count and reports accuracy/latency wins on VisDrone/xView —
  evidence that merge-strategy work on sliced inference is publishable in a Q1/Q2 remote-sensing venue.
- Confluence (arXiv 2012.00257, appears unaccepted): +2.3–3.8% AP / +5.3–7.2% AR on COCO+CrowdHuman —
  cite as related work, don't build on it.
- The "WBF or CBF" you half-remembered: it's **WBF**. No established method named "CBF" exists
  (searched; nothing found).
- CAVEAT on the June-29 doc's "~+10% mAP from WBF": that number is for *multi-model ensembles* in the
  original paper — for single-model tile fusion expect a much smaller, but free, gain. Report your own.

### 3c. Competitor landscape on C2A (novelty check)
- C2A paper (Nihal et al., **ICPR 2024**, arXiv 2408.04922) benchmark: best is YOLOv9-e at
  mAP 0.6883 / mAP50 0.8927 (58 M params). Your CBAM+P2 at ~20 M params: 0.615/0.853 single-model,
  0.698/0.890 with TTA — competitive at 1/3 the size; frame it as accuracy-per-param/latency.
- **LightSeek-YOLO** (Mathematics/MDPI 2025, doi 10.3390/math13193231): lightweight YOLOv11 variant,
  reports **AP_small 0.478 on C2A** — your 0.6156 beats it by a wide margin (protocol caveats apply;
  verify their eval setup before claiming in print).
- The 13 citing papers found for C2A are overwhelmingly *application/comparison* papers (SAR system
  demos, YOLO comparisons, surveys). **Nobody has published: (i) tiny-object loss work on C2A,
  (ii) sliced-inference fusion on C2A, (iii) cross-domain C2A→SARD→real-footage deployment analysis.**
  The niche is active enough to cite (14+ papers in 2 years) but the method space is wide open.
- The C2A paper's own cross-domain Table (their protocol): C2A-only→SARD mAP 0.259, C2A+GeneralHuman→
  SARD 0.660. Your measured zero-shot collapse (0.004–0.015 mAP50) is far more severe — reconciling
  this discrepancy (their eval conf/protocol vs yours) is itself a publishable finding. [Their table
  values still flagged UNVERIFIED — confirm against arxiv.org/html/2408.04922v2 before quoting.]

---

## 4. THE PLAN — ranked directions (expected gain × feasibility × novelty)

### #1 — Calibration-weighted box fusion for sliced inference ("C-WBF") — START HERE
**What:** Replace the greedy-NMM/NMS tile-merge in your SAHI pipeline with WBF, weighted by
*calibrated* confidences — you already compute ECE/temperature data in every eval (`opt_thresholds`,
`calibration` blocks in your summaries), so calibrate scores (per-slice-scale temperature) before the
weighted average. Ensemble whole-frame + sliced + TTA predictions in the same fusion.
**Why it can be called novel:** SAHI verifiably lacks WBF; calibration-aware fusion for tile merging
has no published instance we could find; your F1-drop-under-slicing is the documented failure it fixes.
**Cost:** inference-only — no training, runs on anything, `ensemble-boxes` pkg + ~200 lines.
**Risk:** gain may be small (1–3 pt F1/AP on sliced configs). Mitigation: it's 2–3 days of work; even
a null feeds the ablation.

### #2 — NWD-hybrid loss for the tiny-object regime
**What:** Add an NWD term to Ultralytics' bbox loss (blend with CIoU, e.g. `L = (1-α)·CIoU + α·NWD`),
optionally NWD-based assignment metric in TAL. Fine-tune from epoch125 first (cheap signal), then one
full C2A retrain if the pilot is positive; final = 3 seeds per your protocol.
**Why:** C2A is 47% sub-10px — NWD's exact design target; Q1-published basis + multiple replications
in YOLOv8-family; nobody has done it on C2A.
**Cost:** loss-level patch (~1 day) + your existing training pipeline; PC-1/PC-4 capable.
**Risk:** gains on C2A may be muted by the label-noise ceiling — measure on AP_small/VT-recall and on
SARD/real footage, not mAP50-95. RFLA is the stretch-goal upgrade if NWD shows signal.

### #3 — The synthetic→real deployment study (the paper's spine — already half-built)
Unchanged from June-29 §8, now with assets #1/#2 feeding it: label 15–20 frames per altitude
(10/30/50 m) → real test set; evaluate {epoch125, +enriched, +NWD, ±C-WBF} on it; run the SAHI-guided
self-training loop; finish the pending drone-FP re-test of the enriched model. The 3-altitude
stratification of your own footage is a micro-benchmark nobody else has.

### #4 — (Stretch) Adaptive slicing
ASAHI-style resolution-adaptive slice count on top of #1, cite ASAHI, or a density-guided variant
(slice only where a cheap low-res pass finds candidates). Only if #1–#3 land early.

### The paper, assembled
*"Deployment-grade tiny-human detection for UAV search-and-rescue: tiny-object-aware training (NWD) +
calibration-weighted sliced-inference fusion (C-WBF), validated across synthetic→real domains (C2A →
SARD → self-collected 3-altitude drone footage), with SSM/Mamba necks reported as an instrumented null
result."* Every block has verified published precedent; the combination + the cross-domain protocol +
the honest negatives = coherent Q2 submission, Q1-adjacent (Remote Sensing / ISPRS-tier) if the real-
footage deltas are strong.

### 4-week schedule
| Week | Work | Machine |
|---|---|---|
| 1 | C-WBF implementation + C2A/SARD ablation vs GreedyNMM/NMS; label drone test frames (parallel, manual) | any (inference) |
| 2 | NWD loss patch → fine-tune pilot from epoch125 → full retrain if positive | PC-1 or PC-4 |
| 3 | Self-training Phases 2–4 (June-29 §8); pending drone-FP re-test of enriched model | PC-4 |
| 4 | 3-seed confirmations, cross-domain master table, paper draft | — |

---

## 5. What NOT to do (all previously adjudicated, now confirmed)
Mamba/SSM in any form (§2) · ViT/DETR/RT-DETR swaps (too heavy; never built here; June-29 §1 shows
CBAM+P2 recipe is prior art anyway) · federated/RL/continual/TENT (June-29 §4) · UDA/CycleGAN
(June-29 §9.2) · more enrichment blending without first finishing the pending drone-FP validation.

## 6. Citation shortlist for the new directions
- WBF: Solovyev, Wang, Gabruseva — *Image and Vision Computing* 2021 (arXiv 1910.13302). Peer-reviewed.
- NWD: Wang et al. (arXiv 2110.13389); extended version in ISPRS J. P&RS. Peer-reviewed (extension).
- RFLA: Xu et al., ECCV 2022 (arXiv 2208.08738). Peer-reviewed.
- ASAHI: *Remote Sensing* 15(5):1249, 2023 (doi 10.3390/rs15051249). Peer-reviewed.
- SAHI: Akyon et al. (arXiv 2202.06934, ICIP 2022). Peer-reviewed.
- MambaOut: Yu & Wang, CVPR 2025 (arXiv 2405.07992). Peer-reviewed.
- Mamba-YOLO: arXiv 2406.05835. **Preprint only — cite as such.**
- Confluence: arXiv 2012.00257. **Preprint only.**
- CEASC: CVPR 2023 (drone-image sparse conv — related work for adaptive inference).
- LightSeek-YOLO: *Mathematics* 13(19):3231, 2025 — the C2A competitor to beat in print.
- C2A: Nihal et al., ICPR 2024 (arXiv 2408.04922; Springer LNCS 10.1007/978-3-031-78341-8_10).
- RMH-YOLO: *Sensors* 25(22):7088 (NWD+InnerCIoU on YOLOv8n, VisDrone).
