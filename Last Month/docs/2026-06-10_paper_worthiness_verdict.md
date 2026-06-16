# Is this thesis paper-worthy? — Evidence-based verdict (web-researched 2026-06-10)

> Method: two deep-research passes (10 search angles, 38 sources fetched) + a 7-claim
> targeted verification pass. Facts below are marked **[VERIFIED]** (checked directly
> against the source page) or **[UNVERIFIED]** (extracted from a fetched page but the
> adversarial check couldn't run / page paywalled). Nothing here is from model memory alone.

---

## TL;DR — the honest verdict

1. **Your numbers are GOOD on C2A.** You rank **2nd of 9** against the C2A paper's own
   benchmark table, beat the only other published C2A improvement work by a wide margin,
   and beat your size-class peer (YOLOv9-c) by **+6 points mAP50-95**. Only YOLOv9-e
   (~3× your params) is above you.
2. **CBAM+P2 as a *recipe* is commodity** (published many times, 2024-2026; the attention
   choice is interchangeable within noise). **But nobody has published your specific study**
   — attention+P2+SSM component attribution for aerial SAR human detection on C2A.
3. **Your Mamba null-result is scientifically consistent with the literature** (MambaOut,
   arXiv 2405.07992: SSM buys only ~1.4 AP in detection at scale; for your task it buys ~0
   at 2.8× latency). Stated honestly with the efficiency analysis, it is a *contribution*,
   not an embarrassment.
4. **Realistic venue: Q2 journal** (Scientific Reports, MDPI Drones/Remote Sensing/Sensors,
   PeerJ CS, IEEE Access) — these publish exactly this shape of paper. Q1 (ISPRS, TGRS,
   Pattern Recognition) is out of reach without a genuinely new mechanism.
5. **What gets you from "another improved-YOLO" to "publishable":** the **C2A→SARD
   zero-shot transfer study** (nobody has published it) + **SAHI inference ablation** +
   the honest component-attribution story. All three are already in your plan.

---

## Q1. The Nihal et al. head-to-head **[VERIFIED — fetched from arXiv:2408.04922]**

C2A paper: *"UAV-Enhanced Combination to Application: Comprehensive Analysis and
Benchmarking of a Human Detection Dataset for Disaster Scenarios"* — Nihal, Yen, Itoyama,
Nakadai. arXiv:2408.04922; ICPR 2024 (Springer LNCS 15314). Their protocol: 640px,
**50 epochs**, Adam, batch 24, A100.

| Model (their Table) | mAP50-95 | mAP50 |
|---|---|---|
| Faster R-CNN | 0.3656 | 0.6340 |
| RetinaNet | 0.3834 | 0.6933 |
| RTMDet | 0.4420 | 0.7080 |
| DINO (transformer) | 0.4710 | 0.7890 |
| Cascade R-CNN | 0.4860 | 0.7350 |
| YOLOv5 | 0.4920 | 0.8080 |
| **YOLOv9-c (~25M — your size class)** | **0.5562** | **0.7996** |
| **YOURS: CBAM+P2 (19.6M)** | **0.6153** | **0.8533** |
| **YOURS: Mamba+CBAM+P2 (22M)** | **0.6313*** | **0.8678*** |
| YOLOv9-e (~58M) | 0.6883 | 0.8927 |

\* ultralytics protocol; COCO protocol ≈ 0.614/0.852. Caveats to state in the thesis:
their runs were 50-epoch (yours trained to convergence — part of your edge is training
budget), and the exact split protocol may differ. Frame as: **"second only to YOLOv9-e,
which has ~3× the parameters; +6 mAP50-95 over the size-class peer YOLOv9-c."**

## Q2. Who else publishes on C2A — and do you beat them?

- **LightSeek-YOLO** (MDPI *Mathematics*, Oct 2025) **[VERIFIED]**: YOLOv11-based
  lightweight detector benchmarked on C2A. **mAP50-95 = 0.473, AP_small = 0.478, 1.86M
  params.** Your CBAM+P2: **0.615 / 0.615** — far above (different size class, but it
  proves C2A has external adopters and your numbers lead them).
- **IEEE DICCT 2025** (10986412, transfer-learning YOLOv11 on C2A) **[VERIFIED]**:
  reports **precision 92%** only (no mAP). Application-level conference paper. Your
  precision: 0.877-0.879 at a full metric suite — comparable, and your study is far more
  rigorous.
- Conclusion: **C2A is an active benchmark with only weak follow-up work so far — you'd
  currently hold the strongest published-quality numbers on it below 58M params.**

## Q3. Calibration — why your ~0.86 mAP50 is high, and what real data gives

Real aerial person benchmarks are far harder than semi-synthetic C2A (47% of C2A humans
are <10px, but compositing makes them learnable):

- TinyPerson: YOLOv11n baseline **21.2 mAP50**, improved model 27.1 (Sci Reports 2026)
  **[VERIFIED]**; YOLOv8n 27.5→29.3 (Frontiers in Physics 2025) **[UNVERIFIED]**.
- VisDrone person classes: pedestrian 33.5-36.2, people 24.8-27.4 mAP50 (LAF-YOLOv10,
  arXiv 2602.13378) **[VERIFIED]**.
- SARD: YOLOv5L reportedly mAP50 0.969 / mAP50-95 0.643 **[UNVERIFIED — paywalled;
  likely inflated by video-frame train/test overlap]**. HERIDAL: ~0.75-0.86 mAP50.

**Implication:** reviewers know C2A-level numbers are dataset-friendly. The credible,
differentiating result is the **C2A→SARD zero-shot drop** — that's the experiment that
makes the thesis say something nobody else has measured.

## Q4. Novelty audit — the part you were afraid of

**Saturated (do NOT claim as novel):**
- Attention + P2 head in YOLO: published repeatedly — e.g. **YOLOv8-MPEB** (Heliyon,
  Apr 2024: EMA-in-C2f + P2 head + BiFPN) **[VERIFIED]**; LAF-YOLOv10 (2026); CF-YOLO
  (Sci Reports 2025); PC-YOLO11s (MDPI 2025) **[UNVERIFIED list items]**.
- Worse: the attention *choice* is interchangeable within noise — SE 35.1 / CBAM 34.9 /
  CA 35.0 mAP50 with everything else fixed **[VERIFIED]**. So "we used CBAM" carries no
  novelty weight at all.
- Mamba-in-YOLO: published — Mamba-YOLO (AAAI 2025), MambaNeXt-YOLO (2025) **[VERIFIED
  existence]**. Attention+Mamba+BiFPN in one detector also exists (Sci Reports Oct 2025 —
  but for vehicles/IR fusion, not SAR humans, no P2, no CBAM) **[UNVERIFIED]**.

**Genuinely yours (claimable):**
1. **No published work combines attention + P2 + SSM neck in one detector for aerial SAR
   human detection, and none on C2A** (searched; nearest neighbors listed above).
2. **The component-attribution finding itself**: a controlled additive ablation showing the
   P2 head is the dominant driver and the SSM neck adds nothing at 2.8× latency. Externally
   corroborated: in the Sci Reports 2026 ablation, P2/neck = +3.6 mAP50 (largest) while
   another bolted-on module = **+0.0** **[VERIFIED]** — null components are real and journals
   print them.
3. **C2A→SARD zero-shot generalization measurement** (not yet run — this is the gap in the
   literature you can own).
4. The first **efficiency-accounted SSM study on a SAR benchmark** (MambaOut showed SSM's
   detection benefit is ~1.4 AP at COCO scale **[VERIFIED]**; you show it's ~0 on C2A
   tiny-humans — a domain-specific boundary result).

## Q5. Is the Mamba null-result publishable?

Yes, with the right framing. Precedent:
- **MambaOut** (arXiv 2405.07992) **[VERIFIED]** — an entire highly-cited paper about SSM
  being unnecessary for classification and only marginally useful for detection.
- Published ablations include zero-gain components (SRepD +0.0 in Sci Reports 2026)
  **[VERIFIED]**.
- Dedicated negative-results venues exist (NeurIPS "I Can't Believe It's Not Better!"
  workshop) **[UNVERIFIED but well-known]** — though you won't need them; frame it inside
  an applied paper as *component attribution*, not as a confession.

**Framing for the paper:** "We perform a controlled component-attribution study for tiny-
human SAR detection. The stride-4 P2 head accounts for the bulk of very-tiny recall gains;
attention placement is secondary; a bidirectional local-window SSM neck — despite 2.4M
extra parameters and 2.8× latency — does not improve accuracy, consistent with MambaOut's
hypothesis that SSMs pay off only in long-sequence regimes."

## Q6. What to add — prioritized by evidence

1. **C2A→SARD zero-shot transfer** (script already built: `cross_dataset_SARD/sard_eval.py`).
   Highest novelty-per-hour. No retraining.
2. **SAHI slicing ablation** at inference on both datasets — standard for aerial papers;
   reviewers expect it; cheap (inference only).
3. *(Optional, time permitting)* NWD loss or P5-head-drop variant — single highest-yield
   architectural change per LAF-YOLOv10's ablation (+1.3 from P2-with-P5-dropped)
   **[VERIFIED]** — but only if the month allows; it's a retrain.
4. Keep the baseline AdamW retrain (in progress) — the clean 4-row table is the spine.

## Venue verdict

| Tier | Examples | Verdict for this work |
|---|---|---|
| Q1 (ISPRS, TGRS, Pattern Recognition) | — | Not without a new mechanism. Don't aim here. |
| **Q2 journals** | **Scientific Reports, MDPI Drones / Remote Sensing / Sensors, PeerJ CS, IEEE Access** | **Realistic target.** These publish exactly this paper shape (improved-YOLO + solid ablation + aerial application), e.g. the verified Sci Reports / Heliyon / MDPI papers above. |
| Conferences | ICPR/ICIP main or workshops; IEEE regional | Fallback; note C2A's own paper is ICPR. |

**What separates acceptance from rejection at Q2 (based on the published examples):** a
complete ablation (you have 5 models), honest efficiency numbers (you have them), more than
one dataset (→ run SARD), and inference-time analysis (→ run SAHI). You are 2 experiments
away from the full package.

---

## Bottom line for Mofazzal

You did not waste your year. You hold the strongest sub-58M-param numbers on an active
benchmark, an independently-corroborated finding about *why* (P2, not attention, not SSM),
and a null result that the field's own literature (MambaOut) predicts. The two experiments
that turn this into a submittable Q2 paper — SARD zero-shot and SAHI — are already scripted
or trivially cheap. Finish the baseline retrain, run those two, and write.

### Sources (key)
- arXiv:2408.04922 (C2A paper, numbers fetched directly)
- MDPI Mathematics 13(19):3231 — LightSeek-YOLO (C2A adopter)
- IEEE DICCT 2025, 10.1109/DICCT64131.2025.10986412 (C2A adopter)
- Heliyon 10(8):e29501 — YOLOv8-MPEB (attention+P2+BiFPN prior art, 2024)
- arXiv:2602.13378 — LAF-YOLOv10 (attention interchangeability; P2 ablation)
- Nature Sci Reports s41598-026-35301-2 (TinyPerson/VisDrone calibration; P2 +3.6; null component +0.0)
- arXiv:2405.07992 — MambaOut (SSM null/marginal precedent)
- arXiv:2406.05835 — Mamba-YOLO (AAAI 2025)
- ScienceDirect S1674862X24000119 (SARD/HERIDAL calibration — paywalled, unverified)
