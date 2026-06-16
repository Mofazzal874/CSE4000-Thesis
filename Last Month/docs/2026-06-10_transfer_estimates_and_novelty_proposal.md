# Realistic transfer estimates + novelty proposal (image-grounded, web-verified 2026-06-10)

> Inputs: visual inspection of C2A val/train batches + label-distribution plots from the
> completed Mamba run; 22-agent verified research pass (8 claims checked against sources,
> 42 more in pool). Markers: [VERIFIED] = checked against the source page.

---

## 1. What the C2A images actually show (my inspection)

From `val_batch0_labels/pred.jpg`, `labels.jpg`, and the CBAM attention overlays:

1. **Uniform-random human placement.** The x/y location heatmap over 215,820 train
   instances is uniform noise — people pasted on rooftops, in pools, mid-rubble at even
   density. Real humans cluster. → A C2A-trained model learns **appearance only, zero
   context prior**.
2. **No shadows, no occlusion, no lighting agreement.** Figures composited ON TOP of the
   scene; nobody half-hidden. Real SAR's hardest cases (shadow, occlusion, blur) are
   absent from training.
3. **Sports poses** (LSP/MPII athletics) scattered in disaster scenes.
4. **Tiny + dense:** w,h mass < 0.05 of image; ~35 persons/image; C2A images are small
   (median width 428 px — so "640 training" upscales many of them).
5. The model detects this regime well (val preds conf 0.3–0.8, dense and mostly correct)
   — it is an excellent appearance-based tiny-blob detector.

## 2. Realistic zero-shot estimates — calibrated by published numbers

**The C2A authors already measured C2A→SARD themselves** (Table 5 of arXiv:2408.04922)
[VERIFIED]:

| Train → Test | mAP (their Table 5) |
|---|---|
| C2A → C2A val | 0.784 |
| **C2A → SARD (zero-shot)** | **0.259** (−67% rel.) |
| SARD → SARD | 0.931 |
| General-Human + C2A → SARD | **0.660** (2.5× the zero-shot) |
| SARD → C2A (reverse) | 0.168 |
| SARD → General Human | 0.036 |

(Metric not stated as @0.5 vs @0.5:0.95 — magnitudes suggest mAP50-style.)

**Expectation for YOUR models:**
- **C2A→SARD zero-shot: mAP50 ≈ 0.25–0.40.** Your detector is stronger than their Table-5
  model (you: 0.853–0.868 mAP50 on C2A vs their 0.784 val), and SARD persons are larger
  (easier), but the dominant factor is the domain gap, not model quality. Plan around
  ~0.3. **A low number is NOT failure — it's the paper's measurement.** Corroborating
  context: rendered-synthetic zero-shot is even worse (SynPlay→VisDrone 9.12 AP50 vs
  49.52 real-trained, ~−80% [VERIFIED]); FlyPose's COCO-pretrained zero-shot on aerial
  person sets = 14.33 weighted mAP.
- **C2A→VisDrone-person zero-shot: AP50 ≈ 0.05–0.20.** Even VisDrone-TRAINED models get
  only 25–36 AP50 on person classes [VERIFIED earlier]; the gap direction (composited
  disaster → dense occluded urban oblique) is maximal. Expect single digits to ~0.15.
  Report it as a domain-gap measurement, don't chase it.
- **Asymmetry worth reporting** [VERIFIED]: SARD-trained transfers WORSE (0.168 on C2A,
  0.036 on General-Human) than C2A-trained (0.259 on SARD) — semi-synthetic data
  generalizes asymmetrically better than small real datasets. That's a thesis-friendly
  framing: C2A is a generalization aid, not an in-domain crutch.

## 3. The novelty proposal — what's open, feasible, and evidence-backed

**The bar:** zero-shot C2A→SARD alone is already published (one number, one unnamed
detector, no analysis). To be novel you must go beyond it. Three layered contributions,
all feasible in ~1 month on the 4070 Ti:

### Contribution A — Architecture × Generalization (zero extra training)
Nobody has measured whether the *architecture* changes synthetic→real transfer. You have
4 trained models (baseline/CBAM/CBAM+P2/Mamba). Run `sard_eval.py` on all → per-model
transfer-drop table + per-size recall on SARD. Open questions you answer first:
- Does the P2 head's tiny-object gain survive the domain shift?
- Does the SSM neck (global context) help OR hurt generalization? (Either answer is a
  finding — and it extends your null-result story into the transfer regime.)

### Contribution B — The few-shot reality curve (the headline)
Fine-tune the best model on N ∈ {0, 20, 50, 100, 200} SARD images; evaluate on SARD test.
~5 short fine-tunes ≈ 1–2 GPU-days total. Why this is strong:
- Published anchors exist but NOT from C2A: VisDrone→SARD few-shot tops out at
  **6.98 AP (200 real, no synthetic) → 17.73 AP with rendered-synthetic pretraining**
  (Lee et al., arXiv:2405.15203, Table 6a) [VERIFIED]. If C2A-pretrained + 200 SARD
  images beats 17.73 AP@[.5:.95], you've shown **composited-real synthetic data beats
  rendered synthetic data as a SAR pretraining corpus** — a clean, falsifiable, novel claim.
- Rendered synthetic can actively HURT (Archangel+real: −0.95 AP vs real-only on
  VisDrone — SynPlay paper). Testing whether C2A avoids negative transfer is itself novel.
- Practical SAR framing reviewers like: "how many real images does a rescue team need?"

### Contribution C — SAHI inference ablation (cheap, expected by reviewers)
Inference-only slicing: +3.2–5.3 AP50 on VisDrone-class data, and gains GROW as objects
shrink (xView: +15–18 AP50) (SAHI, ICIP 2022). Run on C2A test + SARD test, both with/
without. Zero training. Config from the paper: 640px slices, 25% overlap.

### Optional Contribution D — data-centric C2A-R (only if Weeks 1–2 go fast)
Your images expose exactly what C2A lacks: shadows, blending, plausible placement,
occlusion. Evidence that fixing these matters: randomized lighting/pose dominate
synthetic-to-real gaps (Vanherle, BMVC 2022: best-case gap 12.7 AP, naive compositing
46–53 AP worse) [VERIFIED]; occlusion-insertion + artifact filtering improved transfer
(SynPoseDiv, ICIP 2025); diffusion style-alignment with only 20 real reference images
gives avg +7.3 mAP50 (CFHA 2025, unverified). A minimal version: re-composite C2A with
(a) Poisson/alpha blending + (b) synthetic shadows + (c) ground-plausible placement →
retrain CBAM+P2 once (~6 h) → re-measure SARD transfer. If transfer improves, that's a
data-centric result worth its own section ("what makes composited SAR data transfer").
If time runs out: future work.

### What I would NOT do
- Chase VisDrone numbers (max domain gap, will look bad, adds little).
- NWD loss retrains (FlyPose's +13.6 mAP is confounded with multi-dataset training;
  a retrain gamble in your last month).
- Any new architecture module. The field's own ablations show bolted-on modules often
  add ~0 (verified SRepD +0.0) and attention choice is interchangeable.

## 4. One-month schedule

| Week | Work | Output |
|---|---|---|
| 1 | Finish baseline AdamW retrain; download SARD; run `sard_eval.py` (4 models); SAHI on/off on C2A+SARD | Contribution A + C tables |
| 2 | Few-shot SARD fine-tune curve (0/20/50/100/200) on CBAM+P2 (+ Mamba if time) | Contribution B (headline figure) |
| 3 | (Optional) C2A-R recompositing arm OR start writing | Contribution D or draft |
| 4 | Write. Tables/figures already exist from the pipeline | Thesis + Q2-journal draft |

## 5. The paper this becomes

**Title shape:** "From composited disasters to real rescues: how architecture and data
realism shape synthetic-to-real transfer for tiny-human SAR detection."
**Story:** strong C2A ablation (2nd of 9 published, P2 dominates, SSM null at 2.8×
latency) → architecture does NOT rescue the domain gap (zero-shot ~0.3) → but C2A as a
pretraining corpus + tens of real images recovers most of it, beating rendered-synthetic
baselines → SAHI adds a further inference-time boost → data-realism analysis explains why.
**Venue:** Q2 (Scientific Reports / MDPI Drones / Remote Sensing / PeerJ CS), consistent
with the 2026-06-10 venue verdict.

### Key sources
- arXiv:2408.04922 Table 5 (C2A→SARD 0.259; +General-Human 0.660) [VERIFIED]
- arXiv:2405.15203 Lee et al. 2024 (few-shot→SARD curves; synthetic helps cross-domain ×8) [VERIFIED]
- arXiv:2408.11814 SynPlay (zero-shot −80%; rendered synthetic can hurt) [VERIFIED]
- arXiv:2211.16066 Vanherle BMVC 2022 (what makes synthetic transfer; 12.7 AP best-case gap) [VERIFIED]
- arXiv:2202.06934 SAHI ICIP 2022 (+3–18 AP50, inference-only)
- arXiv:2512.13869 CFHA 2025 (diffusion alignment, +7.3 mAP50 avg) [unverified]
- arXiv:2405.15939 SynPoseDiv ICIP 2025 (pose diversity, occlusion insertion)
- arXiv:2601.05747 FlyPose WACV 2026 (1280px lever; multi-dataset+NWD)
