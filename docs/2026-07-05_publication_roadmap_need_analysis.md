# Publication roadmap — need analysis, compatibility, and step-by-step plan (2026-07-05)

Branch: `novelty-direction`. This document is grounded in a fresh measurement pass over the actual
datasets on this machine (C2A `new_dataset3` all splits via their COCO JSONs; SARD test via
`sard_test_coco_gt.json`; visual inspection of C2A images). Every number below was computed today —
raw output in the session scratchpad (`dataset_analysis.json`). Companion doc:
`2026-07-04_novelty_direction_research.md` (literature verdicts + citations).

---

## PART 1 — What the data actually says (measured need analysis)

### 1.1 ★ HEADLINE FINDING: C2A's official split leaks backgrounds between train and test

Measured today, verified visually:

- File-level overlap between splits: **0** (clean at first glance).
- But filenames encode scene + variant: `fire_image0353_0 … _4` = up to 5 composites built on the
  **same background photo**, and these variants are scattered across splits.
- **Scene-level overlap: 1375 of 1399 test scenes (98.3%) also appear in train.**
  **1981 of 2043 test images (97.0%) share their exact background with a training image.**
  **0 test images come from a scene absent from train+val** — unseen-background generalization is
  unmeasurable under the official split, period.
- Visual proof: `test/images/collapsed_building_image0031_0.png` and
  `train/images/collapsed_building_image0031_2.png` are the **same rubble photo, pixel-for-pixel
  background** — only the pasted human sprites differ.

Consequences:
1. Every published C2A benchmark number — the C2A paper's YOLOv9-e (mAP50 0.8927), LightSeek-YOLO,
   and our own 0.874–0.890 — is partly **background memorization**, not generalization.
2. This mechanistically explains two of our standing mysteries: the mAP50-95 saturation at ~0.615
   across all architectures, and the catastrophic 99% zero-shot collapse on SARD (C2A-test never
   asked the model to generalize; SARD did).
3. Fixing it is cheap and rigorous: **a scene-disjoint re-split** (group by scene prefix, split
   scenes 60/20/20 — the crop-suffix structure makes this a bookkeeping exercise) plus a re-benchmark
   quantifying the inflation. This is a *dataset-audit contribution* of the kind journals actively
   publish, and it makes our cross-domain story coherent end-to-end.

Verification still owed before claiming in print (be sure): (a) re-read the C2A paper's split
methodology to confirm they split at image level, not scene level [their arXiv HTML, Table/§dataset];
(b) spot-check ~5 more sibling pairs across categories; (c) confirm suffix semantics with the dataset
README if present.

### 1.2 C2A is even more of a tiny-object dataset than the paper claims — and it has a blind spot

Measured over 360,467 instances (train+val+test consistent to ±0.4%):

| Property | Measured value | Implication |
|---|---|---|
| Median box size | **12.0 px** (sqrt-area) | dead-center in the "IoU breaks down" regime |
| Under 10 px | **42.8%** (paper says 47% — close, version difference; cite our own number) | NWD/tiny-aware loss is not optional polish, it's the core regime |
| Under 16 px | **62.8%** | P2 head (stride 4) justified empirically |
| Over 64 px | **~0.1%** | large-object branches are dead weight on C2A |
| Objects/image | mean **35**, max **240** | dense-crowd regime; fusion/NMS behavior matters a lot |
| Resolutions | heterogeneous: 240×240 and 300×300 are the most common; up to 1280×720+ | see adaptive slicing, §1.5 |
| **Effective** size at 640 letterbox | 23.9% < 8 px; **6.5% < 4 px** | the sub-4px tier is invisible even to P2 — a recall ceiling we should *state*, not fight |
| Images with zero annotations | **0 of 10,215** | ★ the model never sees a person-free disaster scene → explains drone false positives; negatives must come from elsewhere (enrichment/SARD/real footage) |
| Category balance | fire/flood/collapsed_building/traffic ≈ 25% each | per-scenario breakdown table is free and reviewers like it |

Visual inspection (fire_image0001_1, collapsed_building_image0031_*): pasted sprites float in smoke
and sky at impossible positions/scales with hard cut edges — confirming the paste-artifact critique
and the multi-blend enrichment rationale (Angle B).

### 1.3 SARD is not a tiny-object dataset — it's the *scale-regime* transfer target

Measured on the local SARD test GT (570 images, all 640×640, 732 instances):

| Property | C2A | SARD-test | Implication |
|---|---|---|---|
| Median box | 12 px | **53.4 px** | C2A→SARD is a *scale-regime shift*, not just appearance shift |
| < 16 px | 62.8% | **1.1%** (8 boxes total) | our "SARD tiny/very-tiny recall = 0.0" rows are **statistically meaningless** (recall over ≤8 boxes) — stop reporting them as failure; annotate as n<10 |
| Objects/image | 35 | **1.28** (max 7) | sparse-scene regime; different FP dynamics |
| Empty images | 0% | **14.7%** (84/570) | SARD supplies the real negatives C2A lacks |
| Duplicate base-names | — | present (`gss1006` ×2+ with different Roboflow hashes) | ★ audit for near-duplicate/augmented copies within and across SARD splits before publishing SARD numbers |

The joint model's SARD AP_small = 0.294 should be read against the 1.1% figure: tiny objects are a
C2A problem; SARD tests medium-object, sparse, real-appearance transfer. The paper should say this
explicitly — it reframes our "weakness" as a measured dataset property.

### 1.4 Real drone footage (our unique asset)
3 videos at 10/30/50 m, 4K, unlabeled. Sliced-vs-whole detection counts disagree per frame (e.g.
frame 1680: 21 vs 44) — an unresolved fusion/consistency problem that C-WBF and the labeled test set
will settle. ~15–20 labeled frames per altitude are still owed (June-29 plan §8, Phase 0).

### 1.5 Slicing need is resolution-conditional (measured)
Most C2A images are ≤640 px on the long side — slicing does nothing for them; SAHI's very-tiny-recall
win (+6.2 pt) is earned on the large-resolution minority and the 4K drone frames. A simple,
defensible **resolution-adaptive slicing rule** ("slice only above threshold R, choose slice size by
R") is directly supported by this measurement and costs almost nothing — cite ASAHI as precedent.

---

## PART 2 — Compatibility check (each component against our exact stack)

Stack facts: Ultralytics YOLO11m + CBAM + P2 (custom injection scripts), AdamW lr0=0.001 protocol,
PATIENCE=50/F2_PATIENCE=40, SEEDS=[0..4] for final claims, Windows on all boxes (PC-1 4070 Ti S 16GB,
PC-2 2×A6000, PC-3 lab, PC-4 4070 12GB fp32-only), COCO-eval + per-size-recall harness, calibration
(ECE/thresholds) already computed in every run summary.

| Component | Touches | Compatibility verdict | Watch-outs |
|---|---|---|---|
| **Scene-disjoint re-split** | dataset YAMLs only; no model/code change | ✅ trivial — group by filename prefix, re-emit split lists | keep official-split numbers too (comparability with prior art); document the mapping |
| **NWD-hybrid loss** | Ultralytics bbox-loss term (+ optionally TAL metric) | ✅ proven on anchor-free YOLOv8-family (RMH-YOLO, GS-YOLO); pure PyTorch, Windows-safe; trains on PC-1/PC-4 | α (CIoU↔NWD blend) needs a 2–3 point sweep; keep optimizer protocol fixed; expect gains on AP_small/VT-recall, NOT on the saturated mAP50-95 |
| **C-WBF fusion** | inference merge step only (currently plain torchvision NMS / SAHI GreedyNMM) | ✅ `ensemble-boxes` is pure numpy; zero training; works everywhere incl. drone script | dense scenes (up to 240 obj/img) stress cluster thresholds — tune IoU-cluster on val only; use existing per-model calibration (temperature from ECE data) before weighting |
| **Self-training on real footage** | Ultralytics predict(save_txt, save_conf) → merge → fine-tune | ✅ native; PC-4 capable (fp32, batch 4) | pseudo-labels NEVER in val/test; conf threshold sweep 0.5–0.7; disjoint from the hand-labeled test frames |
| **Adaptive slicing rule** | inference wrapper only | ✅ trivial | frame as engineering + ablation row, not a headline claim |
| RFLA-style assignment | TAL assigner internals | ⚠️ real integration work (~1 wk), do only if NWD shows signal | stretch goal |
| Pose-aware anything | `All labels with Pose information/` exists locally | ⚠️ scope creep — park it | mention as future work only |

No component requires Linux, new CUDA kernels, or architecture surgery. Nothing conflicts with the
frozen ablation protocol. Total new training compute: 2–3 full C2A runs + 2–4 fine-tunes — within
PC-1+PC-4 capacity in the window.

---

## PART 3 — Step-by-step process (no code here — sequence, gates, owners)

**Phase 0 — Lock the evidence (1–2 days, this laptop)**
1. Re-verify the split-leakage finding: C2A paper's split methodology; 5 sibling-pair visual checks;
   suffix semantics. → freeze a one-page "leakage memo" with the three numbers (98.3% / 97.0% / 0).
2. SARD hygiene audit: base-name duplicate scan within/across SARD splits; decide keep/dedupe.
3. Emit the scene-disjoint split lists (60/20/20 by scene, stratified by disaster category; same
   instance-size distribution check afterwards). Freeze as `c2a_scenesplit_v1`.

**Phase 1 — Quantify the leakage (the cheapest headline result; ~3–4 days GPU, PC-1)**
4. Retrain baseline YOLO11m and CBAM+P2 (s0 only, standard protocol) on `c2a_scenesplit_v1` train.
5. Evaluate on scene-disjoint test; compare against official-split results.
   **Gate G1:** if scene-disjoint numbers drop materially (expected), the paper's motivation section
   is done — proceed. If they don't drop, that's *also* publishable (split is safe) — pivot emphasis
   to Phase 2–4 and say so honestly.

**Phase 2 — Tiny-object training upgrade (parallel with Phase 1 labeling; ~4–6 days GPU)**
6. NWD-hybrid loss pilot: fine-tune from epoch125 (cheap), α sweep {0.3, 0.5, 0.7} — judge on
   AP_small + very-tiny/tiny recall on C2A val (official split first for comparability).
   **Gate G2:** ≥ +1 pt AP_small or VT-recall on val → do one full retrain on scene-split train
   (headline config); else record as null and drop (the paper survives on Phases 1/3/4).
7. If G2 passes: 3-seed confirmation for the final table only.

**Phase 3 — Inference fusion (2–3 days, any machine, zero training)**
8. Implement calibrated WBF merge for: tile-merge (vs GreedyNMM/NMS), whole+sliced ensemble, ±TTA.
   Ablate on C2A val large-resolution subset + SARD test + drone frames.
   **Gate G3:** recovers ≥ half of the sliced-F1 loss (0.850→0.829) at equal recall → keep as
   contribution; else demote to ablation row.
9. Adaptive slicing rule (resolution-conditional) as one extra ablation row.

**Phase 4 — Real-world validation (interleaved; manual labeling is the long pole — start day 1)**
10. Label 15–20 frames per altitude (10/30/50 m) → frozen real test set (never trained on).
11. Evaluate the model ladder on it: epoch125 → +enriched → +NWD → ±C-WBF. Also finish the pending
    drone-FP re-test of the enriched model.
12. SAHI-guided self-training loop (June-29 §8 Phases 2–4) with the best model from above.
    **Gate G4:** any positive delta on the real test set = the deployment claim; a flat/negative
    delta is still a reportable finding on pseudo-labeling limits.

**Phase 5 — Assembly (3–4 days, no GPU)**
13. Master table: {official split, scene split} × {baseline, CBAM+P2, +NWD} × {plain, SAHI, C-WBF,
    TTA} on C2A + SARD + real footage, plus the existing Mamba null and few-shot SARD curve.
14. Write. Target length: 8–10 pages + supplementary.

**Sequencing note:** Phases 1, 2-pilot, and 4-labeling can run simultaneously (different machines +
manual work). The critical path is Phase 1 (motivation) → Phase 5.

---

## PART 4 — Publication guideline

**The paper in one sentence:** *"We show the standard synthetic benchmark for UAV disaster human
detection rewards background memorization (98% scene leakage), introduce a scene-disjoint protocol,
and close the measured synthetic→real gap with tiny-object-aware training (NWD), calibration-weighted
sliced-inference fusion, and self-training — validated on SARD and self-collected 3-altitude drone
footage, with an instrumented SSM/Mamba null result."*

**What to CLAIM:** (1) first scene-disjoint evaluation protocol + leakage quantification for C2A;
(2) measured scale-regime analysis C2A vs SARD (12 px vs 53 px median) reframing cross-dataset
transfer; (3) calibrated WBF tile-fusion for sliced inference (first in the SAHI setting);
(4) deployment-grade validation on altitude-stratified real footage; (5) honest negatives (Mamba,
copy-paste, enrichment trade-off). **What NOT to claim:** architectural novelty of CBAM+P2 (prior
art — cite P2-YOLOv8n-ResCBAM etc.); any "+X mAP" from literature as our expectation; tiny-recall
failure on SARD (n=8 boxes).

**Venues (in order):**
1. *Remote Sensing* (MDPI, IF~5, Q1/Q2) — ASAHI and SAR-drone papers live here; dataset-audit +
   method + real validation is their exact profile; fast turnaround.
2. *Drones* (MDPI, Q2) — slightly easier, same fit.
3. *ISPRS Journal of P&RS* (Q1) — only if Phase 1 delta is dramatic and writing is strong; NWD's
   extension published here, so the topic fits, but the bar is high.
4. Fallback: *IEEE Access* (Q2), or ICPR/ICIP companion for speed.

**Reviewer-proofing checklist:** scene-split lists + code released; seeds and protocol stated
(SEEDS 0–4 for headline rows); official-split numbers reported alongside for comparability; C2A/SARD
version hashes; per-category (fire/flood/collapse/traffic) breakdown; calibration reported (we
already have ECE — few detection papers do this, cheap differentiator); limitations section owns the
sub-4px invisibility tier (6.5% of instances) and the synthetic-label ceiling.

**Risks:** (i) C2A authors already flagged split caveats somewhere → Phase 0 verification kills this;
(ii) NWD null on C2A → paper stands on leakage + fusion + real validation; (iii) time — the only
manual long pole is frame labeling, so it starts on day 1; (iv) a concurrent paper scoops the leakage
finding → submit the moment Phase 1 + Phase 4 minimum are done; everything else can be revision-added.
