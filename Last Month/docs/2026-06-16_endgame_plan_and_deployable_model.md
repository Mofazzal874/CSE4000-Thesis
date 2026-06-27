# End-of-month plan + deployable model (detailed) — 2026-06-16

State of play (all verified earlier):
- C2A ablation COMPLETE & consistent (baseline/CBAM/CBAM+P2/Mamba, all AdamW).
  mAP50-95 saturated ~0.615; P2 is the only real gain (+1.5pt very-tiny recall, +1.0 AP50,
  ~zero param cost); CBAM marginal; Mamba nil at 3× latency.
- Zero-shot C2A→SARD = near-zero (mAP50 ~0.006) — REAL (scale non-overlap: C2A <10px vs SARD
  32–96px, + domain gap). Few-shot smoke (20 imgs/2 ep) already recovered 5–12×.
- Deployable model needs MORE than C2A — your own zero-shot proves it; the C2A paper agrees
  (C2A-only→SARD 0.259; +real human data→0.660).

The thesis story is set: **"C2A as a pretraining corpus for data-efficient real-world SAR
detection; P2 drives small-object recall; an SSM neck does not help; zero-shot transfer
collapses but minimal real data + joint training recovers it."** Two more experiments finish it.

---

## PHASE 1 — Few-shot reality curve (Contribution B, the headline). RUNNING.
**Script:** `cross_dataset_SARD/sard_fewshot.py` (now leakage-safe).
**What:** fine-tune CBAM+P2 (and Mamba) on N∈{0,20,50,100,200} *distinct, leak-safe* SARD
images; evaluate every variant on the untouched SARD test split.
**Steps:**
1. Copy `cross_dataset_SARD/` D:→E:.
2. `SMOKE_TEST=True` → run → confirm the line `[sample] ... K distinct leak-safe source photos
   available`. Need K ≥ 200; if K<200 set `N_SHOTS=[0,20,50,100]`.
3. `SMOKE_TEST=False` → `python sard_fewshot.py`.
**Output:** `ablation_master/fewshot_curve.{csv,png}` (vs Lee et al. anchors 11.53@20 / 17.73@200).
**Decision rule:** if C2A-pretrained @ N=200 beats **17.73 COCO-AP**, claim "composited-real
(C2A) > rendered-synthetic (Archangel) as a SAR pretraining corpus." Either way the curve
(zero-shot → recovers) is the headline.
**Time:** few-shot fine-tunes on 20–200 imgs are minutes each; whole curve a few hours.
**Caveat to write down:** SARD is medium-scale (32–96px), not tiny — frame as "tiny-trained →
medium-real" scale-shift + domain gap. Roboflow augmentation duplication handled by the
source-disjoint sampler.

## PHASE 2 — The DEPLOYABLE model (joint C2A + SARD). NEW — script built this session.
**Script:** `cross_dataset_SARD/joint_c2a_sard_train.py`.
**What & why:** one model strong on BOTH disaster scenes (C2A) AND real humans (SARD) — what a
drone needs. Mirrors the C2A paper's "General-Human + C2A" recipe (→ 0.874 C2A & 0.660 SARD).
**Design:**
- Architecture = **CBAM+P2** (the deployment choice: ~14.6 ms, best recall; NOT Mamba — 41 ms,
  no gain, disqualified for edge real-time).
- Init from the **C2A-trained CBAM+P2 best.pt** (keeps the strong C2A model; CBAM is YAML-native
  so it survives train()'s rebuild — no injection patch needed). Option to start fresh from
  yolo11m.pt.
- Train on **C2A-train + SARD-train combined** via an explicit image-list .txt (reliable
  oversampling: SARD repeated `SARD_OVERSAMPLE×` to offset C2A's ~6:1 size dominance).
- Val on combined C2A-val + SARD-val (checkpoint selected for joint performance).
- After training: evaluate best.pt SEPARATELY on **C2A-test AND SARD-test** (same protocol as
  the chain), report both + per-size recall + efficiency.
**Steps:**
1. (after Phase 1) `python joint_c2a_sard_train.py` — auto-finds C2A + SARD + the C2A CBAM+P2 ckpt.
2. `SMOKE_TEST=True` first (2-epoch tiny), confirm, then full.
**Decision rule / success:** joint model keeps C2A-test high (~0.85 mAP50) AND lifts SARD-test
far above zero-shot (target: clearly > the few-shot N=200 point, since it uses the full SARD
train). That single model = your deployable artifact + a clean "good on both domains" result.
**Time:** one training run; C2A(6k)+SARD(1k×oversample) ≈ similar to a chain run (a few–several h).
**Tuning if SARD-test stays low:** raise `SARD_OVERSAMPLE` (3→5→8); the more SARD signal, the
more it adapts (watch C2A-test doesn't drop much).

## PHASE 3 — Production / paper continuation (NOT this month; future-work section)
A genuinely field-robust drone detector = combined **real** aerial-human corpus:
SARD + HERIDAL + VisDrone-person + Okutama-Action (+ AIDER for disaster context), all collapsed
to one `person` class, multi-scale. Precedent: FlyPose (WACV 2026) trained on 9 real aerial
datasets. Then: export ONNX→TensorRT, test on Jetson, consider 1280px input for tiny persons
(FlyPose's lever; 13 ms on Orin). Write this as the deployment roadmap / future work.

---

## What is the "deployable model" you hand over?
The **Phase-2 joint CBAM+P2 checkpoint**: real-time-capable (~15 ms), strong on disaster scenes,
and adapted to real SAR humans. State its limits honestly (trained on C2A+SARD only; wider
real-data corpus = Phase 3). That's a defensible thesis deliverable AND a usable drone model —
not the fragile C2A-only model whose zero-shot collapse you measured.

## Writing (the actual critical path)
Compute serves the report. Start writing the methods + the complete C2A ablation table NOW (it's
done); slot in the few-shot curve and the joint-model result as they land. Don't block writing on
Phase 2/3.
