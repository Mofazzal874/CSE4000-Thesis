# Ranking & Publication Targets — lap 3 (2026-07-10)

Inputs: idea catalog (58 audited items, same folder) + verification log Lanes 1–3 + lap-2 needs
N1–N8 + lap-2 FCCG proposal + our own falsified-locally list. Lanes 4–5 (top-venue mining) are
running kill-safe; their findings refine §4 (venues) but do not change §1–3 tiers — the tier
logic rests on verified primary sources already in the log.

## 1. Scoring rubric
Each candidate DIRECTION (not raw catalog item) scored 0–5:
- **Need** — how directly it answers measured needs N1–N8 (lap-2 doc A).
- **Feas** — feasibility on our stack (Ultralytics YAML/Windows/16 GB protocol GPU, kill-safe).
- **Nov** — novelty defensibility AFTER this audit (prior-art distance, no-plug-and-play rule).
- **Evid** — probability of producing reviewer-proof numbers (incl. beating/contextualizing the
  verified C2A bar YOLOv9-e 0.8927/0.6883).
- **Cost⁻¹** — 5 = hours, 1 = multi-week GPU commitment.

## 2. The ranking

| # | Direction (catalog refs) | Need | Feas | Nov | Evid | Cost⁻¹ | Σ/25 | Tier |
|---|---|---|---|---|---|---|---|---|
| D3 | **Sim-to-real pillar**: seam probe → C2A-H harmonization (CFHA-lite) → SF-UT ladder on own footage → joint 3-set model (#46-48,51-53) | 5 | 4 | 4 | 5 | 4 | **22** | **T1** |
| D2 | **Tiny-aware assignment in TAL**: STAL min-anchor floor (<8 px) + SimD/RFLA-style similarity term inside Ultralytics TaskAlignedAssigner (#4,55) | 5 | 4 | 3.5 | 4 | 4 | **20.5** | **T1** |
| D1 | **FCCG-YOLO core**: HF-evidence branch × large-kernel context gate + FreqFusion-lite (#35,56,57; lap-2 proposal) | 5 | 4 | 3.5 | 4 | 3 | **19.5** | **T1** |
| D4 | **Occlusion stack**: repulsion + Slide + (stretch) synthesis-aware visibility supervision; NOMAD occlusion-graded eval (#38) | 4 | 3.5 | 4 | 3.5 | 3 | 18 | T1.5 |
| D5 | **Pose-prior aux supervision** (C2A's unused official pose labels) (#37) | 3 | 4 | 3.5 | 3 | 4 | 17.5 | T2-PROBE |
| D11 | Combined-dataset final recipe (joint C2A+SARD+drone-labeled, balanced sampling) (#51) | 4 | 5 | 2 | 4 | 4 | 19* | folded into D3 |
| D6 | Dual-head o2o + ProgLoss port to our lineage (#1,5,23) | 3 | 2.5 | 2 | 3 | 2 | 12.5 | T2-PARK |
| D8 | Differentiable tile-zoom inside YOLO (#36; ZoomDet adjacent) | 4 | 2 | 3 | 3 | 1.5 | 13.5 | T2-PARK |
| D9 | Adversarial DA (GRL/MS-DAYOLO/"DAMA") (#39,40) | 3 | 3 | 2 | 3 | 3 | 14 | T3 (baseline inside D3 at most) |
| D7 | Hypergraph neck (#16,21,22) | 3 | 3 | 1.5 | 3 | 2 | 12.5 | T3-REJECT |
| D10 | Base-model swap as "novelty" (v12/v13/26) (#8,9,16) | — | — | 0 | — | — | — | BASELINES only (S4) |
| D12 | Test-time adaptation (#50) | 2 | 2 | 2 | 2 | 3 | 11 | T3-REJECT (future work ¶) |

*D11 scores high but is a training recipe, not a contribution — it ships inside D3.

**Why the top three are these:**
- **D3** is the only direction where we own assets nobody publishing on C2A has (3-altitude 4K
  footage + frozen test frames + scene-disjoint protocol + measured paste-artifact evidence from
  lap-2 needs analysis), the 2025/26 literature just handed us a validated recipe (CFHA +14.1
  mAP50; SF-UT collapse-proof ladder; S3OD tiny-object pseudo-label fixes), and every step is
  cheap on our fleet. It also weaponizes what was previously our biggest WEAKNESS (synthetic
  training data).
- **D2** is the sharpest verified unclaimed slot: the entire AI-TOD lineage says assignment (not
  loss) is the tiny-object lever, our own G2 negative agrees, and nobody has published a
  SimD/STAL-style term inside YOLO11's TAL for aerial persons. Small surgery, big literature
  backing.
- **D1** survives the audit but with a tightened mandate: the generic "frequency module in a UAV
  detector" slot filled in 2025 (SET, UAV-DETR, Freq-DETR, EFSI, wavelet-YOLOs) and DERNet
  (2606.23825, 18 days old) is adjacent — the defensible claim is specifically the
  **evidence-vs-plausibility GATED two-stream design, in a YOLO neck, for tiny occluded persons,
  coupled to D2/D4 and evaluated under the D3 protocol.** Mandatory: DERNet differentiation note
  before S0 freeze.

## 3. The recommended composite (what we'd take to a paper)

**One system, one story, two coupled contribution axes:**

> **FCCG-YOLO + FreqDA protocol** — *"A tiny human is a high-frequency anomaly in a low-frequency
> scene — but in synthetic composites, so are the paste seams. We (i) architecturally separate
> high-frequency EVIDENCE from large-kernel contextual PLAUSIBILITY (gated two-stream neck +
> tiny-aware assignment + occlusion stack), and (ii) show the same frequency lens exposes and
> fixes the synthetic-to-real gap (seam-bias quantification → training-set harmonization →
> self-training on real unlabeled footage), validated on a 3-altitude real-world drone benchmark
> under a leakage-audited scene-disjoint protocol."*

Axis 1 (architecture) = D1 + D2 + D4. Axis 2 (domain/protocol) = D3 + lap-1 assets (scene-split,
leakage audit, metric harness). The frequency theme welds the two axes together so neither reads
as a bolt-on — that coupling is the answer to "why is this one paper".

**Paper table skeleton this implies:** ablation (base → +assignment → +FCCG → +occlusion stack →
+harmonization → +self-training) × eval grid (C2A-official, C2A-scene-disjoint, SARD zero-shot,
own-drone 3-altitude) + anchors (YOLOv9-e repro, YOLO26m-AdamW, optional YOLOv12m/D-FINE-M/
FBRT-YOLO) + latency/params + calibration. Every cell already has a script or a PC slot.

## 4. Venue reality check (honest; Lane-4 evidence now in; Lane 5 refines journal bars)

- **CVPR / ICCV / ECCV main:** confirmed not a realistic target — Lane 4 found ZERO CVPR 2026
  main-conference papers on tiny/aerial-human detection (the one UAV-detection main-conf datapoint
  demanded a new sensing modality). The subfield lives in workshops/WACV/AAAI.
- **★ NEW (Lane 4): AERO-HPR — 1st CVPR 2026 workshop on aerial human perception** — CFP matches
  our story nearly verbatim (small objects, synthetic imagery, aerial person pipeline); its
  proceedings include **SAFE-Net (flood aerial person detection — our closest 2026 neighbor,
  must-read/cite/beat)** and a P2-in-YOLOv12 aerial-pedestrian paper (proof P2-alone still clears
  a workshop in 2026 — our composite must out-claim it). **2nd edition at CVPR 2027 = realistic
  workshop target.**
- **★ NEW (Lane 4): WACV-main pattern-proof — RealDroneVision (WACV 2026)**: [own real dataset +
  architecture mods] in one paper clears WACV MAIN. Our own-footage 3-altitude benchmark + FCCG
  composite replicates that recipe → **WACV 2027 = primary conference target.** Bonus framing:
  cast the C2A→own-drone lane in **CD-FSOD vocabulary** (NTIRE 2026 challenge, CVPRW) so reviewers
  recognize the protocol as a first-class problem setting.
- **ICML / NeurIPS / ICLR:** formally ruled out with receipts — ICML 2025/26 accepted lists show
  zero applied small-object/aerial detection (Lane 4 queries logged); ICLR 2026 main had nothing
  beyond RF-DETR; ML4RS workshop is the ICLR-adjacent home if ever needed.
- **Journal ladder (Lane-5 evidence bars, all DOI-verified exemplars):**
  1. **IEEE JSTARS — PRIMARY.** Demonstrated bar = 2 novel modules on a YOLO base (WE-YOLO,
     JSTARS 2026); our package (3 datasets incl. real SARD + own footage, latency, leakage audit)
     already EXCEEDS typical JSTARS rigor.
  2. **Pattern Recognition (Q1) — reachable REFRAME target.** PR's 2025-26 acceptances (AMSF-YOLO,
     DN-TOD, domain-consistency, dynamic scale-aware assignment) all weld architecture to a
     FORMAL problem. Our formal problems already exist: paste-label noise + benchmark leakage +
     occlusion evidence. If S1-S3 deltas land, a PR-shaped write-up ("tiny-object evidence under
     synthetic label noise and benchmark leakage") is genuinely in reach.
  3. **IEEE TGRS — stretch.** Bar = FFCA-YOLO template: ≥3 components, 3+ datasets, ~10 baselines,
     lite/efficiency variant. Attempt only if we beat/reframe the C2A bar cleanly.
  4. **IEEE GRSL (Q1, IF 4.4)** — compact letter SPIN-OFF option (the context gate alone, or the
     leakage audit as a short protocol paper).
  5. **MDPI Drones (IF 4.8) / Remote Sensing (IF 4.3)** — fast floor, use only against deadlines.
  (ISPRS JPRS: only if the PROTOCOL becomes the headline; IEEE TIP/IJCV: no for this package;
  MVA dropped to Q3 in 2025 JCR — removed.)
- **Conference ladder:** **WACV 2027 main** — pattern-proven by RealDroneVision (own dataset +
  architecture = accepted); **AERO-HPR @ CVPR 2027 workshop** — CFP matches us verbatim;
  BMVC/ICIP backups. Frame the C2A→own-drone lane in **CD-FSOD vocabulary** (NTIRE 2026).
- Sober probability read (single student, 16 GB fleet, niche benchmark, but: unchallenged C2A bar
  + unique real-footage benchmark + leakage audit): JSTARS/WACV = solid shot if §3 executes;
  PR = real if the formal-problem reframe is taken seriously; TGRS/ISPRS = stretch; CVPR-main =
  ruled out with receipts (Lane 4). **Both lanes independently converge on the same #1 missing
  experiment: reproduce YOLOv9-e on C2A under BOTH splits (official + scene-disjoint) — it tests
  whether the printed 0.8927/0.6883 bar survives a clean protocol, upgrades the leakage audit to
  a headline if it doesn't, and defines the honest number FCCG must beat. Already P0/S4.**

## 5. Amended gated plan (delta to lap-2 S0–S5; costs respect PC fleet)

| Gate | What | Where | Cost | Pass criterion |
|---|---|---|---|---|
| **P0 (this week, NEW)** | (a) read DERNet 2606.23825 **+ SAFE-Net (CVPRW 2026, AERO-HPR) + SRTSOD-YOLO (RS 17(20):3414 — gated fusion neck ON YOLO11) + AFGLFF-YOLO abstract (JSTARS 2026)** → 1-page differentiation note; (b) pose-label audit on ~100 C2A images (#37); (c) extract unlabeled drone train-pool frames, dedup, altitude-stratified (script 05; 60 test frames untouchable) | laptop | 0 GPU | note exists; pose labels usable y/n; ≥500 clean unlabeled frames |
| **P0.5 (NEW, optional but cheap)** | DN-TOD (PR 2026, code ZhuHaoranEIS/DN-TOD) feasibility skim: can CLC/TLS bolt onto Ultralytics trainer? If yes → queue as S2 row (targets our paste-label-noise ceiling directly) | laptop | 0 GPU | go/no-go note |
| **P1 seam probe (NEW)** | low-pass/re-JPEG C2A test images → eval CBAM+P2 (script 04); compare degradation slope vs SARD | PC-4 | ~2 h eval | ANY outcome reportable; if AP collapses ≫ SARD ⇒ seam reliance QUANTIFIED (feeds paper §, motivates C2A-H) |
| S0 | FCCG modules + YAML + selftests (shape/grad/pickle/param) + 2-ep smoke — unchanged from lap-2, now with DERNet-differentiation checklist | laptop + PC-4 | 1–2 d | modules ACTIVE, ≤22.5M params |
| S1 | paired 50-ep pilots scene-split: control vs +FCCG | PC-4/PC-2 | ~5 h each | +1.5 AP_small or +2 VT-recall |
| S2 | loss/assignment stack stepwise: **STAL floor → SimD-in-TAL → repulsion → Slide** (D2/D4; order = cheapest-first) | PC-4 | ~5 h each | each keeps/improves S1 |
| **S2.5 DA ladder (NEW, D3)** | AdaBN → fixed-PL self-training (SF-UT recipe + S3OD size-aware thresholds) on unlabeled drone frames; optional DINOv2-labeler pass | PC-2 GPU1 | 0.5–1 d | drone-frame FP rate / VT-recall improves without C2A regression |
| **S2.6 C2A-H pilot (NEW, D3)** | CFHA-lite harmonization of a C2A subset (blend-diversity re-paste first — cheap; diffusion pass only if justified) → 50-ep pilot | PC-2 | 1–2 d | transfer delta on SARD/drone > control |
| S3 | full 300-ep protocol runs (winning config), scene-split + official | PC-1 (after G1 baseline finishes) | ~1.5 d each | beats CBAM+P2 full runs; SARD zero-shot no collapse |
| S4 | anchors: YOLOv9-e repro + YOLO26m (AdamW pinned) | PC-1 queue | ~3 d total | table completeness |
| S5 | final: joint 3-set model (C2A[-H]+SARD+labeled drone frames, balanced sampling), 3 seeds, drone-bench eval (60 labels), optional NOMAD transfer, **+ RealDisaster-mini R-set row (eval-only; user's real Venezuela-earthquake/Bangladesh-flood videos; pipeline + provenance rules in `RealDisaster\README_PROVENANCE.md`, extractor = scripts/12_rset_extract.py)** | PC-1/PC-4 | — | paper numbers |
| **Annotation gate** | ONLY after S2.5 shows lift: label 50–150 stratified non-test frames (user approves budget) | Roboflow | user time | few-shot-DAOD evidence says this is enough to matter |

Failure floors unchanged from lap-2 (protocol + ablation corpus + real-footage benchmark survive
any gate failure; Direction A sparse-P2 remains the efficiency fallback).

## 6. Decision menu — ★RESOLVED 2026-07-12: user chose **Option A**.
Baselines locked: YOLOv9-e repro + YOLO26m + YOLOv12m (all native/verified); D-FINE-M optional
after anchors on the A6000. Annotation approved and started same-day (see
`..\ANNOTATION_TODO_2026-07-12.md`): 60 test frames finish first, then 120-target/240-stretch
selftrain pool. Pose-label audit PASSED (P0b) → pose-aux promoted to optional S2 row.
User note: PC resources not a constraint (16 GB + 12 GB + 12 GB + A6000 48 GB).
*(original menu kept below for the record)*

- **Option A (recommended): the composite of §3** — FCCG architecture axis + FreqDA sim-to-real
  axis, target JSTARS/WACV (stretch TGRS/ISPRS). Most work, highest ceiling, both axes reuse
  existing scripts/PCs.
- **Option B: architecture-only** — D1+D2+D4 under the existing protocol; sim-to-real shrinks to
  one experiment. Faster; loses the unique-asset story; venue ceiling drops toward GRSL/Drones.
- **Option C: data/protocol-first** — D3 + D2 only, minimal architecture change (CBAM+P2 stays).
  Cheapest GPU path, very defensible ("first sim-to-real study on C2A + real benchmark"), but
  abandons the lap-2 architecture ambition — weaker fit to the "novel architecture" thesis goal.
- Also decide: (i) annotation budget approval in principle (50–150 frames, gated on S2.5);
  (ii) baseline set for S4 beyond YOLOv9-e/YOLO26m (add YOLOv12m? D-FINE-M?); (iii) whether the
  60 frozen test-frame labels (in progress on Roboflow) can be prioritized — S5 depends on them.
