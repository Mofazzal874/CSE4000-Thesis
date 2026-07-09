# Idea Catalog & Audit — every idea from the 2026-07 Google/Gemini conversation, verified (2026-07-10)

**What this is:** the user brought a long Gemini conversation with "50+" 2025/26 ideas. Three
web-verification agents checked every one against primary sources (arXiv/DOI/official docs/
fetch-verified GitHub). Full raw reports with all URLs: `2026-07-10_web_verification_log.md`
(Lanes 1–3). Two more lanes (top-venue mining) are running; their additions land in §H.

**Verdict legend:** ✅ VERIFIED real · ⚠ GARBLED (real thing, claim as stated wrong) ·
❌ FABRICATED/UNFINDABLE · 🔬 GEMINI-REMIX (unpublished invention by the search AI, not literature) ·
🧪 FALSIFIED-LOCALLY (we already ran it; negative).
**Decision legend:** ADOPT-CORE (headline design) · ADOPT-COMP (component of our stack) ·
CITE+BEAT (real prior art we must cite and outperform/differentiate) · BASELINE (comparison row) ·
PROBE (cheap experiment first) · PARK (viable, not now) · REJECT · OOS (out of scope).

**Score:** of the concrete Gemini claims audited: ~70% verified real (often with garbled details),
5 outright fabrications, 5 unpublished remixes presented as if established, 2 items we had
already falsified locally. The single biggest UNDER-sell in that conversation: it missed that the
real 2025/26 sim-to-real consensus is data harmonization + self-training (CFHA/SF-UT line), and
pitched 2021-era adversarial GRL as novel instead.

---

## A. YOLO26 components (Ultralytics, arXiv 2606.03748, ships in pip ultralytics v8.4.0+, AGPL-3.0)

1. **Native NMS-free end-to-end (dual-head, deploy o2o)** — ✅ (lineage = YOLOv10 NeurIPS 2024,
   consistent dual assignments) → **BASELINE** (comes free with the YOLO26m anchor run) + PARK as
   component: o2o supervision is sparse (1 positive/GT) — bad default for tiny recall unless
   STAL-compensated; deployment win, not an accuracy lever for us.
2. **DFL removal** — ✅ → REJECT for us. Counter-evidence: D-FINE (ICLR'25 Spotlight) shows
   *distributional* box refinement HELPS under localization uncertainty — which is exactly the
   tiny-object regime. Keep DFL; consider FDR-style refinement as PARK.
3. **MuSGD optimizer** — ✅ real (Muon lineage: KellerJordan/Muon; Moonshot arXiv 2502.16982) but
   🧪 FALSIFIED-LOCALLY: diverged on our P2 config (2026-06) → **REJECT** (AdamW lr0=0.001 stays
   pinned; run YOLO26m baseline with AdamW too).
4. **STAL — small-target-aware label assignment** — ✅; only official mechanic published anywhere:
   **"minimum 4 anchor assignments for objects <8 px"** (Ultralytics blog). ⚠ the equations
   floating in arXiv 2601.12882 are third-party reconstructions that contradict the official
   ProgLoss description — do NOT cite as Ultralytics math; grep local v8.4.x for ground truth.
   → **ADOPT-COMP** (lap-2 plan already lists it, component 4): min-anchor floor for <8 px inside
   our assigner. Directly answers N1 (34.5% of C2A boxes <8 px).
5. **ProgLoss — progressive o2m→o2o loss weighting** — ✅ (official description) → PARK (only
   matters if we adopt the dual-head; revisit at S4).
6. **"43% faster CPU inference"** — ✅ official claim (26n vs 11n, ONNX) → OOS (deployment story,
   not accuracy; note for paper's efficiency table only).
7. **"YOLOE-26" open-vocab** — ⚠ GARBLED: YOLOE is real (ICCV 2025, THU-MIG/yoloe); "YOLOE-26"
   exists only as a third-party mashup preprint (arXiv 2602.00168), NOT an official model → OOS.
   One stealable pattern: RepRTA's *train-time aux branch that re-parameterizes to zero deploy
   cost* — relevant template for our HFE branch (ADOPT-pattern).
8. **YOLO26 as a whole (m-scale)** — ✅ COCO 53.1 vs YOLO11m 51.5. Independent aerial check
   (arXiv 2605.24831): YOLO26-x ≈ YOLOv8-x on VisDrone (0.224 vs 0.214) — **the new assignment
   toys alone do NOT fix aerial tiny objects** → **BASELINE (S4, mandatory)** + motivation
   paragraph for our composite thesis.

## B. YOLOv12 components (arXiv 2502.12524, NeurIPS 2025 badge, code sunsmarterjie/yolov12)

9. **Area Attention (A²)** — ✅ (split map into 4 areas, plain attention per area, ½ cost) →
   PARK. Real but plug-and-play as a block; Ultralytics port has documented instability/memory
   caveats. Our 4070 Ti S supports FlashAttention (Ada) if ever revisited.
10. **R-ELAN (residual-scaled aggregation)** — ✅ → PARK (stability trick worth remembering if we
    ever train attention-heavy variants).
11. **FlashAttention in the detector** — ✅ optional in v12 → OOS for the CNN-based FCCG design.
12. **FFN/MLP ratio 4→1.2–2** — ✅ → ADOPT-pattern (if we add any attention/FFN blocks, use thin
    FFNs; free parameter savings).
13. **No positional encodings + 7×7 separable-conv position perceiver** — ✅ → PARK (same bucket
    as 9–11).
14. **"Transformer-enhanced detection head" / "Ghost+Swin hybrid backbone"** — ❌ FABRICATED
    (exact phrases absent from paper and repo; the head is the standard YOLO conv head) → REJECT.
15. **Gemini's "Decoupled-Projection A² bottleneck inside C3k2 + FFN 1.5"** — 🔬 GEMINI-REMIX of
    #9+#12; unpublished, derivative, plug-and-play flavor → REJECT as novelty (fails the
    no-plug-and-play rule); components live on via #12 pattern.

## C. YOLOv13 / hypergraph line

16. **HyperACE (hypergraph adaptive correlation)** — ✅ (arXiv 2506.17733; iMoonLab/yolov13;
    NOT in pip ultralytics — standalone fork) → CITE+BEAT / BASELINE-if-cheap. Note NO m-scale
    variant exists (N/S/L/X only).
17. **FullPAD (3-tunnel redistribute)** — ✅ → CITE (design-pattern awareness only).
18. **DS blocks (depthwise-separable large-kernel replacements)** — ✅ → OOS.
19. **"YOLO-TCM temporal context module"** — ❌ FABRICATED (absent from paper+repo; v13 is
    single-frame) → REJECT. The *real* temporal line is StreamYOLO (CVPR 2022) — OOS for our
    single-frame protocol.
20. **"Dynamic Task Routing (DTR)"** — ❌ FABRICATED (absent from paper+repo) → REJECT.
21. **Hyper-YOLO (TPAMI 2025, arXiv 2408.04804)** — ✅ → CITE. Hypergraph-neck origin.
22. **Gemini's "NovelHyperACE replacing C2PSA in YOLO11"** — 🔬 GEMINI-REMIX; and the direction is
    CROWDED (Hyper-YOLO, YOLOv13, HyperEdge-DETR JRTIP'25, MSGHNet, HyperTea) → **REJECT as
    headline** — would read as derivative iMoonLab work.

## D. Other families & trends named in the conversation

23. **YOLOv10 consistent dual assignments** — ✅ (NeurIPS 2024, THU-MIG) → origin of A1; PARK.
24. **RT-DETR / v2** — ✅ (CVPR 2024 / arXiv 2407.17140; in pip ultralytics) → CITE. Literature
    treats vanilla RT-DETR as weak on UAV tiny objects (UAV-DETR fixes it with FREQUENCY modules
    — supports our premise).
25. **D-FINE (FDR + GO-LSD)** — ✅ ICLR 2025 Spotlight, code Peterande/D-FINE → CITE + PARK
    (distributional refinement is tiny-friendly; D-FINE-M 19M = our weight class → optional
    BASELINE row).
26. **DEIM Dense-O2O + MAL (+DEIMv2 2025-09)** — ✅ CVPR 2025 → CITE (the DETR-side mirror of
    STAL: densify positives for tiny objects; conceptual support for our assignment stack).
27. **RF-DETR** — ✅ ICLR 2026 (arXiv 2511.09554; roboflow/rf-detr; DINOv2 + NAS) → CITE only
    (different compute class; PML license on big variants).
28. **Open-vocab YOLO-World/YOLOE as base** — ✅ real → REJECT as base (kills YOLO11m
    comparability; open-vocab orthogonal to tiny/occluded specialization).
29. **"EFEM-YOLO" (PV defects)** — ❌ UNFINDABLE as a model (EFEM exists only as a module name in
    unrelated papers) → REJECT.
30. **Medical/agri/XAI/edge YOLO trends** — ✅ trends exist → OOS (different domains; XAI could be
    one paper-figure via attention maps, not a contribution).
31. **FBRT-YOLO (AAAI 2025)** — ✅ (arXiv 2504.20670, code galaxy-oss/FCM) — *added by our sweep,
    not in Gemini convo*: the peer-reviewed 2025 YOLO-family aerial-small-object reference →
    **CITE + candidate BASELINE**.
32. **Dome-DETR (ACM MM 2025)** — ✅ AI-TODv2 34.6 AP — density-gated sparse compute frontier →
    CITE+BEAT-in-spirit (a YOLO-neck analogue of density/frequency-gated routing is exactly the
    slot FCCG occupies).
33. **UAV-DETR (arXiv 2501.01855)** — ✅ (repo archived June 2026) → CITE (frequency-in-DETR
    precedent; the YOLO-neck gated version remains open).

## E. Gemini's five custom pitches (the "novel" proposals) — audited

34. **CG-DSA (Context-Guided Deformable Spatial Attention)** — 🔬 GEMINI-REMIX. Ingredients real
    (deformable conv v2, spatial attention); nearest published: DOMino's deformable alignment
    (Remote Sensing 2026, vehicles). As pitched = plug-and-play attention block → **REJECT as
    headline**; deformable alignment stays a PARK option for the occlusion head.
35. **BP-FFP (Fourier boundary-preserving pyramid)** — 🔬 GEMINI-REMIX that *converges on a real
    2024-26 line* (FreqFusion TPAMI'24, HS-FPN/HFP AAAI'25, SET CVPR'25, wavelet-YOLOs, DERNet).
    → Superseded by our own FCCG plan (which predates this convo) — treat as independent
    validation of the frequency direction, and use the Lane-2 must-cite list. **ADOPT-CORE (as
    FCCG, not as BP-FFP)**.
36. **Differentiable tile-scouting / dynamic-resolution routing** — 🔬 remix of a real family
    (ClusDet, DMNet, AdaZoom(RL), QueryDet, CZ-Det, ESOD TIP'25) + **new adjacent prior art:
    ZoomDet (ISPRS JPRS 2026, learnable end-to-end zoom, code)** → PARK. Technically still open
    "inside a real-time YOLO", but high engineering risk on Windows/Ultralytics and the window is
    narrowing; lap-2 Direction A (density-gated sparse P2) remains our fallback version of this.
37. **APP-Loss (pose-prior aspect-ratio regression)** — 🔬 GEMINI-REMIX, BUT audit UPGRADE:
    **C2A officially ships pose labels** ("All labels with Pose info": bent/kneeling/lying/
    sitting/upright, YOLO+COCO formats) and NO published follow-up uses them. Labels inherit
    paste noise; our splits are nc=1. → **PROBE→PARK: pose-conditioned aux supervision is a real,
    unclaimed, cheap-to-try micro-contribution IF label audit passes** (small: parse pose labels,
    check consistency on 100 images).
38. **Visibility/part-aware aux head ("VE-DINO-inspired")** — ⚠ IMPORTANT CORRECTION: **VE-DINO is
    REAL** (Smart Cities 8(1):12, Jan 2025 — earlier fabrication suspicion refuted). But: ground-level
    COCO+custom training, UAV only as case study, no code, low-prestige venue → CITE+BEAT.
    Aerial visibility-aware supervision remains lightly occupied (Psych-Occlusion = RetinaNet on
    NOMAD). **Our synthesis-aware visibility supervision from C2A paste provenance is still
    unclaimed → keep as ADOPT-COMP/stretch (lap-2 component 5), now with NOMAD (WACV 2024,
    5 altitudes × 10 visibility grades) as the ready-made occlusion-conditional eval.**
39. **GD-ANN (GRL domain classifier on YOLO neck)** — ⚠ GARBLED-as-novel: this IS MS-DAYOLO
    (ICIP 2021 + IEEE TIP 2023). → REJECT as novelty; keep as cheap DA BASELINE if we run the DA
    ladder.
40. **DAMA (Domain-Adversarial Masked Attention)** — 🔬 GEMINI-REMIX in a CROWDED space (AWADA
    CVIU'24 = attention-weighted adversarial DA; SCAN; DA-DETR; masked-DETR-alignment). Narrow
    unclaimed delta: *evidence-gated alignment specialized to tiny objects at P2/P3, tied to
    paste-seam bias*. → PARK (only as a later extension of the sim-to-real pillar; must position
    vs AWADA/SCAN explicitly).
41. **"3D photogrammetry study, July 2026, 88%→95%"** — ❌ UNFINDABLE (4 query variants logged).
    Probable garble of: MDPI RS 18(2):361 (photogrammetry synthetic framework — VEHICLES) and/or
    arXiv 2411.09077 (drone-*detection*, has 88/95 figures) → REJECT the citation; the underlying
    *direction* is real and better served by #47.
42. **Altitude/angle perspective + degradation augmentation** — ✅ real family (domain
    randomization: Tobin'17, Tremblay'18; UAV-Sim NeRF 2310.16255) → ADOPT-COMP (cheap loader-side
    augmentation for the DA pillar; not a novelty claim).

## F. Sim-to-real & data strategy (the axis the Gemini convo got directionally right but 4 years stale)

43. **DANN/GRL anchor (Ganin & Lempitsky 2015)** — ✅ → CITE (anchor only).
44. **SSDA-YOLO (CVIU 2023, code hnuzhy/SSDA-YOLO)** — ✅ → CITE; template if we ever want
    mean-teacher+style-transfer on YOLO11 (engineering port, not novelty).
45. **Teacher-student UDA line (Unbiased/Adaptive Teacher, ConfMix, CMT, Harmonious Teacher)** —
    ✅ all → CITE (related-work spine for the DA section).
46. **2024-26 UDA SOTA: SF-YOLO (ECCV-W 2024), SF-UT/simple-SFOD (ECCV 2024), DINO Teacher
    (CVPR 2025), Dual-Rate Dynamic Teacher (ICCV 2025)** — ✅ all, three with code →
    **ADOPT-COMP (the "SF-UT ladder"): AdaBN → fixed one-shot pseudo-labels → (optional) mean
    teacher, on our unlabeled drone frames; DINOv2-as-labeler pass for hard/tiny instances.**
    Highest credibility per GPU-hour of anything in the DA space; collapse-resistant.
47. **CFHA (arXiv 2512.13869, ARL line: Archangel→UAV-Sim→CFHA)** — ✅ **the most on-point paper
    for C2A→real found by any lane**: diffusion style transfer + small-instance SR refinement +
    hallucination filtering on the SYNTHETIC TRAINING SET, labels preserved, +14.1 mAP50 sim2real.
    → **ADOPT-COMP: "C2A-H" harmonized-training-set experiment (CFHA-lite)**; also the
    related-work spine for our sim-to-real story.
48. **Paste-seam bias literature (Cut-Paste-Learn ICCV'17; Simple Copy-Paste CVPR'21
    counter-evidence; Synthetic Object Compositions arXiv 2510.09110 +10.9 AP)** — ✅ →
    **ADOPT-PROBE (cheapest high-value experiment in the whole catalog): (i) frequency seam probe
    — low-pass/re-JPEG C2A test images, if AP collapses vs SARD we have QUANTIFIED the paste
    shortcut (needs N7, FCCG's declared risk, zero training); (ii) Dwibedi-style mixed re-blending
    of C2A pastes → transfer delta.** Either outcome is reportable.
49. **Archangel (IEEE Access 2023) + UAV-Sim (2310.16255) + maritime-SAR EAAI line (synthetic +
    10% real > 100% real by 13.7%)** — ✅ → CITE (evidence base for hybrid training + the
    altitude-stratified eval template our 10/30/50 m bench already matches).
50. **Test-time adaptation (AMROD 2406.16439 etc.)** — ✅ real, fragile on detection → REJECT for
    headline; one line of future work.
51. **Multi-dataset joint training (UniDet CVPR'22, Detection Hub, ScaleDet, CerberusDet-YOLO)** —
    ✅; consistent published pattern: **joint mix ≥ sequential fine-tune for cross-domain
    robustness** (our Angle-B fine-tune SARD dip 0.917→0.898 is the textbook sequential symptom;
    our joint C2A+SARD 0.878/0.917 matches the pattern) → **ADOPT-COMP: final-model recipe =
    joint C2A + SARD + (50–150 labeled drone frames), per-dataset balanced sampling, per-domain
    reporting.** This answers the user's "combined dataset" question: YES, with balanced sampling,
    keeping per-domain test sets separate.
52. **Label-budget evidence (Unbiased Teacher 0.5–2% COCO ≈ +10 mAP; AsyFOD CVPR 2023 few-shot
    DAOD with single-digit target images, code; maritime 10%-real result)** — ✅ →
    **DECISION INPUT: 50–150 labeled drone frames (altitude-stratified) is the evidence-backed
    annotation ask; 60 frozen test frames stay untouched; rest of footage = unlabeled fuel.**
53. **S3OD (ISPRS JPRS 2025)** — ✅ quantifies tiny-object pseudo-label recall collapse +
    size-aware fixes → ADOPT-COMP inside the SF-UT ladder for 50 m footage (size-aware thresholds,
    size-rebalanced assignment).
54. **VTSaR benchmark + APD survey (J. Remote Sensing 2025, 0474)** — ✅ → PARK: optional third
    cross-domain eval set (real+synthetic aerial persons, RGB-T); survey = citation anchor.

## G. Tiny-object mechanism clusters (verified consensus relevant to our stack)

55. **Assignment-level > loss-level for <16 px** — ✅ consensus (AI-TOD lineage DotD→NWD-RKA→
    RFLA→DCFL→SimD is ALL assignment-side; SimD IROS 2024 +4.1 AP_vt, code). Our G2 NWD-as-loss
    negative is *consistent with the literature*. **Unclaimed micro-slot: Gaussian/similarity
    (RFLA/SimD-style) term inside Ultralytics TaskAlignedAssigner for YOLO11 on aerial persons —
    nobody has published this** → **ADOPT-COMP (upgrade of lap-2 component 4: STAL min-anchor
    floor + SimD-style similarity inside TAL).**
56. **Frequency-domain cluster** — ✅ crowded (SET CVPR'25 = top-venue proof tiny objects live in
    frequency bands; FreqFusion; UAV-DETR/Freq-DETR/EFSI-DETR/UFO-DETR/FMC-DETR; wavelet-YOLOs
    WCDB/MS-YOLOv11; **DERNet 2606.23825 = closest neighbor, 18 days old, no code**) → FCCG's
    remaining gap (verbatim from Lane 2): *HF **evidence** branch (i) GATED by large-kernel
    context (ii) in an Ultralytics YOLO neck (iii) for tiny occluded PERSONS (iv) + assignment/
    loss stack (v)*. **ADOPT-CORE with mandatory differentiation section**; read DERNet in full
    before S0 freeze.
57. **Large-kernel context (LSKNet ICCV'23/IJCV'24, PKINet CVPR'24, UniRepLKNet CVPR'24)** — ✅
    all backbones; none used as a *neck gate over an evidence branch* → ADOPT-COMP (CXG gate
    ingredient; cite all three + UFO-DETR's LSK use).
58. **Sparse high-res compute (QueryDet CVPR'22, ESOD TIP'25 code alibaba/esod, CEASC, Dome-DETR)**
    — ✅ → PARK = lap-2 Direction A fallback (efficiency story) if FCCG S1 gate fails.

---

## H. Top-venue additions (Lanes 4–5; full checkpoint files: `2026-07-10_lane4_top_conferences.md`, `2026-07-10_lane5_top_journals_positioning.md`)

59. **AERO-HPR — 1st CVPR 2026 Workshop on Human Perception & Recognition in Aerial Surveillance**
    — ✅ (official page aero-hpr.github.io, held 2026-06-03, Denver; organizers QUT/MSU). CFP
    explicitly covers small-object detection, multi-scale, SYNTHETIC imagery, aerial person
    pipeline. → **VENUE TARGET: our story matches the CFP nearly verbatim; 2nd edition (CVPR 2027
    workshop) is a calendar-realistic home.** Field signal: top venues take aerial-human work via
    WORKSHOPS, not main conference.
60. **SAFE-Net (CVPRW 2026, AERO-HPR track)** — ✅ "Scale-Aware Feature Enhancement for Aerial
    Person Detection in Flood Disaster Imagery" (IIT Tirupati) — **the closest published 2026
    neighbor to our exact problem.** PDF not yet fetched (agent budget). → **CITE+BEAT +
    must-read at P0** (add to DERNet reading task); possible comparator row.
61. **High-Res P2 in YOLOv12 for aerial pedestrians (CVPRW 2026, AERO-HPR track)** — ✅ →
    evidence that P2-alone is STILL workshop-publishable in 2026 — i.e., our CBAM+P2 base was the
    right instinct AND our composite must out-claim exactly this (frequency gate + assignment +
    protocol) to be more than parity. CITE.
62. **NTIRE 2026 CD-FSOD challenge (CVPRW 2026, arXiv 2604.11998)** — ✅ cross-domain few-shot
    detection (source → unseen shifted target, k-shot labels; 31 active teams). → **ADOPT-FRAMING:
    cast our C2A(synthetic)→own-drone(real) lane in CD-FSOD vocabulary — turns our eval from
    "extra experiment" into a recognized problem setting**; winners' write-ups = transfer-trick
    mining list.
63. **RealDroneVision (WACV 2026 main, verified wacv.thecvf.com/virtual/2026/poster/670)** — ✅
    173k-image real drone-detection dataset (semi-automatic labeling) + architecture mods — same
    IIT Tirupati lab as SAFE-Net. → **PATTERN-PROOF: [own dataset + tuned detector] clears WACV
    MAIN. Our own-footage benchmark + FCCG composite replicates the recipe → WACV 2027 primary
    conference target.**
64. **ScaleBridge-Det (arXiv 2512.01665, venue unconfirmed)** — ✅ preprint: Routing-Enhanced
    Mixture attention (scale-expert routing) + density-guided queries; SOTA claims AI-TOD-v2. →
    CITE; conceptual support: **a gate IS a 2-expert router** — adopt their claim axis "lift tiny
    WITHOUT hurting medium/large" as an FCCG evaluation criterion (our N2 need, OOD-large).
65. **ICML venue-fit — ❌ NO-FIT confirmed** (2025/2026 accepted lists: zero applied small-object
    /aerial detection for our keywords; ICML detection wants learning-paradigm claims). ICLR 2026
    main: nothing beyond RF-DETR; ML4RS workshop = the ICLR-adjacent home. AAAI 2026/NeurIPS 2025:
    no confirmed hits beyond the known AAAI 2025 trio; nearest-shaped: EV-UAV event-camera tiny
    benchmark (arXiv 2506.23575, venue unverified). → Venue ladder for us: WACV main / CVPR-ICCV
    workshops (AERO-HPR, NTIRE) / AAAI + journals for depth. **CVPR/ICML main formally ruled out
    with receipts.**

66. **DN-TOD (Pattern Recognition 2026, DOI 10.1016/j.patcog.2026.113448, code
    ZhuHaoranEIS/DN-TOD — from the NWD/AI-TOD group)** — ✅ label-noise-robust tiny detection:
    Class-aware Label Correction + Trend-guided Learning reweighting; +4.9 AP50 under 40% noise;
    plugs into one-stage detectors. → **ADOPT-COMP CANDIDATE: directly targets C2A's paste-label
    noise (our measured ~0.615 AP ceiling) — training-time, orthogonal to FCCG modules.** Also a
    PR-novelty-pattern template: formalize an ignored data pathology, fix with training strategy.
67. **Frequency-neck crowding — journal wave (Lane 5 side-finding, hardens catalog #56):**
    ✅ AFGLFF-YOLO (JSTARS 2026, 10.1109/JSTARS.2025.3649074, adaptive frequency global-local
    fusion) · WE-YOLO (JSTARS 2026, wavelet+Mamba) · MicroDETR (PR 2026, frequency-spatial DETR)
    · FANet (RS 17(24):4066) · **SRTSOD-YOLO (RS 2025, 17(20):3414 — GATED-activation fusion neck
    ON YOLO11, +7.9 mAP50 vs own baseline)** → all CITE+DIFFERENTIATE; SRTSOD-YOLO added to the
    P0 must-read list (closest neck-mechanism neighbor on our exact base model).
68. **FFCA-YOLO (IEEE TGRS 2024, 10.1109/TGRS.2024.3363057)** — ✅ the TGRS evidence-bar
    exemplar: 3 modules + lite variant, 3 datasets, ~10+ baselines, efficiency analysis → CITE +
    use as the structural template if we aim at the TGRS tier.
69. **PR 2025-26 tiny-object cluster** — ✅ AMSF-YOLO (113303), domain-consistency optimization
    (113463 — protocol-first precedent for our sim-to-real lane), dynamic scale-aware label
    assignment + context (112449 — **competitor to our D2; must be in related work**), SADet,
    FSENet → CITE. Pattern: PR accepts UAV tiny composites ONLY when welded to a formal problem
    (label noise / assignment / domain consistency) — our leakage+paste-noise+occlusion story
    fits that mold.
70. **Journal evidence bars + quartile map (Lane 5 Parts A/B/D)** — ✅ TGRS: ≥3 components /
    3+ datasets / ~10 methods / efficiency+deployment framing. **JSTARS: 2 modules on a YOLO base
    suffices (WE-YOLO bar) — our package already exceeds it.** PR: reachable via formal-problem
    reframe. MDPI (Drones IF 4.8 / RS 4.3, 2025 JCR): fast floor. GRSL (Q1, IF 4.4): compact
    letter spin-off option (context gate alone, or the leakage audit). MVA dropped to Q3 —
    remove from list. **Verdict: JSTARS > PR(reframed) > Drones/RS floor; ceiling-raiser #1 =
    YOLOv9-e reproduction on BOTH splits (already P0/S4 in our plan).**

---

## What this audit changes (delta to lap-2 plan)
1. **FCCG-YOLO core survives** — independently validated (Gemini's BP-FFP converges on it;
   SET/CVPR-level endorsement of the frequency premise) but the generic-frequency-module slot
   filled in 2025 → the *gate/evidence-separation + person/occlusion + protocol* framing is now
   mandatory, and **DERNet (2606.23825) must be read + differentiated before S0 freeze**.
2. **A second pillar materialized that the lap-2 plan under-weighted:** sim-to-real with OUR data
   — seam probe (#48) → C2A-H harmonization (#47) → SF-UT ladder on own footage (#46) → joint
   3-set final model (#51). Cheap, evidence-backed, uses assets nobody else has (3-altitude 4K
   footage + frozen test frames + scene-disjoint protocol).
3. **Assignment upgrade sharpened** (#55): STAL min-anchor floor + SimD-style similarity inside
   TAL = concrete unclaimed slot, replaces vague "STAL-style" wording.
4. **Occlusion stretch upgraded** (#38): VE-DINO exists → cite+beat; NOMAD = ready occlusion-
   graded eval; synthesis-aware visibility supervision still unclaimed.
5. **New cheap probes queued:** seam probe (eval-only), pose-label audit (#37), DERNet full read.
6. **Rejected for good, with receipts:** hypergraph headline (#22), MuSGD (#3), NWD-as-loss
   (G2 closed, now literature-consistent #55), plain adversarial-GRL-as-novelty (#39–40),
   plug-and-play A²/CBAM-class swaps (#9, #15, #34), fabrications (#14, #19, #20, #29, #41).
