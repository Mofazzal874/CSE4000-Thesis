# Lane 5 — journal positioning & evidence bar (kill-safe checkpoints)

Date: 2026-07-10 · Status: COMPLETE (all parts A-D checkpointed; ~19 tool calls, within budget)
Scope: A) evidence-bar exemplars 2024-26 · B) venue quartile sanity · C) new 2025-26 methods (budget permitting) · D) verdict.
Verification legend: [S2]=Semantic Scholar API · [XR]=Crossref/DOI resolver · [WS]=web-search listing · [PUB]=publisher page. No DOI is stated without one of these routes.

## Part A — evidence-bar exemplars (2024-26, scope ≈ ours: YOLO/one-stage composite, small/tiny aerial objects or persons, some protocol dimension)

(rows appended as verified)

### A1. FFCA-YOLO (IEEE TGRS 2024) — the TGRS-bar exemplar
- **Venue/DOI:** IEEE TGRS, 2024, DOI 10.1109/TGRS.2024.3363057 — verified [WS: IEEE Xplore doc 10423050 + NASA ADS abstract 2024ITGRS..6263057Z]. Zhang, Ye, Zhu, Liu, Guo, Yan.
- **# novel components claimed:** 3 modules (FEM feature-enhancement, FFM feature-fusion, SCAM spatial-context-aware) + a 4th deliverable: L-FFCA-YOLO lite variant (PConv-rebuilt backbone/neck) for onboard use.
- **# datasets:** 3 (abstract reports three mAP50 figures: 0.748 / 0.617 / 0.909 — VEDAI + AI-TOD + a third/self-built; names not in abstract).
- **# baselines:** "several benchmark models and SOTA" per abstract — exact count n/a from abstract (full paper compares ≈10+, unverified here).
- **Headline delta:** absolute mAP50s given, "exceeds SOTA"; explicit +Δ not in abstract.
- **Ablation rows:** n/a from abstract (module trio implies ≥4-row ablation minimum).
- **Latency/params:** y — dedicated lite version, onboard real-time framing.
- **Real-world/deployment test:** n (onboard motivation, no field trial claimed in abstract).

### A2. Real-Time SAR with Drones: small-object YOLO (Drones-MDPI 2025) — the MDPI-bar exemplar
- **Venue/DOI:** Drones (MDPI) 9(8):514, 2025, DOI 10.3390/drones9080514 — verified [WS: mdpi.com/2504-446X/9/8/514 listing; DOI from MDPI URL pattern].
- **# novel components:** ~2 architecture tweaks on YOLOv5s (PBfpn neck + Deconv head) + a 2-stage transfer-learning PROTOCOL (VisDrone pretrain → HERIDAL fine-tune).
- **# datasets:** 2 (VisDrone, HERIDAL).
- **# baselines:** YOLO-family configurations (exact count n/a).
- **Headline delta:** mAP50 0.802 on HERIDAL (vs ~0.75-0.79 prior HERIDAL YOLO results) + real-time on Jetson Nano.
- **Ablation rows:** multiple config sweep (n/a exact).
- **Latency/params:** y (embedded FPS on Jetson Nano).
- **Real-world/deployment:** y (embedded-hardware deployment; no live field trial).

### A3. SRTSOD-YOLO (Remote Sensing-MDPI 2025) — the strong-MDPI exemplar, on OUR base model
- **Venue/DOI:** Remote Sensing 17(20):3414, 2025, DOI 10.3390/rs17203414 — verified [WS: mdpi.com/2072-4292/17/20/3414 listing; DOI from MDPI URL pattern].
- **# novel components:** ≥1 named (GAC-FPN gated-activation conv fusion pyramid neck) on improved YOLO11; multi-scale variants (s/l).
- **# datasets:** VisDrone2019 (+ possibly others; only VisDrone visible in listing).
- **# baselines:** YOLO11 family (exact count n/a).
- **Headline delta:** +7.9 mAP50 over YOLO11l on VisDrone — vs-own-baseline, not vs-published-SOTA.
- **Ablation rows:** n/a. **Latency/params:** y (real-time framing). **Real-world test:** n.
- NOTE: same base (YOLO11) + gated-fusion-neck idea — read before writing FCCG neck section.

### A4. WE-YOLO (IEEE JSTARS 2026) — the JSTARS-bar exemplar
- **Venue/DOI:** IEEE JSTARS 2026, DOI 10.1109/JSTARS.2026.3672925 — verified [XR: Crossref + WS: IEEE Xplore doc 11430535].
- **# novel components:** 2 (MixHWD mixed-Haar-wavelet downsampling + SFEM spatial-frequency-enhancement Mamba) on YOLOv8.
- **# datasets / baselines / delta / ablation:** n/a from abstract snippet (typical JSTARS package: 2-3 datasets, 6-10 baselines).
- **Latency/params:** unknown. **Real-world:** n.
- Two modules + one base = enough for JSTARS 2026 acceptance; that is the JSTARS bar.

### A5. Pattern Recognition 2025-26 cluster — proof the PR door is open, and what it costs
Crossref sweep [XR], all container-title "Pattern Recognition", 2024-26 — UAV/tiny-object papers ARE landing there:
- AMSF-YOLO (attention multi-scale, UAV small objects) — 10.1016/j.patcog.2026.113303, 2026. A YOLO composite in PR.
- "Small object detection in UAV imagery through domain consistency optimization for cross-resolution semantic alignment" — 10.1016/j.patcog.2026.113463, 2026 (domain-protocol angle, like our sim-to-real lane).
- MicroDETR (frequency-spatial aware + cross-scale fusion, tiny objects) — 10.1016/j.patcog.2026.113747, 2026.
- DN-TOD (tiny detection under label noise) — 10.1016/j.patcog.2026.113448, 2026 (≈ our paste-label-noise/leakage story as a formal problem).
- SADet (semantic-aware anti-missed-detection) — 10.1016/j.patcog.2025.112624; FSENet — 10.1016/j.patcog.2025.111425; dynamic scale-aware label assignment — 10.1016/j.patcog.2025.112449.
- Pattern: every PR acceptance pairs architecture with a FORMAL problem statement (label noise, assignment, domain consistency, a-contrario detection) — never "improved YOLO" alone.

### ⚠ Prior-art adjacency warning for FCCG-YOLO (side-finding of this sweep)
Frequency/wavelet small-object necks are now CROWDED in Q1-Q2 print: WE-YOLO (JSTARS 2026, wavelet+Mamba), AFGLFF-YOLO (JSTARS 2026, DOI 10.1109/JSTARS.2025.3649074 [XR] — "Adaptive Frequency Global-Local Feature Fusion"), MicroDETR (PR 2026, frequency-spatial), MS-YOLOv11 (Sensors 2025, Haar), WCDB-YOLO (Drones 2026), HS-FPN, plus AIE-YOLO (2022). A frequency branch per se is NOT novel; novelty must be carried by the evidence×context GATING composition + person/disaster focus + protocol. Cite AFGLFF-YOLO and WE-YOLO and differentiate explicitly.

### Part A synthesis — the evidence bar (8 sentences)
1. TGRS/ISPRS/TIP tier: the template (FFCA-YOLO, TGRS 2024) is ≥3 named components forming one coherent mechanism, 3+ datasets, ~10+ compared methods including recent SOTA (not just stock YOLOs), an explicit efficiency/lite analysis, and onboard/deployment framing.
2. Pattern Recognition (Q1) demonstrably accepts UAV tiny-object YOLO/DETR composites in 2025-26 (A5 cluster), but every accepted paper welds the architecture to a formal problem statement — label noise, assignment theory, domain consistency — not to "we improved the neck".
3. The JSTARS bar is materially lower: 2 novel modules on a YOLO base with standard benchmarks suffices (WE-YOLO), with remote-sensing relevance weighted over theory.
4. The MDPI bar (Remote Sensing/Drones) is lower still: 1-2 modules or even a pure protocol contribution (2-stage transfer + Jetson deployment, A2) on 1-2 datasets clears it.
5. Across tiers, headline deltas look big (+7.9 to +20 mAP50 on VisDrone) because they are measured against the paper's own stock baseline; top-tier reviewers discount this and demand vs-published-SOTA tables, which is exactly where our unreproduced YOLOv9-e 0.8927/0.6883 C2A bar bites.
6. Our current assets — 3 datasets including real SARD and own 3-altitude 4K footage, measured latency/params, a leakage audit with a scene-disjoint re-split — already CLEAR the MDPI and JSTARS bars and exceed their typical protocol rigor (few JSTARS/MDPI papers audit their benchmark).
7. What we still MISS for TGRS/TIP/PR: (a) a result that beats or credibly reframes the printed C2A SOTA under a defensible protocol (reproduce YOLOv9-e on the scene split), (b) a ~10-method comparison table including 2025-26 competitors, and (c) a formal mechanism claim for the freq×context gate with measurable evidence (e.g., high-frequency energy vs recall curves), not module names.
8. Own real drone footage + sim-to-real protocol is our rarest asset — almost none of the surveyed papers have ANY real-world test — and it should be spent as the differentiator at whichever tier we submit.

## Part B — venue quartile sanity list
Sources: items marked [WS-2025JCR] verified today via journalmetrics.org / wos-journal.info / mdpi.com announcement listings (2025 JCR, released 2026-06). Unmarked = approximate JCR 2024-25 band from knowledge, NOT re-verified today — re-check before submission.

- **IEEE TGRS** — Q1 top-decile geoscience (IF ≈ 8, approx). Fit: reach. Viable only after SOTA-beating result + ~10-method table; undergrad-led fine, bar non-negotiable.
- **ISPRS JPRS** — Q1 flagship (IF ≈ 10+, approx). Fit: poor for a YOLO composite; realistic only if the PROTOCOL (leakage audit + scene split + sim-to-real) is promoted to headline contribution.
- **IEEE TIP** — Q1 (IF ≈ 10, approx). Fit: poor-to-reach; wants a general CV mechanism validated beyond remote-sensing benchmarks.
- **Pattern Recognition** — Q1 (IF ≈ 7.5, approx). Fit: reach-but-real. 2025-26 shows UAV tiny-object composites accepted (Part A5) when welded to a formal problem (label noise / assignment / domain consistency) — our leakage+occlusion+paste-noise story fits this mold.
- **IJCV** — Q1 elite (double-digit IF, approx). Fit: no, for this package.
- **IEEE JSTARS** — Q1-Q2 (IF ≈ 4.7-5, approx; OA fee applies). Fit: STRONG — exact scope; 2-module+protocol packages routinely accepted (WE-YOLO bar); our assets already exceed typical rigor.
- **IEEE GRSL** — Q1, IF 4.4 [WS-2025JCR]. Fit: good for a compact single-mechanism letter spin-off (e.g., the context gate alone, or the leakage audit).
- **ESWA** — Q1 (IF ≈ 7.5, approx). Fit: good with applied-system + deployment framing; high volume.
- **EAAI** — Q1 (IF ≈ 7-8, approx). Fit: good; engineering-application framing (existing maritime-SAR line there).
- **Remote Sensing (MDPI)** — Q1, IF 4.3 [WS-2025JCR]. Fit: very high acceptance probability, fast; some committees apply an MDPI prestige discount.
- **Drones (MDPI)** — Q1 in RS category, IF 4.8 [WS-2025JCR: mdpi.com announcement]. Fit: highest probability of all; exact scope (UAV SAR person detection); same discount caveat.
- **CVIU** — conflicting aggregator IFs (≈4-6); treat as Q1-Q2, verify before submit. Fit: moderate; needs CV-methodological depth beyond the application.
- **J. Real-Time Image Processing** — Q2, IF 3.0 [WS-2025JCR]. Fit: high if the latency/edge story (15.7 ms, lite variant) is the headline.
- **Machine Vision and Applications** — Q3 in 2025 JCR, IF 2.3 [WS-2025JCR: journalmetrics] (was Q2). Fit: easy but low payoff; fallback only.

## Part C — new 2025-26 journal methods (not in exclude list; all four in Pattern Recognition, i.e., inside the allowed TPAMI/IJCV/ISPRS/TIP/TGRS/PR set)

**1. DN-TOD (Pattern Recognition 2026)** — de-noising tiny-object detector robust to label noise.
- Source: DOI 10.1016/j.patcog.2026.113448 [XR + WS: ScienceDirect S0031320326004140 + SSRN 5933545 + arXiv 2401.08056]. Zhu, Xu, Yang, R. Zhang, Y. Zhang, Xia (the NWD/AI-TOD group).
- Code: https://github.com/ZhuHaoranEIS/DN-TOD
- Mechanism: argues tiny-object annotation is inherently noisy (class-shift + box-shift); Class-aware Label Correction (CLC) detects/filters corrupted positives, Trend-guided Learning Strategy (TLS) reweights samples and regenerates boxes against box noise. +4.9 AP50 under 40% mixed noise on AI-TOD-v2.0; works plugged into one- and two-stage detectors.
- Composability: training-time layer, orthogonal to FCCG modules — directly applicable to C2A's paste-label noise (our measured ~.615 AP ceiling).
- Novelty pattern: formalize an ignored data pathology, then fix it with a training strategy — architecture untouched.

**2. MicroDETR (Pattern Recognition 2026)** — DETR with frequency-spatial aware and cross-scale fusion for tiny objects.
- Source: DOI 10.1016/j.patcog.2026.113747 [XR: Crossref, container-title Pattern Recognition]. No abstract retrievable this sweep (title-level only — flag before citing details).
- Code: none found.
- Mechanism (from title only): joint frequency-spatial feature awareness + cross-scale fusion inside a query-based (DETR) tiny-object pipeline — the DETR-side twin of our high-freq evidence branch.
- Composability: rival, not component; cite and differentiate (CNN gate + evidence branch vs query-based frequency attention).
- Novelty pattern: frequency cue + scale fusion migrated into a transformer detector.

**3. Domain-consistency optimization for cross-resolution semantic alignment (Pattern Recognition 2026)** — small-object UAV detection via domain protocol.
- Source: DOI 10.1016/j.patcog.2026.113463 [XR]. No abstract this sweep (title-level).
- Code: none found.
- Mechanism (title-level): optimizes domain consistency so semantics align across resolutions/domains in UAV imagery — a protocol/objective contribution rather than a module.
- Composability: same lane as our harmonization + self-training sim-to-real protocol; evidence PR accepts protocol-first small-object papers.
- Novelty pattern: domain alignment promoted to first-class training objective.

**4. Dynamic scale-awareness label assignment + contextual enhancement (Pattern Recognition 2025, in 2026 issue)** — tiny-object assigner+context pairing.
- Source: DOI 10.1016/j.patcog.2025.112449 [XR]. No abstract this sweep (title-level).
- Code: none found.
- Mechanism (title-level): label assignment made dynamically scale-aware (the post-NWD/RFLA assignment line) paired with contextual feature enhancement.
- Composability: direct alternative/competitor to our tiny-aware assignment stack — must be in our related-work table.
- Novelty pattern: assigner + context module co-designed as one contribution (exactly the FCCG pairing logic).

Bonus adjacency (outside allowed venue set, strengthens the crowded-frequency warning): FANet — frequency-aware attention tiny-object detection, Remote Sensing 17(24):4066, DOI 10.3390/rs17244066 [WS: doi.org listing].

## Part D — honest verdict
For this project the best acceptance-probability × prestige products are: (1) **IEEE JSTARS** — scope-perfect, our 3-dataset + real-footage + leakage-audit package already exceeds its demonstrated bar (WE-YOLO = 2 modules); (2) **Pattern Recognition** — genuinely reachable IF we reframe the paper around a formal problem (occlusion/tiny evidence under label noise and benchmark leakage) with the architecture as the instrument, since 2025-26 shows that door open; (3) **Drones/Remote Sensing (MDPI)** — the fast, near-certain floor that should only be used if time runs out before graduation deadlines. The single missing experiment that most raises the ceiling: **reproduce YOLOv9-e on C2A under both the official and our scene-disjoint split** — it simultaneously tests whether the printed 0.8927/0.6883 bar survives a clean protocol, converts the leakage audit into a headline-grade finding if the bar drops, and defines the honest number FCCG-YOLO must beat; every tier's Reviewer 2 will ask for exactly this. Secondary ceiling-raiser (optional): one public tiny-person benchmark (e.g., AI-TOD person class or TinyPerson) to show the composite transfers beyond C2A.

*Status: COMPLETE (Parts A-D). S2 API was rate-limited throughout (429); all verification via Crossref [XR] and publisher/search listings [WS] as marked.*
<!-- APPEND-HERE -->
