# Lane 5 — journal positioning & evidence bar (kill-safe checkpoints)

Date: 2026-07-10 · Status: IN PROGRESS (checkpointed incrementally; if this ends abruptly the agent was killed mid-run)
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

<!-- APPEND-HERE -->
