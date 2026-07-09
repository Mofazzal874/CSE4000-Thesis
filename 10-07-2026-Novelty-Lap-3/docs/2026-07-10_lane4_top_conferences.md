# Lane 4 — top-conference mining (kill-safe checkpoints)

Date: 2026-07-10. Budget-limited web verification (primary sources only). Scope: CVPR 2026 → ICLR/WACV/AAAI 2026 + NeurIPS 2025 → ICML venue-fit → ICCV 2025 (if budget).
Note on CVPR 2026 main-conference open access: as of 2026-07-10, topic searches did NOT surface an indexed openaccess.thecvf.com CVPR2026 list for our keywords; the strongest verified CVPR 2026 material found is the AERO-HPR workshop (official page lists accepted papers). Status noted per item.

## P1 — CVPR 2026

**1. AERO-HPR Workshop (CVPR 2026, venue itself)** — 1st Workshop on Human Perception and Recognition in Aerial Surveillance; the field's first dedicated top-venue home for exactly our problem.
- Source: "AERO-HPR: The 1st Workshop on Human Perception and Recognition in Aerial Surveillance", CVPR 2026 workshop, June 3 2026, Denver. URL: https://aero-hpr.github.io/ (fetched 2026-07-10; lists accepted papers, organizers: Kien Nguyen QUT, Arun Ross MSU, Clinton Fookes, et al.)
- Code: n/a (venue)
- Mechanism: n/a — but call-for-topics explicitly includes small object detection, multi-scale detection, low-resolution matching, SYNTHETIC IMAGERY, and the full aerial person pipeline (detection → re-ID). Ties to IARPA BRIAR / AG-ReID / DetReIDX ecosystems.
- YOLO11m composability: n/a — this is a publication TARGET: our tiny-occluded-human + synthetic-to-real story matches the CFP almost verbatim; 2nd edition (CVPR 2027) or similar WACV/ICCV workshops are realistic homes.
- Novelty pattern: venue signal — top conferences now accept aerial-human perception as a first-class applied track via workshops, not main conference.

**2. SAFE-Net (CVPRW 2026, AERO-HPR proceedings track)** — Scale-aware feature enhancement for aerial person detection in FLOOD DISASTER imagery — the closest published neighbor to our exact problem found at any 2026 venue.
- Source: "SAFE-Net: Scale-Aware Feature Enhancement for Aerial Person Detection in Flood Disaster Imagery", Arun Kumar S, Komuravelli Prashanth, J. Ganesh Mouli, Gorthi R. K. Sai Subrahmanyam (IIT Tirupati group). Verified via official workshop page https://aero-hpr.github.io/ (proceedings track = will appear in CVF CVPRW 2026 open access). arXiv/CVF PDF not yet located (budget).
- Code: not checked (budget)
- Mechanism: per title/track: scale-aware feature enhancement targeted at small persons in flood scenes (disaster-domain aerial person detection). PDF not yet fetched — mechanism details unverified; treat as "must-read + must-cite + possible baseline to beat".
- YOLO11m composability @640/16GB: unknown until PDF read, but "scale-aware enhancement" modules are typically neck-level and composable; more importantly it is a COMPARATOR, not a component.
- Novelty pattern: narrow, named disaster domain (flood) + scale-awareness claim + workshop proceedings — shows a focused applied claim on disaster person detection clears a CVPR-workshop bar without giant benchmarks.

**3. High-Res P2 YOLOv12 pedestrian detection (CVPRW 2026, AERO-HPR proceedings track)** — P2 high-resolution feature integration in YOLOv12 for aerial pedestrian detection; direct 2026 parallel of our lap-0 CBAM+P2 design.
- Source: "Enhancing Aerial Pedestrian Detection via High-Resolution P2 Feature Integration in YOLOv12", Sukesh Babu V S, Rahul Raman, Sambit Bakshi. Verified via https://aero-hpr.github.io/ (proceedings track → CVF CVPRW 2026).
- Code: not checked (budget)
- Mechanism: adds/exploits the stride-4 P2 pyramid level in YOLOv12 for small aerial pedestrians (same family of intervention as our CBAM+P2: recover high-res detail the P3-P5 neck discards).
- YOLO11m composability @640/16GB: trivially — we already run P2 on YOLO11m; this paper is evidence P2-for-aerial-humans alone is STILL publishable at workshop level in 2026, i.e., our composite must go beyond it (frequency branch + gating + assignment) to claim more than parity.
- Novelty pattern: single well-motivated architectural lever (P2) + a specific aerial-human niche = workshop paper; a MAIN-conference claim needs a composite or a new axis (their being at a workshop confirms the bar).

STATUS NOTE (P1): openaccess.thecvf.com CVPR2026 IS live (main conf published 2026-05-23; workshops menu at /CVPR2026_workshops/menu). But domain-restricted searches surfaced NO main-conference CVPR 2026 paper on tiny/aerial-human detection for our keywords — consistent with lap-2 finding that this subfield publishes at AAAI/WACV/workshops, not CVPR main.

**4. NTIRE 2026 CD-FSOD Challenge (CVPRW 2026)** — 2nd Cross-Domain Few-Shot Object Detection challenge report; the closest CVPR-2026-official thing to our synthetic→real transfer axis.
- Source: "The Second Challenge on Cross-Domain Few-Shot Object Detection at NTIRE 2026: Methods and Results", NTIRE workshop @ CVPR 2026. arXiv:2604.11998 (fetched: confirms "accepted by CVPRW 26 @ NTIRE"; 128 registrants, 31 active teams, 19 valid finals).
- Code: challenge GitHub linked in paper (URL not extracted — budget); 1st edition (NTIRE 2025) repo is lqy58123/NTIRE2025_CDFSOD-style public benchmark (not re-verified).
- Mechanism: benchmark/challenge on "detecting objects in unseen target domains under limited annotation" — i.e., train on source domain, adapt to visually-shifted targets with k-shot labels. (1st-edition CD-FSOD benchmark included aerial DIOR as a target; 2026 target list not re-verified from abstract.)
- YOLO11m composability @640/16GB: not a module — but its protocol (source→shifted-target, k-shot) is a citable template for our C2A(synthetic)→own-drone(real) evaluation lane; winning-method writeups = mining list for transfer tricks.
- Novelty pattern: cross-DOMAIN detection under label scarcity is an active, contested CVPR-workshop axis in 2026 — framing our synthetic→real drone protocol in CD-FSOD vocabulary buys instant novelty legibility.

**5. Tri-Modal Fusion Transformers for UAV-based Object Detection (CVPR 2026 — UNCONFIRMED TIER)** — multi-modal (tri-modal) fusion transformer for UAV detection.
- Source: attributed to "Iaboni and Abichandani, CVPR 2026" only in a search-result summary; no openaccess.thecvf.com URL captured. Queries tried: CVF-domain-restricted "CVPR2026 tiny/small/aerial/UAV object detection". Treat as UNVERIFIED (could be workshop track) until CVF page is fetched.
- Code: not checked (budget)
- Mechanism (from title only): fuses three modalities (likely RGB + IR + depth/event) in a transformer detector for UAV platforms.
- YOLO11m composability: low — multi-modal input contradicts our single-RGB 4K pipeline; relevance is as a "what CVPR-main accepts for UAV detection" datapoint: NEW SENSING AXIS, not better RGB necks.
- Novelty pattern: main-conference UAV detection slots appear to demand a new input/sensing modality or paradigm, not incremental RGB architecture — supports our decision to target strong applied venues instead.

## P2 — ICLR 2026 / WACV 2026 / AAAI 2026 / NeurIPS 2025

STATUS NOTE (ICLR 2026): OpenReview-restricted searches found NO accepted ICLR 2026 main-conf paper on tiny/aerial object detection (RF-DETR, already excluded, remains the known detection entry). Found instead: 4th ML4RS (Machine Learning for Remote Sensing) workshop @ ICLR 2026 — the realistic ICLR-adjacent home for aerial detection work. One candidate ("Query Optimization Detection Transformer for Small Objects in Remote Sensing Images", openreview.net/forum?id=T6hhDEnAoo) is UNVERIFIABLE (OpenReview bot-wall blocked fetch; venue/decision unknown) — do not cite without manual check.

**6. RealDroneVision (WACV 2026)** — large-scale real-world small-object DRONE-detection dataset + architecture advancements, presented as a WACV 2026 poster.
- Source: "RealDroneVision: Dataset and Architecture Advancements for Small-Object Drone Detection", Arun Kumar Sivapuram, Pranav Peddinti, Harish Puppala, Komuravelli Prashanth, Jaladi Sri Harsha, Gorthi Subrahmanyam. Verified on the OFFICIAL WACV virtual site: https://wacv.thecvf.com/virtual/2026/poster/670 (poster session 2026-03-10).
- Code: not checked (budget)
- Mechanism: joint dataset+method contribution — 173,023-image real-world drone-detection dataset built via a semi-automatic labeling pipeline, plus architecture modifications for small targets (drones-as-targets, i.e., small-object regime like ours but different class). NOTE: same IIT Tirupati group (Gorthi Subrahmanyam, K. Prashanth) as SAFE-Net (item 2) — one lab landing WACV main + CVPRW in the same year with dataset+architecture combos on small aerial objects.
- YOLO11m composability @640/16GB: class differs (drones vs humans) so modules are secondary; the transferable part is the CONTRIBUTION SHAPE (own dataset + tuned detector) which we replicate with own 3-altitude drone-human footage + composite YOLO11m.
- Novelty pattern: dataset + architecture co-contribution clears WACV main conference — strong precedent for our "own drone test set + FCCG-style composite" bundle.

**7. ScaleBridge-Det (arXiv 2512.01665 — VENUE UNCONFIRMED, AAAI-2026-timing preprint)** — mixture-of-scale-experts routing + density-guided dynamic queries to balance tiny vs general objects in one detector.
- Source: "Bridging the Scale Gap: Balanced Tiny and General Object Detection in Remote Sensing Imagery", Zhicheng Zhao, Yin Huang, Lingma Sun, Chenglong Li, Jin Tang. arXiv:2512.01665 (submitted 2025-12-01; abs page fetched — NO venue note; treat as preprint).
- Code: none found (not in abstract)
- Mechanism: (a) Routing-Enhanced Mixture attention (REM): adaptive routing dynamically selects/fuses scale-specific EXPERT features; (b) Density-Guided dynamic Query (DGQ): predicts object density to adjust query positions and counts. Claims SOTA on AI-TOD-V2 + DTOD and "superior cross-domain robustness on VisDrone" (numbers not in abstract).
- YOLO11m composability @640/16GB: DGQ is DETR-query-specific (not applicable); the REM idea (route features to scale-specialized branches instead of uniform FPN mixing) IS translatable to a YOLO neck and rhymes with our high-freq-branch × context-gate design — a gate IS a 2-expert router.
- Novelty pattern: "balance tiny AND general objects" framing (don't sacrifice one scale for the other) + density-adaptive compute — a claim axis we could adopt: our gate should show it does NOT hurt medium/large humans while lifting tiny ones.

STATUS NOTE (AAAI 2026 / NeurIPS 2025): budgeted searches surfaced NO confirmed AAAI 2026 or NeurIPS 2025 accepted paper on tiny/aerial-human detection beyond the already-excluded AAAI 2025 trio (FBRT-YOLO, RemDet, HS-FPN). Queries tried: "AAAI 2026 tiny/small object detection UAV (arXiv + general)", "NeurIPS 2025 small object detection aerial/UAV/benchmark". Nearest NeurIPS-shaped candidate: "Event-based Tiny Object Detection: A Benchmark Dataset and Baseline" (EV-UAV, arXiv:2506.23575, targets avg 6.8x5.4 px — event camera, benchmark+baseline pattern; venue unverified). Manual follow-up: official AAAI 2026 proceedings (ojs.aaai.org) and NeurIPS 2025 D&B track list.

## P3 — ICML venue-fit verdict

ICML is a NO-FIT for this work. Queries tried (2026-07-10): "ICML 2025 / ICML 2026 accepted paper object detection small objects aerial applied" (+ the AAAI/NeurIPS sweeps above, which also index ICML pages). The ICML 2026 accepted list (~6,634 papers per third-party aggregator aiconfpaper.com — not primary, but directionally reliable) surfaced ZERO applied small-object/aerial detection papers for our keywords; ICML detection papers historically require a learning-paradigm claim (e.g., new training theory), not an applied architecture. Conclusion: do not target ICML; the realistic top-venue ladder for this thesis is WACV main / AAAI main / CVPR-ICCV workshops (AERO-HPR, NTIRE, Anti-UAV, ML4RS), with journals (ISPRS, TGRS, TIP) as the depth route.

## P4 — ICCV 2025

Not mined — budget exhausted at P3 (per mission priority order). Note from earlier lanes: Dual-Rate Dynamic Teacher (ICCV25) already covered in exclude list.

## Steal-the-pattern verdict

1. **SAFE-Net (CVPRW 2026, AERO-HPR)** — claim structure: [named disaster niche: flood] + [scale-aware feature enhancement] + [aerial person detection]; our composite matches this shape but adds a measured-needs-driven design (N1-N8) and a synthetic→real protocol, i.e., we can out-claim the nearest 2026 neighbor at the same venue family.
2. **RealDroneVision (WACV 2026)** — claim structure: [own large real dataset via semi-automatic pipeline] + [architecture advancements] in ONE paper; bundling our 3-altitude drone-human test set with the FCCG composite is the proven WACV-main recipe, and WACV 2027 is the right calendar target.
3. **NTIRE 2026 CD-FSOD (CVPRW 2026)** — claim structure: [source domain → unseen shifted target under label scarcity] as a first-class evaluation protocol; recasting C2A(synthetic)→own-drone(real) in cross-domain vocabulary turns our eval lane from "extra experiment" into a recognized problem setting reviewers already value.
