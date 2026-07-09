# Web-verification log — 2026-07-10 (raw agent reports, traceability layer)

Five web-search agent lanes (WebSearch + WebFetch, primary sources only, repo URLs
fetch-verified). This file preserves Lanes 1–3 nearly verbatim so every verdict in the idea
catalog and ranking traces to a source. Lanes 4–5 (user task 2: top-venue mining) write their own
kill-safe checkpoint files after a quota incident killed their first spawns:
- **Lane 1 — YOLO lineage & official components** ✅ (below)
- **Lane 2 — tiny/occluded aerial-human SOTA + mechanism prior art** ✅ (below)
- **Lane 3 — sim-to-real / domain adaptation / data strategy** ✅ (below)
- **Lane 4 — top-conference mining (CVPR/ICLR/WACV/AAAI 2026, NeurIPS 2025, ICML fit)** ✅ →
  `2026-07-10_lane4_top_conferences.md` (key: AERO-HPR CVPRW 2026 venue; SAFE-Net = closest 2026
  neighbor; RealDroneVision WACV-main pattern; CD-FSOD framing; ICML no-fit)
- **Lane 5 — top-journal methods + evidence-bar/positioning analysis** ✅ →
  `2026-07-10_lane5_top_journals_positioning.md` (key: JSTARS bar = 2 modules, already cleared;
  PR door open via formal-problem framing; TGRS bar = FFCA-YOLO template; frequency-neck crowding
  confirmed in journals too — AFGLFF/WE-YOLO/SRTSOD-YOLO/MicroDETR/FANet; DN-TOD = label-noise
  component candidate with code; ceiling-raiser #1 = YOLOv9-e repro on both splits)

---

## LANE 1 — YOLO lineage & official components (completed 2026-07-10)

**Provenance:** every Source/Code URL live-fetched 2026-07-10 unless noted.

**1. YOLOv10 (NMS-free dual assignments) — VERIFIED**
- Source: "YOLOv10: Real-Time End-to-End Object Detection", Wang et al. (Tsinghua), NeurIPS 2024, arXiv 2405.14458 — https://arxiv.org/abs/2405.14458
- Code: https://github.com/THU-MIG/yolov10 (11.3k stars)
- Mechanism: one-to-many head (dense TAL supervision) + one-to-one head trained with **consistent dual assignments** (shared matching metric); deploy o2o only → no NMS. Plus efficiency package (lightweight cls head, spatial-channel decoupled downsampling, rank-guided blocks, large-kernel conv + PSA). YOLOv10-B: −46% latency, −25% params vs YOLOv9-C.
- Fit: the lineage YOLO26 productized. o2o = sparser supervision (1 positive/GT) → historically hurts small-object recall unless compensated (DEIM Dense-O2O, YOLO26 STAL exist to fix exactly this). Composable training principle, not a bolt-on.

**2. YOLOv12 (attention-centric) — VERIFIED; Medium claims FABRICATED**
- Source: "YOLOv12: Attention-Centric Real-Time Object Detectors", Tian, Ye, Doermann, arXiv 2502.12524 (repo badges NeurIPS 2025) — https://arxiv.org/abs/2502.12524
- Code: https://github.com/sunsmarterjie/yolov12 ("Turbo" default since 2025-03-09)
- Mechanism (full text verified): **Area Attention (A2)** = split (H,W) into l=4 vertical/horizontal segments, plain attention per area (2n²hd → ½n²hd, no complex ops). **R-ELAN** = block-level residual shortcuts with small scaling (0.01 for large models) + bottleneck-style aggregation. FlashAttention adopted (~0.3–0.4 ms saved on N/S). **MLP ratio 4 → 1.2** (2.0 for N/S/M). Positional encodings removed; **7×7 large separable conv "position perceiver"**.
- Fabrication check: "transformer-enhanced detection head", "Ghost", "Swin" appear NOWHERE in paper/repo. Head is standard YOLO conv head. Confirmed fabrications from a Medium post.
- Fit: A2/R-ELAN/thin-FFN/conv-position-perceiver = composable principles. As a base model: Ultralytics docs warn of training instability + memory; authors say the ultralytics port is inefficient vs their repo. YOLO12m COCO 52.5 (+1.0 over YOLO11m, −3% speed). 4070 Ti S (Ada) is FlashAttention-eligible.

**3. YOLOv13 (hypergraph) — core VERIFIED; "YOLO-TCM" & "Dynamic Task Routing" FABRICATED**
- Source: "YOLOv13: Real-Time Object Detection with Hypergraph-Enhanced Adaptive Visual Perception", Lei et al., arXiv 2506.17733 (v2 2025-09-05) — https://arxiv.org/abs/2506.17733
- Code: https://github.com/iMoonLab/yolov13 (standalone Ultralytics FORK — not in pip package)
- Mechanism: **HyperACE** = multi-scale pixels as hypergraph vertices, LEARNABLE hyperedge construction, linear-complexity high-order message passing across scales. **FullPAD** = HyperACE output redistributed via 3 tunnels (backbone→neck, within-neck, neck→head). **DS blocks** replace large-kernel convs to pay for it. COCO: N 41.6 / S 48.0 / L 53.4 / X 54.8 (NO M variant).
- Fabrication check: "Temporal Context Module"/"TCM"/"Dynamic Task Routing"/"DTR" appear NOWHERE in paper or repo. YOLOv13 is single-frame. Confirmed hallucinations.
- Fit: high-order cross-scale correlation is relevant to occlusion (part/context evidence aggregation); port into YOLO11m feasible (it's an ultralytics fork). But published → component to cite/compare, not novelty.

**4. Hyper-YOLO — VERIFIED**
- Source: "Hyper-YOLO: When Visual Object Detection Meets Hypergraph Computation", Feng et al., IEEE TPAMI 2025, arXiv 2408.04804 — https://arxiv.org/abs/2408.04804
- Code: https://github.com/iMoonLab/Hyper-YOLO (224 stars, YOLOv8-based; v1.1 repo exists)
- Mechanism: HGC-SCS (semantic-space hypergraph, distance-threshold hyperedges, propagate, scatter back); MANet backbone; HyperC2Net cross-level+cross-position hypergraph neck over 5 scales. COCO: N 41.8 / S 48.0 / M 52.0 (33.8M params — heavy).
- Fit: citable origin of hypergraph necks in YOLO; any cross-scale context gating we build must cite Hyper-YOLO + YOLOv13 and differentiate.

**5. YOLO26 — VERIFIED (released & trainable)**
- Sources: https://docs.ultralytics.com/models/yolo26/ ; official paper "Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models", Jocher, Qiu, Liu, Lyu, Akyon, Kalfaoglu, arXiv 2606.03748 (2026-06-02) — https://arxiv.org/abs/2606.03748 ; blog https://www.ultralytics.com/blog/how-ultralytics-yolo26-trains-smarter-with-progloss-stal-and-musgd ; release discussion https://github.com/orgs/ultralytics/discussions/22214
- (a) NMS-free: native end-to-end by default, dual-head (YOLOv10 lineage), inference = o2o head, `iou` arg ignored; `end2end=False` available at export.
- (b) DFL removed: plain regression head, unconstrained range, better export.
- (c) MuSGD: per-parameter-group hybrid Muon+SGD (groups undisclosed).
- (d) **STAL: only official mechanic published anywhere = "enforces a minimum of four anchor assignments for objects smaller than 8 pixels"** (official blog). Docs/paper: "guarantees positive coverage for small objects".
- (e) ProgLoss: progressively shifts loss weight from o2m head (early: dense supervision/recall) toward o2o inference head (late: match deploy). ⚠ Third-party arXiv 2601.12882 (single author, KIIT) prints equations for STAL/ProgLoss/MuSGD — its ProgLoss (cls→box) CONTRADICTS the official o2m→o2o description → likely author reconstruction; DO NOT cite as Ultralytics' math. Ground truth = grep local v8.4.x site-packages on PC-1.
- (f) CPU: "up to 43% faster CPU ONNX (26n vs 11n)" — official.
- (g) Release: shipped ultralytics **v8.4.0**, confirmed by 2026-01-23; current ~v8.4.82 (Jul 2026). Drop-in trainable.
- (h) COCO mAP50-95: 26 n/s/m/l/x = 40.9/48.6/**53.1**/55.0/57.5 vs YOLO11m 51.5 → +1.6 at m.
- (i) License: AGPL-3.0 + Enterprise.
- Fit: STAL (min-4 anchors <8 px) targets exactly our 34.5%-<8px regime and is in our toolchain via v8.4.x retrain. Independent aerial eval (2605.24831) shows YOLO26 ≈ YOLOv8 on VisDrone → assignment tweaks alone don't close the aerial tiny gap → supports composite-architecture thesis; YOLO26m = mandatory baseline row. MuSGD stays falsified locally (train 26m with pinned AdamW).

**6. Muon lineage — VERIFIED**
- Code: https://github.com/KellerJordan/Muon (2.7k stars). "Muon is Scalable for LLM Training", Moonshot AI, arXiv 2502.16982 (Moonlight; ~2× compute efficiency vs AdamW).
- Mechanism: momentum matrix orthogonalized by Newton-Schulz; applies only to 2D hidden weight matrices (embeddings/heads/gains stay AdamW/SGD). MuSGD = per-group hybrid with SGD. Lineage verified; our local divergence stands.

**7. YOLOE / "YOLOE-26" — YOLOE VERIFIED; "YOLOE-26" GARBLED (not official)**
- Source: "YOLOE: Real-Time Seeing Anything", THU-MIG, ICCV 2025, arXiv 2503.07465 — https://arxiv.org/abs/2503.07465
- Code: https://github.com/THU-MIG/yoloe (also native in ultralytics docs)
- Mechanism: open-vocab detect+segment; **RepRTA** (re-parameterizable region-text alignment — aux branch folds away at deploy, zero overhead), SAVPE (visual prompts), LRPC (prompt-free lazy region-prompt contrast). +3.5 AP LVIS over YOLO-Worldv2 at 3× less training.
- "YOLOE-26": no such official model (repo has YOLOE-v8-S/M/L, YOLOE-11-S/M/L). A third-party mashup paper exists: Sapkota & Karkee, arXiv 2602.00168 (2026-01-29).
- Fit: open-vocab orthogonal to our problem; **RepRTA's "train-time aux branch that re-parameterizes to zero deploy cost" is a pattern worth stealing** for any train-only evidence branch.

**8. RT-DETR / RT-DETRv2 — VERIFIED**
- Sources: arXiv 2304.08069 (CVPR 2024) ; RT-DETRv2 arXiv 2407.17140. In pip ultralytics.
- Mechanism: hybrid encoder = intra-scale attention (AIFI, P5 only) + cross-scale CNN fusion (CCFF); uncertainty-minimal query selection; NMS-free by construction. 53.1/54.3 AP @108/74 FPS T4. v2: per-scale deformable sampling counts, discrete sampling (no grid_sample), dynamic aug.
- Small objects: NEITHER paper claims small-object strength; UAV literature treats vanilla RT-DETR as weak on UAV tiny objects (UAV-DETR improves it +3.1 AP on VisDrone via frequency modules).
- Fit: expensive baseline on 16 GB; the decouple-intra/cross-scale idea is composable.

**9. D-FINE / DEIM / RF-DETR — all VERIFIED**
- **D-FINE**: ICLR 2025 Spotlight, arXiv 2410.13842; code https://github.com/Peterande/D-FINE (3.2k stars, Apache-2.0). **FDR**: box regression as iterative refinement of per-edge probability distributions (uncertainty-aware); **GO-LSD**: self-distill final-layer distributions into shallow layers. COCO M 52.3 @19M/57G. Irony vs YOLO26 (which deleted DFL): distributional localization is GOOD for tiny-object localization noise. Composable loss/head principle.
- **DEIM**: CVPR 2025, arXiv 2412.04234; code https://github.com/ShihuaHuang95/DEIM. **Dense O2O** (densify one-to-one supervision via synthesized targets) + **MAL** (matchability-aware loss). DEIM-D-FINE-X 56.5 AP @78 FPS. DEIMv2 released 2025-09-26 (8 sizes). Dense-O2O = DETR-side mirror of STAL; citable for our loss stack.
- **RF-DETR**: code https://github.com/roboflow/rf-detr (8.4k stars); paper arXiv 2511.09554, ICLR 2026. DINOv2 backbone + LW-DETR lineage + weight-sharing NAS. RF-DETR-L 56.5 AP @6.8 ms; XL/2XL under PML-1.0 license (base Apache-2.0). Pitch = domain adaptability from DINOv2.
- Fit: 2025/26 non-YOLO frontier for our comparison table; D-FINE-M (19M) is the same weight class as our 19.6M CBAM+P2.

**10. "EFEM-YOLO" (photovoltaic defects) — UNFINDABLE as a named model**
- Queries tried: `"EFEM-YOLO" photovoltaic cell defect detection` · `"EFEM" YOLO "edge feature enhancement" solar OR photovoltaic OR EL defect detection 2025` · bare `"EFEM-YOLO"`.
- EFEM exists only as a module name inside other models (ESL-YOLO RS'24, RFCS-YOLO Sensors'25, a Pest-YOLO variant). No "EFEM-YOLO" paper. Consistent with search-AI mashup. Zero relevance to us regardless.

**11. StreamYOLO — VERIFIED (context only)**
- Source: CVPR 2022 Oral, arXiv 2203.12338 (Megvii). Streaming-perception sAP metric; Dual-Flow Perception + Trend-Aware Loss; +4.9 sAP Argoverse-HD.
- Fit: the REAL temporal-context YOLO line (video streams, paired-frame training) — unrelated to YOLOv13; out of scope for our single-frame protocol.

**12. Native pip-ultralytics runnability (docs fetched)**
- YOLOv10 ✅ native · YOLO12 ✅ native (instability/memory caveats; authors prefer own repo) · **YOLOv13 ❌ (standalone fork — separate venv if benchmarked)** · **YOLO26 ✅ native flagship (v8.4.0+, current v8.4.82)** · YOLOE, RT-DETR also native.

### Lane-1 Extras (all fetched)
1. **UAV-DETR** — arXiv 2501.01855: frequency-enhanced multi-scale fusion + frequency-focused downsampling on RT-DETR; VisDrone 26.7→29.8 AP (R18). Published precedent that HIGH-FREQUENCY PRESERVATION is the active ingredient for UAV tiny objects (in DETRs) — supports FCCG premise, differentiation = gated HF evidence in YOLO CNN pipeline.
2. **DERNet "From Spatial to Spectral"** — arXiv 2606.23825 (2026-06-22, 18 days old). Decompose-Enhance-Reconstruct frequency operator: Wavelet-Difference Gate + Log-Gabor Enhancer + Frequency-Driven Head across backbone/neck/head; evaluated VisDrone/UAVDT/**TinyPerson**/DOTAv1; beats YOLOv11 at ~1/6 params. **CLOSEST PUBLISHED NEIGHBOR TO FCCG-YOLO. Read in full before freezing S0–S5.** Simultaneously our strongest supporting citation and biggest novelty-overlap threat. Remaining differentiators: large-kernel context GATE (they enhance, we veto), occlusion loss stack, disaster-domain + sim-to-real protocol, human-specific evidence.
3. **YOLO26 vs YOLOv8 aerial benchmark** — Oguine et al., arXiv 2605.24831: VisDrone (75% objects <2000 px²): YOLO26-x 0.224 vs YOLOv8-x 0.214 mAP50-95 — "minimal gap", both "struggle significantly". STAL/NMS-free alone ≠ aerial tiny solution → motivation paragraph for composite architecture; YOLO26m baseline row is cheap insurance.
4. **Psych-Occlusion / Psych-ER** — arXiv 2412.05553 (NOMAD aerial SAR dataset): human psychophysics (accuracy vs distance×occlusion) reweights detection loss; improves occluded-person detection at range (RetinaNet). Citable prior art for occlusion-aware loss WEIGHTING (perceptual-difficulty, ≠ NWD).
5. **DEIMv2** — released 2025-09-26 (DEIM repo): 8 sizes down to "Atto"; with D-FINE-M defines the non-YOLO efficiency frontier reviewers will expect in our table.

**Lane-1 takeaways:** (i) fabrications resolved: v12 Ghost/Swin/transformer-head ✗, v13 TCM/DTR ✗, YOLOE-26-as-official ✗ (only 3rd-party 2602.00168), EFEM-YOLO ✗. (ii) Only official STAL mechanic anywhere: "min 4 anchor assignments for <8 px objects"; equations in 2601.12882 are third-party reconstructions contradicting the official ProgLoss description — grep local v8.4.x for ground truth. (iii) YOLO26 lives in OUR toolchain → yolo26m-AdamW retrain = near-mandatory baseline row (already anticipated as S4 in lap-2 plan).

---

## LANE 2 — tiny/occluded aerial-human SOTA + mechanism prior art (completed 2026-07-10)

**Provenance:** live fetches of arXiv abs/html, publisher pages, Semantic Scholar API, GitHub (all repos fetch-confirmed 2026-07-10). MDPI full texts / IEEE Xplore / some CVF PDFs blocked fetching (403) — flagged where relevant.

**1. C2A dataset paper — VERIFIED**
- "UAV-Enhanced Combination to Application: … Human Detection Dataset for Disaster Scenarios", Nihal, Yen, Itoyama, Nakadai; **ICPR 2024** (LNCS, DOI 10.1007/978-3-031-78341-8_10); arXiv 2408.04922. Code/data: https://github.com/Ragib-Amin-Nihal/C2A ; Kaggle rgbnihal/c2a-dataset.
- (a) Table 4 confirmed EXACTLY: **YOLOv9-e 0.8927 mAP50 / 0.6883 mAP50-95** (best in paper); YOLOv9-c 0.7996/0.5562; YOLOv5 0.8080/0.4920; Cascade R-CNN 0.7350/0.4860; Faster R-CNN 0.6340/0.3656. Split methodology NOT documented → numbers are on the official (leaky, per our audit) split. Our YOLOv9-e reproduction under our protocol remains the only way to make 0.6883 comparable.
- (b) Classes: standard annotations = **single "person"**; release ALSO contains a folder **"All labels with Pose info" with 5 pose categories (bent/kneeling/lying/sitting/upright), YOLO + COCO formats** → pose-prior supervision is FEASIBLE with official labels (noise inherited from paste synthesis). No C2A follow-up has used them.
- (c) Native sizes 123×152 … 5184×3456; ~50% of images in the 322–600 px width band.

**2. 2025–26 papers training/evaluating on C2A — VERIFIED, field is THIN (important)**
- Semantic Scholar: 13 citing works; only ONE verifiable C2A benchmark number beyond the original: **LightSeek-YOLO** (Mathematics 13(19):3231, 2025, DOI 10.3390/math13193231): YOLOv11n-based lightweight (HGNetV2, Seek-DS, Seek-DH); **C2A AP_small 0.478**; COCO 0.473; 571 FPS. Well below our CBAM+P2 AP_small 0.6156.
- Citing-but-unverifiable (IEEE paywalled, likely low-tier): ICSEDIS 2026 multimodal YOLOv11+terrain; RCSM 2025 AIoT victim detection; INSPECT 2025 comparative; DICCT 2025 transfer learning; AICT 2025 optimization; 1 IRJMETS PDF (non-peer-reviewed).
- Related, NOT C2A-benchmark: Applied System Innovation 9(1):6 (Dec 2025) flood-scene YOLO12 mAP50 0.95 on a custom 4-source set; VE-DINO cites C2A but trains COCO+custom.
- **Bottom line: NOBODY has published a number beating YOLOv9-e 0.8927/0.6883 on official C2A. The bar is the original paper's. Comparison field nearly empty → good for a claim; reviewers will demand our YOLOv9-e repro.**

**3. SARD/HERIDAL/WiSAR SOTA 2024–26 — VERIFIED (fragmented, little code)**
- **VTSaRNet** — IEEE JSTARS 18:5082 (2025), DOI 10.1109/JSTARS.2025.3526995: Union Transformer RGB+thermal + ISCP aug; mAP50 98.73/mAP50-95 73.98 on own VTSaR set; also runs HERIDAL. No code.
- Drones 9(8):514 (2025, open PDF): YOLOv5s-PBfpn-Deconv, VisDrone→HERIDAL transfer, **HERIDAL mAP50 0.802**, Jetson real-time. No code.
- APD survey: J. Remote Sensing 2025, DOI 10.34133/remotesensing.0474 (403-blocked; confirmed via DOAJ).
- Context: YOLOv5L HERIDAL 0.834 (JEST 2024); classic SARD Faster R-CNN AP50 90.8. Our joint model SARD 0.917 ≥ published numbers (SARD test 96% near-dupes = soft leaderboard).

**4. Tiny-object assignment vs loss — VERIFIED; consensus = ASSIGNMENT-level**
- **NWD** arXiv 2110.13389: biggest gains when replacing IoU in ASSIGNMENT, not loss. **RFLA** ECCV 2022, arXiv 2208.08738, code https://github.com/Chasel-Tsui/mmdet-rfla (303★): receptive-field Gaussian + hierarchical assignment, ~24.8 AP AI-TOD. **DotD** CVPRW 2021 (EarthVision). **NWD-RKA** ISPRS JPRS 190 (2022), arXiv 2206.13996, code Chasel-Tsui/mmdet-aitod. **SimD** IROS 2024, arXiv 2407.02394, code https://github.com/cszzshi/SimD: adaptive location+shape similarity for assignment/NMS, **+4.1 AP_very-tiny** over SOTA on AI-TOD.
- AI-TOD progression 2021→24 (DotD→NWD-RKA→RFLA→DCFL→SimD) is ENTIRELY assignment-side; no 2024–26 work claims a loss-only fix for <16 px. **Supports our G2 negative (NWD-as-loss didn't scale).** Composable unclaimed slot: **SimD/RFLA-style Gaussian/similarity term INSIDE Ultralytics TaskAlignedAssigner for YOLO11 on aerial persons — nobody has published this.**

**5. Frequency-domain prior art — VERIFIED; space is CROWDED in 2025/26**
- **FreqFusion** TPAMI 2024, arXiv 2408.12879, code https://github.com/Linwei-Chen/FreqFusion (501★): ALPF+offset+AHPF at fusion/upsampling. NOT: real-time YOLO, tiny objects, UAV.
- **FcaNet** ICCV 2021, arXiv 2012.11879, code cfzd/FcaNet: DCT channel attention in backbone. NOT: spatial HF evidence, neck, tiny.
- **SET** — **CVPR 2025**, "Spectral Enhancement for Tiny Object Detection" (CVF html verified): hierarchical background smoothing + adversarial perturbation injection; **+3.2 AP over RFLA on AI-TOD**. Training-side, two-stage/assignment methods. KEY citation: CVPR-level proof that tiny objects live in frequency bands. NOT: real-time YOLO, fused HF branch.
- **UAV-DETR** arXiv 2501.01855 (code archived June 2026): frequency-enhanced fusion + frequency-focused downsampling in RT-DETR; VisDrone +3.1 AP. NOT: gating by context, not YOLO-neck. (Name-clash warning: MDPI Sensors 25(15):4582 "UAV-DETR" is a DIFFERENT paper.)
- **Freq-DETR** ESWA 2025, DOI 10.1016/j.eswa.2025.129710: dual-branch spatial-frequency conv + HF/LF decoupled interaction; +4.9 mAP50 VisDrone over RT-DETR. No code.
- Also: EFSI-DETR arXiv 2601.18597 (+1.6 AP VisDrone, 188 FPS); **UFO-DETR arXiv 2602.22712 (DynFreq-C3 + LSKNet backbone — nearest existing freq×large-kernel combo, but DETR, module-level, NO gating)**; FMC-DETR arXiv 2509.23056; AUHF-DETR RS 17(11):1920 (WTConv); LF-DETR RS 18(3):531 (Laplacian, aerial RGB-IR pedestrians); wavelet-YOLO swaps: WCDB-YOLO Drones 10(3):155 (2026), MS-YOLOv11 Sensors 25(19):6008, **DERNet arXiv 2606.23825** (wavelet+Log-Gabor triplet, TinyPerson/VisDrone, no code).
- **Gap we can still occupy: (i) explicit high-frequency EVIDENCE branch (ii) GATED by learned large-kernel CONTEXT (iii) inside an Ultralytics YOLO neck (iv) for tiny occluded PERSONS (v) with an assignment/loss stack. Must cite SET, FreqFusion, UAV-DETR/Freq-DETR/EFSI, wavelet-YOLO cluster, DERNet — and show mechanism-level difference (gating/evidence separation) or it reads as "another frequency module".**

**6. Large-kernel prior art — VERIFIED (all backbones, none used as a neck GATE)**
- LSKNet ICCV 2023 arXiv 2303.09030, code zcablii/LSKNet (693★; IJCV 2024). PKINet CVPR 2024 arXiv 2403.06258, code PKINet/PKINet (86★; PKINet-v2 → ECCV 2026). UniRepLKNet CVPR 2024 arXiv 2311.15599, code AILab-CVC/UniRepLKNet (1.1k★; TPAMI 2025 ext 2410.08049). Cite all three + UFO-DETR's LSK use as the receptive-field ingredient; the *gate-over-evidence* use is ours.

**7. Occlusion-aware person detection + VE-DINO — VERIFIED; ⚠ VE-DINO EXISTS (fabrication suspicion REFUTED)**
- **VE-DINO** — "Enhancing Human Detection in Occlusion-Heavy Disaster Scenarios: A Visibility-Enhanced DINO Model with Reassembled Occlusion Dataset", Zhao et al., **Smart Cities 8(1):12, Jan 2025**, DOI 10.3390/smartcities8010012 (verified via Semantic Scholar API; MDPI 403). DINO + visibility-aware loss weighting 4 body-region keypoint groups; trained **COCO2017 + custom "reassembled" occlusion set — NOT C2A, NOT aerial-trained** (UAV = case study only); AP 0.615 vs DINO 0.491 on their set. **No code found.** Low-prestige venue; beatable prior art for "visibility-weighted loss for disaster victims" — cite, don't fear; do NOT call it nonexistent.
- Classics confirmed: Bi-box (ECCV 2018), V2F-Net (arXiv 2104.03106), PedHunter (AAAI 2020).
- **NOMAD** — WACV 2024, arXiv 2309.09518, repo https://github.com/ArtRuss/NOMAD: 42,825 frames, 100 actors, **5 altitudes × 10 graded visibility levels** — the ONLY aerial person dataset with occlusion labels; includes Psych-ER behavioral data. **Psych-Occlusion** arXiv 2412.05553: psychophysics-shaped loss improves occluded/distant person detection (RetinaNet). Also OGMN arXiv 2304.11805; "Insight any invisible" ESWA 2025.
- **Aerial occlusion-visibility loss = lightly occupied (VE-DINO ground-level; Psych-Occlusion RetinaNet+NOMAD). C2A lacks visibility labels → synthesis-aware visibility supervision from paste provenance still UNCLAIMED; NOMAD = ready occlusion-conditional eval vehicle.**

**8. Coarse-to-fine / tile-zoom — VERIFIED; window narrowing**
- ClusDet ICCV 2019 (aged code); DMNet CVPRW 2020 code Cli98/DMNet (dormant); AdaZoom arXiv 2106.10409 (RL, NOT differentiable, no official code); QueryDet CVPR 2022 code ChenhongyiYang/QueryDet-PyTorch (481★, updated); CZ Det CVPRW 2023 code akhilpm/DroneDetectron2; Focus&Detect 2022 (no code); **ESOD IEEE TIP 2025 arXiv 2407.16424, code https://github.com/alibaba/esod (133★, YOLOv5/v8/RTMDet): objectness masking + patch slicing + sparse head, ~+8% AP on VisDrone/UAVDT/TinyPerson at lower FLOPs.**
- Differentiable train-time routing: **ZoomDet — ISPRS JPRS 2026, arXiv 2602.07512, code twangnh/zoomdet_code (mmdet/FRCNN): learnable end-to-end non-uniform zoom (+8.4 mAP SeaDronesSee @~3 ms)** + Dome-DETR density-sparse attention. SAHI-style inference tiling still the applied norm. **"Differentiable tile scouting inside a real-time YOLO" technically still open, but ZoomDet (Feb 2026) is adjacent prior art — free-real-estate window narrowing.**

**9. UAV-DETR — VERIFIED** (details in Lane 1 Extras; repo archived read-only June 2026.)

**10. Hypergraph density — VERIFIED: CROWDED**
- Hyper-YOLO TPAMI 2025 (iMoonLab, 224★) + YOLOv13 (1.7k★) + HyperEdge-DETR (JRTIP 2025, UAV small objects) + MSGHNet (RS small objects) + HyperTea arXiv 2508.10678. A hypergraph block on YOLO11 would read as DERIVATIVE. Avoid as headline; YOLOv13 = baseline family at most.

**11. TinyPerson / AI-TOD current SOTA — VERIFIED**
- AI-TOD(-v2) top line: **Dome-DETR ACM MM 2025, arXiv 2505.05741: 34.0/34.6 AP (M/L)** (+3.3); DQ-DETR ECCV 2024 arXiv 2404.03507 code hoiliu-0801/dq-detr (30.2); DNTR TGRS 2024; SET CVPR 2025 (+3.2 over RFLA, CNN track); D3R-DETR arXiv 2601.02747; Scale-Aware Relay Layer arXiv 2511.09891 (29.0 on YOLO-family). **Pattern: tiny SOTA = density-aware compute/query allocation + assignment, now ~34 AP.**
- TinyPerson: no active 2024–26 race; SSPNet-era ~59 AP50 stands; ESOD/DERNet use it as transfer testbed; CFINet ICCV 2023 (code shaunyuan22/CFINet) moved action to SODA.

### Lane-2 Extras
1. **FBRT-YOLO — AAAI 2025**, arXiv 2504.20670, DOI 10.1609/aaai.v39i8.32937, code https://github.com/galaxy-oss/FCM (146★, YOLOv8-based): Feature Complementary Mapping + Multi-Kernel Perception; VisDrone/UAVDT/AI-TOD. **Closest peer-reviewed 2025 YOLO-family aerial-small-object SOTA → mandatory baseline/citation.**
2. **ZoomDet** (ISPRS JPRS 2026) — differentiable resolution allocation is publishable NOW in top RS journals; YOLO-internal router still open but must differentiate.
3. **NOMAD + Psych-ER** — the only public asset to QUANTIFY occlusion-conditional recall for aerial persons; maps onto our 10/30/50 m protocol as secondary eval.
4. **SET (CVPR 2025)** — top-venue endorsement of frequency reasoning for tiny objects; frame FCCG's HF branch as its real-time, neck-level, GATED counterpart.
5. **Dome-DETR** — density-gated sparse compute = where tiny SOTA is going; a YOLO-neck analogue (density/frequency-gated routing) is exactly FCCG's un-occupied slot; cite as the DETR-side frontier.

**Lane-2 takeaways:** (1) C2A bar 0.8927/0.6883 unchallenged in print → reproduce it and we own the table. (2) Assignment-level has the literature's weight (RFLA/SimD): put a Gaussian/similarity term into YOLO11's TAL, not another NWD loss. (3) Frequency branch defensible ONLY via gating/evidence-separation + person/occlusion specialization (generic frequency-module slot filled in 2025). (4) **VE-DINO is real** — cite and beat (no code, ground-level training, private test set). (5) Hypergraph crowded; differentiable tiling newly contested (ZoomDet); **occlusion-visibility loss for aerial tiny persons = still open, NOMAD = eval vehicle.**

---

## LANE 3 — sim-to-real / domain adaptation / data strategy (completed 2026-07-10)

**Provenance:** all verdicts from live searches/fetches 2026-07-10; every GitHub URL fetch-confirmed.

**1. DANN / Gradient Reversal — VERIFIED**
- "Unsupervised Domain Adaptation by Backpropagation", Ganin & Lempitsky, ICML 2015, arXiv 1409.7495. GRL + domain classifier → domain-indistinguishable but task-discriminative features. Anchor citation only; 2014-era, never a novelty claim. (GRL ≈ 10 lines PyTorch.)

**2. MS-DAYOLO — VERIFIED (direct prior art for the "GD-ANN" pitch)**
- "Multiscale Domain Adaptive YOLO", Hnewa & Radha, IEEE ICIP 2021, arXiv 2106.01483; extended "Integrated MS-DAYOLO", IEEE TIP 2023, arXiv 2202.03527. Code: https://github.com/Mazin-Hnewa/MS-DAYOLO (Darknet/YOLOv4).
- GRL+domain-classifier heads on the three multiscale backbone maps feeding the YOLOv4 neck. Cityscapes→Foggy 43.04 vs 35.64 mAP. Also "Domain Adaptive YOLO" arXiv 2106.13939 (same year/family).
- Fit: **"GRL domain heads on YOLO neck" was published 2021 + extended 2023 — Gemini's "GD-ANN" re-invents MS-DAYOLO.** Cheap baseline, not a novelty claim.

**3. SSDA-YOLO — VERIFIED**
- Zhou, Jiang, Lu, CVIU 229 (2023), arXiv 2211.02213, DOI 10.1016/j.cviu.2023.103649. Code: https://github.com/hnuzhy/SSDA-YOLO (YOLOv5).
- Mean-teacher distillation + CUT style transfer (cross-generated source-like/target-like images) + prediction-consistency loss. Needs NO target labels despite the name.
- Fit: closest ready-made template for C2A(synthetic)→drone(real) with our unlabeled footage; porting to YOLO11 = engineering + strong supporting experiment, not novelty.

**4. Teacher-student UDA line + 2024-26 SOTA (one-stage/YOLO) — all VERIFIED**
- Unbiased Teacher (ICLR'21, arXiv 2102.09480, code facebookresearch/unbiased-teacher): EMA teacher + focal-loss vs pseudo-label imbalance; ~+10 mAP at 0.5–2% labeled COCO.
- Adaptive Teacher (CVPR'22, code facebookresearch/adaptive_teacher, archived): teacher-student + weak-strong aug + GRL in student.
- ConfMix (WACV'23, code giuliomattolin/ConfMix, YOLOv5): confident target-region mixing, progressive gating.
- Contrastive Mean Teacher (CVPR'23, arXiv 2305.03034, code Shengcao-Cao/CMT): object-level contrastive on pseudo-crops.
- Harmonious Teacher (CVPR'23, code kinredon/Harmonious-Teacher, dense one-stage): cls-loc consistency reweighting — targets the IoU-blind pseudo-label problem of dense detectors.
- **2024-26 strongest with code:**
  1. **SF-YOLO** — ECCV 2024 W (OOD-CV), arXiv 2409.16538, code https://github.com/vs-cv/sf-yolo (YOLOv5): teacher-student + learned target-style Target Augmentation Module, NO adversarial loss; only maintained YOLO-specific source-free DA repo.
  2. **SF-UT / simple-SFOD** — ECCV 2024, arXiv 2407.07586 (EPFL), code https://github.com/epfl-imos/simple-sfod: **AdaBN-only is already strong; FIXED one-shot pseudo-labels rival full mean-teacher without collapse; +4.7 AP50 Cityscapes→Foggy over prior SOTA. The discipline to copy.**
  3. **DINO Teacher** — CVPR 2025, arXiv 2503.23220, code https://github.com/TRAILab/DINO_Teacher: frozen DINOv2 labeler beats EMA-of-self teachers for target pseudo-labels + DINO-feature alignment; SOTA DAOD.
  4. **Dual-Rate Dynamic Teacher** — ICCV 2025 (CVF verified): asynchronous two-rate EMA for source-free DAOD (code only anonymous).
  - Also real, no code found: CLDA-YOLO (arXiv 2412.11812); Hierarchical MS-DA YOLO (Sensors 2025, C→F 45.9).
- Field arc: GRL alignment (2021) → mean-teacher + weak-strong (2022-23) → **source-free recipes + foundation-model labelers (2024-25)**. For our compute: SF-UT-style fixed-PL self-training + a DINOv2-labeler pass over unlabeled drone frames = highest credibility per GPU-hour.

**5. Attention/mask-guided adversarial DA ("DAMA" pitch) — VERIFIED as CROWDED**
- SIGMA (CVPR'22, arXiv 2203.06398, code CityU-AIM-Group/SIGMA): alignment as cross-image graph matching. SCAN (AAAI'22 oral, code CityU-AIM-Group/SCAN): semantic-conditioned kernels — alignment already "gated" by semantics. DA-DETR (CVPR'23, arXiv 2103.17084): attention modulates which CNN features align (no official code). MTTrans (ECCV'22, arXiv 2205.01643). **AWADA (arXiv 2208.14662; CVIU 2024): foreground-attention-weighted adversarial losses — CLOSEST to "DAMA".** Plus Masked Feature Alignment DETR (arXiv 2310.15646); WACV 2025 class-conditioned attention alignment.
- Gap assessment: mask/gate-WHERE-alignment-happens exists in ≥4 published forms. **Genuinely unclaimed: evidence-gated alignment specialized to TINY objects at P2/P3 on a modern anchor-free YOLO, tied to a measured failure mode (paste-seam bias in synthetic composites).** A delta, not a new mechanism — must be positioned against AWADA/SCAN explicitly.

**6. Synthetic→real for aerial person / SAR:**
- **(a) Archangel — VERIFIED**: Shen et al. (ARL+UMD), IEEE Access 11:80958-80972 (2023), arXiv 2209.00128, data https://a2i2-archangel.vision. Real + mannequin + ~4.4M synthetic UAV frames with matched altitude/radius/pose metadata; hybrid real+synthetic fine-tuning analysis stratified by altitude/angle. Template for our 10/30/50 m stratified evaluation.
- **(b) 2024-26 synthetic-composite→real UAV person — VERIFIED, active line:**
  - **CFHA** — "Coarse-to-Fine Hierarchical Alignment for UAV-based Human Detection using Diffusion Models", Li, Wu, Chen, Eum, Kwon, Qu (ARL line), arXiv 2512.13869 (Dec 2025, rev Jun 2026). 3 stages on the SYNTHETIC TRAINING SET: global diffusion style transfer → local super-resolution refinement of small human instances → hallucination filtering; labels preserved; **up to +14.1 mAP50 over raw-synthetic training** on UAV sim2real benchmarks. THE most on-point paper for C2A→real. (Code claimed released; repo not independently fetch-verified.)
  - **Maritime SAR line** — Martinez-Esteso et al., EAAI 2024/25 (S0952197624017445; SynBASe dataset) + 2025 follow-up (incremental UDA + pseudo-labeling): **synthetic pretrain + only 10% real fine-tune BEAT 100%-real training by 13.7%** — measures AND partially closes the gap with a small real budget.
- **(c) "July 2026 photogrammetry 88%→95% drone SAR study" — UNFINDABLE (fabrication confirmed as unfindable).** 4 query variants tried (logged). Plausible garble sources: MDPI RS 18(2):361 Jan-2026 (photogrammetry synthetic framework — but VEHICLES; MDPI 403-blocked) and arXiv 2411.09077 (drone-DETECTION, contains 88.0/95-97 figures). Do not cite.
- **(d) Copy-paste seam-artifact literature — VERIFIED, nuanced:**
  - "Cut, Paste and Learn" (Dwibedi, Misra, Hebert, ICCV 2017, arXiv 1708.01642): THE seam-bias primary source; fix = MIXTURE of blending modes per instance so the net can't use seams. +21% relative with synthetic+real.
  - "Simple Copy-Paste" (Ghiasi et al., CVPR 2021, arXiv 2012.07177): counter-evidence at scale (large objects, real-to-real) — seam sensitivity is regime-dependent.
  - Modern: "Synthetic Object Compositions" (arXiv 2510.09110, rev Jan 2026): diffusion harmonization + mask-area-weighted blending, +10.9 AP LVIS over prior synthetic pipelines; FFT-based blending precedents exist (Electronics 2023).
  - **Fit: sharpest external-validity threat to C2A — a 0.85 mAP50 model may partly be a SEAM detector. Two cheap reportable probes: (1) Dwibedi-style re-blend of C2A pastes → compare real-drone transfer; (2) frequency probe: low-pass/re-JPEG C2A test images → if AP collapses disproportionately vs SARD, seam reliance is demonstrated quantitatively.** Ties directly to need N7 and FCCG's declared high-pass risk.

**7. Pseudo-label self-training + LABEL BUDGET — VERIFIED**
- **S3OD** — ISPRS J. Photogramm. Remote Sens. 2025 (S0924271625000425); precursor arXiv 2310.14718: quantifies the TINY-OBJECT pseudo-label recall collapse (too few positive assignments from pseudo-boxes; negative mining suppresses small foreground); fixes = size-aware adaptive thresholds, size-rebalanced assignment, teacher-guided negative learning. Directly portable to an Ultralytics self-training loop for our 50 m frames.
- SF-UT (above): fixed-PL + AdaBN ladder = collapse-proof first move. DINO Teacher (above): frozen-foundation labeler for hard/small instances.
- **Label-budget numbers:** Unbiased Teacher: +~10 mAP at 0.5–2% labeled COCO. Few-shot DAOD: FAFR-CNN (CVPR 2019, arXiv 1903.09372) works with a handful of target images; **AsyFOD (CVPR 2023, code https://github.com/Hlings/AsyFOD, YOLOv5): +3.1 mAP C→F under single-digit-image few-shot protocol.** Maritime SAR: 10% real was enough to beat all-real.
- **Fit: 50–150 labeled drone frames (altitude-stratified) is an evidence-backed budget; 60 frozen test frames stay untouched; the rest of the footage = unlabeled self-training fuel.**

**8. Test-time adaptation 2024-26 — VERIFIED but FRAGILE**
- AMROD (arXiv 2406.16439): continual TTA-OD with adaptive thresholds + randomized restoration (+3.2 mAP Cityscapes-C); TENT-style entropy minimization degrades on detection (BN/batch sensitivity); error accumulation/forgetting still the story. NOT headline-stable for us; our protocol is offline anyway → offline self-training gives the benefit reproducibly. Future-work mention only.

**9. Multi-dataset joint training — VERIFIED (mechanisms real; exact head-to-head thin)**
- UniDet (CVPR 2022, arXiv 2102.13086, code xingyizhou/UniDet): learned unified label space; joint ≥ per-dataset experts, better OOD. Detection Hub (arXiv 2206.03484); ScaleDet (arXiv 2306.04849): generalization scales with #datasets. CerberusDet (arXiv 2407.12632): YOLO-based multi-dataset precedent.
- **No clean published head-to-head for "big synthetic + medium real + small real third set". Consistent pattern: keep synthetic and real in the SAME mix (joint) rather than sequential fine-tune** — sequential trades off the source domain (our Angle-B fine-tune SARD 0.917→0.898 dip = textbook symptom). Recipe: per-dataset balanced sampling (oversample small set), unified single "person" class (trivial for us), per-domain val/test reporting, optional short low-LR final tune on the mix.
- Fit: our C2A+SARD joint model (0.878/0.917) already matches the pattern; extend to C2A+SARD+(50-150 labeled drone frames) joint.

**10. Domain randomization classics + UAV follow-up — VERIFIED**
- Tobin IROS 2017 (arXiv 1703.06907); Tremblay CVPR-W 2018 (arXiv 1804.06516): randomization forces shape/structure reliance; synthetic+real fine-tune > real-only. UAV-Sim (arXiv 2310.16255, ARL): NeRF-rendered aerial data; hybrid real+synthetic boost. 2025-26 shift: high-fidelity/photogrammetric generation + post-hoc harmonization (CFHA; MDPI RS 18(2):361; SimD3 arXiv 2601.14742).

### Lane-3 Extras
- **E1. The ARL research arc (Archangel 2022 → UAV-Sim 2023 → CFHA 2025-26)** is a ready-made related-work spine; positioning C2A→own-drone against it gives reviewers a known problem frame.
- **E2. Diffusion harmonization of composite training data = 2025-26 consensus fix for paste artifacts** (Synthetic Object Compositions +10.9 AP LVIS; CFHA +14.1 mAP50 tiny aerial humans). A harmonized "C2A-H" variant is our cheapest high-credibility experiment.
- **E3. SF-UT's negative result is a gift for a compute-limited thesis**: AdaBN → fixed-PL → mean-teacher ladder runs on one 4070 Ti S with honest ablation value.
- **E4. VTSaR** — "Aerial Person Detection for SAR: Survey and Benchmarks", J. Remote Sensing (SPJ) 2025, DOI 10.34133/remotesensing.0474; repo https://github.com/zxq309/VTSaR (19,956 real + 54,749 synthetic instances, aligned RGB-T). Second real+synthetic aerial-person benchmark beyond SARD; survey = current APD citation anchor.
- **E5. S3OD** re-flagged: the only work quantifying tiny-object pseudo-label recall failure — size-aware thresholds + size-rebalanced assignment portable into our loop; combined with our NWD/VT-recall tooling = defensible micro-contribution.

**Lane-3 takeaways:** (i) Gemini's GD-ANN = MS-DAYOLO (2021/2023); DAMA ≈ AWADA/SCAN/masked-DETR alignment (2022-24) — both crowded. (ii) The "July 2026 photogrammetry" study does not exist. (iii) Unclaimed, narrow territory that fits OUR assets: tiny-object evidence-gated alignment at P2/P3 on modern YOLO + **measuring & suppressing paste-seam bias in C2A with harmonization (CFHA-lite)** — fresher support, cheaper compute. (iv) Label budget: 50–150 stratified frames justified by few-shot-DAOD + maritime-SAR evidence.
