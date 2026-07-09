# Agent B — 2025/26 SOTA + mechanism research (2026-07-08)
Legend: [V]=verified from primary source · [A]=abstract only · [S]=secondary/UNVERIFIED.
MDPI full texts (LightSeek/DOMino/SRTSOD) blocked fetchers → their numbers [A]/[S].
Companion: `2026-07-08_A_image_needs_analysis.md` (the measured needs this must answer).

## 1. SOTA anchors

### COCO (architecture reference, m-scale)
| Model | Status | mAP50-95 | P/GFLOPs |
|---|---|---|---|
| YOLO11-M (our base) | — | 51.5 | 20.1M/68G [V] |
| YOLOv12-M (2502.12524) | real; NeurIPS'25 claim [S] | 52.5 | 20.2M/67.5G [V] — Area Attention + R-ELAN |
| YOLOv13 (2506.17733) | preprint; NO M variant | S 48.0 / L 53.4 | HyperACE hypergraph [V] |
| YOLO26-m (2606.03748) | Ultralytics Jun'26 | 53.1 (52.5 E2E NMS-free) | 20.4M [V] — STAL tiny-assignment, ProgLoss, DFL removed |
| Hyper-YOLO-M | TPAMI 2025 (peer-rev) | 52.0 | 33.8M/103G [V] |
| MHAF-YOLO-m (2502.04656) | preprint (v1 PRCV'24) | 52.7, AP_s 35.2 | **15.3M/65.2G** [V] — MAFPN |
| D-FINE-M/L (2410.13842) | ICLR 2025 Spotlight | 52.3/54.0 | 19M/57G, 31M/91G [V] |
| DEIM (2412.04234) | CVPR 2025 | train framework (Dense O2O+MAL), 2x convergence | [A] |

### VisDrone2019-DET
| Model | Status | AP / AP50 | P/GFLOPs |
|---|---|---|---|
| RemDet-X @1024 (2412.10040) | AAAI 2025 | **40.0 / 61.9** | 74M/182G [V] |
| Dome-DETR-L (2505.05741) | ACM MM 2025 | 39.0 val | 36M/359G [V] — density-gated queries on D-FINE |
| SOD-YOLO (2507.12727) | preprint | 35.1 / 52.6 | 22.6M/95G [V] — its +9.4 Soft-NMS delta = implausible, flagged |
| RTUAV-YOLO-M | Sensors 2025 (peer-rev) | 28.9 / 46.8 | **7.2M/56G** [V PMC] — the recent m-scale bar |
| FBRT-YOLO-X (2504.20670) | AAAI 2025 | 30.1 / 48.4 | 22.8M [V] |
| UAV-DETR-R50 (2501.01855) | preprint | 31.5 / 51.1 | 44M/161G [V] |
| YOLO11-M baseline | — | 26.9 / 44.1 | 20.1M/68.5G [V via RTUAV] |

### AI-TOD(v2) (mean box ~12px — statistically closest to C2A)
| Model | Status | AP | AP_vt |
|---|---|---|---|
| ScaleBridge-Det (2512.01665) | preprint Dec'25 | 35.7 | — [S] |
| **Dome-DETR-L** | ACM MM 2025 | **34.6 (+3.3 over D-FINE)** | **19.0** [V] |
| HS-FPN (2412.10116) | **AAAI 2025** | 23.6 Cascade (+3.4 vs FPN); **HFP alone +2.2** | 11.6 vs 9.9 [V] |
| DNTR (2406.05755) | TGRS 2024 | ≥+3.5 AP_vt over RFLA | [S] |
| NWD / RFLA | — | +6.7 / +4.0 AP | [A] — now commodity in 2025 UAV papers |

### TinyPerson / SeaDronesSee / CrowdHuman (occlusion refs)
TinyPerson: R-AFPN-L 50.7 AP50 (SciRep'25) [S]; ESOD (IEEE TIP 2025, 2407.16424) up to +8% AP at high-res via feature-level patch gating [A]. SeaDronesSee ODv2 top AP .6152 [S]. CrowdHuman: DDQ-DETR 93.8 AP; **FeatComp++ (2405.01311) MR-2 37.46 (>3pp SOTA), CityPersons heavy-occl 39.92→31.78** [V].

## 2. Mechanism catalog (per-module verified gains)
**Downsample/stem:** SPD-Conv (2208.03641, ECML'22): AP_s +19%(n)/+9.5%(s)/+6%(m) relative [V README]; CED (RemDet AAAI'25) +0.7 AP [V]; LDSM (RTUAV) +0.7 mAP50 [V].
**Upsample/fusion:** DySample (ICCV'23) +1.1 AP FRCNN, ~0 params, pure PyTorch [V]; **FreqFusion (TPAMI 2024, 2408.12879) +1.9 AP FRCNN** (ALPF+AHPF+offset resampling; mmcv parts replaceable) [V]; MAFPN +1.6 AP, +2.4 AP_s (v10n) [V]; GD (Gold-YOLO NeurIPS'23) +2.6 AP [A]; ASF +0.4 mAP50 only [V via SOD-YOLO].
**Context/large-kernel/frequency:** **HFP (HS-FPN AAAI'25): high-pass filter mask, +2.2 AP alone on AI-TOD** [V]; LSK (ICCV'23+IJCV'24) DOTA SOTA, lightweight [A]; CAA (PKINet CVPR'24) [A]; WTConv (ECCV'24) drop-in wavelet conv, det gains UNVERIFIED [V README]; GatedFFN (RemDet) +1.2 AP [V]; MKP (FBRT) +1.6 [V]; A2 area attention (v12) +1.0 over YOLO11-M [V]; HyperACE +3.0(N)→+0.4(L) [V].
**Sparse high-res compute (P2 enablers):** QueryDet (CVPR'22) +2.0 AP_s, 2.3-3x HR speedup — official impl needs spconv CUDA → reimplement as masked dense [A]; **ESOD (TIP 2025)** objectness-gated patches, backbone reuse [A]; CEASC (CVPR'23) sparse-conv head, pure-PyTorch-able masking [A]; Dome-DETR DeFE+MWAS+PAQI +3.3 AP AI-TODv2 [V].
**Assignment/losses:** **STAL (YOLO26): tiny-target assignment, +0.6 AP_S on YOLO11s — Ultralytics-native, liftable** [V]; NWD +6.7 [A]; RFLA +4.0 [A]; Dense O2O+MAL (DEIM CVPR'25) [A]; Wise-IoU v3 (used by LAF-YOLOv10 for label noise — C2A-relevant) [A]; Soft-NMS claims of +9pt = protocol artifact, budget ≤+1.
**Occlusion (2024-26):** FeatComp++ channel-completion [V]; **DOMino-YOLO (Remote Sensing 2026, 18(1):66): YOLOv11 + deformable align DCEM +3.2 mAP50 + visibility-aware aggregation + occlusion-aware REPULSION loss — but VEHICLES; the human version is UNCLAIMED** [S/A]; SEAM (YOLO-FaceV2, PatRec 2024): separated attention + NWD + Repulsion + Slide — the "occlusion+tiny" stack exists in FACES, not aerial humans [A]; V2F/EPM visible-part aux [A]; Psych-Occlusion (2412.05553): distance/occlusion-shaped loss works in aerial SAR [V abs].

## 3. C2A citation sweep (complete, 13 works) — the niche is OPEN
Only architecture entry: **LightSeek-YOLO** (Mathematics 2025): AP_small 0.478 @1.86M — we already beat by +0.14 at 10x params. Rest = system/comparison papers with unverifiable protocols (e.g. INSPECT'25 "YOLOv7 96.76 mAP50" = custom subset, cite-and-dismiss). Dataset authors' own bar: **YOLOv9-e 0.8927 mAP50 / 0.6883 mAP @57.3M** (640px/50ep/Adam, official=leaky split) [V]. Nobody has published scene-disjoint C2A. **CAVEAT: their YOLOv9-e 0.6883 mAP CONTRADICTS our "~0.615 ceiling" (measured on our 4 archs) → ceiling is NOT absolute; reconcile by reproducing YOLOv9-e under our protocol.**

## 4. Used-up vs unclaimed
**Used-up as plug-and-play (≥5 papers each, 2025):** P2 head, CBAM/CA/EMA/SimAM, BiFPN/AFPN/ASF, SPD-Conv, DySample/CARAFE, NWD/Wise-IoU/Shape-IoU, Soft-NMS, Ghost/PConv/HGNetV2, SAHI.
**Unclaimed for aerial tiny HUMANS:** density-gated sparse P2 inside YOLO (Dome did DETR-only) · high-frequency contrast gating (HFP only on R-CNNs) · occlusion-decomposed head for aerial humans (DOMino = vehicles) · synthesis-aware visibility supervision (nobody — C2A's paste provenance uniquely enables) · scene-disjoint C2A protocol (ours).

## 5. Three composite directions (agent's synthesis)
**B (primary) FCCG:** high-pass evidence branch (HFP-style, P2/P3) × large-kernel context gate (LSK/CAA, P4/P5) with FreqFusion-style fusion + STAL/RFLA assignment. Expected C2A: mAP50 +1.5-3.0, AP_s +2-4, VT-recall +2-5 (per-module numbers discounted ~50%). Budget ≤22.5M/~100G, +2-3ms. Pure PyTorch. RISK: high-pass may learn PASTE SEAMS → gains vanish on scene-split/SARD; mitigation = those exact gates (doubles as paper subsection).
**C (differentiator) SODH:** synthesis-aware occlusion-decomposed head — visible-part aux supervision recoverable from C2A compositing + SEAM-style separated attention + occlusion-aware repulsion + Slide weighting. Gains concentrate on occluded/VT recall (+2-4 recall<8px). Risk: pseudo-visibility mask quality.
**A (efficiency fallback) DGSP:** SPD stem + DeFE-style density heatmap gating masked-dense P2 (CEASC-style, no spconv) + DySample. AP_s +2-4 at neutral-or-less latency. Risk: gating recall for <8px; block-wise masking engineering on Windows.
Practical: add YOLO26m baseline (pin AdamW — MuSGD diverged on our P2 before); YOLOv12-M = fair attention baseline; Dense O2O is train-only and composable.

## 6. What "beat SOTA on C2A" concretely means
(a) Clear **0.8927 mAP50 / 0.6883 mAP at ≤20M params** on official split, AND/OR reproduce YOLOv9-e under our protocol and show it collapses on scene-disjoint while ours doesn't; (b) AP_small > 0.478 (done: 0.6156); (c) report scene-disjoint as the protocol the field should adopt. External anchors if asked: VisDrone m-bar 46.8 mAP50 @7.2M (RTUAV), AI-TODv2 34.6 (Dome), TinyPerson ~50.7 AP50.
