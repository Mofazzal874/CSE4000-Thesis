# P0 — rival differentiation notes (2026-07-12)

Purpose: mechanism-level differentiation of FCCG-YOLO vs the 4 most dangerous rivals (ranked).
Source texts fetched 2026-07-12; per-paper "Our delta" = what we do that they don't + what we must NOT claim.
Caveat: extraction via arXiv HTML / abstracts through a summarizing fetch — exact numbers re-check before citing in the paper.

---

## Rival 1 — DERNet (arXiv 2606.23825, "From Spatial to Spectral", 2026-06-22) — CLOSEST ARCHITECTURE RIVAL

**Source:** arXiv HTML v1 (full text reachable). No code released (as of lap-3 audit).

**Paradigm:** "Decompose–Enhance–Reconstruct" (DER) applied at THREE sites: backbone (WDG), neck (LGE), head (FDHead). Base = YOLOv11-S/M; also shown on RTMDet, PP-PicoDet, RT-DETR. Heavily lightweight-focused: DERNet-S = 1.3M params / 13.3 GFLOPs (−86% params vs YOLOv11-S 9.4M); DERNet-M = 2.9M / 29.4 GFLOPs.

**Their three modules (exact mechanisms):**
1. **Wavelet-Difference Gate (WDG)** — backbone bottlenecks (C2/C3, pre-stride). Fixed 2D **Haar DWT** → LL + (LH,HL,HH). Gate is **derived FROM the HF subbands** of the SAME feature: g = σ(BN(1×1 conv([LH;HL;HH]))); it **gates the LL (low-freq) band**: x̃_LL = y_LL ⊙ (1+g), where y_LL = RepCDC(LL) (re-param central-difference conv). Reconstruct via inverse DWT + residual. Gate learned (1×1 conv), wavelet fixed. <5% backbone FLOPs.
2. **Log-Gabor Enhancer (LGE)** — neck, after multi-scale fusion. **Fixed Log-Gabor filter bank** as depthwise conv (K=2 orientations, S=1 scale default), learnable softmax aggregation over scale/orientation, output added to skip path: y = x_skip + f_mix(σ(γ)·h). LGE-W variant at P2 swaps the mixer for wavelet-transform conv (bigger ERF).
3. **Frequency-Driven Head (FDHead)** — P2 head only. DEConv shared stem + **Spectral Haar Gate**: fixed Haar DWT → HF magnitudes → channel gate h = Σ_k softmax(ω)_k|f_k|; applied **only to the regression branch** f̃ = f ⊙ (1+αg); classification stays ungated ("box-only modulation"). Uses DFL decoding.

**Numbers (val/test mAP50):** VisDrone 0.398/0.316 (S), 0.447/0.362 (M) vs YOLOv11 baselines 0.384/0.311 and 0.442/0.353 — gains modest (+0.5–1.4 pts) BUT AP_S jumps +3.15/+2.27 pts (15.72/18.40). TinyPerson: 0.298/0.252 (S), 0.324/0.274 (M) vs 0.264/0.222, 0.283/0.239 (+3.0–4.1 pts — their strongest set). UAVDT and DOTAv1: DERNet does NOT beat baseline (0.844 vs 0.875 UAVDT val; 0.420 vs 0.447 DOTA val) — they trade accuracy for −85% params there. Ablation (VisDrone val): +P2 head +0.026(test), +WDG +0.015, +LGE +0.013, +FDHead +0.031 (biggest single), full +0.035 test.

**What DERNet does NOT do:**
- **No cross-scale semantic gating.** Every gate is intra-level and frequency-derived: HF subbands of a feature gate the LL band of the SAME feature. There is no signal from coarse/deep semantics deciding WHERE fine detail is plausible.
- **No large-kernel context branch** (no LSK/decomposed-DW context aggregation).
- **No assignment or loss changes** — standard YOLO TAL assignment, no RFLA/NWD/SimD, no occlusion losses (repulsion/Slide). Purely representation-level.
- **No disaster/person focus** — generic aerial multi-class (VisDrone/UAVDT/TinyPerson/DOTA); no C2A, no occlusion analysis.
- **No sim-to-real, no self-training, no own-footage protocol.**
- Enhancement is **fixed-basis** (Haar, Log-Gabor) + learned mixing — not a learnable DW filter bank producing explicit evidence masks.

**Our delta (FCCG-YOLO vs DERNet):**
- **Gate direction is inverted and cross-scale:** DERNet's gate = HF→LL within one level ("let edges amplify the base"). Ours = coarse-scale large-kernel CONTEXT (k7-dil3+k11 DW) → plausibility gate g on the fine-scale HF EVIDENCE branch (F' = F + HFE(F)·g). Semantics decide where high-frequency evidence is trustworthy — DERNet has no top-down/coarse-to-fine gating anywhere.
- **Evidence extraction is learnable and spatial-domain:** X − AvgPool residual + learnable DW filter bank yielding explicit evidence masks, vs their fixed Haar/Log-Gabor bases with only mixing weights learned.
- **We couple architecture with tiny-aware ASSIGNMENT (STAL min-anchor floor + SimD-in-TAL) + occlusion losses (repulsion, Slide)** — DERNet explicitly leaves assignment/loss untouched; the <16px assignment-over-loss consensus (RFLA/SimD line) is our lever they ignore.
- **Problem+protocol delta:** disaster-person single-class (C2A/SARD/own drone), occlusion-explicit, plus sim-to-real pillar (seam probe → C2A-H → SF-UT self-training). DERNet is generic-benchmark only.
- **Must NOT claim:** "first frequency/high-frequency enhancement for tiny aerial objects," "first HF gating in a YOLO detector," or novelty of a P2/frequency-aware head per se — DERNet occupies backbone+neck+head frequency enhancement including a gated P2 head. Also cannot claim "first to show HF cues drive AP_small" — their +3.15 AP_S does that. Our FreqFusion-lite top-down must be framed as *fusion anti-aliasing*, distinct from their intra-level enhancement, and we should not oversell "frequency" as our headline word — "context-gated evidence" is the defensible frame.

---

## Rival 2 — SAFE-Net (CVPRW 2026, AERO-HPR workshop) — CLOSEST PROBLEM RIVAL

**Sources checked (2026-07-12):** AERO-HPR workshop page (https://aero-hpr.github.io/), IIT Tirupati SPCV/ViSAL lab page (https://sites.google.com/iittp.ac.in/ee-spcv-lab), arXiv + general web search. **Paper text NOT reachable: workshop page says "PDF (coming soon)"; nothing on arXiv, CVF open access, or the lab page beyond an acceptance announcement.**

**VERIFIED (primary sources):**
- Full title: "SAFE-Net: Scale-Aware Feature Enhancement for Aerial Person Detection in Flood Disaster Imagery." Authors: Arun Kumar S (Sivapuram A.K.), Komuravelli Prashanth, Janipireddy Ganesh Mouli, Gorthi Rama Krishna Sai Subrahmanyam — IIT Tirupati.
- Proceedings-track POSTER at AERO-HPR @ CVPR 2026 (pre-recorded video at poster session). It is the ONLY disaster-related aerial-person paper at that workshop.
- Same lab also has "RealDroneVision: Dataset and Architecture Advancements for Small-Object Drone Detection" (WACV 2026) — an actively publishing small-object aerial group; treat as a continuing rival group, not a one-off.

**UNVERIFIED (could not fetch):** mechanism of the "scale-aware feature enhancement," base detector, dataset (C2A? own flood set?), all numbers and baselines. Title-level read: scope is FLOOD-only and the method name sits in the generic multi-scale feature-enhancement family; no title-level signal of frequency processing, gating, assignment/loss work, occlusion handling, or sim-to-real.

**ACTION ITEM (standing):** re-poll CVF open access (CVPR2026_workshops) + arXiv for the PDF before any submission; redo this delta at mechanism level once text drops. Until then our differentiation rests on scope + protocol, not mechanism.

**Our delta (FCCG-YOLO vs SAFE-Net — provisional, title/venue level):**
- **Broader disaster scope + benchmark discipline:** flood-only vs our multi-disaster C2A (fire/flood/collapse/traffic) + SARD + own 3-altitude drone footage under a scene-disjoint, leakage-audited protocol.
- **Composite mechanism vs (apparent) enhancement-only:** no title-level signal of cross-scale context gating, an explicit HF evidence branch, tiny-aware assignment (STAL floor + SimD-in-TAL), or occlusion losses — the exact levers our composite stacks.
- **Second pillar with no counterpart signal:** sim-to-real protocol (seam probe → C2A-H harmonization → SF-UT self-training ladder → joint 3-set training) on own drone footage.
- **Must NOT claim:** "first/only aerial PERSON detection work for DISASTER imagery at a 2026 CVPR-affiliated venue" — SAFE-Net occupies that exact problem framing (and the C2A paper defined the task). Position as advancing the problem, not inventing it; cite SAFE-Net in related work once public.

---

## Rival 3 — SRTSOD-YOLO (Remote Sensing 17(20):3414, DOI 10.3390/rs17203414, 2025) — CLOSEST SAME-BASE-MODEL RIVAL

**Route:** MDPI page returned 403 to direct fetch; full abstract obtained via **Semantic Scholar API** (DOI lookup) + search snippets; a Preprints.org mirror exists (DOI 10.20944/preprints202507.2594.v1) but also 403'd. **Verified = abstract + snippet level; the exact GAC gate formula (what tensor the gate multiplies, gate input, activation) is UNVERIFIED — pull the MDPI HTML from a browser/other network before citing mechanism details in print.**

**What it is (verified from abstract):** model series (n/s/m/l) on **YOLO11** with:
1. **MFCAM** (backbone): channel+spatial attention + multi-scale conv extraction, against small-target information loss with depth.
2. **GAC-FPN** (neck): "Gated Activation Convolutional Fusion Pyramid Network" — enhances multi-scale fusion by "emphasizing salient features while suppressing irrelevant background," via three strategies: (a) **adds a small-receptive-field detection head and removes the original largest one** (i.e., +P2 head, −P5 head), (b) leverages large-scale features more effectively, (c) inserts **gated activation convolutional modules** (snippets: a "dynamic gating mechanism"). Reading: the gate is a salience/background-suppression modulation applied to the fused pyramid features — i.e., **self-gating of a single stream** (the fused features gate themselves); no separate evidence signal, no cross-scale gate source is described.
3. **Adaptive threshold focal loss** replaces BCE in the head (positive/negative imbalance, faster convergence) — a classification-loss reweighting, NOT an assignment change.

**Numbers (verified from abstract/snippets):** VisDrone2019 mAP50 +3.1 (n vs YOLO11n), **+7.9 (l vs YOLO11l)**; UAVDT +1.2 (n), +3.3 (l); missed-target metric E_missed −1.08% (l). Params/GFLOPs: not in abstract — unverified.

**What SRTSOD-YOLO does NOT do (abstract-level):** no frequency/HF-detail extraction anywhere; no large-kernel context operators mentioned; no cross-scale gating (gate source = the same fused features); no occlusion losses; no assignment change (TAL untouched — ATFL is loss-side); generic multi-class UAV benchmarks; no disaster/person focus; no sim-to-real/self-training.

**Our delta (FCCG-YOLO vs SRTSOD-YOLO):**
- **Two-stream cross-scale gating vs single-stream self-gating:** GAC-FPN's gate is computed from and applied to the same fused features (salience filter). Ours factorizes into an explicit HF EVIDENCE branch (X − AvgPool + learnable DW filter bank → evidence masks at P2/P3) gated by a PLAUSIBILITY signal synthesized from a DIFFERENT (coarse) scale via decomposed large-kernel DW convs (k7-dil3 + k11): F' = F + HFE(F)·g. The gate source, gate target, and their scales all differ from GAC-FPN.
- **Assignment-level lever they lack:** STAL min-anchor floor + SimD-style similarity inside TAL — the <16px assignment>loss consensus (RFLA/SimD line). Their only supervision change is ATFL, a focal-loss variant.
- **Occlusion-explicit losses (repulsion, Slide) + disaster-person scope + sim-to-real pillar** — none present in SRTSOD-YOLO.
- **Must NOT claim:** "first gated fusion neck on YOLO11," "first to add a P2 head / drop the P5 head on YOLO11," or "largest YOLO11 gain on VisDrone" — SRTSOD-YOLO owns gated-neck-on-YOLO11 with a P2-head swap and claims +7.9 mAP50 (l). Our headline must live on C2A/tiny-person/occlusion metrics under our clean protocol, and if we cite their +7.9 we must note it is vs YOLO11l on VisDrone, not a tiny-person disaster setting.

---

## Rival 4 — AFGLFF-YOLO (IEEE JSTARS 2026, DOI 10.1109/JSTARS.2025.3649074)

**Route:** IEEE full text not fetched (paywall); abstract + metadata via **Semantic Scholar API** (DOI lookup). Verified = abstract level; module-placement specifics beyond the abstract are UNVERIFIED.

**What it is:** "Adaptive Frequency Global-Local Feature Fusion Model on YOLO for Remote Sensing Object Detection" — Junpeng Wu, Xinguang Tao, Pan Gao. Base = **YOLOv8** (NOT YOLO11 — weaker same-base threat than SRTSOD). Frequency transform = **wavelets: Daubechies-4 (DB4) + Haar** for "feature frequency differentiation"; includes an **AFWD (adaptive frequency wavelet downsample)** module in the backbone (frequency-aware downsampling, i.e., sub-band reweighting when reducing resolution) and dual-channel frequency-response structures; global-local fusion via **graph convolution + depthwise-separable convolution**. So: frequency use = adaptive sub-band weighting inside downsample/fusion — adaptive-weighted fusion, not a plausibility gate from a separate semantic stream (abstract gives no gate-source/gate-target separation; treat "dual-channel" claim as unverified detail).

**Numbers (abstract):** mAP50 — NWPU VHR-10 95.6, RSOD 97.6, DIOR 79.6, RS-STOD 83.1. Note the benchmark character: VHR-10/RSOD/DIOR are generic mid/large remote-sensing objects (planes, ships, storage tanks…); only RS-STOD is small-target. **No person class, no disaster imagery, no UAV-altitude tiny humans, no occlusion analysis.**

**What AFGLFF-YOLO does NOT do (abstract-level):** no coarse-context→fine-evidence gating (frequency weighting happens where the transform is applied, same scale); no large-kernel spatial context operator; no assignment or occlusion-loss work; not YOLO11; no disaster/person setting; no sim-to-real.

**Our delta (FCCG-YOLO vs AFGLFF-YOLO):**
- **Gated evidence vs weighted sub-bands:** they adaptively reweight fixed wavelet sub-bands (DB4/Haar) during downsampling/fusion; we construct a learnable spatial-domain HF evidence signal (X − AvgPool residual + learnable DW filter bank) and multiply it by a semantics-derived cross-scale plausibility gate — the gating signal comes from ANOTHER scale's large-kernel context, which no wavelet-reweighting scheme has.
- **Supervision + protocol stack:** tiny-aware assignment (STAL floor + SimD-in-TAL), occlusion losses, scene-disjoint disaster-person benchmarks, sim-to-real ladder — all absent in AFGLFF.
- **Base + problem:** YOLO11m on tiny (<16px) occluded HUMANS in disaster imagery vs YOLOv8 on generic RS objects — their numbers (95%+ on VHR-10/RSOD) reflect saturated large-object sets, not our regime.
- **Must NOT claim:** novelty of "adaptive frequency fusion" or frequency-aware/wavelet DOWNSAMPLING as such — AFGLFF (and DERNet) own that phrasing; avoid a module name colliding with "adaptive frequency ... fusion," and do not present wavelet-style sub-band reweighting as a contribution.

---

## Differentiation summary

All four rivals enhance multi-scale features, but none of them uses coarse semantic context to decide WHERE fine-grained detail should be trusted: DERNet's gates are frequency-derived and intra-level (HF sub-bands amplify the LL band of the same feature), SRTSOD-YOLO's GAC-FPN self-gates its own fused activations for salience, AFGLFF-YOLO adaptively reweights fixed wavelet sub-bands during downsampling, and SAFE-Net (text still unpublished) signals only generic scale-aware enhancement. FCCG-YOLO's defensible core is therefore a cross-scale, two-stream factorization: an explicit learnable high-frequency EVIDENCE branch on the fine laterals (P2/P3: X − AvgPool residual + learnable DW filter bank → evidence masks) multiplied by a PLAUSIBILITY gate synthesized from coarse scales through decomposed large-kernel context (k7-dil3 + k11 DW), F' = F + HFE(F)·g — semantics vetting evidence, rather than evidence amplifying itself or features self-gating. Second, uniquely among the four, we couple this representation change to the sub-16px assignment consensus (STAL min-anchor floor + SimD-style similarity inside TAL) and to occlusion-explicit losses (repulsion, Slide) — all four rivals leave label assignment untouched and none addresses occlusion. Third, we are the only one with a domain-transfer pillar — seam probe, C2A-H harmonization, SF-UT self-training on our own 3-altitude drone footage — and with a leakage-audited, scene-disjoint disaster-person evaluation (C2A + SARD + own set). Accordingly, our related-work positioning must concede: frequency/HF enhancement for tiny aerial objects exists across backbone-neck-head (DERNet), gated fusion necks on YOLO11 with P2-head swaps exist (SRTSOD-YOLO), adaptive wavelet fusion exists (AFGLFF-YOLO), and aerial disaster-person detection is an occupied problem (SAFE-Net, C2A) — we claim none of these as firsts. The one-paragraph claim that survives all four simultaneously: FCCG-YOLO is the first detector for tiny occluded humans in aerial disaster imagery that factorizes fine-scale perception into learnable high-frequency evidence and large-kernel cross-scale contextual plausibility, and aligns its supervision (tiny-aware assignment + occlusion losses) and its sim-to-real transfer protocol with that same factorization. Lead with "context-gated evidence," not "frequency."
