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
