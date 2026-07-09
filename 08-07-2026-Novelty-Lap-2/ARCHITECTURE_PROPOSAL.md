# LAP-2 ARCHITECTURE PROPOSAL — FCCG-YOLO (working name)
**Frequency-Contrast, Context-Gated YOLO for tiny/occluded aerial humans**
2026-07-08 · branch `novelty-lap-2` · Inputs: `docs/2026-07-08_A_image_needs_analysis.md` (measured
needs N1–N8) × `docs/2026-07-08_B_sota_research.md` (verified 2025/26 mechanisms).

## 0. The design thesis (one paragraph)
A tiny human in disaster imagery is a **high-frequency anomaly inside a low-frequency scene** —
our own images show median 12 px bodies (34.5% <8 px) whose CBAM attention smears scene-wide
instead of peaking on them, texture-only false positives on grass/rubble/boulders, and merged
detections in crowds. FCCG-YOLO therefore separates **evidence** from **plausibility**: a high-pass
evidence branch at the fine scales says *what pops out*; a large-kernel context branch at the
coarse scales says *where a person is plausible* and **gates** the evidence — killing texture FPs
without suppressing true tiny targets; a frequency-aware fusion preserves detail through the neck;
and a tiny/occlusion-aware training stack (assignment + repulsion + hard-example weighting)
teaches separation in crowds and recovery of low-contrast bodies. Every element answers a numbered,
measured need. No mechanism requires custom CUDA; all integrate as `nn.Module`s in the existing
Ultralytics YAML pipeline (same injection pattern as the thesis' CBAM).

## 1. Components → needs → evidence
| # | Component (what we build) | Answers | Evidence base (from B) |
|---|---|---|---|
| 1 | **HFE — High-Frequency Evidence branch** at P2/P3 laterals: X − AvgPool_k(X) + learnable DW filter bank → spatial+channel evidence masks (HFP-style) | N1 tiny pop-out, N6 low-contrast | HFP alone +2.2 AP on AI-TOD (AAAI'25) [V]; our attention-map finding (diffuse CBAM) |
| 2 | **CXG — Context Gate branch** from P5/P4: decomposed large-kernel DW convs (LSK-style, k7-dil3 + k11) → plausibility gate g∈[0,1], upsampled; fusion F' = F + HFE(F)·g | N4 texture-FP veto, N5 (partial) depicted-human suppression | LSK IJCV'24 [A]; RemDet GatedFFN +1.2 [V]; our counts.csv FP evidence |
| 3 | **FreqFusion-lite** on the top-down path: adaptive low-pass on upsampled deep features + high-pass-enhanced skips (offset-free, pure PyTorch) | N1/N2 detail preservation through neck | FreqFusion TPAMI'24 +1.9 AP [V] |
| 4 | **Training stack**: STAL-style tiny-aware assignment (lifted from YOLO26, Ultralytics-native) + NWD α=0.5 term (our own pilot: +0.73 VT-recall) + occlusion-aware **repulsion loss** + Slide hard-example weighting | N3 crowd separation, N6 occlusion/contrast, N1 | STAL +0.6 AP_S [V]; our G2 pilot; DOMino repulsion +3.2 mAP50 on occluded aerial vehicles [S/A]; SEAM/FaceV2 stack [A] |
| 5 | (STRETCH, only if time) **visibility-aux head** using pseudo-visibility from C2A paste provenance | N6/N7; unclaimed novelty | nobody has done synthesis-aware visibility supervision (B §4) |
| 6 | **Evaluation gates as design constraints**: scene-disjoint C2A + SARD zero-shot + own 3-altitude drone bench are MANDATORY pass criteria, not afterthoughts | N7 anti-paste-shortcut, N8 calibration (04 harness) | our lap-1 infrastructure |
Coverage: N1✔(1,3,4) N2✔(3+existing P2..P5; OOD-large watched in eval) N3✔(4) N4✔(2) N5◐(2+drone
bench hard-negatives — honest partial) N6✔(1,4,5) N7✔(6+SARD gate) N8✔(harness+optional C-WBF at deploy).

## 2. Novelty claim (why this is not plug-and-play)
Plug-and-play = dropping a published block into a YAML slot. FCCG is a **coupled two-stream design**:
the high-pass branch is only safe *because* the context gate suppresses its false pop-outs, and the
gate is only useful *because* the evidence branch surfaces sub-16px responses that vanilla FPN
loses. Per B §4, high-frequency gating exists only on R-CNNs (HS-FPN), density/context gating only
in DETRs (Dome-DETR), the occlusion stack only in faces (FaceV2) and vehicles (DOMino) — **the
combination, on YOLO, for aerial tiny HUMANS, is unclaimed**, and it is *derived from measured
failure evidence* (A §E), which is the design story reviewers reward. The stretch visibility-aux
(synthesis-aware supervision) is unclaimed by anyone on any dataset.

## 3. Budgets
Params ≤ 22.5M (base 19.6M + ~1.5–3M) · GFLOPs ≤ ~100 (base 87) · latency ≤ ~19 ms on 4070 Ti S
(base 15.7) · trains in 16 GB at batch 8 · pure PyTorch (Windows-safe) · Ultralytics YAML + the
proven register-inject-verify pattern (incl. pickle-roundtrip + module-active assertions — Mamba
and CBAM lessons are baked into the test list).

## 4. Honest expectations (per-module numbers, ~50% stacking discount)
C2A: mAP50 +1.5–3.0 · AP_small +2–4 · VT-recall +2–5 · biggest visible win expected on **drone
footage FP rate** (texture veto) and **scene-disjoint/SARD deltas** (shortcut robustness).
SOTA bar (B §6): (a) YOLOv9-e 0.8927 mAP50 / 0.6883 mAP @57M on the leaky official split — clearing
it outright from 0.8533 needs the optimistic end of the range; the *defensible* claim is
**match-or-beat YOLOv9-e's mAP50-95/AP_small at 1/3 the params AND dominate it on the scene-disjoint
split + real footage** (we reproduce YOLOv9-e under our protocol — mandatory anchor run, which also
settles the 0.6883-vs-our-0.615-ceiling contradiction). (b) LightSeek AP_small 0.478 — already beaten.

## 5. Gated execution plan (fail cheap, prove hard)
| Stage | What | Where | Time | GATE (numeric) |
|---|---|---|---|---|
| S0 | Implement HFE/CXG/FreqFusion-lite modules + YAML + selftests (shape/grad/pickle/param-count) + 2-ep smoke | laptop code, PC-4 smoke | 1–2 d | smoke trains, modules verified ACTIVE, params ≤22.5M |
| S1 | **Paired 50-ep pilots on scene-split**, seed 0: control (CBAM+P2) vs +FCCG, identical protocol | PC-4 (fp32 b4) or PC-2 GPU1 | ~5 h each | **+1.5 AP_small or +2 VT-recall** vs control → go; else ONE config iteration, then stop |
| S2 | Loss-stack pilots (repulsion, Slide, STAL) added stepwise, 50-ep each | PC-4 | ~5 h each | each keeps or improves S1 gains; drop any that regress |
| S3 | Full 300-ep protocol runs: FCCG on scene-split + official | PC-1 (after baseline) | ~1.5 d each | beats CBAM+P2 full runs on AP_small/VT-recall AND SARD zero-shot does not collapse vs CBAM+P2's |
| S4 | Anchors: YOLOv9-e under our protocol; YOLO26m baseline (AdamW pinned) | PC-1 queue | ~1.5 d + 1.5 d | table completeness (reviewer-proof) |
| S5 | 3 seeds on final config only + drone-bench eval (needs user's 60 labels) + paper tables | PC-1/PC-4 | — | final |
Failure floor: if S1 gates fail after one iteration, we still hold the scene-disjoint protocol,
the ablation corpus (CBAM/P2/Mamba/NWD/ECA/enrichment), the real-footage benchmark, and pivot the
paper to Direction A (density-gated sparse P2 — efficiency story) with parts already built.

## 6. Decisions locked / open
LOCKED: base = YOLO11m CBAM+P2 lineage (continuity with all existing numbers); protocol = thesis
metric contract; clean benchmark = scene-split; anchors = YOLOv9-e + YOLO26m.
OPEN (user): confirm FCCG as primary (vs A=sparse-P2-first, C=occlusion-first); working name.
