# Novelty research findings — what to pursue, what to drop (2026-06-29)

Compiled from 4 web-research agents (some sub-searches were throttled by a session limit; items not
fully verified are flagged). Honest framing throughout: this is an APPLIED thesis; the contribution is
empirical, not a new architecture.

## TL;DR
- **Architecture (YOLO11m+CBAM+P2) is NOT novel** — it's established prior art. Cite it; don't claim it.
- **Your novelty bet = SAHI-guided self-training** on your own real DJI drone footage (uses an asset
  nobody else has; mitigates the documented small-object failure mode of naive pseudo-labeling).
- **Reliable metric wins (not novel): SAHI + TTA.**
- **Mamba = honest null result, but FRAME IT CORRECTLY** (we were mis-citing MambaOut — see §5).
- **Drop: federated learning, reinforcement learning, test-time adaptation (TENT/CoTTA).**

## 1. Architecture is established prior art (cite as justification, not novelty)
- P2-YOLOv8n-ResCBAM (P2 head + CBAM attention, small-object): IEEE https://ieeexplore.ieee.org/document/10577135/ (mAP 90.3→92.6).
- Real-Time SAR with Drones (P2 + CBAM + BiFPN on HERIDAL — closest "same problem, same toolbox"): https://www.mdpi.com/2504-446X/9/8/514
- LAF-YOLOv10 (auxiliary P2 head + attention FPN): https://arxiv.org/pdf/2602.13378
=> State plainly: the recipe is known; the thesis contribution is the empirical deployment study.

## 2. THE NOVELTY: SAHI-guided self-training (synthetic→real adaptation)
Idea: use SLICED (SAHI) inference to pseudo-label your real drone footage — so small aerial humans are
detected cleanly — then fine-tune; measure on a small hand-labeled real test set. The "SAHI-guided"
part directly attacks the confirmation-bias-on-small-objects failure mode the literature flags, which
makes it a tailored, defensible contribution (not generic pseudo-labeling).
- Self-training recipe is easy in Ultralytics: `model.predict(save_txt=True, save_conf=True, conf=…)`
  writes YOLO-format pseudo-labels → merge into train split → retrain. (https://www.ultralytics.com/glossary/semi-supervised-learning)
- Method family to CITE (don't reimplement — no YOLO11 support): **Efficient Teacher** (YOLOv5/v6/v7/v8),
  +0.9–1.45 mAP on COCO, big gains in low-label regime.
- **Honest risk — confirmation bias**: pseudo-labeling reinforces the model's own errors, worst on
  small objects (your weak spot). Mitigate: tuned high conf threshold (sweep ~0.5–0.7), EMA-teacher
  optional, and NEVER put pseudo-labels in val. (Arazo et al.: https://www.researchgate.net/publication/347038727)
- Gains are dataset-dependent; **no clean YOLO-specific "+X mAP" benchmark number exists** (unverified —
  don't quote a number; report your own measured delta). A flat/negative delta is still a real finding.
- Prior synthetic→real aerial precedent to cite: "Exploring the Impact of Synthetic Data for Aerial-view
  Human Detection" https://arxiv.org/pdf/2405.15203

## 3. Reliable metric boosters (NOT novelty — for the results chapter)
- **SAHI tiled inference** (strongest small-object win; Ultralytics-native, supports YOLO11):
  inference-only **+5–7% AP** on aerial (VisDrone/xView), **+12–14%** with slicing-aided fine-tuning.
  Paper: https://arxiv.org/abs/2202.06934 ; guide: https://docs.ultralytics.com/guides/sahi-tiled-inference
  CAVEAT: those numbers are for FCOS/VFNet/TOOD, NOT YOLO — cite as "reported for one-stage detectors
  on aerial data," report your own YOLO number.
- **TTA** (`augment=True` in val/predict): free **+~1 mAP / +1 small-object AP**, 2–3× slower, one flag.
  Tip: raise imgsz ~30% with TTA. https://docs.ultralytics.com/yolov5/tutorials/test-time-augmentation
- Also reliable: higher imgsz (640→1024/1280), mosaic (on by default). NOTE: Ultralytics `copy_paste`
  is **segmentation-only** (needs masks) — won't apply to your bbox datasets without masks.

## 4. Learning paradigms — fit verdicts (web-confirmed 2026-06-29)
- **Federated learning — DROP.** FL-for-UAV exists but ONLY for the multi-drone/swarm/privacy-preserving
  distributed setting; its value is **privacy, not accuracy** (FL usually slightly *underperforms*
  centralized due to data heterogeneity). One student / one dataset / one machine = no motivation;
  contrived and won't improve metrics. (https://arc.aiaa.org/doi/10.2514/1.I011655)
- **Reinforcement learning — DROP.** In detection RL appears only in *meta* roles (active-learning
  sample selection, tracking, image-attribute tuning), never the core detector; complex, no base-mAP
  gain. (https://arxiv.org/pdf/2310.08387)
- **Continual learning — DROP.** Targets incremental new classes + catastrophic forgetting; you have a
  fixed single "person" class and fixed datasets. No fit. (https://www.mdpi.com/1424-8220/20/23/6777)
- **Active learning — MARGINAL** (optional footnote). Legit for "which frames to label," but you need
  only ~15 test frames and self-training uses the unlabeled data better. Not a headline.
- **Test-time adaptation (TENT/CoTTA) — DROP.** Classification-focused, needs large batches + BatchNorm
  surgery, no turnkey YOLO repo, catastrophic-forgetting risk. (TENT: https://github.com/DequanWang/tent ;
  detection-TTA is hard: https://arxiv.org/html/2406.16439v4)
- **Bonus reliable booster — Weighted Boxes Fusion (WBF):** ensembling boxes (Solovyev et al.) gives
  ~+10% mAP50 over the best single model; easy via the `ensemble-boxes` package; ensemble your joint
  model + a higher-imgsz/TTA variant. (https://ar5iv.labs.arxiv.org/html/1910.13302)

## 5. Mamba — CORRECTED framing (important; current ablation doc mis-cites MambaOut)
- **MambaOut (arXiv:2405.07992) does NOT say SSM fails at detection.** It says SSM is unnecessary for
  *classification*, but DETECTION/segmentation ARE long-sequence, and its no-SSM model CANNOT match
  VMamba on COCO (MambaOut-Tiny 45.1 vs VMamba-T 46.5 AP^box). It argues FOR Mamba on detection
  (backbone level). A committee member who knows the paper will catch a misframing.
- **Defensible framing of YOUR null result:** (a) yours was a NECK SSM block at small feature-map
  resolutions — NOT a full SSM backbone at COCO scale, so MambaOut's backbone gains don't apply;
  (b) the honest support is **latency**: pure vision-Mamba has no FPS advantage below ~1024px, selective
  scan only beats attention past ~2K tokens and doesn't use tensor cores, so your **~3× latency is
  exactly what theory predicts**. NVIDIA's MambaVision calls pure Mamba a poor fit for spatial data.
  (MambaVision: https://arxiv.org/html/2407.08083v2 ; Mamba-2 latency: https://tridao.me/blog/2024/mamba2-part3-algorithm/ ;
  Mamba-in-Vision survey: https://arxiv.org/html/2410.03105v1)
- **Do NOT claim a "Mamba detection gains don't replicate" consensus — no such paper exists.**
- Conclusion unchanged: don't spend the month on Mamba (backbone swap = modest ~1.4 AP at throughput
  cost). Keep it as the honest, correctly-framed null result. ACTION: fix the line in
  `2026-06-13_complete_ablation_table.md` that says "Consistent with MambaOut: SSMs don't help
  non-long-sequence vision."

## 6. Dataset / citation map for the thesis
- **C2A**: Nihal et al., "UAV-Enhanced Combination to Application," arXiv:2408.04922 / Springer LNCS
  10.1007/978-3-031-78341-8_10. 10,215 imgs, 360k instances; LSP/MPII-MPHB poses on AIDER backgrounds;
  **47% of instances <10px, 52% 10–50px, 1% 50–300px**; benchmark best YOLOv9-e 0.6883/0.8927.
- **SARD**: Sambolek & Ivasic-Kos 2021 (IEEE Access); ~1,980 FHD imgs, ~6,525 person instances, real UAV,
  wilderness. IEEE DataPort record. (Contrast benchmark: HERIDAL.)
- **Cross-dataset (train-C2A→test-SARD) is the C2A paper's own protocol** (their Table 5: C2A-only→SARD
  0.259; General-Human+C2A→SARD 0.660). Your zero-shot/joint numbers are directly comparable.

## 7. Flagged unverified (confirm before quoting verbatim)
- C2A Table 5 figures (0.259 / 0.660) and the pixel-bin percentages were read via a summarizer from the
  arXiv HTML — confirm against the rendered table at https://arxiv.org/html/2408.04922v2.
- SAHI +5–7%/+12–14% are for FCOS/VFNet/TOOD on VisDrone/xView, not YOLO11 — report your own number.
- No primary-source "+X mAP YOLO pseudo-labeling" number — report your measured delta.

## 8. The self-training experiment (the actual plan)
Phase 0: extract ~15–20 frames (10/30/50m); model pre-labels → correct in Roboflow → REAL test set.
Phase 1: eval epoch125 (sliced) on that test set → baseline ("synthetic-trained, no adaptation").
Phase 2: SAHI-pseudo-label the REMAINING unlabeled frames (disjoint from test!), high conf ≥0.5.
Phase 3: fine-tune epoch125 on C2A + SARD + pseudo-labeled real frames (low LR, ~30–50 ep).
Phase 4: re-eval on the SAME real test set → Δ vs Phase 1 = the contribution; check C2A/SARD no-regression.

## 9. DECISIVE UPDATE — all 10 agents in (2026-06-29)

### (1) Mamba is a confirmed dead-end — now beyond doubt (3 independent nails)
- **It won't even BUILD on your Windows box.** `mamba-ssm` / `causal-conv1d` / `selective_scan` CUDA
  kernels are **Linux-only** (no Windows wheels) → forces WSL2/Docker; install reports span "hours to
  days." On a crash-prone single Windows A6000 this alone could eat the month.
  ([mamba #662](https://github.com/state-spaces/mamba/issues/662), [VMamba #26](https://github.com/MzeroMiko/VMamba/issues/26))
- **Independent aerial test: Mamba LOSES to plain YOLO.** SeaDronesSee (UAV small objects):
  MambaYOLO-L 25.5 < YOLOv8-m 26.1 < YOLO11-m 26.3; authors: SSM "optimizations do not translate to
  multi-scale small object detection." ([PLOS One](https://pmc.ncbi.nlm.nih.gov/articles/PMC12212484/))
- **Backbone Mamba shows ZERO small-object gain:** LocalMamba (locality is its entire pitch) gets
  AP_small 26.0 vs Vim 26.1 — no benefit on the exact axis you need. Mamba-YOLO's COCO "win" is vs a
  smaller/faster YOLOv8-N; size-matched it's break-even/negative and never tested vs YOLO11.
=> Keep Mamba as the honest null result (framed via latency + neck-vs-backbone, NOT MambaOut). Do not reopen it.

### (2) DROP — UDA / CycleGAN (high-risk, infeasible in 1 month, mostly not YOLO)
- Landmark UDA detectors (DA-Faster CVPR'18, SWDA CVPR'19) are **Faster R-CNN / Caffe / Detectron2**,
  not YOLO. YOLO-native UDA is immature — the one YOLOv8 gradient-reversal repo self-warns results
  **"vary even with the same random seed"** (fatal for your SEEDS=[0..4] reproducibility).
- CycleGAN synth→real translation: unstable, **multi-day per run**, can hallucinate objects, and is
  documented to **LOSE to plain augmentation** (32.8 vs 38.7 mAP). Its weakness (geometry preservation)
  is exactly your need (pasted humans must not warp). ([UDA survey](https://arxiv.org/abs/2105.13502),
  [CycleGAN<aug result](https://arxiv.org/pdf/2210.15176))

### (3) A SECOND feasible novelty emerged: attack the C2A paste-box artifact directly
C2A's weakness is documented — detectors overfit the **paste boundary** instead of the human
([Cut-Paste-Learn](https://arxiv.org/abs/1708.01642), [Kisantal](https://arxiv.org/abs/1902.07296)).
Cheap, well-evidenced fix: **regenerate/augment C2A with multi-blend compositing** (no-blend + Gaussian
blur + `cv2.seamlessClone` MIXED_CLONE, randomly per paste) so no single boundary signature is learnable.
Dwibedi reports **~+8 mAP** from "varied blending on identical scenes"; CPU-only, at dataset-build time.
Rule (Kisantal): mix real+synthetic, never synthetic-only. (Numbers are GMU/COCO, not aerial — direction
is solid, magnitude is your own to measure.)

### => TWO complementary, feasible novelty angles (both close the synthetic→real gap, no Mamba/GAN):
- **(A) SAHI-guided self-training** on your unlabeled real DJI footage (adapt *using* real data).
- **(B) Multi-blend compositing** to de-bias C2A's paste artifacts (fix the *synthetic* data).
Thesis spine: *"Closing the synthetic→real gap for UAV disaster human detection — better synthetic
compositing (B) + self-training on self-collected real footage (A), validated on real DJI footage at
10/30/50 m, with a state-space (Mamba) neck reported as a null result."* Honest, feasible, and yours.
