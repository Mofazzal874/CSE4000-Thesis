# Open-items follow-up (2026-07-23)

_Web-verification follow-up on two open items. Primary sources only; UNFINDABLE is a valid answer._

---

## ITEM 1 — SAFE-Net PDF re-poll — STILL UNAVAILABLE (as of 2026-07-23)

**Verdict: PDF NOT yet released.** Unchanged from the 2026-07-12 "coming soon" state. No mechanism can be extracted; only title + venue + author list are public.

**Sources polled (all primary):**
- **AERO-HPR workshop site** (https://aero-hpr.github.io/): paper is listed in the **Proceedings Track**, status literally `"PDF (coming soon)"`. No link to PDF, CVF page, arXiv, or project page.
- **CVF open access** (openaccess.thecvf.com, CVPR 2026 workshops): search returns nothing — CVPR 2026 workshop proceedings not yet posted for AERO-HPR.
- **arXiv**: no preprint found for the title or author set.
- **Authors' lab page — ViSAL / EE-SPCV Lab, IIT Tirupati** (https://sites.google.com/iittp.ac.in/ee-spcv-lab): only a congratulatory announcement — _"Congrats Sivapuram.A.K, P. Komuravelli, G. J, and R. K. S. Gorthi 'SAFE-Net: Scale-Aware Feature Enhancement for Aerial Person Detection in Flood Disaster Imagery' 2026 CVPRW"_. No abstract, PDF, dataset ref, base-detector, or module description. Contact: visal@iittp.ac.in.

**What IS public:** title, the four authors (Arun Kumar S / "Sivapuram A.K", Komuravelli Prashanth, J. Ganesh Mouli, Gorthi R.K. Sai Subrahmanyam), IIT Tirupati affiliation, CVPRW 2026 AERO-HPR workshop, presented as a poster with pre-recorded video. **No abstract is anywhere public** — cannot even confirm the base detector, whether C2A is used, or any numbers/baselines.

**FCCG-YOLO delta — UNDETERMINED (cannot claim delta yet).** Because no abstract/mechanism is available, we cannot responsibly state which of {cross-scale context gating, high-frequency/evidence branch, tiny-aware label assignment, occlusion-specific loss, sim-to-real} SAFE-Net does or does not do. The name ("Scale-Aware Feature Enhancement") signals a multi-scale feature enhancement, but that is inference from the title only, NOT verified. ACTION: re-poll when CVF CVPR2026 workshop proceedings drop (likely alongside the main-conf camera-ready release); this is the ONE closest-venue rival whose mechanism we still can't see. Do not write a comparison paragraph against SAFE-Net until the PDF is in hand.

---

## ITEM 2 — DN-TOD feasibility for our Ultralytics YOLO11 stack — VERDICT: GO-WITH-EFFORT

**Paper:** "Robust Tiny Object Detection in Aerial Images amidst Label Noise", Haoran Zhu, Chang Xu, Wen Yang, Ruixiang Zhang, Yan Zhang, Gui-Song Xia (the AI-TOD/NWD group). arXiv 2401.08056; Pattern Recognition 2026 (DOI 10.1016/j.patcog.2026.113448). Repo: https://github.com/ZhuHaoranEIS/DN-TOD.

**(a) Framework:** Built on **MMDetection** (README: _"this repository is based on MMDetection"_). Dependencies: Python 3.6+, PyTorch 1.3+, CUDA 9.2+, MMCV, cocoapi-aitod, **RFLA**. Trained with total batch size 1. Datasets: AI-TOD-v1.0 (real), AI-TOD-v2.0 + DOTA-v2.0 (synthetic noisy). NOT a YOLO/Ultralytics codebase.

**(b) The two mechanisms — both are TRAINING-LOOP components, NOT architecture modules:**
- **Class-aware Label Correction (CLC):** _"mitigates inaccurate class supervision by identifying and filtering out class-shifted positive samples."_ = a label/sample-filtering step in supervision. (For our single-class "human" C2A, the class-shift half is largely a no-op — the value for us is the box-noise half.)
- **Trend-guided Learning Strategy (TLS):** _"reduces noisy box-induced erroneous supervision through sample reweighting and bounding box regeneration."_ = sample reweighting + on-the-fly box regeneration during training. This is the part that directly targets our C2A paste-label-noise ceiling (~0.615 AP saturation).
- Paper explicitly states DN-TOD is **detector-agnostic**: _"can be seamlessly integrated into both one-stage and two-stage object detection pipelines."_ +4.9 AP on RFLA baseline @ 40% mixed noise.

**(c) Portability into Ultralytics YOLO11 trainer:** The IDEAS are training-loop (loss reweighting + label correction), which is exactly the layer we already patch for NWD/CBAM — so conceptually compatible and NOT architecture-coupled. BUT the shipped CODE is MMDetection-native and RFLA-coupled (assigner-dependent), batch-size-1, and reuses MMDet's label-assignment plumbing. There is no drop-in; CLC/TLS must be RE-IMPLEMENTED against Ultralytics' TaskAlignedAssigner + v8DetectionLoss, not lifted. TLS's "bounding box regeneration" needs access to per-sample assigned targets across epochs (a trend/history buffer), which is a non-trivial addition to the Ultralytics trainer.

**VERDICT: GO-WITH-EFFORT.** Mechanistically it is the right tool for our paste-label-noise ceiling and lives at the training-loop layer we already modify, so it is feasible — but the effort is a from-scratch re-implementation of CLC+TLS logic in the Ultralytics trainer (est. medium lift, on the order of the NWD patch but larger because TLS needs a cross-epoch target-history buffer and box-regeneration hook), not a code port. For a single-class human task, prioritize TLS (box-noise reweighting/regeneration) and treat CLC as low-priority (class-shift is near-moot with one class). Recommend scheduling as an S2 experiment: reproduce the noise-robustness gain on a small C2A subset first to confirm it moves the ~0.615 ceiling before committing full engineering.

