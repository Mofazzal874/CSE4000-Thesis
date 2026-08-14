# Paper skeleton — start writing the DONE parts NOW (draft 2026-07-27)

Target: **IEEE JSTARS** (primary) / **WACV 2027**. Two-pillar story. Write against this;
fill the two pending cells (matched control, disaster retrain) when they land.

**Working title:** *Tiny-Aware Assignment and Supervised Sim-to-Real Adaptation for Aerial
Human Detection in Disaster Imagery*

## 1. Introduction  — WRITE NOW (all inputs final)
- SAR context: tiny/occluded humans in aerial disaster imagery; **recall-critical → F2 primary**.
- Gaps in the go-to synthetic benchmark (C2A): (a) **scene leakage**, (b) **paste-label noise
  ceiling ~0.615 AP**, (c) **sim-to-real gap**, (d) **void/black-hole false positives**.
- Contributions: (1) leakage-audited **scene-disjoint protocol**; (2) **tiny-aware NWD-in-assignment**
  (D2, a tunable α dial); (3) **supervised sim-to-real** validated on our **own real 3-altitude
  drone benchmark (+40 pp AP50)** [+ disaster set]. Honest negatives: FCCG architecture = null;
  real-disaster (R-set) remains hard.

## 2. Related Work  — WRITE NOW (refs verified in lap-3 web_verification_log)
YOLO lineage (YOLO11/26) · tiny-object **assignment** line (AI-TOD → RFLA/DCFL/**SimD**/NWD) ·
sim-to-real (CFHA / SF-UT) · C2A + aerial-human detectors (YOLOv9-e bar, SAFE-Net, DERNet).

## 3. Datasets & Protocol  — WRITE NOW
C2A (official leaky + our scene-disjoint re-split; leakage −1.6 AP50) · own drone 3-altitude
(60 test / 101 train, labeled + QC'd) · **R-set** disaster (25) · campus (occlusion) · SARD.
Metric contract: **F2 primary**, per-size recall (VT<8/tiny/small), AP_small, ECE/calibration,
efficiency on the 4070 Ti S.

## 4. Method  — WRITE NOW
4.1 Tiny-aware assignment: NWD blended into TaskAlignedAssigner's localization metric; α trade dial.
4.2 Supervised sim-to-real: joint fine-tune (C2A + labeled real) with oversampling + downscale.
4.3 FCCG (described) + why it was a null → motivates the assignment route.

## 5. Experiments (result → subsection)
| result | status | source |
|---|---|---|
| leakage audit (−1.6 AP50) | ✅ | G1, MANIFEST |
| ablation base → +assignment α-sweep {0,0.5,1.0}; +FCCG null | ✅ | S1/S2, MANIFEST |
| assignment at full protocol (S3) | ✅ | S3, MANIFEST |
| **matched control (α=0) full protocol** — clean D2 pair | ⏳ PC-1 running | s3_control_full |
| **sim-to-real drone: zero-shot 0.324 → 0.725 AP50 (+40pp)** | ✅ | D3, MANIFEST |
| disaster fine-tune: within-video R-set 3× BUT cross-event (unseen AP clip) only +1.7pp over base (within noise) → target-specific, honest limit; C2A/drone/campus held | ✅ | MANIFEST 08-06 |
| baselines: G1 CBAM+P2 (thesis, yolo11m) + cite YOLOv9-e C2A bar | ✅ | MANIFEST (YOLO26m moved to lap-5) |
| efficiency (params/GFLOPs/latency) | ✅ | run folders |

## 6. Results & Discussion  — honest findings
FCCG null (assignment > architecture here) · sim-to-real works with target labels (+40pp) ·
**domain-specificity**: drone-tuning ≠ disaster transfer · R-set hardness (tiny+clutter).

## 7. Conclusion & Future Work
disaster-specific + cross-video data, harmonization (CFHA), occlusion losses.

---
**Do now:** draft §1–§4 + §5 rows marked ✅ (they're final). Leave the two ⏳ cells blank; drop
numbers in when the matched control + disaster retrain finish. Don't wait on experiments to write.
