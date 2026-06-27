# mAP50-95 benchmark calibration — how other papers score (2026-06-16)

Purpose: calibrate the student's numbers (C2A ~0.62; SARD few-shot 0.39 @200) against the
literature. **Markers:** [V] = verified against source this session; [s] = from search snippet
(treat as indicative). **Big caveat:** cross-paper mAP50-95 is NOISY — dataset version/split,
all-class vs person-only, IoU/conf protocol, and train/test definitions all differ. Use as
CONTEXT, not a leaderboard.

## A. SEMI-SYNTHETIC C2A (the student's dataset — easier → higher numbers)
| Model | mAP50-95 | Source |
|---|---|---|
| Faster R-CNN | 0.366 | C2A paper, ICPR 2024 (arXiv:2408.04922) [V] |
| RetinaNet | 0.383 | C2A paper [V] |
| RTMDet | 0.442 | C2A paper [V] |
| LightSeek-YOLO (YOLOv11n, 1.86M) | 0.473 | MDPI Mathematics 2025 [V] |
| Cascade R-CNN | 0.486 | C2A paper [V] |
| YOLOv5 | 0.492 | C2A paper [V] |
| YOLOv9-c (~25M) | 0.556 | C2A paper [V] |
| **STUDENT baseline / CBAM / CBAM+P2 / Mamba** | **0.615 / 0.616 / 0.615 / 0.614** | this work |
| YOLOv9-e (~58M, best) | 0.688 | C2A paper [V] |
→ Student sits **2nd of 9**, between YOLOv9-c and YOLOv9-e (which is 3× the params).

## B. REAL aerial / UAV person & small-object data (HARDER → much lower numbers)
| Dataset | Model | mAP50-95 | Source |
|---|---|---|---|
| SARD | RetinaNet + Archangel rendered-synth + 200 real | 0.177 | Lee et al. 2024 (arXiv:2405.15203) [V] |
| SARD | YOLOv5l6 (trained on SARD) | 0.247 | SAR person-detection paper [s] |
| **SARD** | **STUDENT CBAM+P2 (C2A-pretrain + 200 real)** | **0.392** | this work, few-shot |
| VisDrone (all 10 cls) | enhanced YOLOv11n | 0.172 | Sci Reports 2026 (s41598-026-35301-2) [V] |
| VisDrone (all 10 cls) | YOLO-MARS (YOLOv8n-based) | 0.234 | YOLO-MARS 2025 [s] |
| VisDrone (all 10 cls) | YOLOv8-m baseline | 0.258 | SOD-YOLO 2025 (arXiv:2507.12727) [V] |
| VisDrone (all 10 cls) | SOD-YOLO | 0.351 | SOD-YOLO 2025 [V] |
| TinyPerson | various recent | ~0.06–0.10 | TinyBenchmark / SP-YOLOv8s [s] |

## Takeaways
1. **Real-aerial mAP50-95 lives at ~0.15–0.35**, even for models *trained on* the target set.
   TinyPerson is brutal (~0.06–0.10). SARD ~0.18–0.25. VisDrone (all-class) ~0.17–0.35.
2. **Student's SARD 0.392 (C2A-pretrain + 200 real) is at/above the TOP of that real-data band**
   and ~2× the Lee et al. SARD synthetic-pretraining anchor (0.177). Strong — but the SARD
   number is cross-paper (different SARD version/split, Roboflow medium-scale export), so report
   it against the student's OWN zero-shot baseline (the controlled curve), and cite the others
   as context.
3. **Never compare a C2A mAP50-95 to a VisDrone/SARD one directly.** C2A (semi-synthetic, pasted
   humans) is intrinsically easier → 0.6+. Real data → 0.2–0.35. A reviewer who sees "0.62 on
   C2A" vs "0.69 on X" must be told the datasets aren't the same difficulty. This difficulty gap
   is itself the thesis's motivation for the cross-dataset study.
4. For the student's own work the honest framing: **competitive-to-strong on C2A (2nd of 9);
   real-world SARD performance only emerges with real-data fine-tuning (zero-shot ≈ 0 → 0.39 @
   200), which beats published synthetic-pretraining anchors.**

Sources: arXiv:2408.04922; arXiv:2405.15203; arXiv:2507.12727; nature s41598-026-35301-2;
mdpi.com/2227-7390/13/19/3231; TinyBenchmark; SAR person-detection / YOLO-MARS search snippets.
