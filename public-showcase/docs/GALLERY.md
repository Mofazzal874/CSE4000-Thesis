# Measured successes and failure cases

These examples come from the official C2A test set and the architectural ablation. They compare the YOLO11m baseline with CBAM + P2. They are separate from the later assignment and adaptation experiments.

The archive selected cases by measured F2 gain/loss or recall at **confidence 0.25**, restricted to images with at least ten annotated people. These are intentionally extreme examples, not a random estimate of average performance. [Original ranked selection at 0.25](../results/failure_and_comparison_cases.csv).

**Overlay legend:** green = matched detection (TP), blue = unmatched detection (FP), red = missed ground truth (FN). Matching uses IoU at least 0.5. The baseline uses confidence 0.19 and CBAM + P2 uses 0.16, their respective archived F2-optimal settings. These are model-specific operating points.

**Audit correction:** the original ranking CSV and the pictures use different thresholds. The captions below use [newly recomputed overlay counts](../results/gallery_overlay_counts.csv) from the cached detections at **0.19/0.16**, so their counts correspond to the pictures. The original 0.25 table is preserved separately.

## Improved case

`collapsed_building_image0509_0.png` has 24 annotated person instances. At the displayed thresholds, recall rises from 0.6250 to 0.8333 and per-image F2 from 0.6250 to 0.8130. CBAM + P2 matches 20 instances, misses 4, and produces 7 false positives. The baseline matches 15, misses 9, and produces 9 false positives.

| YOLO11m baseline | CBAM + P2 |
|---|---|
| ![Baseline detections in the improved case](../assets/examples/improved-baseline.jpg) | ![CBAM and P2 detections in the improved case](../assets/examples/improved-cbam_p2.jpg) |

## Regressed case

`collapsed_building_image0124_3.png` has 35 annotated instances. At the displayed thresholds, recall falls from 0.8857 to 0.7714 and per-image F2 from 0.8708 to 0.7542. CBAM + P2 matches 27 instances, misses 8, and produces 12 false positives. The baseline matches 31, misses 4, and produces 7 false positives.

| YOLO11m baseline | CBAM + P2 |
|---|---|
| ![Baseline detections in the regressed case](../assets/examples/regressed-baseline.jpg) | ![CBAM and P2 detections in the regressed case](../assets/examples/regressed-cbam_p2.jpg) |

The architecture improves the aggregate tiny-target result while still making some individual scenes worse. This pair makes that limitation visible.

## Severe missed-detection case

`flood_image0087_1.png` contains 30 annotated person instances. At confidence **0.16**, the displayed CBAM + P2 overlay matches **1**, misses **29**, and produces **12 false positives**. Recall is 0.0333 and precision is 0.0769. At confidence **0.25**, the original ranking table records zero matches and six false positives; those different counts must not be attached to this picture.

![CBAM and P2 failure case: twenty-nine of thirty annotated people missed at confidence 0.16](../assets/examples/missed-people.jpg)

These overlays were copied from the archive. They were drawn from cached detections rather than produced by a fresh inference run for this companion. C2A is credited to Nihal et al.; see [attribution and availability](AVAILABILITY.md). Click an image to inspect its original resolution.
