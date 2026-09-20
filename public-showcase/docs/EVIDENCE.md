# Evaluation notes

The tables are saved experiment outputs. Training, inference, and bootstrap resampling were not repeated when preparing this package.

## Tables

| File | Experiment |
|---|---|
| [Architectural ablation](../results/tab_4_4_additive_ablation.csv) | Four models on the official C2A split |
| [Per-size recall](../results/tab_4_5_per_size_recall.csv) | Ground-truth and matched-instance counts |
| [Scene split](../results/tab_4_9_scene_leakage_cost.csv) | Official versus scene-disjoint C2A |
| [Matched comparison](../results/tab_4_11_matched_pair_bootstrap.csv) | Assignment versus control, seed 0 |
| [Three seeds](../results/tab_4_11b_three_seed_replication.csv) | Assignment and control across three seeds |
| [Held-out sets](../results/tab_4_13_final_model_all_heldout_sets.csv) | Final fine-tuned model on five sets |
| [Adaptation stages](../results/tab_4_13b_adaptation_stages.csv) | Before and after real-data fine-tuning |

## Definitions

AP50 uses IoU 0.50. AP50:95 averages over IoU thresholds from 0.50 to 0.95. Target size is sqrt(box width x box height) in evaluation coordinates.

Extended-study per-size recall uses each model's own F1-optimal threshold. Best F2 is the maximum over a confidence sweep; it is different from F2 at a fixed threshold. The original matched-pair CSV's "operational" F2 label refers to Best F2.

Official C2A has overlapping background scenes across splits. The scene-disjoint experiment uses 6,135 training, 2,040 validation, and 2,040 test images. Its results should not be mixed with the official-split ablation.

## Uncertainty

The three-seed recall gain is 1.62 +/- 0.49 percentage points (mean +/- sample standard deviation).

The [seed-0 paired bootstrap](../results/bootstrap_s3_control_full_vs_s3_assign_full.json) reports +2.08 points, 95% interval [1.77, 2.38], over 1,000 resamples of 2,040 test images. Rounded seed-table values give +2.09. A saved p-value of zero means p < 0.001 at this resampling resolution.

For the [unseen event](../results/bootstrap_s3_xevent_vs_d3all_xevent.json), the AP50 change is +1.69 points, interval [-3.19, 5.69], p = 0.468. Images are the resampling units; the intervals do not account for all within-video or within-scene correlation.

## Corrected counts

The architectural comparison adds 18,991 - 18,622 = **369** very-small matched instances. An earlier text used 371.

The ranked gallery CSV uses confidence 0.25, while the pictures use baseline 0.19 and CBAM + P2 0.16. [Overlay counts](../results/gallery_overlay_counts.csv) were recomputed at those display thresholds; the [gallery](GALLERY.md) uses those counts.

## File checks

[SHA-256 hashes](../results/SHA256SUMS.txt) identify the numerical exports. The [package manifest](../RELEASE_MANIFEST.json) covers all included files. The checker verifies integrity and arithmetic, not the correctness of the original predictions.
