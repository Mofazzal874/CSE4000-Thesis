# Evidence and interpretation

Prepared 2026-09-19 from the thesis submission archive's numerical exports and its 2026-09-10 verification report. This companion has not rerun model inference or training.

## Source map

The filenames retain their thesis table identifiers. All CSVs below are copied without altering the source bytes from `05_NUMERICAL_RESULTS/paper_tables/` in the private handover. The matching run records remain in that handover.

| Shared file | Private source record described by the archive |
|---|---|
| [4.4: architectural ablation](../results/tab_4_4_additive_ablation.csv) | Four official-split runs' `metrics/summary.json` |
| [4.5: size-bin counts and recall](../results/tab_4_5_per_size_recall.csv) | Four runs' `metrics/per_size.csv`, including matched and ground-truth counts |
| [4.9: split comparison](../results/tab_4_9_scene_leakage_cost.csv) | Official-split summary and scene-disjoint G1 retrain record |
| [4.11: assignment pair](../results/tab_4_11_matched_pair_bootstrap.csv) | Control/assignment seed-0 evaluation JSONs and paired-bootstrap JSON |
| [4.11b: seed replication](../results/tab_4_11b_three_seed_replication.csv) | Six evaluation JSONs: control and assignment for seeds 0, 1, 2 |
| [4.13: final model](../results/tab_4_13_final_model_all_heldout_sets.csv) | Five `s1_eval_d3all_*.json` evaluation records |
| [4.13b: adaptation stages](../results/tab_4_13b_adaptation_stages.csv) | Unadapted, drone-only, and all-real evaluation records |

The [result SHA-256 manifest](../results/SHA256SUMS.txt) identifies the packaged CSV and bootstrap JSON bytes. The [complete release manifest](../RELEASE_MANIFEST.json) also records the documents, figures, and media. These establish file identity, not experimental validity. The public script checks arithmetic and consistency of archived statistical summaries, not the correctness of the original detections.

The [unseen-event bootstrap summary](../results/bootstrap_s3_xevent_vs_d3all_xevent.json) reports AP50 change +1.69 percentage points, 95% interval [−3.19, 5.69], and p = 0.468 (0.47 rounded). The [assignment bootstrap summary](../results/bootstrap_s3_control_full_vs_s3_assign_full.json) records the seed-0 intervals. Both are byte-identical original summary files. They do not contain the underlying resampled predictions.

## Reading the statistics

- **+1.62 ± 0.49 points:** mean and sample standard deviation across the three seed deltas in Table 4.11b. This is not a pooled confidence interval or an uncertainty interval on every future run.
- **+2.08 points, [1.77, 2.38]:** seed-0 bootstrap summary in Table 4.11. Rounded point estimates in Table 4.11b imply +2.09. Preserve the distinction rather than silently editing an exported value.
- **Zero bootstrap p-values in the CSV:** report as `p < 0.001` for the archived 1,000-resample analysis, not as a literal zero probability.
- **F2 (operational) label in the original Table 4.11 CSV:** the archive verification identifies these values as `Best_F2`, not F2 at confidence 0.25. The CSV is preserved verbatim; the README uses the more precise interpretation.
- **AP-small is not sub-8-pixel recall.** COCO small-object AP covers areas below 32² pixels; the sub-8-pixel recall statistic isolates a smaller subset.

## Documented archive discrepancies

The September 10 verification report records 369, rather than the older prose's 371, additional very-tiny matches in the official-split architectural comparison. This companion uses 18,991 − 18,622 = 369.

That report also notes two unlocated ECA screening values, a benchmark caption describing a single batch size where runs used several, and a two-instance difference between a printed drone-training count and the annotation export. Those contested values are not used in the headline tables here. The missing seed-2 control evaluation had already been recovered before the three-seed export was assembled.

The September 19 visual audit found an additional mismatch in the gallery: the case-selection CSV inherits the training runner's per-image metrics at confidence **0.25**, while the overlay generator uses **0.19 for the baseline and 0.16 for CBAM + P2**. The public [gallery counts](../results/gallery_overlay_counts.csv) were recomputed from cached detections using the overlay generator's matching procedure. For example, the displayed flood failure has 1 TP / 12 FP / 29 FN at 0.16, while its original ranking row has 0 TP / 6 FP / 30 FN at 0.25. The source archive is unchanged. Gallery captions now use the overlay counts and identify both protocols.

## Figure provenance

`research-overview.svg` and `method.svg` are new explanatory schematics prepared for this companion. They describe the study design and the retained method at a conceptual level. They are not experimental images or exact implementation graphs and contain no third-party footage.

`results-overview.png` is a new plot of Tables 4.11b and 4.13b. The [gallery](GALLERY.md) links five original archived overlays to the shared ranked-case CSV. Slide previews are renders of the original [presentation PDF](../presentation/defense-presentation.pdf). Video previews are frames from the included annotated clips; conversion settings and source hashes are recorded in [media provenance](../demos/media.json).

## Document audit

The 88-page thesis download matches the final internal report and the pendrive copy byte-for-byte. The older version-history PDF differs in the cover's project-number field. The 43-page presentation download matches both archived copies exactly. Its ten embedded XML files are equation markup; they are not detector source code. The PPTX, which is not included, contains two embedded videos.

The presentation predates the extended study. Its [reading notes](../presentation/README.md) distinguish historical scope and flag the incorrect one-point comparison against YOLOv9-e: the slide's displayed AP50 numbers imply approximately four points. The original presentation and thesis are preserved; the companion provides corrections alongside them.

## What readers can independently check

The included data support checking table arithmetic, the three-seed summary, size-count ratios, and adaptation differences. Re-evaluating predictions, recomputing bootstrap intervals, or reproducing training additionally requires the underlying prediction, annotation, configuration, and model artifacts. Those are not shipped in the public companion.
