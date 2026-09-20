# Tiny human detection in aerial images

This project studies how to detect people who occupy only a few pixels in drone and disaster imagery. It compares changes to YOLO11m, training assignment for small targets, and fine-tuning on real footage.

[Results](#results) | [Video demos](demos/README.md) | [Detection examples](docs/GALLERY.md) | [Evaluation notes](docs/EVIDENCE.md)

## Method

The detector uses CBAM attention and a stride-4 P2 detection branch. A separate training experiment blends CIoU and normalized Wasserstein distance in the assignment score, with a weight of 0.5. The real-data experiment fine-tunes the model on labelled drone, campus, and disaster images.

![Detector architecture and training assignment](assets/method.svg)

The diagram summarizes the method; it is not a layer-by-layer network specification.

## Results

### Architecture

Results on the official C2A split:

| Model | AP50 | AP50:95 | Parameters | GFLOPs |
|---|---:|---:|---:|---:|
| YOLO11m | 0.8432 | 0.6151 | 20.03M | 67.7 |
| + CBAM | 0.8473 | 0.6161 | 19.08M | 66.9 |
| + CBAM + P2 | 0.8533 | 0.6153 | 19.57M | 86.7 |
| + Mamba + CBAM + P2 | 0.8521 | 0.6143 | 22.01M | 98.4 |

CBAM replaces an existing attention component, so the parameter count decreases. Mamba adds computation without improving AP50 over CBAM + P2. [Full ablation table](results/tab_4_4_additive_ablation.csv).

The official split contains overlapping background scenes. Later experiments use a scene-disjoint split; scores from the two splits should be read separately.

### Assignment for the smallest targets

On the scene-disjoint split, the assignment change improved recall for targets below 8 pixels by **1.62 +/- 0.49 percentage points** across three seeds. Here, size means sqrt(width x height), and +/- is the sample standard deviation.

| Seed | Recall change |
|---|---:|
| 0 | +2.09 points |
| 1 | +1.66 points |
| 2 | +1.11 points |

Overall AP50 and F2 did not consistently improve. Recall decreased in larger size groups. [Seed results](results/tab_4_11b_three_seed_replication.csv) | [Matched comparison](results/tab_4_11_matched_pair_bootstrap.csv).

### Fine-tuning on real images

| Evaluation set | AP50 before | AP50 after |
|---|---:|---:|
| Drone: 60 held-out frames | 0.3354 | 0.7950 |
| Disaster: 25 held-out frames from training-source videos | 0.1326 | 0.4109 |
| Unseen disaster event: 23 frames | 0.3957 | 0.4126 |
| C2A scene-disjoint: 2,040 images | 0.8441 | 0.8370 |

The drone improvement is substantial. The unseen-event change is not statistically significant (paired bootstrap p = 0.468). These results support adaptation to the collected domains, not reliable transfer to arbitrary disasters. [Adaptation table](results/tab_4_13b_adaptation_stages.csv) | [All held-out results](results/tab_4_13_final_model_all_heldout_sets.csv).

![Recall changes across seeds and AP50 before and after adaptation](assets/results-overview.png)

## Demos

[![Annotated drone footage at 50 m](demos/drone-50m.jpg)](demos/drone-50m.mp4)

[10 m: plain inference](demos/drone-10m.mp4) | [50 m: tiled inference with test-time augmentation](demos/drone-50m.mp4)

These clips use the earlier CBAM + P2 model at confidence 0.40, not the final fine-tuned model. Boxes are predictions. The clips illustrate detections and misses; playback speed is not inference speed. [Settings](demos/README.md) | [Successes and failures](docs/GALLERY.md).

## Limits

- Improving recall for the smallest targets costs recall in other size groups.
- The real-image test sets are small, and frames from the same video are correlated.
- Occlusion was not evaluated as a separately labelled subgroup.
- The experiments do not establish reliability for operational rescue decisions.

## Check the exported results

From this folder, with Python 3.10 or newer:

```powershell
python analysis/verify_results.py
```

The script checks file hashes, table arithmetic, and consistency of the saved summaries. It does not rerun training, inference, or bootstrap resampling.

[Evaluation and threshold notes](docs/EVIDENCE.md) | [Data and method credits](docs/AVAILABILITY.md)
