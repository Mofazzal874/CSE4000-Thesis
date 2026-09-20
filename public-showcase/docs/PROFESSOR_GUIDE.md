# Professor's guide to the thesis and evidence

**Md Mofazzal Hosen · Roll 2007074 · Project CSER-26-88**

Department of Computer Science and Engineering, KUET

Supervisor: Prof. Dr. Sk. Md. Masudul Ahsan

The thesis investigates tiny human detection in aerial search-and-rescue imagery through architectural ablation, benchmark auditing, tiny-aware assignment, and supervised adaptation to real imagery. This guide links the material available in the current repository. The complete local submission archive has a separate `pendrive/README.md` with source, environment, and checkpoint navigation.

## A ten-minute review

1. **Read the [research overview](../README.md#what-the-study-establishes).** Start with the findings table and the two method diagrams. The headline is a measured size-specific recall improvement, with explicit trade-offs.
2. **Inspect the [three-seed results](../results/tab_4_11b_three_seed_replication.csv).** Sub-8-pixel recall improves in all three seeds: +2.09, +1.66, and +1.11 percentage points. The mean is +1.62 ± 0.49 points, where ± is sample standard deviation.
3. **Compare the [adaptation results](../README.md#3-synthetic-to-real-adaptation).** The large drone improvement and the non-significant unseen-event improvement distinguish target-domain adaptation from broad generalization.
4. **Watch the [50 m demonstration](../demos/drone-50m.mp4).** The [demo notes](../demos/README.md) identify the earlier defense checkpoint, tiling/TTA, confidence, and video conversion. These clips do not demonstrate the later adapted model's measured scores.
5. **Inspect the [successes and failures](GALLERY.md).** The examples include both regression and missed detections. Their captions use counts recomputed at the displayed confidence thresholds.

## Read the report and presentation

| Material | Recommended focus |
|---|---|
| [Final thesis, 88 pages](../documents/thesis.pdf) | Methodology, implementation/results, and conclusions; references establish the relationship to prior methods |
| [Original defense presentation, 43 pages](../presentation/defense-presentation.pdf) | Architectural motivation and visual explanation |
| [Presentation previews and reading notes](../presentation/README.md) | Six selected slides and corrections needed when reading the historical deck |
| [Evidence guide](EVIDENCE.md) | Source tables, bootstrap intervals, metric conventions, and archived discrepancies |

The defense deck predates the assignment/adaptation extension. Some future-work statements therefore describe the earlier stage of the project. Its YOLOv9-e comparison also has a prose discrepancy: the displayed table implies about four AP50 percentage points, not one. The original PDF is preserved and the reading notes explain this.

## Questions the evidence can answer

| Assessment question | Evidence and limit |
|---|---|
| Is the selected architecture justified? | [Official-split ablation](../results/tab_4_4_additive_ablation.csv): CBAM + P2 reaches 0.8533 AP50 at 19.57M parameters and 86.7 GFLOPs; recorded latency is 14.6 ms under its archived protocol |
| Is the smallest-target gain repeatable? | Three matched seeds and a [seed-0 paired-bootstrap summary](../results/bootstrap_s3_control_full_vs_s3_assign_full.json); overall AP50/F2 do not consistently improve |
| Was benchmark overlap examined? | [Scene-split comparison](../results/tab_4_9_scene_leakage_cost.csv); official and scene-disjoint results must remain separate |
| Does adaptation generalize to another event? | [Unseen-event bootstrap](../results/bootstrap_s3_xevent_vs_d3all_xevent.json): +1.69 AP50 points, 95% interval [−3.19, 5.69], p = 0.468; no significant gain established |
| Are unsuccessful experiments disclosed? | Mamba and frequency gating are documented negative/null findings, not claimed successful components |
| Can the shared numbers be checked? | The independent checker verifies export integrity, arithmetic, and summary consistency; it does not rerun training or inference |

## Verify the package

From the repository root:

```powershell
python public-showcase/analysis/verify_results.py
```

From an extracted standalone companion:

```powershell
python analysis/verify_results.py
```

Python 3.10 or newer is sufficient; no detector dependencies or GPU are needed. Full experimental reproduction additionally requires the datasets, checkpoints, implementation, and recorded environment in the complete research handover.

The current repository is public. This guide is a reading route, not a restriction on access to other tracked files. See [availability and attribution](AVAILABILITY.md) for the artifact scope and third-party credits.
