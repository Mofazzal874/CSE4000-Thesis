# Agent A — Image-level & failure-evidence analysis (2026-07-08)
Purpose: measurable, image-grounded NEEDS that the new (lap-2) architecture must solve.
Method: agent VIEWED 23 C2A test images (all 4 categories, small + native-res), parsed 15 label
files for exact pixel sizes, viewed val_batch GT/pred mosaics + CBAM attention maps from the
CBAM+P2 run, 9 raw drone frames (3 per altitude) + 6 annotated 10m inference outputs + counts.csv,
and 2 real SARD images. No web. Every claim carries a file path (bottom).

## (A) C2A tiny-human sample (~55 resolvable humans sampled)
| Attribute | Share | Notes |
|---|---|---|
| Low contrast vs local background | ~25–30% | dark clothing on rubble/soil/night/smoke |
| Occluded (partial+heavy) | ~40% (≈30% partial, ≈10% heavy) | rubble/water/smoke/vehicles. CAVEAT: true occlusion UNDER-represented — sprites pasted ON TOP of scenes |
| Floating / paste artifact | ~30–35% (fire/flood wide shots ~55%) | people suspended on sky/smoke/water/roof faces |
| Clustered / overlapping | ~45% | typical layout: one dense cluster + scattered singles |

**Decisive quantitative finding (label-parse of 15 images):** sprites are pasted at a NEAR-FIXED
ABSOLUTE pixel size — median √area stays ~9.5–20 px whether the image is 178 px or 4720 px wide
(e.g., flood_image0155_2 @4720×2988: median 20.1 px; traffic_incident_image0201_4 @2269×2150:
median 9.5 px; 30–45% of boxes <8 px in most images; min 3.0 px). So at 640-px training resolution
the native-res images downscale their humans to ~1–4 px — effectively invisible. Single outlier
>64 px in the whole sample: one 85-px hi-vis worker (traffic_incident_image0412_1).

## (B) FN causes (ranked) — what the model misses
1. **Too tiny (<8–16 px absolute)** — dominant; per-size recall VT .7575 vs small .8857; native-res images have sub-pixel humans at 640.
2. **Low contrast / camouflage** — dark-shirt players unboxed in drone sliced_f000720; SARD person blends into boulders (gss124).
3. **Occlusion by smoke/water/rubble** — fire/smoke tiles visibly sparser in val_batch0_pred vs labels.
4. **Dense-cluster merge** — overlapping tiny GT collapses (val_batch mosaics; drone sideline crowds).
5. **OOD-large + motion blur + edge (deployment)** — the LARGEST, closest runners left unboxed in sliced_f000720 while smaller mid-field people detected.

## FP causes (ranked) — what it falsely detects
1. **Printed/depicted human figures** — footballer cut-outs on the "SPA 2026" arch + cricketer on "ANWAR ISPAT" banners get boxed (sliced_f000360/000600). Real-world FP class ABSENT from C2A training. NOVEL deployment failure.
2. **Texture over-detection at whole-image inference** — counts.csv: whole_count > sliced_count on 16/17 near-empty-grass frames (e.g., 24 vs 6) → downscaled 4K→640 whole-frame inference hallucinates people on grass/tree texture.
3. **Debris/clutter shaped like a person** — litter/cones/specks; train wreckage; scoreboards, blue tents, white pads, bicycles, building windows (raw drone frames).
4. **Duplicate boxes in dense clusters / slice overlap** (val_batch1_pred; sideline groups).

**CBAM attention maps are DIFFUSE** — reddish overlays spread scene-wide, not sharp peaks on tiny
bodies → the current attention does NOT localize sub-16 px targets (architecture-relevant).

## (C) Own drone footage (3840×2160)
| Altitude | person √area (native) | equivalence |
|---|---|---|
| 10 m | ~45–55 px | ≈ SARD (53 px), far above C2A |
| 30 m | ~22–28 px | C2A "small" band |
| 50 m | ~13–16 px | C2A tiny/very-tiny band |
Real-world properties C2A lacks: cast shadows, perspective foreshortening, motion blur; contrast
bimodal (bright jerseys reliably hit; dark clothing on shade missed).

## (D) SARD (2 real images viewed)
Boulder scree = simultaneous camouflage-FN + rock-FP hazard; bodies 35–50 px, real shadows.
Confirms 4–5× larger targets than C2A + real cluttered terrain (domain gap).

## (E) THE NEEDS TABLE (ranked, architect-facing)
| # | Need (measurable) | Architectural implication |
|---|---|---|
| N1 | Detect 4–16 px bodies (median 12 px; 34.5% <8 px); raise VT-recall from .7575 | <8 px spans ≤2×2 cells at P2 stride 4; sub-cell at P3 → **feature/spatial-resolution PRESERVATION for sub-16 px** |
| N2 | One model across 4–90 px absolute sizes (50 m≈13 px … 10 m≈50 px … rare 85 px) | size decoupled from image res → **explicit multi-scale handling incl. OOD-large** (single RF band fails: sliced_f000720) |
| N3 | Separate touching/overlapping tiny instances (≤240/img; ~45% clustered) | IoU-NMS merges adjacent tiny boxes → **crowd-aware instance separation** |
| N4 | Person-vs-texture discrimination (grass/rubble/boulders/smoke) to cut FPs | texture alone triggers detections → **scene-context veto beyond local appearance** |
| N5 | Reject human-DEPICTING distractors (printed figures, person-shaped debris) | silhouette-keying without "real 3-D person" cue → **semantic robustness to depicted humans** (novel FP class) |
| N6 | Infer partially visible / low-contrast bodies (smoke/water/rubble; dark clothing) | full silhouette often unavailable → **part-based/context inference + contrast-robust features** |
| N7 | Don't depend on compositing cues (hard edges, no shadow, floating placement); robust to shadow/blur/true occlusion | C2A paste artifacts (~30–35% of sprites float) train shortcuts absent at deployment → **representation robust to composite→real shift** |
| N8 | Calibrated confidence at the tiny-object operating point (preds skew 0.3–0.5) | recall is bought at low conf → **decision reliability/calibration** |

## Artifacts confirmed ABSENT on the laptop (searched)
No failure_grid*/success_grid*/detection_grid*/*taxonomy* anywhere under Last Month/; no
runs_sahi_tta ... qualitative COMPARE jpgs (those live on PC-1, produced 2026-07-07); SARD
runs_fewshot has no pred visuals; drone_inference_out/30m and /50m are empty (only 10m processed).

## Cited files (verification trail)
- C2A: `c2a/C2A_Dataset/new_dataset3/test/images/` — collapsed_building_image0001_3/0010_3/0032_1/0035_0/0052_0/0056_2, fire_image0044_4/0129_2/0153_0/0388_2/0475_1/0500_1, flood_image0087_2/0119_1/0155_2/0391_3/0404_0/0466_1, traffic_incident_image0201_4/0306_2/0412_1/0469_4/0474_4 (+ labels in sibling test/labels).
- Run artifacts: `Last Month/24_01_26- Benchmarking YOLOs/CBAM_P2Head/runs/20260602_063759_yolo11m_cbam_p2head_s0_nogit/ultra/val_batch{0,1}_{labels,pred}.jpg`; `.../architecture/attention_maps/collapsed_building_image000{1_3,5_2}_cbam_spatial.png`.
- Drone: `Drone Shoot/extracted_v1/test_frames/{10m,30m,50m}/...` (9 frames); `Last Month/deployable_model/A6000 run/drone_inference_out/10m/sliced_f000{000,240,360,480,600,720}.jpg` + counts.csv.
- SARD: `Last Month/cross_dataset_SARD/runs_fewshot/mamba_cbam_p2head_N100/sard_N100/train/images/gss124...jpg, gss1031...jpg`.
