# slide_figures — staged figures for the defense deck (2026-07-09)

Clean COPIES of the report figures, named by their **on-slide figure number**
(`Fig. - 0X` in the deck, per `docs/2026-07-09_defense_presentation_slide_plan.md`).
Figure numbers run in deck order, so sorting by name = build order. The originals in
`Defense/draft1_30_6_26/figures/` are untouched — the report LaTeX references those names;
never rename them.

## Slide → figure map
| Slide | File(s) |
|---|---|
| 1 (title) | `KUET_Logo.png` |
| 3 | `Fig01_Flood_Scene.png` |
| 4 | `Fig02a_Collapsed_Building.png` `Fig02b_Fire.png` `Fig02c_Flood.png` `Fig02d_Traffic_Incident.png` |
| 10 | `Fig03_Ground_Truth_Scene.png` |
| 11, 12-B | `Fig04_Ablation_Chain.png` |
| 12 | `Fig05_Stride_Problem.png` + `Fig06_Size_Distribution.png` (missing) |
| 13 | `Fig07_System_Overview.png` |
| 14 | `Fig08_Running_Example_Input.png` |
| 15 | `Fig09_Backbone_Pyramid_Source.png` — full architecture; **crop the left backbone column** in PowerPoint (or crop the Feature Pyramid stack from Fig07) |
| 16 | `Fig10_CBAM_Module.png` |
| 17 | `Fig11a_CBAM_Effect_Schematic.png` `Fig11b_CBAM_Attention_Map.png` |
| 18 | `Fig12_P2_Branch.png` |
| 19 | `Fig13a_Stride8_Grid.png` `Fig13b_Stride4_Grid.png` `Fig13c_P2_Feature_Response.png` |
| 20 | `Fig14_C3k2Mamba_Block.png` |
| 22 | `Fig15_Output_Detections.png` |
| 23 | `Fig16_Full_Architecture.png` |
| 24 | `Fig17a_Trace_Input.png` `Fig17b_Trace_Attention.png` `Fig17c_Trace_P2_Response.png` `Fig17d_Trace_Detections.png` |
| 26 | `Fig18a_SAHI_Pipeline.png` `Fig18b_SAHI_Slice_Grid.png` `Fig18c_SAHI_Merged_Detections.png` |
| 27 | `Fig19a_TTA_Pipeline.png` `Fig19b_TTA_Detections_Zoom.png` |
| 29 | reuses Fig02a–d and Fig06 (keep the same figure numbers on the slide) |
| 32 | `Fig20_Ablation_Waterfall.png` (missing) |
| 33 | `Fig21_Per_Size_Recall.png` |
| 34 | `Fig22a_PR_Curve.png` `Fig22b_F_vs_Confidence.png` `Fig22c_Reliability_Diagram.png` |
| 35 | `Fig23a_Ground_Truth.png` `Fig23b_Baseline_640px.png` `Fig23c_SAHI_256.png` `Fig23d_SAHI_TTA.png` |
| 37 | `Fig24_Inference_Mode_Recall.png` |
| 39 | `Fig25_Gantt_Timeline.png` (missing) |
| 45 | `ThankYou_Background.png` |
| Hidden H3 | `Hidden_H3_Training_Dynamics.png` |
| Hidden H6 | `Hidden_H6_Confusion_Matrix.png` `Hidden_H6_Reliability_Diagram.png` |
| Hidden H9 | `Hidden_H9_SAHI_Full_Input.png` `Hidden_H9_SAHI_Merged_Zoom.png` |

## Still missing (4 — save here with exactly these names)
| Target name | Where it comes from |
|---|---|
| `Fig06_Size_Distribution.png` | export `figures/fig_size_distribution.drawio` → PNG 300 dpi (or crop report PDF) |
| `Fig20_Ablation_Waterfall.png` | export `figures/fig_ablation_waterfall.drawio` → PNG |
| `Fig25_Gantt_Timeline.png` | crop from compiled report PDF, Chapter I timeline |
| `Bridge_HBPA_Proposal.png` | screenshot slide 18 of the pre-defense PPTX (HBPA block diagram), for Slide 12-B |

## Notes
- ⚠ `Fig07_System_Overview.png` has a typo: the P5 head tile says "stride 16" — should be
  **"stride 32"**. Fix in `figures/fig_overall_framework.drawio`, re-export, replace this
  copy and the original PNG before the defense.
- Duplicates are intentional (e.g. the running-example detections appear as Fig15, Fig17d,
  and ThankYou_Background) so each slide's assets are self-contained.
