# slide_figures — staged copies for the defense deck (2026-07-09)

Clean, slide-ordered COPIES of the report figures. The originals in
`Defense/draft1_30_6_26/figures/` are untouched (the report LaTeX references those names —
never rename them). Naming: `sNN_figXX_description.png` = deck slide NN, on-slide figure
number XX (matches `docs/2026-07-09_defense_presentation_slide_plan.md`). `hNN_` = hidden
slide. Sort by name = build order.

## Still missing (4 items — add here when made)
| Target name | Where it comes from |
|---|---|
| `s12_fig06_size_distribution.png` | export `figures/fig_size_distribution.drawio` → PNG 300 dpi (or crop report PDF) |
| `s32_fig20_waterfall.png` | export `figures/fig_ablation_waterfall.drawio` → PNG |
| `s39_fig25_gantt.png` | crop from compiled report PDF, Chapter I timeline |
| `s12B_hbpa_proposal.png` | screenshot slide 18 of the pre-defense PPTX (HBPA block diagram) |

## Notes
- `s15_fig09_SOURCE_crop_backbone_column.png` is the FULL architecture figure — crop the
  left backbone column in PowerPoint (Picture Format → Crop). Alternative: crop the
  "Feature Pyramid" tile stack from `s13_fig07_overall_framework.png`.
- ⚠ `s13_fig07_overall_framework.png` has a typo: the P5 head tile says "stride 16" —
  should be "stride 32". Fix in `figures/fig_overall_framework.drawio`, re-export, and
  replace this copy (and the original PNG) before the defense.
- Duplicates are intentional: the same source image appears once per slide that uses it
  (e.g. `arch_detections.png` → s22, s24d, s45), so each slide's assets are self-contained.
- Reused figures keep their first on-slide number (scene grid = Fig. 02, size dist = Fig. 06
  when reused on slide 29).
