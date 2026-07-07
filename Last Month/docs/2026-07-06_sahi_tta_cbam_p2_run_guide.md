# SAHI + TTA on CBAM+P2 — run guide + figure-asset plan (2026-07-06)

The March SAHI/TTA run (`31-03-26(Mamba-ViT-CNN)/sahi_tta_eval_noaug.py`) evaluated the
**Mamba** model on Kaggle. The report needs those numbers for **CBAM+P2** (the recommended
model). Two new scripts were built in `Last Month/24_01_26- Benchmarking YOLOs/CBAM_P2Head/`
(both compile-checked; the laptop has no GPU — run them on a GPU PC):

| Script | Produces | Runtime |
|---|---|---|
| `sahi_tta_cbam_p2_thesis.py` | the full SAHI/TTA ablation table + curves (Chapter IV numbers) | ~2–3.5 h on PC-1 |
| `make_report_figure_images.py` | detgrid (#14) + SAHI insets (#15) + TTA insets (#16) + zoom crops | ~3 min |
| `make_arch_figure_images.py` (existed) | arch input/detections/CBAM overlay/P2 map (#10–13) | ~2 min |

## Which PC → **PC-1 (RTX 4070 Ti SUPER 16 GB)**
1. **Spec §5 requires latency on the 4070 Ti SUPER** — the primary ablation table's latency
   (6.5 ms pure / 14.6 ms e2e) was measured there; SAHI/TTA latency must be comparable.
2. Everything is already on PC-1: C2A at `E:\Thesis_mofazzal_2007074\common\c2a\...`, the
   canonical run `Benchmarking YOLOs\CBAM_P2Head\runs\20260602_063759_yolo11m_cbam_p2head_s0_nogit\`.
3. 16 GB fits TTA@1920 (12 GB PC-4 would OOM-skip it). PC-2 = wrong-hardware latency + shared.
   PC-2/PC-4 remain free for the fine-tune/drone work in parallel.

## Model + protocol (locked)
- Weights: canonical s0 run best.pt (C2A-test mAP50 **0.8533**, F2opt 0.8442, vt-recall 0.7575).
  Auto-discovered; KNOWN_BEST_PT list covers PC-1/laptop paths.
- Spec §5: op-metrics conf=0.25/IoU=0.5; val() AP conf=0.001/IoU=0.7; SAHI rows = per-image
  protocol (+ COCO AP from cached dets, floor 0.10 — footnote); TTA rows = official val(augment=True).
- Configs: baseline-640 · SAHI 256/320/512/640 (GREEDYNMM, IOS, standard-pred) · TTA 640/832/1280/1920
  · best-TTA custom per-size pass · SAHI+TTA (best slice + per-tile augment).

## Failproof mechanics
Auto-smoke (10 imgs, all stages) → full run. Per-image JSONL cache per config → power cut
loses nothing; **re-run the same command to resume**. Per-config metrics JSON = completion
marker. OOM → skip+log (never crash). `grand_summary.partial.json` updated after every config
(monitor intermediate results there or in the console).

## Run commands (PC-1, PowerShell)
```powershell
E:\Thesis_mofazzal_2007074\mofazzal1\Scripts\Activate.ps1
cd "E:\Thesis_mofazzal_2007074\Benchmarking YOLOs\CBAM_P2Head"
# transfer the 2 new scripts into this folder first
python make_arch_figure_images.py        # 2 min  -> arch_fig_out\  (#10-13)
python make_report_figure_images.py      # 3 min  -> report_fig_out\ (#14-16 + MANIFEST.md)
python sahi_tta_cbam_p2_thesis.py        # 2-3.5 h -> runs_sahi_tta\<id>\  (needs internet once: pip sahi)
```
Copy back to the laptop: `arch_fig_out\`, `report_fig_out\`, `runs_sahi_tta\<id>\metrics\` + `plots\` + `qualitative\`.

## draw.io slots that need these images (audit of all 12 .drawio)
Only 3 of 12 figures need rasters — 8 slots; all currently EMPTY placeholders. 640² outputs
exceed every minimum. Other 9 figures are pure vector (export as-is).

| Figure | Slot label | Min px (2× geometry) | Fill with |
|---|---|---|---|
| fig_cbam_p2_architecture | input aerial image | 720×180 (slot is 4:1 — reshape slot or letterbox) | `arch_input.png` |
| fig_cbam_p2_architecture | detections | 720×180 | `arch_detections.png` |
| fig_cbam_p2_architecture | CBAM spatial overlay | 300×180 | `cbam_overlay.png` (regenerate — don't trust the old `cbam_spatial_overlay.png` provenance) |
| fig_cbam_p2_architecture | P2 feature map | 300×180 | `p2_featuremap.png` |
| fig_sahi | full image | 260×220 | `sahi_input_full.png` (or `sahi_slice_grid.png`) |
| fig_sahi | merged detections | 200×220 | `sahi_merged_detections.png` / `_zoom.png` |
| fig_tta | input image | 240×220 | `tta_input.png` |
| fig_tta | detections | 220×220 | `tta_detections.png` / `_zoom.png` |

Standalone (not embedded): `detgrid_c2a_s4/s8.png` — real companion of fig_stride_problem.

## Expected outcome (calibrate against the March Mamba run — same test split/bins)
Mamba got: TTA@1280 mAP50 0.8736→**0.8900** (+0.0164), mAP50-95 +0.054; SAHI-256 vt-recall
0.7668→**0.8292** (+0.062) at precision cost (F1 −0.021, F2 +0.007), latency 45→534 ms.
CBAM+P2 starts at mAP50 0.8533 / vt 0.7575 → expect TTA@1280 ≈0.87, SAHI-256 vt ≈0.81–0.83.
If gains land in that band, the Chapter IV story is: *inference-time enhancement recovers the
tiny-object recall at a stated latency cost — quantified on the recommended model.*
