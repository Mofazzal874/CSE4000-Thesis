# Defense Demo — Self-Contained Folder

Everything needed for the thesis-defense demo lives here. No other folder is required.
Created 2026-07-10 (defense: 2026-07-11).

## What is where (provenance)

| Item | Source (do not delete originals) | Notes |
|---|---|---|
| `models\baseline_yolo11m.pt` | `Last Month\24_01_26- Benchmarking YOLOs\Yolo11m\runs\20260615_230315_yolo11m_baseline_s0_nogit\weights\best.pt` | Final AdamW-retrained baseline, seed 0, 2026-06-16 |
| `models\cbam_p2head.pt` | `Last Month\24_01_26- Benchmarking YOLOs\CBAM_P2Head\runs\20260602_063759_yolo11m_cbam_p2head_s0_nogit\weights\best.pt` | Final AdamW CBAM+P2 run, seed 0, 2026-06-02 (matches report ablation) |
| `models\cbam_p2head_finetune_enriched.pt` | `Last Month\deployable_model\A6000 run\runs_joint\20260702_144117_finetune_enriched\ultra\weights\best.pt` | OPTIONAL. Angle-B enriched fine-tune 2026-07-02 — best candidate for the drone-shoot video (decide before Phase 1) |
| `code_snapshots\*.py` | `code\` snapshot inside each run folder | **REQUIRED to load cbam_p2head weights** — CBAM class is registered into `ultralytics.nn.modules` / `nn.tasks` (see lines ~355–446 of `yolo11m_cbam_p2head_thesis.py`). Plain `YOLO("cbam_p2head.pt")` fails without this registration. |
| `run_metadata\` | yaml/json/csv from both run folders | Training config + results curves, for the Results tab of the demo UI |
| `data\c2a_test\` | `c2a\C2A_Dataset\new_dataset3\test` (2043 images + 2043 labels) | Same split as training `data.yaml` (verified against both runs) |
| `data\drone_shoot\` | `Drone Shoot\` 10m.MP4, 30m.MP4, 50m.MP4 | Own drone footage at 3 altitudes |
| `results\` | (empty) | GPU batch-inference outputs land here (JSON predictions, latency log, annotated video) |
| `app\` | (empty) | Gradio demo app goes here (Phase 2) |

Everything was COPIED, not moved — originals are untouched.

## Demo design (agreed 2026-07-10, see docs\2026-07-10_defense_demo_brainstorm.md)
- Compare: **baseline vs CBAM+P2 vs CBAM+P2+SAHI+TTA** on C2A test set + own drone shoot.
- Cached-inference architecture: GPU PC precomputes raw predictions @ conf=0.01 as JSON;
  laptop UI (Gradio) draws boxes instantly, confidence slider re-filters live.
- Optional single "true live" CPU inference (plain CBAM+P2, 640px, ~2–4 s).
- SARD stays out of the demo; prepared honest line + backup slide instead.
- Hard rule: no live internet/AnyDesk dependency during the defense.
