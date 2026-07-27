# REPRODUCE / CD contents — tiny-aware assignment + sim-to-real (2026-07-27)

Everything needed to re-run the study. Reproducibility note: runs use `seed=0`,
`deterministic=True`; results reproduce to **within noise**, not bit-identical (`cache=ram`,
the non-deterministic `adaptive_max_pool2d` in CBAM, and power-cut resumes prevent bit-exactness
— standard for DL). All headline numbers are in `../05-07-2026-Novelty-Lap/results/MANIFEST.md`.

## 0. Environment  (⚠️ MUST capture from PC-1 — see gap below)
- Python 3.11.9, **ultralytics 8.4.56**, torch 2.12.0+cu126, CUDA 12.6, pycocotools, opencv, Pillow.
- On PC-1 run `pip freeze > requirements.txt` and put it on the CD (the laptop has an OLD ultralytics
  8.0.196 that CANNOT load these checkpoints — freeze the PC-1 venv, not the laptop).
- Recreate: `python -m venv env; env\Scripts\activate; pip install -r requirements.txt`.

## 1. Data (put on CD, or the scripts + a download note)
- **C2A** official + scene-disjoint re-split `new_dataset3_scenesplit_v1` (built by lap-1 script 01
  from the deterministic `evidence\scene_assignment.csv`; md5s frozen in PC1_RUN_STATUS.md).
- **SARD** (Roboflow YOLO export) — external.
- **Own real data** (Roboflow COCO exports + raw videos): drone `test_v1`/`selftrain_v1`, `rset_v1`,
  `campus_*`, `disaster_train_v1`. Frames regenerable from the videos via scripts 05/12/27.

## 2. Reproduce each result (commands = what actually ran; expected = MANIFEST)
Scripts in `../10-07-2026-Novelty-Lap-3/scripts/`. All `--selftest` first.
| stage | command | expected |
|---|---|---|
| G1 baseline (CBAM+P2, 300ep) | thesis script (Benchmarking YOLOs) | AP50 0.8372 |
| S1 pilots control/+FCCG (50ep) | `17_s1_pilot.py --variant {control,fccg}` | FCCG NULL |
| S2 assignment α-sweep | `20_assign_patch.py --train --alpha {0.5,1.0}` | VT +1.89/+3.62pp |
| S3 assignment FULL | `20_assign_patch.py --train --alpha 0.5 --pretrained yolo11m.pt --epochs 300 --patience 50` | AP50 0.8441 |
| control FULL (α=0) | same, `--alpha 0.0 --name s3_control_full` | AP50 0.8449 (pair: +2pp AP_small/VT) |
| D3 sim-to-real | `22_coco_singleclass` → `23_coco_to_yolo` → `24_joint_finetune.py --train` | drone 0.324→0.725 |
| eval (any model) | `18_s1_eval.py --weights <best.pt> --gt-json <coco>` | F2/AP_small/per-size |
| seam probe | `19_seam_probe.py --tag c2a` | HF-reliance curve |
| YOLO26m anchor | `21_yolo26_anchor.py --train` | baseline row |
| disaster pillar | `27_disaster_extract` → SAM3 → `22`→`23`→`24` | R-set + void-FP |

## 3. Trained models to include (best.pt, ~40MB each)
`s3_assign_full`, `s3_control_full`, `d3_joint_A` (+ optionally s1_control/s1_fccg/s2_assign). On PC-1
under `...\scripts\runs_s1\` and `runs_d3\`. (These are gitignored — copy them onto the CD manually.)

## 4. CD folder layout (suggested)
```
CD/
  code/          <- git export of the repo (scripts + docs + MANIFEST + README/START_HERE)
  requirements.txt  <- pip freeze from PC-1
  data/          <- datasets (or a DATA_SOURCES.txt with paths + download links)
  models/        <- the best.pt files from Sec 3
  results/       <- results\ tree (MANIFEST + eval jsons + results.csv + plots)
  REPRODUCE.md   <- this file
  thesis.pdf / paper.pdf
```

## 5. Gaps to close before burning the CD
- [ ] `requirements.txt` — run `pip freeze` on PC-1 (env spec; the ONE thing not yet captured).
- [ ] copy the `best.pt` models (Sec 3) + latest eval jsons off PC-1 into `results\`/`models\`.
- [ ] `git archive` the repo (or copy the working tree minus gitignored data) into `code\`.
