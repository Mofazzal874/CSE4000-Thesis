r"""
21_yolo26_anchor.py -- S4 anchor baseline: YOLO26m on the scene-split, AdamW-pinned.

YOLO26m is a LOCKED baseline row (decision 2026-07-12; verified real -- arXiv 2606.03748,
native in ultralytics 8.4.x). This trains it under OUR protocol so it drops into the paper
table next to CBAM+P2 / FCCG / assignment. Optimizer PINNED to AdamW lr0=0.001 -- YOLO26's
default MuSGD DIVERGED on our P2 config (falsified locally); do NOT use MuSGD.

Resume-safe (last.pt every epoch, --resume) + GPU-slot guard, same house-pattern as 17/20.
No custom modules -> plain YOLO() loads it; nothing to register.

  # 1) confirm it loads on THIS PC first (auto-downloads COCO weights) -- do before the long run:
  python 21_yolo26_anchor.py --check-load
  # 2) train (300-ep cap, patience 50, AdamW) -- ~12h on the 4070 Ti S:
  python 21_yolo26_anchor.py --train --data E:\Thesis_mofazzal_2007074\scenesplit_pc1.yaml --device 0 --batch 12 --mem-frac 0.95 --resume
  # resume after any stop: re-run the SAME --train line (NEVER delete runs_anchor\yolo26m).
Eval (same harness as S1/S2):
  python 18_s1_eval.py --variant control --tag yolo26m --weights runs_anchor\yolo26m\weights\best.pt --images-dir <test\images> --gt-json <test json> --device 0
"""
from __future__ import annotations
import argparse
import sys
from importlib import import_module
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
RUNS = HERE / "runs_anchor"


def _check_load(a) -> int:
    from ultralytics import YOLO
    model = YOLO(a.model)                       # auto-downloads yolo26m.pt on first call
    n = sum(p.numel() for p in model.model.parameters())
    print(f"[check-load] {a.model} loaded OK: {n / 1e6:.2f}M params, task={model.task}")
    print("[check-load] safe to launch --train")
    return 0


def _train(a) -> int:
    from ultralytics import YOLO
    if a.device is not None:
        try:
            guard = import_module("16_gpu_guard")
            guard.claim(device=int(a.device), mem_frac=a.mem_frac, reserve_gb=(a.reserve_gb or None))
        except Exception as e:
            print(f"[yolo26] gpu-guard skipped ({e})")

    name = a.name
    last_pt = RUNS / name / "weights" / "last.pt"
    resumed = bool(a.resume and last_pt.exists())
    if resumed:
        model = YOLO(str(last_pt))
        print(f"[yolo26] RESUMING {name} from {last_pt}")
    else:
        if a.resume:
            print(f"[yolo26] --resume set but no checkpoint at {last_pt}; starting FRESH (safe).")
        model = YOLO(a.model)                   # pretrained COCO weights (transfer, like the CBAM+P2 baseline)
        print(f"[yolo26] loaded pretrained {a.model}")
    n = sum(p.numel() for p in model.model.parameters())
    print(f"[yolo26] {name} built: {n / 1e6:.2f}M params")

    cache_val = False if str(a.cache).lower() in ("false", "0", "none", "") else a.cache
    kw = dict(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
              optimizer="AdamW", lr0=a.lr0, seed=a.seed, workers=a.workers,
              patience=a.patience, amp=not a.no_amp, project=str(RUNS), name=name,
              exist_ok=True, val=True, plots=True, deterministic=True, cache=cache_val,
              save_period=-1, resume=resumed)
    if a.device is not None:
        kw["device"] = a.device
    model.train(**kw)
    print(f"[yolo26] DONE. Eval: python 18_s1_eval.py --variant control --tag {name} "
          f"--weights {RUNS / name / 'weights' / 'best.pt'} --images-dir <test\\images> "
          f"--gt-json <test_annotations.json> --device 0")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-load", action="store_true", help="verify the anchor loads here, then exit")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--model", default="yolo26m.pt", help="pretrained anchor (auto-downloads); or yolo26m.yaml for scratch")
    ap.add_argument("--data")
    ap.add_argument("--name", default="yolo26m")
    ap.add_argument("--epochs", type=int, default=300, help="300-ep cap (matches CBAM+P2 baseline); patience stops early")
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--lr0", type=float, default=0.001, help="AdamW pinned (MuSGD diverged on our P2 config)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-amp", action="store_true", help="only on fp32-only PCs (PC-4)")
    ap.add_argument("--mem-frac", type=float, default=0.90)
    ap.add_argument("--reserve-gb", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default="ram")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true", help="safe to always pass (power-cut recovery)")
    a = ap.parse_args()
    if a.check_load:
        sys.exit(_check_load(a))
    if a.train:
        if not a.data:
            sys.exit("--train needs --data")
        sys.exit(_train(a))
    print(__doc__)
