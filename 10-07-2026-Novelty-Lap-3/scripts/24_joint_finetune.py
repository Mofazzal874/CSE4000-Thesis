r"""
24_joint_finetune.py -- D3 supervised sim-to-real (approach A: fine-tune, drone-heavy).

Takes the best C2A+assignment checkpoint and fine-tunes it on a mix of ALL C2A train +
OVERSAMPLED labeled drone frames, so the model adapts to the real domain without forgetting
C2A. Eval on the FROZEN drone test_v1 (never trained on) + C2A shows the sim-to-real gap close.
The drone frames include void-negative backgrounds (empty labels) -> also fixes the void-FP bias.

Core mechanisms wired to spec (Last Month/system_spec_thesis.md sec 6 + sec 9):
  - init from the C2A+assignment best.pt; assignment patch RE-APPLIED (alpha kept) + FCCG classes
    registered so the custom checkpoint unpickles
  - AdamW, lower lr0 for fine-tune, cos_lr, lrf=0.01, close_mosaic=10, DEFAULT aug only
  - DUAL early stop: fitness patience (Ultralytics built-in) + F2 patience (custom callback)
  - power-failure: save_period=5 + resume from last.pt (re-run same command)
  - GPU-slot guard, cache=ram, deterministic, run folder under runs_d3/
  - eval afterwards with 18_s1_eval (F2 + AP_small + per-size recall) on drone test AND C2A

Run on PC-1 (after S3 frees the GPU); drone data copied to E:\...\drone_data\annotations\:
  python 24_joint_finetune.py --train ^
    --checkpoint runs_s1\s3_assign_full\weights\best.pt ^
    --c2a-root E:\Thesis_mofazzal_2007074\common\c2a\C2A_Dataset\new_dataset3_scenesplit_v1 ^
    --drone-root E:\Thesis_mofazzal_2007074\drone_data\annotations ^
    --device 0 --batch 12 --oversample 20 --epochs 80 --lr0 0.0005 --mem-frac 0.95 --resume
Validate the plumbing (no GPU/ultralytics needed):
  python 24_joint_finetune.py --selftest
"""
from __future__ import annotations
import argparse
import sys
from importlib import import_module
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
RUNS = HERE / "runs_d3"
IMG_EXT = (".jpg", ".jpeg", ".png")


def _imgs(d: Path):
    return sorted(str(p) for p in Path(d).rglob("*") if p.suffix.lower() in IMG_EXT)


def build_combined_list(c2a_images_dir: Path, drone_train_images_dir: Path,
                        oversample: int, out_txt: Path):
    """C2A train imgs (x1) + drone train imgs (x oversample) -> newline image-list for Ultralytics."""
    c2a = _imgs(c2a_images_dir)
    drone = _imgs(drone_train_images_dir)
    lines = c2a + drone * oversample
    Path(out_txt).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(c2a), len(drone), len(lines)


def write_data_yaml(train_txt: Path, val_images: Path, out_yaml: Path):
    out_yaml.write_text(
        f"path: {out_yaml.parent.as_posix()}\n"
        f"train: {Path(train_txt).as_posix()}\n"
        f"val: {Path(val_images).as_posix()}\n"
        f"nc: 1\nnames: [person]\n", encoding="utf-8")


def f2_early_stop_cb(patience: int):
    """Custom F2 early stop (spec 6.1 dual early stop). Fires trainer.stop when val-F2 plateaus."""
    st = {"best": -1.0, "bad": 0, "best_ep": 0}

    def cb(trainer):
        m = getattr(trainer, "metrics", None) or {}
        P = m.get("metrics/precision(B)")
        R = m.get("metrics/recall(B)")
        if P is None or R is None:
            return
        f2 = 5 * P * R / max(4 * P + R, 1e-9)
        ep = int(getattr(trainer, "epoch", 0))
        if f2 > st["best"] + 1e-4:
            st.update(best=f2, bad=0, best_ep=ep)
        else:
            st["bad"] += 1
        print(f"[f2-earlystop] ep{ep} F2={f2:.4f} best={st['best']:.4f}@{st['best_ep']} bad={st['bad']}/{patience}")
        if st["bad"] >= patience:
            print(f"[f2-earlystop] STOP: F2 no improvement for {patience} epochs "
                  f"(best {st['best']:.4f} @ ep{st['best_ep']})")
            trainer.stop = True
    return cb


def _prep_drone(drone_root: Path, out_root: Path):
    """Export drone selftrain train+valid COCO(_sc) -> YOLO structure. Returns (train_imgs, val_imgs)."""
    c2y = import_module("23_coco_to_yolo")
    splits = {}
    for split in ("train", "valid"):
        src = drone_root / "selftrain_v1" / split
        coco = src / "_annotations_sc.coco.json"
        if not coco.is_file():
            raise FileNotFoundError(f"missing {coco} -- run 22_coco_singleclass.py --tree first")
        out = out_root / split
        stats = c2y.export_split(coco, src, out, copy_images=True)
        print(f"[d3] drone {split}: {stats['images']} imgs, {stats['boxes']} boxes, "
              f"{stats['backgrounds']} void-negatives -> {out}")
        splits[split] = Path(stats["images_dir"])
    return splits["train"], splits["valid"]


def _train(a) -> int:
    fccg = import_module("10_fccg_modules")
    if not fccg.register_fccg():
        print("FATAL: ultralytics not importable"); return 1
    assign = import_module("20_assign_patch")
    from ultralytics import YOLO

    if a.device is not None:
        try:
            guard = import_module("16_gpu_guard")
            guard.claim(device=int(a.device), mem_frac=a.mem_frac, reserve_gb=(a.reserve_gb or None))
        except Exception as e:
            print(f"[d3] gpu-guard skipped ({e})")

    assign.apply_assign_patch(alpha=a.alpha, C=a.C)          # keep the D2 assignment lever

    # 1) drone COCO -> YOLO, 2) combined oversampled train list, 3) data.yaml
    drone_root = Path(a.drone_root)
    yolo_root = drone_root.parent / "drone_yolo"
    train_imgs, val_imgs = _prep_drone(drone_root, yolo_root)
    c2a_train_images = Path(a.c2a_root) / "train" / "images"
    if not c2a_train_images.is_dir():
        print(f"FATAL: C2A train images not found: {c2a_train_images}"); return 1
    name = a.name
    run_dir = RUNS / name
    run_dir.mkdir(parents=True, exist_ok=True)
    train_txt = run_dir / "combined_train.txt"
    nC, nD, nT = build_combined_list(c2a_train_images, train_imgs, a.oversample, train_txt)
    print(f"[d3] combined train list: {nC} C2A + {nD}x{a.oversample} drone = {nT} entries -> {train_txt}")
    data_yaml = run_dir / "d3_data.yaml"
    write_data_yaml(train_txt, val_imgs, data_yaml)

    # 4) load checkpoint (resume-safe)
    last_pt = run_dir / "weights" / "last.pt"
    resumed = bool(a.resume and last_pt.exists())
    if resumed:
        model = YOLO(str(last_pt)); print(f"[d3] RESUMING {name} from {last_pt}")
    else:
        if a.resume:
            print(f"[d3] --resume set but no checkpoint at {last_pt}; starting from --checkpoint.")
        if not Path(a.checkpoint).is_file():
            print(f"FATAL: checkpoint not found: {a.checkpoint}"); return 1
        model = YOLO(str(a.checkpoint)); print(f"[d3] fine-tuning from {a.checkpoint}")

    model.add_callback("on_fit_epoch_end", f2_early_stop_cb(a.f2_patience))
    model.add_callback("on_train_epoch_end", assign.make_verify_callback())

    cache_val = False if str(a.cache).lower() in ("false", "0", "none", "") else a.cache
    kw = dict(data=str(data_yaml), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
              optimizer="AdamW", lr0=a.lr0, lrf=0.01, cos_lr=True, close_mosaic=10,
              seed=a.seed, workers=a.workers, patience=a.patience,          # fitness early stop
              amp=not a.no_amp, project=str(RUNS), name=name, exist_ok=True,
              val=True, plots=True, deterministic=True, cache=cache_val,
              save_period=5, resume=resumed)                                 # power-fail: periodic + last.pt
    if a.device is not None:
        kw["device"] = a.device
    model.train(**kw)
    print(f"[d3] DONE. Eval the FROZEN drone test + C2A (both must be reported):")
    print(f"     python 18_s1_eval.py --variant control --tag d3_drone --weights "
          f"{run_dir/'weights'/'best.pt'} --images-dir {drone_root/'test_v1'/'test'} "
          f"--gt-json {drone_root/'test_v1'/'test'/'_annotations_sc.coco.json'} --device 0")
    print(f"     (repeat with the C2A scene-split test to confirm no C2A regression)")
    return 0


def _selftest() -> int:
    import json
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    # fake C2A + drone image dirs
    (tmp / "c2a").mkdir(); (tmp / "drone").mkdir()
    for i in range(5):
        (tmp / "c2a" / f"c{i}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    for i in range(2):
        (tmp / "drone" / f"d{i}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    txt = tmp / "list.txt"
    nC, nD, nT = build_combined_list(tmp / "c2a", tmp / "drone", 10, txt)
    assert nC == 5 and nD == 2 and nT == 5 + 2 * 10, (nC, nD, nT)
    assert len(txt.read_text().strip().splitlines()) == 25
    # data.yaml well-formed
    y = tmp / "d.yaml"
    write_data_yaml(txt, tmp / "drone", y)
    body = y.read_text()
    assert "nc: 1" in body and "names: [person]" in body and "train:" in body and "val:" in body
    # F2 callback math + stop
    cb = f2_early_stop_cb(patience=2)

    class T:
        def __init__(s, p, r, ep):
            s.metrics = {"metrics/precision(B)": p, "metrics/recall(B)": r}; s.epoch = ep; s.stop = False
    t = T(0.8, 0.6, 0)
    cb(t); assert not t.stop
    for ep in (1, 2, 3):                       # F2 not improving -> should stop after patience
        t2 = T(0.8, 0.55, ep); cb(t2)
    assert t2.stop, "F2 early-stop did not fire"
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print("SELFTEST OK: combined oversampled list + data.yaml + F2 dual early-stop all verified")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--checkpoint", help="best C2A+assignment best.pt to fine-tune from")
    ap.add_argument("--c2a-root", help="scene-split root (has train/images + train/labels)")
    ap.add_argument("--drone-root", help="drone annotations root (has selftrain_v1/{train,valid}/_annotations_sc.coco.json)")
    ap.add_argument("--oversample", type=int, default=20, help="repeat each labeled drone frame N times (drone-heavy mix)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr0", type=float, default=0.0005, help="lower than pretrain (fine-tune)")
    ap.add_argument("--alpha", type=float, default=0.5, help="keep the D2 assignment lever at this weight")
    ap.add_argument("--C", type=float, default=12.8)
    ap.add_argument("--patience", type=int, default=50, help="fitness early-stop (Ultralytics)")
    ap.add_argument("--f2-patience", type=int, default=40, help="F2 early-stop (custom, spec dual-stop)")
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--mem-frac", type=float, default=0.90)
    ap.add_argument("--reserve-gb", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default="ram")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default="d3_joint_A")
    ap.add_argument("--resume", action="store_true", help="safe to always pass (power-cut recovery)")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(_selftest())
    if a.train:
        if not (a.checkpoint and a.c2a_root and a.drone_root):
            sys.exit("--train needs --checkpoint, --c2a-root, --drone-root")
        sys.exit(_train(a))
    print(__doc__)
