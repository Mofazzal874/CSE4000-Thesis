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

REPRODUCIBLE run (config lives in the PRESETS block below, NOT on the command line). On PC-1,
after the disaster+campus folders are copied next to selftrain_v1 under drone_root:
  python 24_joint_finetune.py --train --preset d3_all --device 0 --resume
  # d3_all = C2A + drone + campus + disaster (unified real-domain model; lifts R-set, fixes void-FP)
  # d3_joint_A = the original drone-only D3 (+40pp drone headline)
The preset fully defines the scientific config; only --device/--resume come from the CLI. A frozen
copy is written to runs_d3\<name>\run_config.json. Then eval the held-out sets with 18_s1_eval:
  drone test_v1, R-set (disaster), campus_eval_v1, C2A test.

Ad-hoc (no preset) still works with explicit flags (logged by Ultralytics to args.yaml):
  python 24_joint_finetune.py --train --checkpoint ... --c2a-root ... --drone-root ... `
    --extra-set disaster disaster-train-v1 20 --extra-set campus campus_train_v1 20 --name d3_all --device 0

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

# ---------------------------------------------------------------- reproducible run configs (IN CODE)
# The scientific config lives here, versioned, so a run is reproduced by `--preset <name>` alone
# (only --device / --resume come from the CLI). A frozen copy is also dumped to run_dir/run_config.json.
# Paths are PC-1 (THE protocol machine); edit here if the machine changes.
_C2A = r"E:\Thesis_mofazzal_2007074\common\c2a\C2A_Dataset\new_dataset3_scenesplit_v1"
_DRONE = r"E:\Thesis_mofazzal_2007074\drone_data\annotations"
_CKPT = r"runs_s1\s3_assign_full\weights\best.pt"
_COMMON = dict(checkpoint=_CKPT, c2a_root=_C2A, drone_root=_DRONE,
               oversample=20, drone_max_side=1280, epochs=80, lr0=0.0005, alpha=0.5, C=12.8,
               batch=12, imgsz=640, patience=50, f2_patience=40,
               mem_frac=0.95, reserve_gb=0.0, seed=0, cache="disk", workers=4, no_amp=False)
PRESETS = {
    # the original reproducible drone-only sim-to-real (D3 headline: drone +40pp AP50)
    "d3_joint_A": {**_COMMON, "name": "d3_joint_A", "extra_set": None},
    # LAP-4 unified real-domain model: C2A + drone + campus + disaster (lifts R-set, fixes void-FP)
    "d3_all": {**_COMMON, "name": "d3_all",
               "extra_set": [["disaster", "disaster-train-v1", "20"],
                             ["campus", "campus_train_v1", "20"]]},
}
# keys that are OPERATIONAL (stay on the CLI, never baked into a preset)
_CLI_ONLY = {"train", "selftest", "preset", "device", "resume"}


def _imgs(d: Path):
    return sorted(str(p) for p in Path(d).rglob("*") if p.suffix.lower() in IMG_EXT)


def build_combined_list(c2a_images_dir: Path, real_sets, out_txt: Path):
    """C2A train imgs (x1) + each real set's imgs (x its oversample) -> newline image-list.
    real_sets = list of (name, images_dir, oversample). Returns (n_c2a, per_set_counts, n_total)
    where per_set_counts = [(name, n_imgs, oversample), ...]."""
    c2a = _imgs(c2a_images_dir)
    lines = list(c2a)
    counts = []
    for name, imgs_dir, ov in real_sets:
        imgs = _imgs(imgs_dir)
        lines += imgs * int(ov)
        counts.append((name, len(imgs), int(ov)))
    Path(out_txt).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(c2a), counts, len(lines)


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


def _prep_split(split_dir: Path, out_dir: Path, img_max_side: int = 1280):
    """Convert ONE COCO(_sc) split dir -> YOLO images/+labels/. Returns the images dir Path.
    Works for any Roboflow set that has <split_dir>/_annotations_sc.coco.json + images."""
    c2y = import_module("23_coco_to_yolo")
    coco = split_dir / "_annotations_sc.coco.json"
    if not coco.is_file():
        raise FileNotFoundError(f"missing {coco} -- run 22_coco_singleclass.py --tree on {split_dir.parent} first")
    stats = c2y.export_split(coco, split_dir, out_dir, copy_images=True, img_max_side=img_max_side)
    print(f"[d3] {split_dir.parent.name}/{split_dir.name}: {stats['images']} imgs, {stats['boxes']} boxes, "
          f"{stats['backgrounds']} void-negatives -> {stats['images_dir']}")
    return Path(stats["images_dir"])


def _prep_drone(drone_root: Path, out_root: Path, img_max_side: int = 1280):
    """Export drone selftrain_v1 train+valid COCO(_sc) -> YOLO. Returns (train_imgs, val_imgs)."""
    train = _prep_split(drone_root / "selftrain_v1" / "train", out_root / "train", img_max_side)
    valid = _prep_split(drone_root / "selftrain_v1" / "valid", out_root / "valid", img_max_side)
    return train, valid


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

    # 1) real COCO sets -> YOLO, 2) combined oversampled train list, 3) data.yaml
    drone_root = Path(a.drone_root)
    yolo_root = drone_root.parent / "drone_yolo"
    train_imgs, val_imgs = _prep_drone(drone_root, yolo_root, a.drone_max_side)
    real_sets = [("drone", train_imgs, a.oversample)]     # base real set (unchanged D3 default)
    for name_, sub, ov in (a.extra_set or []):            # disaster / campus / ... each = <sub>/train
        set_imgs = _prep_split(drone_root / sub / "train", yolo_root / f"{name_}_train", a.drone_max_side)
        real_sets.append((name_, set_imgs, int(ov)))
    c2a_train_images = Path(a.c2a_root) / "train" / "images"
    if not c2a_train_images.is_dir():
        print(f"FATAL: C2A train images not found: {c2a_train_images}"); return 1
    name = a.name
    run_dir = RUNS / name
    run_dir.mkdir(parents=True, exist_ok=True)
    # freeze the exact resolved config next to the results (spec-6 reproducibility; single source of truth)
    import json as _json
    cfg_keys = ("preset", "checkpoint", "c2a_root", "drone_root", "extra_set", "name", "oversample",
                "drone_max_side", "epochs", "lr0", "alpha", "C", "batch", "imgsz", "patience",
                "f2_patience", "mem_frac", "reserve_gb", "seed", "cache", "workers", "no_amp", "device")
    (run_dir / "run_config.json").write_text(
        _json.dumps({k: getattr(a, k, None) for k in cfg_keys}, indent=2), encoding="utf-8")
    train_txt = run_dir / "combined_train.txt"
    nC, counts, nT = build_combined_list(c2a_train_images, real_sets, train_txt)
    parts = ", ".join(f"{n}: {k}x{o}" for n, k, o in counts)
    print(f"[d3] combined train: {nC} C2A + [{parts}] = {nT} entries -> {train_txt}")
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
    # single real set (drone) == the reproducible D3 default
    nC, counts, nT = build_combined_list(tmp / "c2a", [("drone", tmp / "drone", 10)], txt)
    assert nC == 5 and counts == [("drone", 2, 10)] and nT == 5 + 2 * 10, (nC, counts, nT)
    assert len(txt.read_text().strip().splitlines()) == 25
    # multi real set (drone + disaster + campus) == the lap-4 joint retrain
    (tmp / "disaster").mkdir(); (tmp / "campus").mkdir()
    for i in range(3):
        (tmp / "disaster" / f"x{i}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    for i in range(4):
        (tmp / "campus" / f"y{i}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    m_sets = [("drone", tmp / "drone", 10), ("disaster", tmp / "disaster", 5), ("campus", tmp / "campus", 3)]
    nC2, counts2, nT2 = build_combined_list(tmp / "c2a", m_sets, tmp / "list2.txt")
    assert nC2 == 5 and nT2 == 5 + 2 * 10 + 3 * 5 + 4 * 3, (nC2, nT2)
    assert {n: (k, o) for n, k, o in counts2} == {"drone": (2, 10), "disaster": (3, 5), "campus": (4, 3)}, counts2
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
    # PRESET integrity: every preset must carry the full scientific config _train reads
    required = {"checkpoint", "c2a_root", "drone_root", "name", "oversample", "drone_max_side",
                "epochs", "lr0", "alpha", "C", "batch", "imgsz", "patience", "f2_patience",
                "mem_frac", "reserve_gb", "seed", "cache", "workers", "no_amp", "extra_set"}
    for pname, cfg in PRESETS.items():
        missing = required - set(cfg)
        assert not missing, f"preset {pname} missing keys: {missing}"
        assert not (required & _CLI_ONLY), "a scientific key leaked into _CLI_ONLY"
    assert PRESETS["d3_joint_A"]["extra_set"] is None, "d3_joint_A must stay drone-only (reproducible D3)"
    assert len(PRESETS["d3_all"]["extra_set"]) == 2, "d3_all must fold in disaster + campus"
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"SELFTEST OK: multi-set list + data.yaml + F2 dual early-stop + {len(PRESETS)} presets verified")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--preset", choices=sorted(PRESETS),
                    help="load the FULL scientific config from code (reproducible): d3_joint_A (drone-only D3) "
                         "or d3_all (C2A+drone+campus+disaster). Only --device/--resume stay on the CLI.")
    ap.add_argument("--checkpoint", help="best C2A+assignment best.pt to fine-tune from")
    ap.add_argument("--c2a-root", help="scene-split root (has train/images + train/labels)")
    ap.add_argument("--drone-root", help="drone annotations root (has selftrain_v1/{train,valid}/_annotations_sc.coco.json)")
    ap.add_argument("--oversample", type=int, default=20, help="repeat each labeled drone frame N times (drone-heavy mix)")
    ap.add_argument("--extra-set", action="append", nargs=3, metavar=("NAME", "SUBFOLDER", "OVERSAMPLE"),
                    help="fold an ADDITIONAL real COCO set (train split) into the joint mix; repeatable. "
                         "Uses <drone-root>/<SUBFOLDER>/train/_annotations_sc.coco.json. "
                         "e.g. --extra-set disaster disaster-train-v1 20 --extra-set campus campus_train_v1 20")
    ap.add_argument("--drone-max-side", type=int, default=1280, help="downscale drone imgs to this max side (kills 4K RAM/decode blowup; 0=keep 4K)")
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
    ap.add_argument("--cache", default="disk", help="'disk' (safe for mixed C2A+drone), 'ram' (needs lots of RAM), or 'False'")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default="d3_joint_A")
    ap.add_argument("--resume", action="store_true", help="safe to always pass (power-cut recovery)")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(_selftest())
    if a.preset:                                   # config from code wins (reproducibility)
        for k, v in PRESETS[a.preset].items():
            setattr(a, k, v)
        print(f"[d3] preset '{a.preset}' applied from code; CLI only supplies device/resume.")
    if a.train:
        if not (a.checkpoint and a.c2a_root and a.drone_root):
            sys.exit("--train needs --checkpoint, --c2a-root, --drone-root (or a --preset that sets them)")
        sys.exit(_train(a))
    print(__doc__)
