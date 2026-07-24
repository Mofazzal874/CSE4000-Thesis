r"""
17_s1_pilot.py — S1 paired 50-epoch pilot: control (CBAM+P2) vs treatment (+FCCG).

Purpose: the first experiment that tests whether FCCG actually HELPS. Both arms use the
IDENTICAL protocol (AdamW lr0=0.001, seed 0, 640px, 50 ep, from-scratch) and differ ONLY by the
FCCG insertion, so the delta in AP_small / very-tiny recall is attributable to FCCG alone.

Shared-box protection baked in (user rule 2026-07-24): GPU-slot reservation (16_gpu_guard) +
per-epoch last.pt so a peer's kill costs <=1 epoch (resume with --resume).

RUN BOTH ON PC-2 (sequential; ~4-5 h each on the A6000):
  python 17_s1_pilot.py --variant control --data .\scenesplit_pc2.yaml --device 0 --reserve-gb 40
  python 17_s1_pilot.py --variant fccg    --data .\scenesplit_pc2.yaml --device 0 --reserve-gb 40
Resume after a kill (same command + --resume):
  python 17_s1_pilot.py --variant fccg --data .\scenesplit_pc2.yaml --device 0 --reserve-gb 40 --resume
Quick delta once both finish (Ultralytics-native metrics; full contract via lap-1 script 04 later):
  python 17_s1_pilot.py --compare

Pass criterion (ranking doc S1): +1.5 AP_small OR +2 very-tiny recall for fccg vs control.
NOTE: this runner reports Ultralytics mAP50 / mAP50-95 as a fast signal. The AUTHORITATIVE
AP_small / VT-recall for the go/no-go come from the metric-contract harness (lap-1 script 04)
run on each best.pt afterwards — script 04 must `import_module('10_fccg_modules'); register_fccg()`
before loading the FCCG checkpoint (same pattern as CBAM). Wire that when both runs are done.
"""
from __future__ import annotations
import argparse
import json
import sys
from importlib import import_module
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
fccg = import_module("10_fccg_modules")

RUNS = HERE / "runs_s1"


def _compare() -> int:
    rows = []
    for variant in ("control", "fccg"):
        p = RUNS / f"s1_{variant}" / "s1_metrics.json"
        if p.exists():
            rows.append(json.loads(p.read_text()))
        else:
            print(f"[compare] missing {p} — run --variant {variant} first")
    if len(rows) < 2:
        return 1
    c = next(r for r in rows if r["variant"] == "control")
    f = next(r for r in rows if r["variant"] == "fccg")
    print("\n=== S1 pilot comparison (Ultralytics-native; AP_small/VT-recall via script 04) ===")
    for k in ("mAP50", "mAP50_95", "precision", "recall"):
        d = f[k] - c[k]
        print(f"  {k:10s} control={c[k]:.4f}  fccg={f[k]:.4f}  delta={d:+.4f}")
    print("  (decision on AP_small/VT-recall — run script 04 on both best.pt for the contract)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["control", "fccg"])
    ap.add_argument("--data")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-amp", action="store_true", help="MANDATORY on fp32-only PCs")
    ap.add_argument("--mem-frac", type=float, default=0.90)
    ap.add_argument("--reserve-gb", type=float, default=0.0, help="reserve N GB on shared PC (~40 on PC-2)")
    ap.add_argument("--pretrained", default="", help="'' = from scratch (clean pairing); or 'yolo11m.pt'")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default="ram", help="'ram' (default; beats disk-starvation), 'disk', or 'False'")
    ap.add_argument("--workers", type=int, default=4, help="dataloader workers (bump if data-starved)")
    ap.add_argument("--compare", action="store_true", help="print delta from both runs' saved metrics")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return _selftest()
    if a.compare:
        return _compare()
    if not a.variant or not a.data:
        ap.error("--variant and --data required (or --compare / --selftest)")

    if not fccg.register_fccg():
        print("FATAL: ultralytics not importable in this venv")
        return 1
    from ultralytics import YOLO

    # shared-box guard BEFORE building the model
    if a.device is not None:
        try:
            guard = import_module("16_gpu_guard")
            guard.claim(device=int(a.device), mem_frac=a.mem_frac, reserve_gb=(a.reserve_gb or None))
        except Exception as e:
            print(f"[s1] gpu-guard skipped ({e})")

    # RESUME: load the checkpoint directly (register_fccg() above made CBAM/FCCG classes
    # available so the pickled model unpickles). The bare `yolo` CLI CANNOT do this.
    # resumed = did we ACTUALLY resume? If --resume was set but no checkpoint exists (e.g. the
    # run died before finishing epoch 1), fall back to a clean fresh start and do NOT pass
    # resume=True to train() (which would error with no checkpoint).
    last_pt = RUNS / f"s1_{a.variant}" / "weights" / "last.pt"
    resumed = bool(a.resume and last_pt.exists())
    if resumed:
        model = YOLO(str(last_pt))
        print(f"[s1] RESUMING {a.variant} from {last_pt}")
    else:
        if a.resume:
            print(f"[s1] --resume set but no checkpoint at {last_pt}; starting FRESH (safe).")
        if a.variant == "fccg":
            ypath = HERE / fccg.YAML_NAME
            if not ypath.exists():
                fccg.emit_yaml()
        else:
            ypath = HERE / fccg.CONTROL_YAML_NAME
            if not ypath.exists():
                fccg.emit_control_yaml()
        model = YOLO(str(ypath))
        if a.pretrained:
            try:
                model.load(a.pretrained)
                print(f"[s1] transferred matching layers from {a.pretrained}")
            except Exception as e:
                print(f"[s1] pretrained transfer skipped ({e}) — from scratch")
    n = sum(p.numel() for p in model.model.parameters())
    print(f"[s1] {a.variant} built: {n/1e6:.2f}M params")

    if a.variant == "fccg":
        def _verify(trainer):
            stats = fccg.FCCGActiveCheck.verify(trainer.model)
            pretty = ", ".join("L" + k.split(".")[-1] + ":" + format(v, ".3f")
                               for k, v in stats.items())
            print("[fccg-verify] OK gates={ " + pretty + " }")
        model.add_callback("on_train_epoch_end", _verify)

    name = f"s1_{a.variant}"
    cache_val = False if str(a.cache).lower() in ("false", "0", "none", "") else a.cache
    kw = dict(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
              optimizer="AdamW", lr0=0.001, seed=a.seed, workers=a.workers, patience=a.epochs,
              amp=not a.no_amp, project=str(RUNS), name=name, exist_ok=True,
              val=True, plots=True, deterministic=True, cache=cache_val,
              save_period=-1, resume=resumed)  # cache='ram' beats disk-starvation on a shared box
    if a.device is not None:
        kw["device"] = a.device
    model.train(**kw)

    # fast native metrics (the go/no-go AP_small/VT-recall come from script 04 afterwards)
    try:
        m = model.val(data=a.data, imgsz=a.imgsz, batch=a.batch,
                      device=a.device, verbose=False)
        out = {"variant": a.variant, "params_M": round(n / 1e6, 3),
               "mAP50": float(m.box.map50), "mAP50_95": float(m.box.map),
               "precision": float(m.box.mp), "recall": float(m.box.mr)}
        (RUNS / name / "s1_metrics.json").write_text(json.dumps(out, indent=2))
        print(f"[s1] {a.variant} DONE — mAP50={out['mAP50']:.4f} mAP50-95={out['mAP50_95']:.4f}")
    except Exception as e:
        print(f"[s1] metrics dump skipped ({e})")
    print(f"[s1] best.pt: {RUNS / name / 'weights' / 'best.pt'}")
    print(f"[s1] copy runs_s1/{name} -> results\\pc2\\<date>_S1_{a.variant}\\ + MANIFEST; "
          "after BOTH arms: `--compare`, then script-04 eval for AP_small/VT-recall.")
    return 0


def _selftest() -> int:
    """Parse both YAMLs so a syntax error is caught before a PC run (no ultralytics needed)."""
    fails = 0
    try:
        import yaml
        for txt, tag in ((fccg.FCCG_YAML, "FCCG"), (fccg.CONTROL_YAML, "CONTROL")):
            d = yaml.safe_load(txt)
            det = d["backbone"][-1] if False else d["head"][-1]
            ok = d["nc"] == 1 and isinstance(d["head"], list) and "Detect" in str(det)
            print(f"[{'PASS' if ok else 'FAIL'}] {tag} yaml parses, Detect head present")
            if not ok:
                fails += 1
        # both must end in a 4-tap Detect (P2,P3,P4,P5)
        cd = yaml.safe_load(fccg.CONTROL_YAML)["head"][-1][0]
        fd = yaml.safe_load(fccg.FCCG_YAML)["head"][-1][0]
        ok2 = len(cd) == 4 and len(fd) == 4
        print(f"[{'PASS' if ok2 else 'FAIL'}] both Detect over 4 scales (control taps {cd}, fccg taps {fd})")
        if not ok2:
            fails += 1
    except ImportError:
        print("[skip] pyyaml not installed — yaml parse check skipped (will validate on PC build)")
    print("SELFTEST OK" if fails == 0 else f"SELFTEST FAILED ({fails})")
    return fails


if __name__ == "__main__":
    sys.exit(main())
