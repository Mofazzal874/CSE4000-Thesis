"""
11_fccg_smoke.py — S0 exit gate: 2-epoch FCCG smoke train (PC-4 / PC-2).
Loads 10_fccg_modules via import_module('10_fccg_modules') — the SAME import
name any later reload must use (house pattern, like import_module('02_nwd_loss_patch')
in the lap-1 eval runner). Do not rename this import path after checkpoints exist.

PC-4 (RTX 4070 12GB, fp32-only):
  python 11_fccg_smoke.py --data <path-to-scene-split-data.yaml> --batch 4 --no-amp
PC-2 (A6000 GPU1):
  python 11_fccg_smoke.py --data <...> --batch 16
Pass criteria (S0): trains 2 epochs without error; [fccg-verify] prints healthy
gate means each epoch; params <= 22.5M; checkpoint save/reload OK (check-load).
"""
import argparse
import sys
from importlib import import_module
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
fccg = import_module("10_fccg_modules")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data.yaml (scene-split copy on this PC)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--no-amp", action="store_true", help="MANDATORY on PC-4")
    ap.add_argument("--device", default=None, help="e.g. 0; PC-2 MUST pass the free GPU (ours=1)")
    a = ap.parse_args()

    if not fccg.register_fccg():
        print("FATAL: ultralytics not importable in this venv")
        return 1
    from ultralytics import YOLO

    ypath = HERE / fccg.YAML_NAME
    if not ypath.exists():
        fccg.emit_yaml()
    model = YOLO(str(ypath))
    n = sum(p.numel() for p in model.model.parameters())
    print(f"[smoke] model built: {n/1e6:.2f}M params (budget <=22.5M)")
    if n > 22_500_000:
        print("FATAL: param budget exceeded")
        return 2

    def _verify(trainer):
        stats = fccg.FCCGActiveCheck.verify(trainer.model)
        pretty = ", ".join("L" + name.split(".")[-1] + ":" + format(v, ".3f")
                           for name, v in stats.items())
        print("[fccg-verify] OK gates={ " + pretty + " }")

    model.add_callback("on_train_epoch_end", _verify)
    kw = dict(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
              optimizer="AdamW", lr0=0.001, seed=0, workers=2, patience=50,
              amp=not a.no_amp, project=str(HERE / "runs_smoke"), name="fccg_s0",
              exist_ok=True, val=True, plots=False)
    if a.device is not None:
        kw["device"] = a.device
    model.train(**kw)
    print("[smoke] S0 SMOKE COMPLETE — copy runs_smoke/fccg_s0 to the results inbox + MANIFEST entry")
    return 0


if __name__ == "__main__":
    sys.exit(main())
