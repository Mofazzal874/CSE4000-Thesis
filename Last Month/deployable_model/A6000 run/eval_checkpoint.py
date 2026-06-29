"""
eval_checkpoint.py
==================
Evaluate ONE specific checkpoint (e.g. a healthy pre-NaN epoch125.pt) on C2A-test + SARD-test,
reusing ALL the eval logic from joint_c2a_sard_train.py (no training, ~25 min, low GPU load).

Use this to recover the true deployable model when the NaN divergence corrupted best.pt:
the unconditional epoch*.pt checkpoints from the healthy plateau are intact.

HOW TO RUN (in the (2007074) venv, from the A6000 run folder):
    1. set CKPT below to the checkpoint you want (e.g. the highest epoch*.pt BEFORE the NaN epoch).
    2. python eval_checkpoint.py
    3. read the "C2A-test mAP50 / SARD-test mAP50" lines + the eval_summary.json path it prints.
"""
import joint_c2a_sard_train as J     # importing sets up GPU pick, torch, mem-cap, CBAM registration
# The checkpoint pickles CBAM/ChannelAttention/SpatialAttention under the '__main__' module (they were
# defined in the training script when IT ran as __main__). Now __main__ is THIS script, so torch.load
# can't find them -> bind them into this namespace so unpickling resolves '__main__.CBAM'.
from joint_c2a_sard_train import CBAM, ChannelAttention, SpatialAttention  # noqa: F401
import torch, json
from pathlib import Path
from datetime import datetime

# >>> SET THIS to the checkpoint to evaluate (a healthy plateau epoch, before the NaN) <<<
CKPT = r"D:\student_2k20\2007074\A6000 run\runs_joint\20260627_162506_cbam_p2head_joint_c2a_sard\ultra\weights\epoch125.pt"


def run():
    ck = Path(CKPT)
    if not ck.is_file():
        wdir = ck.parent
        print(f"[eval] CKPT not found: {ck}")
        if wdir.is_dir():
            print(f"[eval] available checkpoints in {wdir}:")
            for p in sorted(wdir.glob("*.pt")):
                print(f"        {p.name}  ({round(p.stat().st_size/1024**2,1)} MB)")
        print("[eval] -> set CKPT to one of the above (prefer the highest epoch*.pt BEFORE the NaN epoch).")
        return

    c2a = J.find_c2a_root(); sard = J.find_sard_root()
    c2a_te_i, c2a_te_l, c2a_te_s = J.pick_split(c2a, ["test", "val"])
    sard_te_i, sard_te_l, sard_te_s = J.pick_split(sard, ["test", "valid", "val"])
    J.collapse_labels_to_person(sard_te_l)
    print(f"[eval] C2A  test = {c2a_te_i}")
    print(f"[eval] SARD test = {sard_te_i}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rdir = J.RUNS_DIR / f"{ts}_EVAL_{ck.stem}"
    (rdir / "metrics").mkdir(parents=True, exist_ok=True)
    (rdir / "weights").mkdir(parents=True, exist_ok=True)
    import shutil; shutil.copy2(ck, rdir / "weights" / "best.pt")

    print(f"[eval] loading checkpoint: {ck}")
    model = J.YOLO(str(ck))
    if torch.cuda.is_available():
        model.to("cuda:0")

    # C2A COCO GT (reuse the dataset's annotations if present, else build from YOLO labels)
    c2a_coco = None
    for cj in (c2a / c2a_te_s / f"{c2a_te_s}_annotations.json",
               c2a_te_l.parent / f"{c2a_te_s}_annotations.json"):
        if cj.is_file():
            c2a_coco = cj; break
    sard_coco = J.build_coco_gt_from_yolo(sard_te_i, sard_te_l, rdir / f"sard_{sard_te_s}_coco_gt.json")

    res_c2a = J.evaluate_on(model, "c2a", c2a_te_i, c2a_te_l, c2a, c2a_te_s, c2a_coco, rdir)
    res_sard = J.evaluate_on(model, "sard", sard_te_i, sard_te_l, sard, sard_te_s, sard_coco, rdir)

    summary = {"checkpoint": str(ck), "c2a_test": res_c2a, "sard_test": res_sard,
               "timestamp": ts}
    (rdir / "metrics" / "eval_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print("\n" + "=" * 70)
    print(f"EVAL RESULT for {ck.name}")
    print(f"  C2A-test : mAP50={res_c2a.get('mAP50')}  mAP50-95={res_c2a.get('mAP50-95')}")
    print(f"  SARD-test: mAP50={res_sard.get('mAP50')}  mAP50-95={res_sard.get('mAP50-95')}")
    print(f"  summary  : {rdir / 'metrics' / 'eval_summary.json'}")
    print("=" * 70)
    print("[done] Healthy checkpoint target: C2A ~0.84-0.85 AND SARD ~0.88-0.92 = the deployable model.")


if __name__ == "__main__":
    run()
