r"""
19_seam_probe.py -- P1 seam-reliance probe (the motivator for the D3 sim-to-real pillar).

Hypothesis: C2A is a SYNTHETIC composite; pasted humans carry high-frequency paste-seam
artifacts a C2A-trained detector can lean on but that do NOT exist in real imagery. If so,
destroying high-frequency content (low-pass Gaussian blur / JPEG re-compression) should
degrade AP on C2A FASTER than on a real set (SARD). The degradation SLOPE = seam reliance,
and it motivates the C2A-H harmonization step. ANY outcome is reportable.

Eval-only, no training. Degrades the test images IN MEMORY (no temp files) at several levels,
runs the C2A model on each, and reports AP_small / very-tiny recall / AP50 vs degradation
level -- reusing the S1 metric harness (18_s1_eval) so the numbers are contract-consistent.
Run it on C2A test now; re-run on a real set later with the SAME --weights, then --compare.

  $T = "E:\Thesis_mofazzal_2007074\common\c2a\C2A_Dataset\new_dataset3_scenesplit_v1\test"
  python 19_seam_probe.py --tag c2a --images-dir "$T\images" --gt-json "$T\test_annotations.json" --device 0
  # later, if a SARD COCO test set is available (same --weights!):
  python 19_seam_probe.py --tag sard --images-dir <sard\images> --gt-json <sard\coco.json> --device 0
  python 19_seam_probe.py --compare c2a sard
Validate plumbing (no GPU needed):
  python 19_seam_probe.py --selftest

--weights default = runs_s1/s1_control/weights/best.pt (the S1 control). For the AUTHORITATIVE
probe pass the G1 300-ep CBAM+P2 best.pt via --weights.
"""
from __future__ import annotations
import argparse
import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
s18 = import_module("18_s1_eval")          # reuse the S1 metric harness
RUNS = HERE / "runs_s1"
PROBE_DIR = HERE / "runs_probe"


# ------------------------------------------------------------ degradations (in memory)
def _lowpass(img, sigma):
    if sigma <= 0:
        return img
    import cv2
    return cv2.GaussianBlur(img, (0, 0), float(sigma))


def _jpeg(img, quality):
    if quality >= 100:
        return img
    import cv2
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else img


# ------------------------------------------------------------ inference + metrics
def _infer_variant(model, names, img_dir, device, imgsz, degrade):
    import cv2
    preds = {}
    for k, fn in enumerate(names):
        img = cv2.imread(str(img_dir / fn))
        if img is None:
            continue
        r = model.predict(degrade(img), imgsz=imgsz, conf=s18.CONF_FLOOR, max_det=300,
                          device=device, verbose=False)[0]
        preds[fn] = {"boxes": r.boxes.xyxy.cpu().numpy().tolist(),
                     "scores": r.boxes.conf.cpu().numpy().tolist()}
        if (k + 1) % 500 == 0:
            print(f"    infer {k + 1}/{len(names)}")
    return preds


def _metrics(preds, gt, gt_raw):
    o = s18.evaluate(preds, gt, gt_raw)
    return {"AP50_allpoint": o["AP50_allpoint"],
            "AP_small": (o["coco"]["AP_small"] if o.get("coco") else None),
            "recall_very_tiny": o["per_size_recall"]["very_tiny"],
            "recall_tiny": o["per_size_recall"]["tiny"],
            "recall_small": o["per_size_recall"]["small"]}


# ------------------------------------------------------------ reporting
def _drop(base, r, k):
    if base.get(k) is None or r.get(k) is None:
        return "  n/a  "
    return f"{(r[k] - base[k]) * 100:+.2f}pp"


def _print_slopes(result):
    print(f"\n=== seam-reliance [{result['tag']}] -- degradation drop from original ===")
    lp = result["lowpass"]
    if lp:
        base = next((r for r in lp if r["sigma"] == 0.0), lp[0])
        print("  low-pass Gaussian blur:")
        for r in lp:
            print(f"    sigma={r['sigma']:<4} AP_small {_drop(base, r, 'AP_small')}  "
                  f"VT {_drop(base, r, 'recall_very_tiny')}  AP50 {_drop(base, r, 'AP50_allpoint')}")
    jp = result["jpeg"]
    if jp:
        base = next((r for r in jp if r["quality"] == 100), jp[0])
        print("  JPEG re-compression:")
        for r in jp:
            print(f"    q={r['quality']:<4} AP_small {_drop(base, r, 'AP_small')}  "
                  f"VT {_drop(base, r, 'recall_very_tiny')}  AP50 {_drop(base, r, 'AP50_allpoint')}")


def _compare(tag_a, tag_b):
    def load(t):
        p = PROBE_DIR / f"seam_probe_{t}.json"
        if not p.is_file():
            print(f"[compare] missing {p} -- run --tag {t} first")
            return None
        return json.loads(p.read_text(encoding="utf-8"))
    ra, rb = load(tag_a), load(tag_b)
    if not ra or not rb:
        return 1
    print(f"\n=== seam-reliance slope: {tag_a} vs {tag_b} (bigger drop = more seam reliance) ===")
    for sweep, key, base_val, label in (("lowpass", "sigma", 0.0, "strongest blur"),
                                        ("jpeg", "quality", 100, "harshest JPEG")):
        def worst_drop(r):
            rows = r[sweep]
            base = next((x for x in rows if x[key] == base_val), rows[0])
            # most-degraded setting = last in list (assumes user passed increasing severity)
            far = rows[-1]
            for metric in ("AP_small", "AP50_allpoint"):
                if base.get(metric) is not None and far.get(metric) is not None:
                    return metric, (far[metric] - base[metric]) * 100, far[key]
            return "AP50_allpoint", 0.0, far[key]
        ma, da, sa = worst_drop(ra)
        mb, db, sb = worst_drop(rb)
        print(f"  {label:15s}: {tag_a} {ma} {da:+.2f}pp (@{sa})   |   {tag_b} {mb} {db:+.2f}pp (@{sb})")
    print("  -> the set with the LARGER AP drop relies more on high-frequency (seam) cues.")
    return 0


# ------------------------------------------------------------ selftest
def _selftest():
    assert hasattr(s18, "evaluate") and hasattr(s18, "load_gt"), "18_s1_eval not importable"
    try:
        import cv2  # noqa: F401
        rng = np.random.default_rng(0)
        img = (rng.random((64, 64, 3)) * 255).astype("uint8")
        assert np.array_equal(_lowpass(img, 0), img)
        blur = _lowpass(img, 2.0)
        assert blur.shape == img.shape and not np.array_equal(blur, img)
        assert np.array_equal(_jpeg(img, 100), img)
        jq = _jpeg(img, 30)
        assert jq.shape == img.shape and not np.array_equal(jq, img)
        print("[selftest] degradation ops OK (cv2 present)")
    except ImportError:
        print("[selftest] cv2 not installed here -> degradation ops run on the PC (skipped)")
    print("SEAM-PROBE SELFTEST OK")
    return 0


# ------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", help="label for this set, e.g. c2a / sard")
    ap.add_argument("--weights", default=str(RUNS / "s1_control" / "weights" / "best.pt"))
    ap.add_argument("--images-dir")
    ap.add_argument("--gt-json")
    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--lowpass", default="0,0.5,1.0,1.5,2.0", help="Gaussian sigmas; 0 = original")
    ap.add_argument("--jpeg", default="100,90,70,50,30", help="JPEG qualities; 100 = original")
    ap.add_argument("--compare", nargs=2, metavar=("TAG_A", "TAG_B"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return _selftest()
    if a.compare:
        return _compare(*a.compare)
    if not (a.tag and a.images_dir and a.gt_json):
        ap.error("--tag, --images-dir, --gt-json required (or --compare / --selftest)")

    fccg = import_module("10_fccg_modules")
    if not fccg.register_fccg():                 # embeds CBAM too -> loads control or fccg ckpts
        print("FATAL: ultralytics not importable in this venv")
        return 1
    from ultralytics import YOLO
    if not Path(a.weights).is_file():
        print(f"FATAL: weights not found: {a.weights}")
        return 1
    model = YOLO(a.weights)

    img_dir = Path(a.images_dir)
    names = sorted(p.name for p in img_dir.iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if a.limit:
        names = names[:a.limit]
    gt, gt_raw = s18.load_gt(a.gt_json)
    sigmas = [float(x) for x in a.lowpass.split(",") if x != ""]
    quals = [int(x) for x in a.jpeg.split(",") if x != ""]

    result = {"tag": a.tag, "weights": a.weights, "n_images": len(names),
              "lowpass": [], "jpeg": []}
    print(f"[probe:{a.tag}] low-pass sweep {sigmas} over {len(names)} imgs")
    for sg in sigmas:
        preds = _infer_variant(model, names, img_dir, a.device, a.imgsz, lambda im, s=sg: _lowpass(im, s))
        m = _metrics(preds, gt, gt_raw); m["sigma"] = sg
        result["lowpass"].append(m)
        print(f"  sigma={sg}: AP_small={m['AP_small']} VT={m['recall_very_tiny']} AP50={m['AP50_allpoint']}")
    print(f"[probe:{a.tag}] JPEG sweep {quals}")
    for q in quals:
        preds = _infer_variant(model, names, img_dir, a.device, a.imgsz, lambda im, qq=q: _jpeg(im, qq))
        m = _metrics(preds, gt, gt_raw); m["quality"] = q
        result["jpeg"].append(m)
        print(f"  q={q}: AP_small={m['AP_small']} VT={m['recall_very_tiny']} AP50={m['AP50_allpoint']}")

    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    outp = PROBE_DIR / f"seam_probe_{a.tag}.json"
    outp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _print_slopes(result)
    print(f"\n[probe] saved -> {outp}   (run a real set with the same --weights, then --compare)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
