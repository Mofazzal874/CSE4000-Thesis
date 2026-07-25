r"""
18_s1_eval.py -- S1 GO/NO-GO: AP_small + per-size recall (very_tiny / tiny / small)
on the scene-split TEST set for BOTH s1 checkpoints. This is the AUTHORITATIVE pass
criterion (+1.5 AP_small OR +2 very-tiny recall) -- the thing overall mAP50 cannot see.

Why a dedicated script (not lap-1 script 04):
  - The FCCG/CBAM checkpoints pickle custom nn.Modules; bare ultralytics YOLO() can't
    unpickle them. This registers CBAM+FCCG (10_fccg_modules.register_fccg) BEFORE load.
  - Self-contained: needs only ultralytics + numpy (both on PC-1). pycocotools is OPTIONAL
    -- used for AP_small; per-size recall is always computed (needs numpy only). So even
    without pycocotools you still get very-tiny recall = half the pass criterion.

Metric semantics match lap-1 script 04 / the thesis contract:
  size = sqrt(w*h) px; bins very_tiny <8, tiny 8-16, small 16-32, medium 32-96, large >=96.
  Detections at conf floor 0.001; greedy IoU>=0.5 matching, each GT used once.
  per-size recall reported at each model's own F1-optimal threshold (OptThr_F1).
  AP_small = pycocotools COCO 'small' (area <32^2), i.e. our very_tiny+tiny+small combined.

Run on PC-1 (the machine that trained them), scene-split TEST, one per variant, then --compare:
  $T = "E:\Thesis_mofazzal_2007074\common\c2a\C2A_Dataset\new_dataset3_scenesplit_v1\test"
  python 18_s1_eval.py --variant control --images-dir "$T\images" --gt-json "$T\test_annotations.json" --device 0
  python 18_s1_eval.py --variant fccg    --images-dir "$T\images" --gt-json "$T\test_annotations.json" --device 0
  python 18_s1_eval.py --compare
Validate the metric math first (no ultralytics/GPU needed):
  python 18_s1_eval.py --selftest
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from importlib import import_module
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
RUNS = HERE / "runs_s1"

# spec bins (identical to lap-1 script 04): size = sqrt(w*h) px
SIZE_BINS = (("very_tiny", 0, 8), ("tiny", 8, 16), ("small", 16, 32),
             ("medium", 32, 96), ("large", 96, 10 ** 9))
CONF_FLOOR = 0.001
IOU_MATCH = 0.5


# ------------------------------------------------------------------ geometry
def iou_matrix(a, b) -> np.ndarray:
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def greedy_match(boxes, scores, gt, iou_thr=IOU_MATCH):
    """Return (correct[per-box 0/1], n_missed). Sort by score desc; each GT used once."""
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    scores = np.asarray(scores, float).reshape(-1)
    correct = np.zeros(len(boxes))
    if len(gt) == 0:
        return correct, 0
    if len(boxes) == 0:
        return correct, len(gt)
    M = iou_matrix(boxes, gt)
    taken = np.zeros(len(gt), bool)
    for i in np.argsort(-scores):
        j = int(np.argmax(np.where(taken, -1.0, M[i])))
        if not taken[j] and M[i, j] >= iou_thr:
            taken[j] = True
            correct[i] = 1.0
    return correct, int((~taken).sum())


# ------------------------------------------------------------------ inference
def run_inference(weights, images_dir, device, imgsz, limit, save_path):
    from ultralytics import YOLO
    model = YOLO(str(weights))
    img_dir = Path(images_dir)
    names = sorted(p.name for p in img_dir.iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if limit:
        names = names[:limit]
    preds: dict = {}
    if save_path and Path(save_path).is_file():
        preds = json.loads(Path(save_path).read_text(encoding="utf-8"))
        print(f"[eval] resume: {len(preds)}/{len(names)} images already predicted")
    for k, fn in enumerate(names):
        if fn in preds:
            continue
        r = model.predict(str(img_dir / fn), imgsz=imgsz, conf=CONF_FLOOR,
                          max_det=300, device=device, verbose=False)[0]
        preds[fn] = {"boxes": r.boxes.xyxy.cpu().numpy().tolist(),
                     "scores": r.boxes.conf.cpu().numpy().tolist()}
        if (k + 1) % 200 == 0:
            print(f"[eval] infer {k + 1}/{len(names)}")
            if save_path:
                Path(save_path).write_text(json.dumps(preds), encoding="utf-8")
    if save_path:
        Path(save_path).write_text(json.dumps(preds), encoding="utf-8")
    return preds


# ------------------------------------------------------------------ GT (COCO)
def load_gt(gt_json):
    d = json.load(open(gt_json, encoding="utf-8"))
    id2fn = {im["id"]: im["file_name"] for im in d["images"]}
    tmp = defaultdict(list)
    for a in d.get("annotations", []):
        x, y, w, h = a["bbox"]
        tmp[id2fn[a["image_id"]]].append([x, y, x + w, y + h])
    gt = {im["file_name"]: np.array(tmp.get(im["file_name"], []), float).reshape(-1, 4)
          for im in d["images"]}
    return gt, d


# ------------------------------------------------------------------ COCO AP_small
def coco_block(preds, gt_raw):
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        print("[eval] pycocotools NOT installed -> AP_small SKIPPED "
              "(per-size recall still valid; `pip install pycocotools` for AP_small)")
        return None
    import os
    import tempfile
    fn2id = {im["file_name"]: im["id"] for im in gt_raw["images"]}
    cat_id = gt_raw["categories"][0]["id"] if gt_raw.get("categories") else 0
    dt = []
    for fn, p in preds.items():
        if fn not in fn2id:
            continue
        for b, s in zip(np.asarray(p["boxes"], float).reshape(-1, 4),
                        np.asarray(p["scores"], float).reshape(-1)):
            dt.append({"image_id": fn2id[fn], "category_id": cat_id,
                       "bbox": [float(b[0]), float(b[1]),
                                float(b[2] - b[0]), float(b[3] - b[1])],
                       "score": float(s)})
    if not dt:
        return None
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(gt_raw, tf)
        gt_path = tf.name
    g = COCO(gt_path)
    d = g.loadRes(dt)
    ev = COCOeval(g, d, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    os.unlink(gt_path)
    keys = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
            "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large"]
    return {k: round(float(v), 4) for k, v in zip(keys, ev.stats)}


# ------------------------------------------------------------------ core eval
def evaluate(preds, gt, gt_raw):
    rows = []          # (score, correct)
    n_gt = 0
    for fn, g in gt.items():
        n_gt += len(g)
        p = preds.get(fn)
        if not p:
            continue
        b = np.asarray(p["boxes"], float).reshape(-1, 4)
        s = np.asarray(p["scores"], float).reshape(-1)
        if len(b) == 0:
            continue
        corr, _ = greedy_match(b, s, g)
        rows.extend(zip(s.tolist(), corr.tolist()))
    out = {"n_images": len(gt), "n_gt": n_gt, "n_det": len(rows)}
    if not rows:
        out["error"] = "no detections"
        return out

    rows.sort(key=lambda r: -r[0])
    scores = np.array([r[0] for r in rows])
    corr = np.array([r[1] for r in rows])

    # AP50 all-point (protocol-matched to the June/lap-1 numbers)
    tp = np.cumsum(corr)
    fp = np.cumsum(1 - corr)
    rec = tp / max(n_gt, 1)
    prec = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.concatenate([[0], rec, [rec[-1]]])
    mpre = np.concatenate([[1], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    out["AP50_allpoint"] = round(float(np.sum(np.diff(mrec) * mpre[1:])), 4)

    # threshold sweep -> F1-optimal operating point
    best_f1, opt_thr = -1.0, 0.25
    for thr in np.arange(0.0, 1.0001, 0.01):
        m = scores >= thr
        TP = float(corr[m].sum())
        FP = float(m.sum() - TP)
        FN = float(n_gt - TP)
        P = TP / max(TP + FP, 1e-9)
        R = TP / max(TP + FN, 1e-9)
        F1 = 2 * P * R / max(P + R, 1e-9)
        if F1 > best_f1:
            best_f1, opt_thr = F1, round(float(thr), 2)
    out["OptThr_F1"] = opt_thr
    out["Best_F1"] = round(float(best_f1), 4)

    # per-size recall at OptThr_F1
    size_tot = {b[0]: 0 for b in SIZE_BINS}
    size_hit = {b[0]: 0 for b in SIZE_BINS}
    for fn, g in gt.items():
        if len(g) == 0:
            continue
        p = preds.get(fn)
        b = np.asarray(p["boxes"], float).reshape(-1, 4) if p else np.zeros((0, 4))
        s = np.asarray(p["scores"], float).reshape(-1) if p else np.zeros(0)
        keep = s >= opt_thr
        taken = np.zeros(len(g), bool)
        if keep.any():
            M = iou_matrix(b[keep], g)
            for i in np.argsort(-s[keep]):
                j = int(np.argmax(np.where(taken, -1.0, M[i])))
                if not taken[j] and M[i, j] >= IOU_MATCH:
                    taken[j] = True
        sz = np.sqrt((g[:, 2] - g[:, 0]).clip(0) * (g[:, 3] - g[:, 1]).clip(0))
        for name, lo, hi in SIZE_BINS:
            mm = (sz >= lo) & (sz < hi)
            size_tot[name] += int(mm.sum())
            size_hit[name] += int(taken[mm].sum())
    out["per_size_recall"] = {n: (round(size_hit[n] / size_tot[n], 4) if size_tot[n] else None)
                              for n, _, _ in SIZE_BINS}
    out["per_size_n"] = dict(size_tot)

    out["coco"] = coco_block(preds, gt_raw)
    return out


# ------------------------------------------------------------------ reporting
def _print_one(o):
    print(f"\n=== S1 eval [{o.get('variant')}] (scene-split TEST, n={o['n_images']}) ===")
    print(f"  AP50_allpoint {o['AP50_allpoint']}   Best_F1 {o['Best_F1']} @thr {o['OptThr_F1']}")
    ap_s = o["coco"]["AP_small"] if o.get("coco") else None
    print(f"  AP_small(coco) {ap_s if ap_s is not None else 'SKIP (no pycocotools)'}")
    r, n = o["per_size_recall"], o["per_size_n"]
    for b in ("very_tiny", "tiny", "small", "medium", "large"):
        print(f"  recall[{b:9s}] {r[b]}   (n={n[b]})")


def _compare():
    rows = {}
    for v in ("control", "fccg"):
        p = RUNS / f"s1_eval_{v}.json"
        if not p.is_file():
            print(f"[compare] missing {p} -- run --variant {v} first")
            return 1
        rows[v] = json.loads(p.read_text(encoding="utf-8"))
    c, f = rows["control"], rows["fccg"]
    print("\n=== S1 GO/NO-GO (scene-split TEST) ===")
    print("  pass criterion: +1.5 AP_small OR +2.0 very-tiny recall (percentage points)\n")

    def line(label, cv, fv):
        if cv is None or fv is None:
            print(f"  {label:22s} control={cv}  fccg={fv}   (delta n/a)")
            return None
        d = (fv - cv) * 100
        print(f"  {label:22s} control={cv:.4f}  fccg={fv:.4f}   delta={d:+.2f} pp")
        return d

    ap_c = c["coco"]["AP_small"] if c.get("coco") else None
    ap_f = f["coco"]["AP_small"] if f.get("coco") else None
    d_ap = line("AP_small (coco)", ap_c, ap_f)
    d_vt = line("recall[very_tiny]", c["per_size_recall"]["very_tiny"], f["per_size_recall"]["very_tiny"])
    line("recall[tiny]", c["per_size_recall"]["tiny"], f["per_size_recall"]["tiny"])
    line("recall[small]", c["per_size_recall"]["small"], f["per_size_recall"]["small"])
    line("AP50_allpoint", c["AP50_allpoint"], f["AP50_allpoint"])

    passed = ((d_ap is not None and d_ap >= 1.5) or (d_vt is not None and d_vt >= 2.0))
    verdict = "PASS -- FCCG lifts tiny objects" if passed else "NULL/FAIL -- criterion not met"
    print(f"\n  VERDICT: {verdict}")
    if d_ap is None:
        print("  (AP_small unavailable -> decided on very-tiny recall alone; "
              "install pycocotools for the full call)")
    return 0


# ------------------------------------------------------------------ selftest
def _selftest():
    rng = np.random.default_rng(0)
    gt, preds = {}, {}
    gt_raw = {"images": [], "annotations": [], "categories": [{"id": 0, "name": "person"}]}
    aid = 0
    for i in range(40):
        fn = f"img{i}"
        n = int(rng.integers(3, 14))
        sz = rng.uniform(4, 60, n)
        xy = rng.uniform(0, 600, (n, 2))
        g = np.concatenate([xy, xy + sz[:, None]], 1)
        gt[fn] = g
        gt_raw["images"].append({"id": i, "file_name": fn, "width": 640, "height": 640})
        for bb in g:
            gt_raw["annotations"].append({"id": aid, "image_id": i, "category_id": 0,
                                          "bbox": [float(bb[0]), float(bb[1]),
                                                   float(bb[2] - bb[0]), float(bb[3] - bb[1])],
                                          "area": float((bb[2] - bb[0]) * (bb[3] - bb[1])),
                                          "iscrowd": 0})
            aid += 1
        # near-perfect detections + a couple of false positives
        b = g + rng.normal(0, 0.8, g.shape)
        s = np.clip(rng.beta(6, 2, n), 0.05, 0.99)
        for _ in range(int(rng.integers(0, 3))):
            c = rng.uniform(0, 600, 2)
            e = rng.uniform(6, 30)
            b = np.vstack([b, [c[0], c[1], c[0] + e, c[1] + e]])
            s = np.append(s, float(rng.uniform(0.05, 0.4)))
        preds[fn] = {"boxes": b.tolist(), "scores": s.tolist()}
    out = evaluate(preds, gt, gt_raw)
    assert set(out["per_size_recall"]) == {b[0] for b in SIZE_BINS}, out["per_size_recall"]
    assert 0.0 <= out["AP50_allpoint"] <= 1.0, out["AP50_allpoint"]
    assert out["Best_F1"] > 0.5, out["Best_F1"]           # near-perfect preds -> high F1
    assert out["n_gt"] == sum(len(g) for g in gt.values())
    # a couple of sanity checks on the greedy matcher
    c1, m1 = greedy_match([[0, 0, 10, 10]], [0.9], [[0, 0, 10, 10]])
    assert c1[0] == 1.0 and m1 == 0
    c2, m2 = greedy_match([[0, 0, 10, 10]], [0.9], [[100, 100, 110, 110]])
    assert c2[0] == 0.0 and m2 == 1
    print("SELFTEST OK:", {"AP50": out["AP50_allpoint"], "Best_F1": out["Best_F1"],
                           "vt_recall": out["per_size_recall"]["very_tiny"],
                           "coco_AP_small": (out["coco"] or {}).get("AP_small", "skip")})
    return 0


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["control", "fccg"])
    ap.add_argument("--weights", help="override; default runs_s1/s1_<variant>/weights/best.pt")
    ap.add_argument("--tag", help="output filename key (default=variant); set for S2/other weights "
                                  "so it does NOT overwrite the saved control/fccg evals")
    ap.add_argument("--images-dir")
    ap.add_argument("--gt-json")
    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return _selftest()
    if a.compare:
        return _compare()
    if not a.variant or not a.images_dir or not a.gt_json:
        ap.error("--variant, --images-dir, --gt-json required (or --compare / --selftest)")

    fccg = import_module("10_fccg_modules")
    if not fccg.register_fccg():
        print("FATAL: ultralytics not importable in this venv")
        return 1

    weights = Path(a.weights) if a.weights else (RUNS / f"s1_{a.variant}" / "weights" / "best.pt")
    if not weights.is_file():
        print(f"FATAL: weights not found: {weights}")
        return 1
    out_key = a.tag or a.variant
    save_path = RUNS / f"s1_eval_{out_key}_preds.json"
    preds = run_inference(weights, a.images_dir, a.device, a.imgsz, a.limit, str(save_path))
    gt, gt_raw = load_gt(a.gt_json)
    out = evaluate(preds, gt, gt_raw)
    out["variant"] = out_key
    (RUNS / f"s1_eval_{out_key}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    _print_one(out)
    print(f"\n[eval] saved -> {RUNS / f's1_eval_{out_key}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
