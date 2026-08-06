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

Now the general eval harness for ANY final checkpoint (S3/D3/anchors/disaster), not just S1.
Per eval it writes, into runs_s1/:
  s1_eval_<tag>.json        - all metrics + curve DATA (pr / f-vs-conf / reliability / conf-hist)
  s1_eval_<tag>_env.json    - spec-6 reproducibility record (versions, weights/gt md5, resolved args)
  s1_eval_<tag>_scorecorr.npz - per-detection (score, correct) for significance later (spec 5)
  s1_eval_<tag>_skipped.txt - the never-silently-drop ledger (what a separate figures pass owns)
  s1_eval_<tag>_*.png       - PR / F-vs-conf / reliability / confidence-hist (unless --no-plots)
Add --profile to also measure efficiency (params/GFLOPs/latency/FPS) on THIS GPU (spec 6).

Run on PC-1 (the machine that trained them), scene-split TEST, one per checkpoint, then --compare:
  $T = "E:\Thesis_mofazzal_2007074\common\c2a\C2A_Dataset\new_dataset3_scenesplit_v1\test"
  python 18_s1_eval.py --variant control --tag s3_assign_full --weights runs_s1\s3_assign_full\weights\best.pt `
      --images-dir "$T\images" --gt-json "$T\test_annotations.json" --device 0 --profile
  python 18_s1_eval.py --compare s3_control_full s3_assign_full
Validate the metric math + curve/dump plumbing first (no ultralytics/GPU needed):
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


# ------------------------------------------------------------------ reproducibility + efficiency
def _md5(path):
    import hashlib
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        return f"NA({e})"


def write_env(path, tag, weights, gt_json, args):
    """spec 6 reproducibility record (once per eval): versions + md5s + resolved args."""
    import platform
    import time
    versions = {"python": platform.python_version()}
    for mod in ("ultralytics", "torch", "numpy", "pycocotools"):
        try:
            versions[mod] = getattr(import_module(mod), "__version__", "installed")
        except Exception:
            versions[mod] = "NA"
    env = {"tag": tag, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "weights": str(weights), "weights_md5": _md5(weights),
           "gt_json": str(gt_json), "gt_md5": _md5(gt_json),
           "conf_floor": CONF_FLOOR, "iou_match": IOU_MATCH,
           "versions": versions, "args": {k: v for k, v in vars(args).items()}}
    Path(path).write_text(json.dumps(env, indent=2), encoding="utf-8")
    return env


def profile_model(weights, device, imgsz, n_warmup=8, n_iter=60):
    """spec 6 efficiency: params/GFLOPs/layers/size + latency (mean/p50/p95) + FPS on THIS GPU.
    Fully guarded -- returns {'error':...} instead of raising, so eval never fails on it."""
    try:
        import time
        from ultralytics import YOLO
        m = YOLO(str(weights))
        info = m.info(detailed=False, verbose=False)      # (layers, params, gradients, GFLOPs)
        layers = params = flops = None
        if isinstance(info, (tuple, list)):
            vals = list(info) + [None] * 4
            layers, params, _grad, flops = vals[0], vals[1], vals[2], vals[3]
        dummy = np.zeros((imgsz, imgsz, 3), dtype="uint8")
        for _ in range(n_warmup):
            m.predict(dummy, imgsz=imgsz, device=device, verbose=False)
        ts = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            m.predict(dummy, imgsz=imgsz, device=device, verbose=False)
            ts.append((time.perf_counter() - t0) * 1000.0)
        ts = np.asarray(ts)
        return {"params_total": (int(params) if params else None),
                "gflops": (round(float(flops), 2) if flops else None),
                "layers": (int(layers) if layers else None),
                "weights_mb": round(Path(weights).stat().st_size / 1e6, 2),
                "latency_ms_mean": round(float(ts.mean()), 2),
                "latency_ms_p50": round(float(np.percentile(ts, 50)), 2),
                "latency_ms_p95": round(float(np.percentile(ts, 95)), 2),
                "fps": round(1000.0 / float(ts.mean()), 1), "imgsz": imgsz}
    except Exception as e:
        return {"error": f"profile failed: {e}"}


def render_pngs(out, save_dir, tag):
    """spec 6/7: a PNG for each curve. Guarded -- skips cleanly if matplotlib is absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        return f"[SKIPPED] plots -- matplotlib unavailable ({e})"
    c = out.get("curves") or {}
    made = []
    if c.get("pr", {}).get("recall"):
        plt.figure(); plt.plot(c["pr"]["recall"], c["pr"]["precision"])
        plt.xlabel("recall"); plt.ylabel("precision"); plt.grid(True, alpha=.3)
        plt.title(f"PR — {tag} (AP50 {out.get('AP50_allpoint')})")
        p = Path(save_dir) / f"s1_eval_{tag}_pr.png"
        plt.savefig(p, dpi=200, bbox_inches="tight"); plt.close(); made.append(p.name)
    if c.get("f_vs_conf", {}).get("conf"):
        fc = c["f_vs_conf"]; plt.figure()
        for k in ("f1", "f2", "precision", "recall"):
            plt.plot(fc["conf"], fc[k], label=k)
        plt.axvline(out.get("OptThr_F2", 0.25), ls="--", c="k", alpha=.4,
                    label=f"F2opt {out.get('OptThr_F2')}")
        plt.xlabel("confidence"); plt.ylabel("score"); plt.legend(); plt.grid(True, alpha=.3)
        plt.title(f"score vs confidence — {tag}")
        p = Path(save_dir) / f"s1_eval_{tag}_fconf.png"
        plt.savefig(p, dpi=200, bbox_inches="tight"); plt.close(); made.append(p.name)
    if c.get("reliability", {}).get("bin_center"):
        rl = c["reliability"]; plt.figure(); plt.plot([0, 1], [0, 1], ls="--", c="gray")
        plt.plot(rl["confidence"], rl["accuracy"], marker="o")
        plt.xlabel("confidence"); plt.ylabel("accuracy (empirical)"); plt.grid(True, alpha=.3)
        plt.title(f"reliability — {tag} (ECE {out.get('calibration', {}).get('ECE')})")
        p = Path(save_dir) / f"s1_eval_{tag}_reliability.png"
        plt.savefig(p, dpi=200, bbox_inches="tight"); plt.close(); made.append(p.name)
    if c.get("conf_hist", {}).get("count"):
        ch = c["conf_hist"]; e = ch["edges"]
        centers = [(e[i] + e[i + 1]) / 2 for i in range(len(e) - 1)]
        plt.figure(); plt.bar(centers, ch["count"], width=(e[1] - e[0]) * 0.9)
        plt.xlabel("confidence"); plt.ylabel("count (log)"); plt.yscale("log")
        plt.title(f"confidence histogram — {tag}")
        p = Path(save_dir) / f"s1_eval_{tag}_confhist.png"
        plt.savefig(p, dpi=200, bbox_inches="tight"); plt.close(); made.append(p.name)
    return f"[plots] wrote {len(made)}: {', '.join(made)}" if made else "[plots] nothing to draw"


# ------------------------------------------------------------------ core eval
def evaluate(preds, gt, gt_raw, dump_scorecorr=None):
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

    # threshold sweep -> F1/F2-optimal points (F2 = thesis PRIMARY: SAR favors recall) + op@0.25
    best_f1, opt_thr = -1.0, 0.25
    best_f2, opt_thr2 = -1.0, 0.25
    op = None
    sw_conf, sw_p, sw_r, sw_f1, sw_f2 = [], [], [], [], []   # F1/F2-vs-confidence curve data
    for thr in np.arange(0.0, 1.0001, 0.01):
        m = scores >= thr
        TP = float(corr[m].sum())
        FP = float(m.sum() - TP)
        FN = float(n_gt - TP)
        P = TP / max(TP + FP, 1e-9)
        R = TP / max(TP + FN, 1e-9)
        F1 = 2 * P * R / max(P + R, 1e-9)
        F2 = 5 * P * R / max(4 * P + R, 1e-9)      # weights recall 2x (spec 6 primary)
        sw_conf.append(round(float(thr), 3)); sw_p.append(round(P, 4)); sw_r.append(round(R, 4))
        sw_f1.append(round(F1, 4)); sw_f2.append(round(F2, 4))
        if F1 > best_f1:
            best_f1, opt_thr = F1, round(float(thr), 2)
        if F2 > best_f2:
            best_f2, opt_thr2 = F2, round(float(thr), 2)
        if abs(thr - 0.25) < 0.005:                # operational point (spec 5: conf=0.25)
            op = (P, R, F1, F2)
    out["OptThr_F1"] = opt_thr
    out["Best_F1"] = round(float(best_f1), 4)
    out["OptThr_F2"] = opt_thr2
    out["Best_F2"] = round(float(best_f2), 4)
    if op:
        out["op_conf25"] = {"precision": round(op[0], 4), "recall": round(op[1], 4),
                            "F1": round(op[2], 4), "F2": round(op[3], 4)}

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

    # calibration of detections (spec 6: ECE 10-bin, MCE, Brier) + reliability-diagram data
    edges = np.linspace(0, 1, 11)
    ece = mce = 0.0
    rel_center, rel_acc, rel_conf, rel_count = [], [], [], []
    for i in range(10):
        mm = (scores > edges[i]) & (scores <= edges[i + 1])
        if mm.any():
            acc, cf = float(corr[mm].mean()), float(scores[mm].mean())
            gap = abs(acc - cf)
            ece += mm.mean() * gap
            mce = max(mce, gap)
            rel_center.append(round(float((edges[i] + edges[i + 1]) / 2), 3))
            rel_acc.append(round(acc, 4)); rel_conf.append(round(cf, 4)); rel_count.append(int(mm.sum()))
    out["calibration"] = {"ECE": round(float(ece), 4), "MCE": round(float(mce), 4),
                          "Brier": round(float(np.mean((scores - corr) ** 2)), 4)}

    # curve DATA for every figure (spec 6/7: data+PNG each; regenerable without re-eval)
    def _sub(arr, k=300):
        arr = np.asarray(arr, float)
        if len(arr) <= k:
            return [round(float(x), 4) for x in arr]
        idx = np.linspace(0, len(arr) - 1, k).astype(int)
        return [round(float(x), 4) for x in arr[idx]]
    hist_c, hist_e = np.histogram(scores, bins=20, range=(0.0, 1.0))
    out["curves"] = {
        "pr": {"recall": _sub(rec), "precision": _sub(prec)},
        "f_vs_conf": {"conf": sw_conf, "precision": sw_p, "recall": sw_r, "f1": sw_f1, "f2": sw_f2},
        "reliability": {"bin_center": rel_center, "accuracy": rel_acc,
                        "confidence": rel_conf, "count": rel_count},
        "conf_hist": {"count": hist_c.tolist(), "edges": [round(float(e), 3) for e in hist_e]},
    }

    # per-detection (score, correct) dumped for significance later (spec 5: save per-image scores)
    if dump_scorecorr is not None:
        try:
            np.savez_compressed(str(dump_scorecorr), score=scores.astype(np.float32),
                                correct=corr.astype(np.int8))
        except Exception as e:
            print(f"[eval] scorecorr dump skipped ({e})")

    out["coco"] = coco_block(preds, gt_raw)        # full 12-stat: AP/AP50/AP75/AP_s/m/l + AR
    return out


# ------------------------------------------------------------------ reporting
def _print_one(o):
    print(f"\n=== S1 eval [{o.get('variant')}] (scene-split TEST, n={o['n_images']}) ===")
    print(f"  AP50_allpoint {o['AP50_allpoint']}   Best_F1 {o['Best_F1']} @{o['OptThr_F1']}")
    print(f"  Best_F2 {o.get('Best_F2')} @{o.get('OptThr_F2')}  (F2=PRIMARY)   op@0.25 {o.get('op_conf25')}")
    print(f"  calibration {o.get('calibration')}")
    ap_s = o["coco"]["AP_small"] if o.get("coco") else None
    print(f"  AP_small(coco) {ap_s if ap_s is not None else 'SKIP (no pycocotools)'}")
    r, n = o["per_size_recall"], o["per_size_n"]
    for b in ("very_tiny", "tiny", "small", "medium", "large"):
        print(f"  recall[{b:9s}] {r[b]}   (n={n[b]})")
    if o.get("efficiency"):
        print(f"  efficiency {o['efficiency']}")


def _dig(o, path):
    cur = o
    for k in path:
        cur = cur.get(k) if isinstance(cur, dict) else None
    return cur


def _compare(tags):
    """Compare any 2+ saved evals; deltas are vs the FIRST tag (baseline).
    e.g. --compare s3_control_full s3_assign_full   |   --compare s3_rset_before d3_rset_after"""
    if not tags or len(tags) < 2:
        tags = ["control", "fccg"]
    rows = {}
    for v in tags:
        p = RUNS / f"s1_eval_{v}.json"
        if not p.is_file():
            print(f"[compare] missing {p} -- run --tag {v} first")
            return 1
        rows[v] = json.loads(p.read_text(encoding="utf-8"))
    base = tags[0]
    print(f"\n=== compare (baseline = {base}) ===")
    metrics = [("AP50", ["AP50_allpoint"]), ("F2(primary)", ["Best_F2"]),
               ("AP_small", ["coco", "AP_small"]), ("VT_recall<8", ["per_size_recall", "very_tiny"]),
               ("tiny_recall", ["per_size_recall", "tiny"]), ("small_recall", ["per_size_recall", "small"]),
               ("precision@25", ["op_conf25", "precision"]), ("recall@25", ["op_conf25", "recall"]),
               ("ECE", ["calibration", "ECE"])]
    for label, path in metrics:
        vals = {v: _dig(rows[v], path) for v in tags}
        b = vals[base]
        s = f"  {label:14s}"
        for v in tags:
            x = vals[v]
            if x is None:
                s += f"  {v}=NA"
            elif v == base or b is None:
                s += f"  {v}={x:.4f}"
            else:
                s += f"  {v}={x:.4f}({(x - b) * 100:+.2f}pp)"
        print(s)
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
    import tempfile
    scdump = Path(tempfile.mkdtemp()) / "sc.npz"
    out = evaluate(preds, gt, gt_raw, dump_scorecorr=scdump)
    assert set(out["per_size_recall"]) == {b[0] for b in SIZE_BINS}, out["per_size_recall"]
    assert 0.0 <= out["AP50_allpoint"] <= 1.0, out["AP50_allpoint"]
    assert out["Best_F1"] > 0.5, out["Best_F1"]           # near-perfect preds -> high F1
    assert out["Best_F2"] > 0.5 and 0.0 <= out["OptThr_F2"] <= 1.0, (out["Best_F2"], out["OptThr_F2"])
    assert out["n_gt"] == sum(len(g) for g in gt.values())
    # curve DATA present + well-formed (the spec-6 additions)
    cu = out["curves"]
    assert cu["pr"]["recall"] and len(cu["pr"]["recall"]) == len(cu["pr"]["precision"]), "PR data broken"
    assert len(cu["f_vs_conf"]["conf"]) == len(cu["f_vs_conf"]["f2"]) > 50, "F-vs-conf data broken"
    assert len(cu["conf_hist"]["count"]) == 20 and len(cu["conf_hist"]["edges"]) == 21, "hist bins broken"
    assert len(cu["reliability"]["accuracy"]) == len(cu["reliability"]["confidence"]), "reliability broken"
    assert scdump.exists(), "scorecorr dump not written"
    _sc = np.load(scdump)
    assert _sc["score"].shape == _sc["correct"].shape and _sc["score"].shape[0] == out["n_det"]
    # a couple of sanity checks on the greedy matcher
    c1, m1 = greedy_match([[0, 0, 10, 10]], [0.9], [[0, 0, 10, 10]])
    assert c1[0] == 1.0 and m1 == 0
    c2, m2 = greedy_match([[0, 0, 10, 10]], [0.9], [[100, 100, 110, 110]])
    assert c2[0] == 0.0 and m2 == 1
    import shutil
    shutil.rmtree(scdump.parent, ignore_errors=True)
    print("SELFTEST OK:", {"AP50": out["AP50_allpoint"], "Best_F2": out["Best_F2"],
                           "vt_recall": out["per_size_recall"]["very_tiny"],
                           "curves": list(out["curves"]),
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
    ap.add_argument("--profile", action="store_true",
                    help="also measure efficiency (params/GFLOPs/latency/FPS) on this GPU (spec 6)")
    ap.add_argument("--no-plots", action="store_true", help="skip PNG rendering (data is still saved in the json)")
    ap.add_argument("--compare", nargs="*", default=None, metavar="TAG",
                    help="compare saved evals, deltas vs first: --compare control fccg")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return _selftest()
    if a.compare is not None:
        return _compare(a.compare)
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
    out = evaluate(preds, gt, gt_raw, dump_scorecorr=RUNS / f"s1_eval_{out_key}_scorecorr.npz")
    out["variant"] = out_key
    if a.profile:
        out["efficiency"] = profile_model(weights, a.device, a.imgsz)
    (RUNS / f"s1_eval_{out_key}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    # spec 6 reproducibility record + never-silently-drop ledger
    write_env(RUNS / f"s1_eval_{out_key}_env.json", out_key, weights, a.gt_json, a)
    skipped = ["[SKIPPED] qualitative grids (detection/failure-16/success) -- dedicated figures pass, not this numeric eval",
               "[SKIPPED] per-stride AP / confusion matrix -- lap-1 script 04 (04_eval_fusion_ablation.py)",
               "[SKIPPED] training-dynamics curves -- live in the run folder's results.csv"]
    if not a.profile:
        skipped.append("[SKIPPED] efficiency (params/GFLOPs/latency) -- re-run with --profile to add it")
    (RUNS / f"s1_eval_{out_key}_skipped.txt").write_text("\n".join(skipped) + "\n", encoding="utf-8")
    if not a.no_plots:
        print(render_pngs(out, RUNS, out_key))
    _print_one(out)
    print(f"\n[eval] saved -> {RUNS / f's1_eval_{out_key}.json'}  (+ _env.json, _scorecorr.npz, _skipped.txt)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
