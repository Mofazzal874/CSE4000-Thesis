"""
04_eval_fusion_ablation.py -- Fusion-mode ablation for sliced inference.

Compares tile/whole/TTA merging via {nms, wbf, cwbf} on a labeled split and
writes an ablation CSV. Runs the model ONCE per config, saves raw predictions,
then evaluates every fusion mode from the same predictions (fair + cheap).

Three ways to run:
  A) REAL (lab PC, GPU, ultralytics installed):
     python 04_eval_fusion_ablation.py --weights <best.pt> \
        --images-dir <root>/test/images --gt-json <root>/test/test_annotations.json \
        --slices 256 --overlap 0.30 --tta-imgsz 1280 --save-preds preds_test.json
  B) OFFLINE (any machine) -- re-evaluate fusion modes from saved predictions:
     python 04_eval_fusion_ablation.py --load-preds preds_test.json --gt-json ...
  C) SELFTEST (any machine, no ultralytics):
     python 04_eval_fusion_ablation.py --synthetic

Calibration for cwbf:
  1) on VAL:  ... --save-preds preds_val.json          (run A on the val split)
  2)          ... --load-preds preds_val.json --gt-json <val json> --fit-cal cal.json
  3) on TEST: ... --load-preds preds_test.json --gt-json <test json> --cal cal.json

Outputs: ablation_results.csv (+ .json) next to the preds file.
Metrics per mode: AP50 (all-point), bestF1/bestF2 (+their conf), mean-IoU of TPs,
per-size recall (vt<8, tiny 8-16, small 16-32, med>=32) at the F1-optimal conf,
detections/image, and COCO AP/AP50/AP_small via pycocotools when available.
"""
from __future__ import annotations
import argparse, csv, json, math, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
_f = import_module("03_cwbf_fusion")  # noqa: N816
nms, wbf, fuse_sources = _f.nms, _f.wbf, _f.fuse_sources
TempScaler, match_greedy, iou_matrix = _f.TempScaler, _f.match_greedy, _f.iou_matrix

SIZE_BINS = (("very_tiny", 0, 8), ("tiny", 8, 16), ("small", 16, 32), ("medium", 32, 10 ** 9))


# ----------------------------------------------------------------- tiling
def tile_grid(W: int, H: int, s: int, overlap: float):
    stride = max(1, int(s * (1 - overlap)))
    xs = list(range(0, max(W - s, 0) + 1, stride)) or [0]
    ys = list(range(0, max(H - s, 0) + 1, stride)) or [0]
    if xs[-1] + s < W:
        xs.append(W - s)
    if ys[-1] + s < H:
        ys.append(H - s)
    return [(x, y) for y in ys for x in xs if x >= 0 and y >= 0]


# ----------------------------------------------------------------- inference
def run_model(args) -> dict:
    """Returns preds = {image_name: {"wh": [W,H], "gt_id": ..., "sources":
    [{"name": ..., "boxes": [[x1,y1,x2,y2],...], "scores": [...]}]}}"""
    from ultralytics import YOLO
    import cv2
    model = YOLO(args.weights)
    img_dir = Path(args.images_dir)
    names = sorted(p.name for p in img_dir.iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if args.limit:
        names = names[: args.limit]
    preds: dict = {}
    t0 = time.time()
    for k, fn in enumerate(names):
        img = cv2.imread(str(img_dir / fn))
        H, W = img.shape[:2]
        sources = []
        # whole frame
        r = model.predict(img, imgsz=args.imgsz_whole, conf=args.conf_floor,
                          device=args.device, verbose=False)[0]
        sources.append({"name": "whole",
                        "boxes": r.boxes.xyxy.cpu().numpy().tolist(),
                        "scores": r.boxes.conf.cpu().numpy().tolist()})
        # TTA whole frame (optional)
        if args.tta_imgsz:
            r = model.predict(img, imgsz=args.tta_imgsz, conf=args.conf_floor,
                              augment=True, device=args.device, verbose=False)[0]
            sources.append({"name": "tta",
                            "boxes": r.boxes.xyxy.cpu().numpy().tolist(),
                            "scores": r.boxes.conf.cpu().numpy().tolist()})
        # tiles (only if image bigger than slice -- measured C2A majority is not)
        for s in args.slices:
            if max(W, H) <= s:
                continue
            for (x, y) in tile_grid(W, H, s, args.overlap):
                crop = img[y: y + s, x: x + s]
                r = model.predict(crop, imgsz=s, conf=args.conf_floor,
                                  device=args.device, verbose=False)[0]
                b = r.boxes.xyxy.cpu().numpy()
                if len(b):
                    b[:, [0, 2]] += x
                    b[:, [1, 3]] += y
                sources.append({"name": f"tile{s}",
                                "boxes": b.tolist(),
                                "scores": r.boxes.conf.cpu().numpy().tolist()})
        preds[fn] = {"wh": [W, H], "sources": sources}
        if (k + 1) % 100 == 0:
            print(f"[infer] {k+1}/{len(names)}  ({(time.time()-t0)/(k+1):.2f}s/img)")
    return preds


# ----------------------------------------------------------------- GT loading
def load_gt(gt_json: str):
    with open(gt_json, encoding="utf-8") as f:
        d = json.load(f)
    by_img: dict[str, list] = defaultdict(list)
    id2fn = {im["id"]: im["file_name"] for im in d["images"]}
    for a in d.get("annotations", []):
        x, y, w, h = a["bbox"]
        by_img[id2fn[a["image_id"]]].append([x, y, x + w, y + h])
    return {fn: np.array(v, float).reshape(-1, 4) for fn, v in by_img.items()}, d


# ----------------------------------------------------------------- evaluation
def evaluate_mode(preds: dict, gt: dict, mode: str, iou_fuse: float,
                  scalers=None, iou_match: float = 0.5) -> dict:
    rows = []            # (score, correct, tp_iou_or_nan)
    n_gt_total = 0
    size_hits = {b[0]: [0, 0] for b in SIZE_BINS}     # at F1-opt conf, filled later
    fused_cache = {}
    for fn, p in preds.items():
        g = gt.get(fn, np.zeros((0, 4)))
        fb, fs = fuse_sources(p["sources"], mode=mode, iou_thr=iou_fuse, scalers=scalers)
        fused_cache[fn] = (fb, fs)
        n_gt_total += len(g)
        if len(fb) == 0:
            continue
        correct, _ = match_greedy(fb, fs, g, iou_match)
        M = iou_matrix(fb, g) if len(g) else np.zeros((len(fb), 1))
        best_iou = M.max(1) if len(g) else np.zeros(len(fb))
        for s_, c_, i_ in zip(fs, correct, best_iou):
            rows.append((float(s_), float(c_), float(i_) if c_ else np.nan))
    if not rows:
        return {"mode": mode, "AP50": 0, "bestF1": 0, "bestF2": 0}
    rows.sort(key=lambda r: -r[0])
    scores = np.array([r[0] for r in rows])
    corr = np.array([r[1] for r in rows])
    tp = np.cumsum(corr)
    fp = np.cumsum(1 - corr)
    rec = tp / max(n_gt_total, 1)
    prec = tp / np.maximum(tp + fp, 1e-9)
    # all-point AP50 (precision envelope)
    mrec = np.concatenate([[0], rec, [rec[-1]]])
    mpre = np.concatenate([[1], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap50 = float(np.sum(np.diff(mrec) * mpre[1:]))
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    f2 = 5 * prec * rec / np.maximum(4 * prec + rec, 1e-9)
    i1, i2 = int(np.argmax(f1)), int(np.argmax(f2))
    conf_f1 = float(scores[i1])
    mean_tp_iou = float(np.nanmean([r[2] for r in rows if r[1] == 1])) if corr.any() else 0.0
    # per-size recall at the F1-optimal confidence
    size_tot = {b[0]: 0 for b in SIZE_BINS}
    size_hit = {b[0]: 0 for b in SIZE_BINS}
    for fn, p in preds.items():
        g = gt.get(fn, np.zeros((0, 4)))
        if len(g) == 0:
            continue
        fb, fs = fused_cache[fn]
        keep = fs >= conf_f1
        corr2 = np.zeros(len(g))
        if keep.any():
            c_det, _ = match_greedy(fb[keep], fs[keep], g, iou_match)
            M = iou_matrix(fb[keep], g)
            taken = np.zeros(len(g), bool)
            for i in np.argsort(-fs[keep]):
                j = int(np.argmax(np.where(taken, -1.0, M[i])))
                if not taken[j] and M[i, j] >= iou_match:
                    taken[j] = True
            corr2 = taken.astype(float)
        sz = np.sqrt(np.clip(g[:, 2] - g[:, 0], 0, None) * np.clip(g[:, 3] - g[:, 1], 0, None))
        for name, lo, hi in SIZE_BINS:
            m = (sz >= lo) & (sz < hi)
            size_tot[name] += int(m.sum())
            size_hit[name] += int(corr2[m].sum())
    out = {"mode": mode, "AP50": round(ap50, 4),
           "bestF1": round(float(f1[i1]), 4), "confF1": round(conf_f1, 3),
           "bestF2": round(float(f2[i2]), 4), "confF2": round(float(scores[i2]), 3),
           "meanIoU_TP": round(mean_tp_iou, 4),
           "dets_per_img": round(len(rows) / max(len(preds), 1), 1)}
    for name, _, _ in SIZE_BINS:
        out[f"recall_{name}"] = round(size_hit[name] / size_tot[name], 4) if size_tot[name] else None
        out[f"n_{name}"] = size_tot[name]
    return out


def coco_eval(preds, gt_raw, mode, iou_fuse, scalers=None):
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        return {}
    import tempfile, os
    fn2id = {im["file_name"]: im["id"] for im in gt_raw["images"]}
    cat_id = gt_raw["categories"][0]["id"] if gt_raw.get("categories") else 0
    dt = []
    for fn, p in preds.items():
        if fn not in fn2id:
            continue
        fb, fs = fuse_sources(p["sources"], mode=mode, iou_thr=iou_fuse, scalers=scalers)
        for b, s in zip(fb, fs):
            dt.append({"image_id": fn2id[fn], "category_id": cat_id,
                       "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
                       "score": float(s)})
    if not dt:
        return {}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(gt_raw, tf)
        gt_path = tf.name
    g = COCO(gt_path)
    d = g.loadRes(dt)
    ev = COCOeval(g, d, "bbox")
    ev.evaluate(); ev.accumulate(); ev.summarize()
    os.unlink(gt_path)
    s = ev.stats
    return {"coco_AP": round(float(s[0]), 4), "coco_AP50": round(float(s[1]), 4),
            "coco_AP75": round(float(s[2]), 4), "coco_AP_small": round(float(s[3]), 4)}


def fit_calibration(preds, gt, out_path, iou_match=0.5):
    fam: dict[str, list] = defaultdict(lambda: ([], []))
    for fn, p in preds.items():
        g = gt.get(fn, np.zeros((0, 4)))
        for s in p["sources"]:
            fam_key = "tile" if s["name"].startswith("tile") else s["name"]
            b = np.asarray(s["boxes"], float).reshape(-1, 4)
            c = np.asarray(s["scores"], float).reshape(-1)
            if len(b) == 0:
                continue
            corr, _ = match_greedy(b, c, g, iou_match)
            fam[fam_key][0].extend(c.tolist())
            fam[fam_key][1].extend(corr.tolist())
    cal = {}
    for k, (c, y) in fam.items():
        c, y = np.array(c), np.array(y)
        ts = TempScaler().fit(c, y)
        cal[k] = {"T": ts.T, "n": len(c),
                  "ece_before": TempScaler.ece(c, y),
                  "ece_after": TempScaler.ece(ts.apply(c), y)}
        print(f"[cal] {k:6s} T={ts.T:.3f}  n={len(c)}  "
              f"ECE {cal[k]['ece_before']:.4f} -> {cal[k]['ece_after']:.4f}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cal, f, indent=2)
    print(f"[cal] saved -> {out_path}")


# ----------------------------------------------------------------- synthetic
def synthetic_selftest() -> None:
    rng = np.random.default_rng(20260705)
    preds, gt = {}, {}
    for i in range(50):
        fn = f"img{i:03d}"
        n = rng.integers(3, 25)
        wh = np.array([640, 640])
        sz = rng.uniform(5, 40, n)
        xy = rng.uniform(0, 600, (n, 2))
        g = np.concatenate([xy, xy + sz[:, None]], 1)
        gt[fn] = g
        srcs = []
        for src_i in range(2):                     # two overlapping tile sources
            bs, ss = [], []
            for b in g:
                if rng.uniform() < 0.86:           # detected
                    jit = rng.normal(0, 1.2, 4)
                    bs.append(b + jit)
                    ss.append(float(np.clip(rng.beta(5, 2), 0.05, 0.99)))  # overconfident
            for _ in range(rng.integers(0, 4)):    # false positives
                c = rng.uniform(0, 600, 2)
                s_ = rng.uniform(6, 25)
                bs.append([c[0], c[1], c[0] + s_, c[1] + s_])
                ss.append(float(np.clip(rng.beta(2, 4), 0.05, 0.9)))
            srcs.append({"name": f"tile{src_i}", "boxes": bs, "scores": ss})
        preds[fn] = {"wh": wh.tolist(), "sources": srcs}
    res = {}
    for mode in ("nms", "wbf"):
        res[mode] = evaluate_mode(preds, gt, mode, iou_fuse=0.55)
        print(f"[synthetic] {mode}: AP50={res[mode]['AP50']} bestF1={res[mode]['bestF1']} "
              f"meanIoU_TP={res[mode]['meanIoU_TP']}")
    assert res["wbf"]["meanIoU_TP"] >= res["nms"]["meanIoU_TP"] - 1e-6, "WBF localization regressed"
    assert res["wbf"]["AP50"] >= res["nms"]["AP50"] - 0.01, "WBF AP50 badly regressed"
    # calibration path end-to-end
    fit_calibration(preds, gt, Path(__file__).parent / "_synthetic_cal.json")
    cal = json.load(open(Path(__file__).parent / "_synthetic_cal.json"))
    scalers = {k: TempScaler(v["T"]) for k, v in cal.items()}
    r3 = evaluate_mode(preds, gt, "cwbf", 0.55, scalers=scalers)
    print(f"[synthetic] cwbf: AP50={r3['AP50']} bestF1={r3['bestF1']}")
    assert r3["AP50"] > 0.5, "cwbf pipeline broken"
    print("SYNTHETIC SELFTEST PASSED")


# ----------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights")
    ap.add_argument("--images-dir")
    ap.add_argument("--gt-json")
    ap.add_argument("--slices", type=lambda s: [int(x) for x in s.split(",")], default=[256])
    ap.add_argument("--overlap", type=float, default=0.30)
    ap.add_argument("--imgsz-whole", type=int, default=640)
    ap.add_argument("--tta-imgsz", type=int, default=0, help="0 disables the TTA source")
    ap.add_argument("--conf-floor", type=float, default=0.01)
    ap.add_argument("--iou-fuse", type=float, default=0.55)
    ap.add_argument("--modes", default="nms,wbf,cwbf")
    ap.add_argument("--device", default="0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-preds")
    ap.add_argument("--load-preds")
    ap.add_argument("--fit-cal", help="fit temperature calibration -> this json (use VAL preds!)")
    ap.add_argument("--cal", help="apply calibration json (fitted on val) for cwbf")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    if args.synthetic:
        synthetic_selftest()
        return

    if args.load_preds:
        with open(args.load_preds, encoding="utf-8") as f:
            preds = json.load(f)
    else:
        if not (args.weights and args.images_dir):
            sys.exit("need --weights + --images-dir (or --load-preds / --synthetic)")
        preds = run_model(args)
        if args.save_preds:
            with open(args.save_preds, "w", encoding="utf-8") as f:
                json.dump(preds, f)
            print(f"[preds] saved -> {args.save_preds}")

    gt, gt_raw = load_gt(args.gt_json) if args.gt_json else ({}, None)

    if args.fit_cal:
        fit_calibration(preds, gt, args.fit_cal)
        return

    scalers = None
    if args.cal:
        cal = json.load(open(args.cal, encoding="utf-8"))
        scalers = {k: TempScaler(v["T"]) for k, v in cal.items()}

    results = []
    for mode in args.modes.split(","):
        r = evaluate_mode(preds, gt, mode, args.iou_fuse,
                          scalers=scalers if mode == "cwbf" else None)
        if gt_raw:
            r.update(coco_eval(preds, gt_raw, mode, args.iou_fuse,
                               scalers=scalers if mode == "cwbf" else None))
        results.append(r)
        print(f"[result] {r}")

    out = Path(args.save_preds or args.load_preds or "ablation").with_suffix("")
    with open(f"{out}_ablation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in results for k in r}))
        w.writeheader()
        w.writerows(results)
    with open(f"{out}_ablation.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[done] -> {out}_ablation.csv")


if __name__ == "__main__":
    main()
