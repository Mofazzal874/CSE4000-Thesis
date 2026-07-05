"""
04_eval_fusion_ablation.py -- Fusion-mode ablation for sliced inference,
emitting the FULL thesis metric contract (system_spec_thesis.md §6 /
system_spec.md §11) per fusion mode, so rows drop straight into the report
tables next to the June ablation numbers.

Per mode {nms, wbf, cwbf} and per split it reports:
  AP-style (conf floor 0.001 per spec §5): AP50_allpoint (protocol-matched to
    the SAHI reports), plus COCO 12-stat block via pycocotools when installed:
    AP / AP50 / AP75 / AP_small / AP_medium / AP_large / AR_1 / AR_10 / AR_100 /
    AR_small / AR_medium / AR_large
  Operational point (conf=0.25, IoU=0.5 per spec §5): precision, recall, F1,
    F2, TP/FP/FN
  Optimal thresholds: OptThr_F1 / Best_F1 / OptThr_F2 / Best_F2 + TP/FP/FN at
    OptThr_F1
  Per-size recall at OptThr_F1, spec bins: very_tiny <8^2, tiny 8-16^2,
    small 16-32^2, medium 32-96^2, large >=96^2 (n per bin; None when n==0 --
    e.g. SARD very-tiny rows, n=8, are meaningless and print as such)
  Calibration of the FUSED detections: ECE (10-bin), MCE, Brier
  Localization: mean IoU of TPs (the C-WBF mechanism metric)
  Efficiency: e2e latency mean/p50/p95 ms + FPS (timings captured at inference
    and stored inside the preds json, so offline replays keep real numbers)
  Curves as CSV (+ PNG when matplotlib exists): PR, F1-vs-conf, F2-vs-conf,
    confidence histogram; per-image TP/FP/FN CSV for later significance tests.
  Anything uncomputable is written to skipped_metrics.txt (spec: never
    silently drop a metric).

Three ways to run:
  A) REAL (lab PC): --weights --images-dir --gt-json [--slices 256 --tta-imgsz 1280]
     [--save-preds preds.json]
  B) OFFLINE replay: --load-preds preds.json --gt-json ... [--cal cal.json]
  C) SELFTEST (no ultralytics needed): --synthetic

Calibration workflow (fit on VAL only):
  ... val run ... --save-preds preds_val.json
  --load-preds preds_val.json --gt-json <val> --fit-cal cal.json
  ... test run ... --load-preds preds_test.json --gt-json <test> --cal cal.json
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
_f = import_module("03_cwbf_fusion")  # noqa: N816
nms, wbf, fuse_sources = _f.nms, _f.wbf, _f.fuse_sources
TempScaler, match_greedy, iou_matrix = _f.TempScaler, _f.match_greedy, _f.iou_matrix

# spec bins: very-tiny <8^2, tiny 8-16^2, small 16-32^2, medium 32-96^2, large >=96^2
SIZE_BINS = (("very_tiny", 0, 8), ("tiny", 8, 16), ("small", 16, 32),
             ("medium", 32, 96), ("large", 96, 10 ** 9))
CONF_FLOOR = 0.001          # spec §5 AP-style
CONF_OPERATIONAL = 0.25     # spec §5 operational F1/F2
IOU_MATCH = 0.5
SKIPPED: list[str] = []


# ------------------------------------------------------- shared-GPU pinning
# Ported from the PC-2 joint script (TC-02/TC-08 tested there): the A6000 box is
# SHARED -- GPU 0 belongs to the other user, GPU 1 is ours. Pick a FREE GPU
# (<1 GB used AND <10% util) BEFORE torch import and pin it via
# CUDA_VISIBLE_DEVICES so device "0" in-process == the picked physical GPU and
# the other user's GPU is invisible. Never auto-grab a busy GPU. On single-GPU
# boxes this resolves to GPU 0. Manual override: set CUDA_VISIBLE_DEVICES
# yourself before launching (then this is a no-op). Spawned DataLoader workers
# inherit the sentinel and skip re-picking.
def pin_free_gpu(free_mem_mb: float = 1024, free_util_pct: float = 10) -> None:
    import os, subprocess
    if os.environ.get("_NLAP_GPU_PINNED") == "1" or os.environ.get("CUDA_VISIBLE_DEVICES"):
        return
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL, timeout=15
        ).decode("utf-8", "ignore")
        gpus = []
        for line in out.strip().splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) >= 4:
                gpus.append({"index": int(p[0]), "mem_used": float(p[1]), "util": float(p[3])})
    except Exception:
        print("[gpu] nvidia-smi unavailable -> leaving device selection to torch defaults")
        return
    if not gpus:
        return
    assigned = 1 if len(gpus) > 1 else 0          # PC-2 convention: GPU 1 is ours
    snap = ", ".join(f"GPU{g['index']}:{int(g['mem_used'])}MB/{int(g['util'])}%" for g in gpus)
    free = [g["index"] for g in gpus if g["mem_used"] < free_mem_mb and g["util"] < free_util_pct]
    if free:
        chosen = assigned if assigned in free else min(free)
        print(f"[gpu] usage [{snap}] -> free: {free} -> picking GPU {chosen}")
    else:
        chosen = assigned
        print(f"[gpu] WARNING: no fully-free GPU [{snap}] -> staying on assigned GPU {chosen} "
              f"(shared right now; consider waiting).")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(chosen)
    os.environ["_NLAP_GPU_PINNED"] = "1"
    print(f"[gpu] CUDA_VISIBLE_DEVICES={chosen} (in-process this is cuda:0; other GPUs invisible)")



def skip(msg: str) -> None:
    SKIPPED.append(msg)
    print(f"[SKIPPED] {msg}")


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
    """preds = {image: {"wh": [W,H], "ms": {"whole":.., "tta":.., "tiles":..},
    "sources": [{"name","boxes","scores"}, ...]}}   boxes = GLOBAL xyxy px."""
    from ultralytics import YOLO
    import cv2
    # The C2A/joint checkpoints pickle the thesis' custom CBAM classes under
    # __main__; register them into ultralytics + THIS script's __main__ before
    # loading, or torch's unpickler raises "Can't get attribute 'CBAM'".
    import_module("02_nwd_loss_patch").ensure_cbam_compat(getattr(args, "compat", None))
    model = YOLO(args.weights)
    img_dir = Path(args.images_dir)
    names = sorted(p.name for p in img_dir.iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if args.limit:
        names = names[: args.limit]
    preds: dict = {}
    ckpt = Path(args.save_preds) if args.save_preds else None
    if ckpt and ckpt.is_file():          # power-cut resume: keep finished images
        preds = json.loads(ckpt.read_text(encoding="utf-8"))
        print(f"[resume] {len(preds)} images already predicted -- skipping them")
    for k, fn in enumerate(names):
        if fn in preds:
            continue
        img = cv2.imread(str(img_dir / fn))
        H, W = img.shape[:2]
        sources, ms = [], {}
        t = time.perf_counter()
        r = model.predict(img, imgsz=args.imgsz_whole, conf=CONF_FLOOR,
                          max_det=args.max_det, device=args.device, verbose=False)[0]
        ms["whole"] = (time.perf_counter() - t) * 1000
        sources.append({"name": "whole",
                        "boxes": r.boxes.xyxy.cpu().numpy().tolist(),
                        "scores": r.boxes.conf.cpu().numpy().tolist()})
        if args.tta_imgsz:
            t = time.perf_counter()
            r = model.predict(img, imgsz=args.tta_imgsz, conf=CONF_FLOOR,
                              augment=True, max_det=args.max_det, device=args.device, verbose=False)[0]
            ms["tta"] = (time.perf_counter() - t) * 1000
            sources.append({"name": "tta",
                            "boxes": r.boxes.xyxy.cpu().numpy().tolist(),
                            "scores": r.boxes.conf.cpu().numpy().tolist()})
        t = time.perf_counter()
        n_tiles = 0
        for s in args.slices:
            if s <= 0 or max(W, H) <= s:
                continue
            for (x, y) in tile_grid(W, H, s, args.overlap):
                crop = img[y: y + s, x: x + s]
                r = model.predict(crop, imgsz=s, conf=CONF_FLOOR,
                                  max_det=args.max_det, device=args.device, verbose=False)[0]
                b = r.boxes.xyxy.cpu().numpy()
                if len(b):
                    b[:, [0, 2]] += x
                    b[:, [1, 3]] += y
                sources.append({"name": f"tile{s}", "boxes": b.tolist(),
                                "scores": r.boxes.conf.cpu().numpy().tolist()})
                n_tiles += 1
        if n_tiles:
            ms["tiles"] = (time.perf_counter() - t) * 1000
        preds[fn] = {"wh": [W, H], "ms": ms, "sources": sources}
        if (k + 1) % 100 == 0:
            print(f"[infer] {k+1}/{len(names)}")
            if ckpt:                      # periodic checkpoint (atomic replace)
                tmp = ckpt.with_suffix(".tmp")
                tmp.write_text(json.dumps(preds), encoding="utf-8")
                tmp.replace(ckpt)
    return preds


# ----------------------------------------------------------------- GT loading
def load_gt(gt_json: str):
    with open(gt_json, encoding="utf-8") as f:
        d = json.load(f)
    by_img: dict[str, np.ndarray] = {}
    id2fn = {im["id"]: im["file_name"] for im in d["images"]}
    tmp = defaultdict(list)
    for a in d.get("annotations", []):
        x, y, w, h = a["bbox"]
        tmp[id2fn[a["image_id"]]].append([x, y, x + w, y + h])
    for im in d["images"]:                       # include zero-GT images
        by_img[im["file_name"]] = np.array(tmp.get(im["file_name"], []),
                                           float).reshape(-1, 4)
    return by_img, d


# ----------------------------------------------------------------- core eval
def _confusion_at(fused_cache, gt, thr):
    TP = FP = FN = 0
    for fn, (fb, fs) in fused_cache.items():
        g = gt.get(fn, np.zeros((0, 4)))
        keep = fs >= thr
        if keep.any():
            corr, missed = match_greedy(fb[keep], fs[keep], g, IOU_MATCH)
            TP += int(corr.sum())
            FP += int((1 - corr).sum())
            FN += missed
        else:
            FN += len(g)
    return TP, FP, FN


def evaluate_mode(preds: dict, gt: dict, mode: str, iou_fuse: float,
                  scalers=None, outdir: Path | None = None,
                  bootstrap: int = 0) -> dict:
    # -- fuse once per image, cache
    fused_cache: dict[str, tuple] = {}
    fuse_ms = []
    per_img_rows = []       # (image, score, correct, best_iou)
    n_gt_total = 0
    for fn, p in preds.items():
        g = gt.get(fn, np.zeros((0, 4)))
        t = time.perf_counter()
        fb, fs = fuse_sources(p["sources"], mode=mode, iou_thr=iou_fuse, scalers=scalers)
        fuse_ms.append((time.perf_counter() - t) * 1000)
        fused_cache[fn] = (fb, fs)
        n_gt_total += len(g)
        if len(fb):
            correct, _ = match_greedy(fb, fs, g, IOU_MATCH)
            best_iou = iou_matrix(fb, g).max(1) if len(g) else np.zeros(len(fb))
            for s_, c_, i_ in zip(fs, correct, best_iou):
                per_img_rows.append((fn, float(s_), float(c_), float(i_)))
    out: dict = {"mode": mode, "n_images": len(preds), "n_gt": n_gt_total}
    if not per_img_rows:
        skip(f"{mode}: no detections at all")
        return out

    rows = sorted(per_img_rows, key=lambda r: -r[1])
    scores = np.array([r[1] for r in rows])
    corr = np.array([r[2] for r in rows])

    # -- AP50 all-point (protocol-matched to the June SAHI reports)
    tp = np.cumsum(corr)
    fp = np.cumsum(1 - corr)
    rec = tp / max(n_gt_total, 1)
    prec = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.concatenate([[0], rec, [rec[-1]]])
    mpre = np.concatenate([[1], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    out["AP50_allpoint"] = round(float(np.sum(np.diff(mrec) * mpre[1:])), 4)

    # -- threshold sweep 0..1 step 0.01 (spec 11.1) -> curves + optima + conf=0.25
    sweep = []
    for thr in np.arange(0.0, 1.0001, 0.01):
        m = scores >= thr
        TP = float(corr[m].sum())
        FPc = float(m.sum() - TP)
        FNc = float(n_gt_total - TP)
        P = TP / max(TP + FPc, 1e-9)
        R = TP / max(TP + FNc, 1e-9)
        F1 = 2 * P * R / max(P + R, 1e-9)
        F2 = 5 * P * R / max(4 * P + R, 1e-9)
        sweep.append((round(float(thr), 2), P, R, F1, F2))
    arr = np.array(sweep)
    i1, i2 = int(arr[:, 3].argmax()), int(arr[:, 4].argmax())
    out.update({"OptThr_F1": float(arr[i1, 0]), "Best_F1": round(float(arr[i1, 3]), 4),
                "OptThr_F2": float(arr[i2, 0]), "Best_F2": round(float(arr[i2, 4]), 4)})
    op = arr[np.argmin(np.abs(arr[:, 0] - CONF_OPERATIONAL))]
    out.update({"precision_conf25": round(float(op[1]), 4),
                "recall_conf25": round(float(op[2]), 4),
                "F1_conf25": round(float(op[3]), 4),
                "F2_conf25": round(float(op[4]), 4)})
    TP, FPc, FNc = _confusion_at(fused_cache, gt, CONF_OPERATIONAL)
    out["confusion_TP_FP_FN_conf25"] = [TP, FPc, FNc]
    TPo, FPo, FNo = _confusion_at(fused_cache, gt, out["OptThr_F1"])
    out["confusion_TP_FP_FN_optF1"] = [TPo, FPo, FNo]

    # -- localization quality of TPs (C-WBF mechanism metric)
    tp_ious = [r[3] for r in rows if r[2] == 1]
    out["meanIoU_TP"] = round(float(np.mean(tp_ious)), 4) if tp_ious else None

    # -- per-size recall at OptThr_F1 (spec 5-bin continuity)
    size_tot = {b[0]: 0 for b in SIZE_BINS}
    size_hit = {b[0]: 0 for b in SIZE_BINS}
    thr = out["OptThr_F1"]
    for fn, (fb, fs) in fused_cache.items():
        g = gt.get(fn, np.zeros((0, 4)))
        if len(g) == 0:
            continue
        taken = np.zeros(len(g), bool)
        keep = fs >= thr
        if keep.any():
            M = iou_matrix(fb[keep], g)
            for i in np.argsort(-fs[keep]):
                j = int(np.argmax(np.where(taken, -1.0, M[i])))
                if not taken[j] and M[i, j] >= IOU_MATCH:
                    taken[j] = True
        sz = np.sqrt(np.clip(g[:, 2] - g[:, 0], 0, None) * np.clip(g[:, 3] - g[:, 1], 0, None))
        for name, lo, hi in SIZE_BINS:
            m = (sz >= lo) & (sz < hi)
            size_tot[name] += int(m.sum())
            size_hit[name] += int(taken[m].sum())
    out["per_size_recall"] = {n: (round(size_hit[n] / size_tot[n], 4) if size_tot[n] else None)
                              for n, _, _ in SIZE_BINS}
    out["per_size_n"] = dict(size_tot)

    # -- calibration of fused detections (spec 11.4: ECE 10-bin, MCE, Brier)
    conf_all = scores
    corr_all = corr
    edges = np.linspace(0, 1, 11)
    ece = mce = 0.0
    for i in range(10):
        m = (conf_all > edges[i]) & (conf_all <= edges[i + 1])
        if m.any():
            gap = abs(corr_all[m].mean() - conf_all[m].mean())
            ece += m.mean() * gap
            mce = max(mce, gap)
    out["calibration"] = {"ECE": round(float(ece), 4), "MCE": round(float(mce), 4),
                          "Brier": round(float(np.mean((conf_all - corr_all) ** 2)), 4)}

    # -- e2e latency from stored timings + fusion overhead
    per_img_ms = []
    for fn, p in preds.items():
        ms = p.get("ms") or {}
        if ms:
            base = ms.get("whole", 0.0)
            if any(s["name"] == "tta" for s in p["sources"]):
                base += ms.get("tta", 0.0)
            if any(s["name"].startswith("tile") for s in p["sources"]):
                base += ms.get("tiles", 0.0)
            per_img_ms.append(base)
    if per_img_ms:
        lat = np.array(per_img_ms) + np.mean(fuse_ms)
        out["latency_ms"] = {"mean": round(float(lat.mean()), 2),
                             "p50": round(float(np.percentile(lat, 50)), 2),
                             "p95": round(float(np.percentile(lat, 95)), 2),
                             "fusion_only_mean": round(float(np.mean(fuse_ms)), 3)}
        out["FPS"] = round(1000.0 / max(lat.mean(), 1e-9), 2)
    else:
        skip(f"{mode}: latency (no timings in preds file -- re-run inference to capture)")

    # -- bootstrap CI over images (optional; paper-grade)
    if bootstrap > 0:
        rng = np.random.default_rng(0)
        keys = list(preds.keys())
        by_img = defaultdict(list)
        for r in rows:
            by_img[r[0]].append(r)
        gt_count = {fn: len(gt.get(fn, ())) for fn in keys}
        ap_s, f1_s = [], []
        for _ in range(bootstrap):
            pick = rng.choice(len(keys), len(keys), replace=True)
            rs, ng = [], 0
            for pi in pick:
                rs += by_img[keys[pi]]
                ng += gt_count[keys[pi]]
            if not rs or ng == 0:
                continue
            rs.sort(key=lambda r: -r[1])
            c2 = np.array([r[2] for r in rs])
            tp2 = np.cumsum(c2)
            fp2 = np.cumsum(1 - c2)
            r2 = tp2 / ng
            p2 = tp2 / np.maximum(tp2 + fp2, 1e-9)
            mr = np.concatenate([[0], r2, [r2[-1]]])
            mp = np.concatenate([[1], p2, [0]])
            for i in range(len(mp) - 2, -1, -1):
                mp[i] = max(mp[i], mp[i + 1])
            ap_s.append(float(np.sum(np.diff(mr) * mp[1:])))
            f1b = 2 * p2 * r2 / np.maximum(p2 + r2, 1e-9)
            f1_s.append(float(f1b.max()))
        if ap_s:
            out["AP50_CI95"] = [round(float(np.percentile(ap_s, 2.5)), 4),
                                round(float(np.percentile(ap_s, 97.5)), 4)]
            out["BestF1_CI95"] = [round(float(np.percentile(f1_s, 2.5)), 4),
                                  round(float(np.percentile(f1_s, 97.5)), 4)]

    # -- artifacts: curves CSV (+PNG best-effort), per-image confusion CSV
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)
        with open(outdir / f"{mode}_curves.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["conf", "precision", "recall", "F1", "F2"])
            w.writerows([[f"{a:.2f}", f"{b:.5f}", f"{c:.5f}", f"{d:.5f}", f"{e:.5f}"]
                         for a, b, c, d, e in sweep])
        hist, hedges = np.histogram(conf_all, bins=20, range=(0, 1))
        with open(outdir / f"{mode}_conf_hist.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["bin_lo", "bin_hi", "count"])
            w.writerows([[f"{hedges[i]:.2f}", f"{hedges[i+1]:.2f}", int(hist[i])]
                         for i in range(len(hist))])
        with open(outdir / f"{mode}_per_image.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["image", "n_gt", "TP_optF1", "FP_optF1", "FN_optF1"])
            for fn, (fb, fs) in fused_cache.items():
                g = gt.get(fn, np.zeros((0, 4)))
                keep = fs >= thr
                if keep.any():
                    c3, miss = match_greedy(fb[keep], fs[keep], g, IOU_MATCH)
                    w.writerow([fn, len(g), int(c3.sum()), int((1 - c3).sum()), miss])
                else:
                    w.writerow([fn, len(g), 0, 0, len(g)])
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=300)
            ax[0].plot(arr[:, 2], arr[:, 1])
            ax[0].set_xlabel("recall"); ax[0].set_ylabel("precision"); ax[0].set_title(f"PR ({mode})")
            ax[1].plot(arr[:, 0], arr[:, 3], label="F1")
            ax[1].plot(arr[:, 0], arr[:, 4], label="F2")
            ax[1].set_xlabel("conf"); ax[1].legend(); ax[1].set_title("F1/F2 vs conf")
            fig.tight_layout()
            fig.savefig(outdir / f"{mode}_curves.png")
            plt.close(fig)
        except Exception as e:
            skip(f"{mode}: curve PNGs ({e}) -- CSVs written, plot later")
    return out


def coco_eval(preds, gt_raw, mode, iou_fuse, scalers=None) -> dict:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        skip(f"{mode}: COCO 12-stat block (pycocotools not installed on this machine)")
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
    keys = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
            "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large"]
    return {f"coco_{k}": round(float(v), 4) for k, v in zip(keys, s)}


def fit_calibration(preds, gt, out_path):
    fam: dict[str, list] = defaultdict(lambda: ([], []))
    for fn, p in preds.items():
        g = gt.get(fn, np.zeros((0, 4)))
        for s in p["sources"]:
            key = "tile" if s["name"].startswith("tile") else s["name"]
            b = np.asarray(s["boxes"], float).reshape(-1, 4)
            c = np.asarray(s["scores"], float).reshape(-1)
            if len(b) == 0:
                continue
            corr, _ = match_greedy(b, c, g, IOU_MATCH)
            fam[key][0].extend(c.tolist())
            fam[key][1].extend(corr.tolist())
    cal = {}
    for k, (c, y) in fam.items():
        c, y = np.array(c), np.array(y)
        ts = TempScaler().fit(c, y)
        cal[k] = {"T": ts.T, "n": len(c),
                  "ece_before": round(TempScaler.ece(c, y, bins=10), 4),
                  "ece_after": round(TempScaler.ece(ts.apply(c), y, bins=10), 4)}
        print(f"[cal] {k:6s} T={ts.T:.3f} n={len(c)} ECE {cal[k]['ece_before']} -> {cal[k]['ece_after']}")
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
        sz = rng.uniform(5, 40, n)
        xy = rng.uniform(0, 600, (n, 2))
        g = np.concatenate([xy, xy + sz[:, None]], 1)
        gt[fn] = g
        srcs = []
        for src_i in range(2):
            bs, ss = [], []
            for b in g:
                if rng.uniform() < 0.86:
                    bs.append(b + rng.normal(0, 1.2, 4))
                    ss.append(float(np.clip(rng.beta(5, 2), 0.05, 0.99)))
            for _ in range(rng.integers(0, 4)):
                c = rng.uniform(0, 600, 2)
                s_ = rng.uniform(6, 25)
                bs.append([c[0], c[1], c[0] + s_, c[1] + s_])
                ss.append(float(np.clip(rng.beta(2, 4), 0.05, 0.9)))
            srcs.append({"name": f"tile{src_i}", "boxes": bs, "scores": ss})
        preds[fn] = {"wh": [640, 640], "ms": {"whole": 10.0}, "sources": srcs}
    outdir = Path(__file__).parent / "_synth_out"
    res = {}
    for mode in ("nms", "wbf"):
        res[mode] = evaluate_mode(preds, gt, mode, 0.55, outdir=outdir, bootstrap=50)
        print(f"[synthetic] {mode}: AP50={res[mode]['AP50_allpoint']} "
              f"Best_F1={res[mode]['Best_F1']} meanIoU_TP={res[mode]['meanIoU_TP']} "
              f"CI95={res[mode].get('AP50_CI95')}")
    assert res["wbf"]["meanIoU_TP"] >= res["nms"]["meanIoU_TP"] - 1e-6
    assert res["wbf"]["AP50_allpoint"] >= res["nms"]["AP50_allpoint"] - 0.01
    # full-contract presence check (the point of this rewrite)
    need = ["AP50_allpoint", "OptThr_F1", "Best_F1", "OptThr_F2", "Best_F2",
            "precision_conf25", "recall_conf25", "F1_conf25", "F2_conf25",
            "confusion_TP_FP_FN_conf25", "confusion_TP_FP_FN_optF1",
            "per_size_recall", "per_size_n", "calibration", "meanIoU_TP",
            "latency_ms", "FPS", "AP50_CI95"]
    missing = [k for k in need if k not in res["wbf"]]
    assert not missing, f"metric contract incomplete: {missing}"
    assert set(res["wbf"]["per_size_recall"].keys()) == {"very_tiny", "tiny", "small", "medium", "large"}
    for f_ in ("wbf_curves.csv", "wbf_conf_hist.csv", "wbf_per_image.csv"):
        assert (outdir / f_).is_file(), f_
    # calibration path
    fit_calibration(preds, gt, outdir / "cal.json")
    cal = json.load(open(outdir / "cal.json"))
    scalers = {k: TempScaler(v["T"]) for k, v in cal.items()}
    r3 = evaluate_mode(preds, gt, "cwbf", 0.55, scalers=scalers, outdir=outdir)
    assert r3["AP50_allpoint"] > 0.5
    print(f"[synthetic] cwbf: AP50={r3['AP50_allpoint']} Best_F1={r3['Best_F1']} "
          f"ECE={r3['calibration']['ECE']} (nms ECE={res['nms']['calibration']['ECE']})")
    print("SYNTHETIC SELFTEST PASSED (full metric contract verified)")


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
    ap.add_argument("--iou-fuse", type=float, default=0.55)
    ap.add_argument("--modes", default="nms,wbf,cwbf")
    ap.add_argument("--max-det", type=int, default=300,
                    help="per-source cap; 300 = ultralytics default = June-runs protocol")
    ap.add_argument("--device", default="0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=0, help="e.g. 1000 for paper CIs")
    ap.add_argument("--save-preds")
    ap.add_argument("--load-preds")
    ap.add_argument("--fit-cal")
    ap.add_argument("--cal")
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
        pin_free_gpu()   # shared-box convention: pin a free GPU before torch loads
        preds = run_model(args)
        if args.save_preds:
            with open(args.save_preds, "w", encoding="utf-8") as f:
                json.dump(preds, f)
            print(f"[preds] saved -> {args.save_preds}")

    gt, gt_raw = load_gt(args.gt_json) if args.gt_json else ({}, None)
    if not gt:
        sys.exit("--gt-json is required for evaluation (or use --fit-cal / --save-preds only)")

    if args.fit_cal:
        fit_calibration(preds, gt, args.fit_cal)
        return

    scalers = None
    if args.cal:
        cal = json.load(open(args.cal, encoding="utf-8"))
        scalers = {k: TempScaler(v["T"]) for k, v in cal.items()}

    stem = Path(args.save_preds or args.load_preds or "ablation")
    outdir = stem.parent / (stem.stem + "_artifacts")
    results = []
    for mode in args.modes.split(","):
        r = evaluate_mode(preds, gt, mode, args.iou_fuse,
                          scalers=scalers if mode == "cwbf" else None,
                          outdir=outdir, bootstrap=args.bootstrap)
        if gt_raw:
            r.update(coco_eval(preds, gt_raw, mode, args.iou_fuse,
                               scalers=scalers if mode == "cwbf" else None))
        results.append(r)
        print(f"[result] {json.dumps(r)}")

    flat = []
    for r in results:
        fr = {}
        for k, v in r.items():
            if isinstance(v, dict):
                fr.update({f"{k}.{k2}": v2 for k2, v2 in v.items()})
            elif isinstance(v, list):
                fr[k] = "|".join(map(str, v))
            else:
                fr[k] = v
        flat.append(fr)
    with open(f"{stem.parent / stem.stem}_ablation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in flat for k in r}))
        w.writeheader()
        w.writerows(flat)
    with open(f"{stem.parent / stem.stem}_ablation.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    if SKIPPED:
        (outdir / "skipped_metrics.txt").parent.mkdir(parents=True, exist_ok=True)
        (outdir / "skipped_metrics.txt").write_text("\n".join(SKIPPED))
    print(f"[done] -> {stem.parent / stem.stem}_ablation.csv  (+ artifacts in {outdir})")


if __name__ == "__main__":
    main()
