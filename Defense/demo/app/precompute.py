"""
precompute.py -- Defense-demo batch inference (run on the GPU PC).

Runs the demo comparison over a sample of the C2A test split and frames
extracted from the drone-shoot videos, caching RAW predictions so the
laptop UI can redraw boxes + sweep confidence thresholds with no GPU.

Configs (C2A test, matches the report's SAHI+TTA ablation protocol,
run 20260707_062217_sahi_tta_cbam_p2):
    baseline          YOLO11m baseline            @640, conf floor 0.10
    cbam_p2           YOLO11m+CBAM+P2             @640, conf floor 0.10
    cbam_p2_sahi_tta  CBAM+P2 + SAHI slice256/ov0.30 GREEDYNMM(IOS,0.5)
                      + per-tile TTA, floor 0.10  (the report's winning config)

Configs (drone frames, no GT):
    cbam_p2, cbam_p2_sahi_tta, enriched, enriched_sahi_tta

Outputs (into <demo>/results/):
    predictions_c2a.json     raw boxes @ floor conf per image per config + GT
    predictions_drone.json   same for extracted drone frames
    drone_frames/*.jpg       the exact frames referenced by the JSON
    latency_summary.json     mean/p50/p95 ms per config + GPU name
    curation_hints.csv       per-image TP/FN @conf .25 -> sort by 'rescue'
                             to find images where SAHI saves missed humans

Usage (from the app folder):
    python precompute.py --smoke      # 30-second sanity check FIRST
    python precompute.py              # full run (~250 C2A imgs + 75 drone frames)
Options: --n-images 250 --frames-per-video 25 --seed 0 --skip-c2a --skip-drone
"""
import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

import cbam_modules
cbam_modules.register()          # MUST precede any ultralytics/sahi model load

from ultralytics import YOLO     # noqa: E402

# ---------------------------------------------------------------- paths
APP_DIR = Path(__file__).resolve().parent
DEMO_ROOT = APP_DIR.parent
RESULTS = DEMO_ROOT / "results"

# c2a_test travels with the demo folder on the laptop but is intentionally
# NOT transferred to the GPU PC -- fall back to the training copy there.
C2A_FALLBACK = Path(r"E:\Thesis_mofazzal_2007074\common\c2a\C2A_Dataset\new_dataset3\test")
_local_c2a = DEMO_ROOT / "data" / "c2a_test"
C2A_TEST = _local_c2a if (_local_c2a / "images").is_dir() else C2A_FALLBACK

DRONE_DIR = DEMO_ROOT / "data" / "drone_shoot"

MODELS = {
    "baseline": DEMO_ROOT / "models" / "baseline_yolo11m.pt",
    "cbam_p2": DEMO_ROOT / "models" / "cbam_p2head.pt",
    "enriched": DEMO_ROOT / "models" / "cbam_p2head_finetune_enriched.pt",
}

# ---------------------------------------------------- protocol constants
# system_spec_thesis.md Sec 5 / sahi_tta_cbam_p2_thesis.py -- DO NOT change.
FLOOR_CONF = 0.10                # collection floor; UI applies thresholds offline
CONF_OP, IOU_OP = 0.25, 0.5      # operational point used for curation hints
SAHI_SLICE, SAHI_OVERLAP = 256, 0.30                 # report's winning config
SAHI_POSTPROCESS = dict(perform_standard_pred=True, postprocess_type="GREEDYNMM",
                        postprocess_match_metric="IOS", postprocess_match_threshold=0.5)
IMGSZ = 640
DEVICE = "cuda:0"


class SahiTTAPatch:
    """Injects augment=True into every prediction SAHI makes (per-tile TTA).
    Verbatim from sahi_tta_cbam_p2_thesis.py."""
    def __enter__(self):
        from sahi.models.ultralytics import UltralyticsDetectionModel
        self.cls = UltralyticsDetectionModel
        self.orig = UltralyticsDetectionModel.perform_inference
        orig = self.orig

        def patched(mself, image):
            oc = mself.model.__call__

            def aug_call(*a, **k):
                k["augment"] = True
                return oc(*a, **k)
            mself.model.__call__ = aug_call
            try:
                orig(mself, image)
            finally:
                mself.model.__call__ = oc
        UltralyticsDetectionModel.perform_inference = patched
        return self

    def __exit__(self, *exc):
        self.cls.perform_inference = self.orig
        return False


# ---------------------------------------------------------------- helpers
def load_plain(tag):
    m = YOLO(str(MODELS[tag]))
    m.to(DEVICE.split(":")[0] if ":" not in DEVICE else DEVICE)
    m.predict(np.zeros((IMGSZ, IMGSZ, 3), np.uint8), verbose=False)  # warmup
    return m


def load_sahi(tag):
    from sahi import AutoDetectionModel
    return AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=str(MODELS[tag]),
        confidence_threshold=FLOOR_CONF, device=DEVICE)


def predict_plain(model, img_path):
    t0 = time.perf_counter()
    r = model.predict(str(img_path), conf=FLOOR_CONF, imgsz=IMGSZ,
                      verbose=False)[0]
    ms = (time.perf_counter() - t0) * 1000
    b = r.boxes
    xyxy = b.xyxy.cpu().numpy() if len(b) else np.zeros((0, 4), np.float32)
    conf = b.conf.cpu().numpy() if len(b) else np.zeros((0,), np.float32)
    h, w = r.orig_shape
    return xyxy, conf, ms, (w, h)


def predict_sahi(sahi_model, img_path, tta=True):
    from sahi.predict import get_sliced_prediction
    ctx = SahiTTAPatch() if tta else _NullCtx()
    with ctx:
        t0 = time.perf_counter()
        res = get_sliced_prediction(str(img_path), sahi_model,
                                    slice_height=SAHI_SLICE, slice_width=SAHI_SLICE,
                                    overlap_height_ratio=SAHI_OVERLAP,
                                    overlap_width_ratio=SAHI_OVERLAP,
                                    verbose=0, **SAHI_POSTPROCESS)
        ms = (time.perf_counter() - t0) * 1000
    boxes, scores = [], []
    for o in res.object_prediction_list:
        bb = o.bbox
        boxes.append([bb.minx, bb.miny, bb.maxx, bb.maxy])
        scores.append(o.score.value)
    return (np.asarray(boxes, np.float32).reshape(-1, 4),
            np.asarray(scores, np.float32), ms, None)


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def read_gt(label_path, w, h):
    """YOLO-format label file -> pixel xyxy list."""
    if not label_path.is_file():
        return []
    out = []
    for ln in label_path.read_text().splitlines():
        p = ln.split()
        if len(p) < 5:
            continue
        cx, cy, bw, bh = (float(v) for v in p[1:5])
        out.append([round((cx - bw / 2) * w, 1), round((cy - bh / 2) * h, 1),
                    round((cx + bw / 2) * w, 1), round((cy + bh / 2) * h, 1)])
    return out


def greedy_match(pred_xyxy, pred_conf, gt, iou_thr):
    """TP/FP/FN with greedy score-ordered IoU matching (mirrors report script)."""
    gt = np.asarray(gt, np.float32).reshape(-1, 4)
    if len(pred_xyxy) == 0:
        return 0, 0, len(gt)
    order = np.argsort(-pred_conf)
    pb = pred_xyxy[order]
    used = np.zeros(len(gt), bool)
    tp = 0
    for box in pb:
        if not len(gt):
            break
        xx1 = np.maximum(box[0], gt[:, 0]); yy1 = np.maximum(box[1], gt[:, 1])
        xx2 = np.minimum(box[2], gt[:, 2]); yy2 = np.minimum(box[3], gt[:, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        a1 = (box[2] - box[0]) * (box[3] - box[1])
        a2 = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
        iou = inter / np.clip(a1 + a2 - inter, 1e-9, None)
        iou[used] = -1
        j = int(np.argmax(iou))
        if iou[j] >= iou_thr:
            used[j] = True
            tp += 1
    fp = len(pb) - tp
    fn = int((~used).sum())
    return tp, fp, fn


def pack(xyxy, conf):
    """[[x1,y1,x2,y2,conf], ...] rounded for compact JSON."""
    return [[round(float(a), 1), round(float(b), 1), round(float(c), 1),
             round(float(d), 1), round(float(s), 4)]
            for (a, b, c, d), s in zip(xyxy, conf)]


def img_size(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.size  # (w, h)


def extract_frames(video_path, n_frames, out_dir):
    """N evenly spaced frames across the video -> JPGs. Returns saved paths."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if total <= 0:
        print(f"  !! cannot read {video_path.name}, skipping")
        return []
    idxs = np.linspace(0, total - 1, min(n_frames, total)).astype(int)
    saved = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        name = f"{video_path.stem}_t{i / fps:07.2f}s.jpg"
        fp = out_dir / name
        cv2.imwrite(str(fp), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved.append(fp)
    cap.release()
    return saved


def lat_stats(ms_list):
    a = np.asarray(ms_list, np.float64)
    if not len(a):
        return {}
    return {"mean": round(float(a.mean()), 1), "p50": round(float(np.percentile(a, 50)), 1),
            "p95": round(float(np.percentile(a, 95)), 1), "n": len(a)}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2 imgs + 2 frames sanity check")
    ap.add_argument("--n-images", type=int, default=250)
    ap.add_argument("--frames-per-video", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-c2a", action="store_true")
    ap.add_argument("--skip-drone", action="store_true")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    print(f"demo root : {DEMO_ROOT}")
    print(f"c2a test  : {C2A_TEST}  (exists: {(C2A_TEST / 'images').is_dir()})")
    print(f"drone dir : {DRONE_DIR}  (exists: {DRONE_DIR.is_dir()})")
    for tag, p in MODELS.items():
        print(f"model {tag:9s}: {p.name}  (exists: {p.is_file()})")
        if not p.is_file():
            sys.exit(f"FATAL: missing {p}")

    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (!!)"
    print(f"device    : {gpu}")
    if not torch.cuda.is_available():
        print("WARNING: no CUDA -- this will be very slow. Ctrl+C if unintended.")

    meta_common = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu": gpu, "floor_conf": FLOOR_CONF, "imgsz": IMGSZ,
        "sahi": {"slice": SAHI_SLICE, "overlap": SAHI_OVERLAP, "tta": True,
                 **SAHI_POSTPROCESS},
        "protocol_source": "sahi_tta_cbam_p2_thesis.py run 20260707_062217 (report)",
    }

    lat = {}

    # ---------------------------------------------------------- C2A test
    if not args.skip_c2a:
        img_dir, lbl_dir = C2A_TEST / "images", C2A_TEST / "labels"
        all_imgs = sorted(img_dir.glob("*.*"))
        n = 2 if args.smoke else min(args.n_images, len(all_imgs))
        sample = sorted(random.Random(args.seed).sample(all_imgs, n),
                        key=lambda p: p.name)
        print(f"\n=== C2A: {n} images, 3 configs ===")

        base_m = load_plain("baseline")
        cbam_m = load_plain("cbam_p2")
        sahi_m = load_sahi("cbam_p2")

        configs = {
            "baseline": ("YOLO11m baseline @640", lambda p: predict_plain(base_m, p)),
            "cbam_p2": ("YOLO11m+CBAM+P2 @640", lambda p: predict_plain(cbam_m, p)),
            "cbam_p2_sahi_tta": ("CBAM+P2 + SAHI(256/0.30)+TTA",
                                 lambda p: predict_sahi(sahi_m, p, tta=True)),
        }
        images, hints = {}, []
        for i, ip in enumerate(sample, 1):
            w, h = img_size(ip)
            gt = read_gt(lbl_dir / (ip.stem + ".txt"), w, h)
            rec = {"w": w, "h": h, "gt": gt}
            row = {"file": ip.name, "gt_n": len(gt)}
            for tag, (_, fn) in configs.items():
                xyxy, conf, ms, _ = fn(ip)
                rec[tag] = {"boxes": pack(xyxy, conf), "ms": round(ms, 1)}
                lat.setdefault(tag, []).append(ms)
                k = conf >= CONF_OP
                tp, fp, fnn = greedy_match(xyxy[k], conf[k], gt, IOU_OP)
                row[f"tp_{tag}"] = tp
                row[f"fp_{tag}"] = fp
                row[f"fn_{tag}"] = fnn
            row["rescue"] = row["tp_cbam_p2_sahi_tta"] - row["tp_baseline"]
            hints.append(row)
            images[ip.name] = rec
            if i % 10 == 0 or i == n:
                print(f"  [{i}/{n}] {ip.name}  "
                      f"gt={len(gt)} sahi_boxes={len(rec['cbam_p2_sahi_tta']['boxes'])}")

        n_zero = sum(1 for r in hints if r["gt_n"] == 0)
        total_gt = sum(r["gt_n"] for r in hints)
        print(f"  GT sanity: {total_gt} GT boxes across {n} images; "
              f"{n_zero} images with 0 GT")
        if n_zero > n * 0.2:
            print("  !! WARNING: too many 0-GT images. The labels folder is "
                  "probably incomplete -- data/c2a_test/labels should hold "
                  "2043 .txt files. Fix the copy and rerun.")

        out = {"meta": {**meta_common,
                        "configs": {k: v[0] for k, v in configs.items()},
                        "image_dir": "data/c2a_test/images",
                        "n_images": n, "seed": args.seed},
               "images": images}
        (RESULTS / "predictions_c2a.json").write_text(json.dumps(out))
        print(f"  -> {RESULTS / 'predictions_c2a.json'}")

        hints.sort(key=lambda r: -r["rescue"])
        with open(RESULTS / "curation_hints.csv", "w", newline="") as f:
            wcsv = csv.DictWriter(f, fieldnames=list(hints[0].keys()))
            wcsv.writeheader()
            wcsv.writerows(hints)
        print(f"  -> {RESULTS / 'curation_hints.csv'} (sorted by SAHI rescue)")

        del base_m, cbam_m, sahi_m
        torch.cuda.empty_cache()

    # ---------------------------------------------------------- drone
    if not args.skip_drone:
        # set-dedupe: Windows glob is case-insensitive, *.MP4 and *.mp4 overlap
        videos = sorted({p.resolve() for ext in ("*.MP4", "*.mp4")
                         for p in DRONE_DIR.glob(ext)})
        if not videos:
            print("\n!! no drone videos found -- skipping drone stage")
        else:
            fdir = RESULTS / "drone_frames"
            fdir.mkdir(exist_ok=True)
            per_vid = 2 if args.smoke else args.frames_per_video
            frames = []
            print(f"\n=== Drone: extracting {per_vid} frames/video ===")
            for v in videos:
                got = extract_frames(v, per_vid, fdir)
                print(f"  {v.name}: {len(got)} frames")
                frames += got

            cbam_m = load_plain("cbam_p2")
            enr_m = load_plain("enriched")
            sahi_c = load_sahi("cbam_p2")
            sahi_e = load_sahi("enriched")
            configs = {
                "cbam_p2": ("CBAM+P2 @640", lambda p: predict_plain(cbam_m, p)),
                "cbam_p2_sahi_tta": ("CBAM+P2 + SAHI(256/0.30)+TTA",
                                     lambda p: predict_sahi(sahi_c, p, tta=True)),
                "enriched": ("CBAM+P2 fine-tuned (enriched) @640",
                             lambda p: predict_plain(enr_m, p)),
                "enriched_sahi_tta": ("Enriched + SAHI(256/0.30)+TTA",
                                      lambda p: predict_sahi(sahi_e, p, tta=True)),
            }
            print(f"=== Drone: {len(frames)} frames, 4 configs "
                  f"(SAHI on 4K frames is slow -- expect a few s/frame) ===")
            images = {}
            for i, fp in enumerate(frames, 1):
                w, h = img_size(fp)
                rec = {"w": w, "h": h}
                for tag, (_, fn) in configs.items():
                    xyxy, conf, ms, _ = fn(fp)
                    rec[tag] = {"boxes": pack(xyxy, conf), "ms": round(ms, 1)}
                    lat.setdefault("drone_" + tag, []).append(ms)
                images[fp.name] = rec
                print(f"  [{i}/{len(frames)}] {fp.name}  "
                      f"sahi={len(rec['cbam_p2_sahi_tta']['boxes'])} "
                      f"enr_sahi={len(rec['enriched_sahi_tta']['boxes'])}")

            out = {"meta": {**meta_common,
                            "configs": {k: v[0] for k, v in configs.items()},
                            "image_dir": "results/drone_frames"},
                   "images": images}
            (RESULTS / "predictions_drone.json").write_text(json.dumps(out))
            print(f"  -> {RESULTS / 'predictions_drone.json'}")

    # ---------------------------------------------------------- latency
    summary = {"gpu": gpu, "created": meta_common["created"],
               "per_config_ms": {k: lat_stats(v) for k, v in lat.items()}}
    (RESULTS / "latency_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nlatency -> {RESULTS / 'latency_summary.json'}")
    for k, v in summary["per_config_ms"].items():
        print(f"  {k:26s} mean {v.get('mean', '?'):>7} ms   p95 {v.get('p95', '?'):>7} ms")
    print("\nDONE." + ("  (smoke only -- now run without --smoke)" if args.smoke else
                       "  Copy the results\\ folder back to the laptop."))


if __name__ == "__main__":
    main()
