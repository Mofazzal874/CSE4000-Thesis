"""
annotate_video.py -- render an annotated (boxes burned in) drone video on the GPU PC.

Examples (from the app folder):
    python annotate_video.py --video 50m.MP4                      # enriched + SAHI (default)
    python annotate_video.py --video 30m.MP4 --model cbam_p2
    python annotate_video.py --video 10m.MP4 --no-sahi --imgsz 1280
    python annotate_video.py --video 50m.MP4 --start 10 --duration 30

Defaults: model=enriched, SAHI slice256/ov0.30 (+per-tile TTA), conf 0.30,
stride 3 (process every 3rd frame, output at src_fps/3 -- looks smooth enough
and cuts render time 3x). Output: <demo>/results/annotated_videos/.

SAHI on 4K frames costs a few seconds per processed frame: a 60 s clip at
stride 3 is ~600 frames -> plan for roughly 20-40 min per video. Use
--duration to render a shorter highlight clip if pressed for time.
"""
import argparse
import time
from pathlib import Path

import numpy as np

import cbam_modules
cbam_modules.register()          # MUST precede any ultralytics/sahi model load

APP_DIR = Path(__file__).resolve().parent
DEMO_ROOT = APP_DIR.parent
DRONE_DIR = DEMO_ROOT / "data" / "drone_shoot"
OUT_DIR = DEMO_ROOT / "results" / "annotated_videos"

MODELS = {
    "cbam_p2": DEMO_ROOT / "models" / "cbam_p2head.pt",
    "enriched": DEMO_ROOT / "models" / "cbam_p2head_finetune_enriched.pt",
}
MODEL_LABELS = {
    "cbam_p2": "YOLO11m+CBAM+P2",
    "enriched": "YOLO11m+CBAM+P2 (fine-tuned)",
}
SAHI_SLICE, SAHI_OVERLAP = 256, 0.30
SAHI_POSTPROCESS = dict(perform_standard_pred=True, postprocess_type="GREEDYNMM",
                        postprocess_match_metric="IOS", postprocess_match_threshold=0.5)


class SahiTTAPatch:
    """Injects augment=True into every prediction SAHI makes (per-tile TTA)."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True,
                    help="filename in data/drone_shoot (e.g. 50m.MP4) or full path")
    ap.add_argument("--model", choices=list(MODELS), default="enriched")
    ap.add_argument("--no-sahi", action="store_true",
                    help="plain full-frame inference instead of SAHI")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--imgsz", type=int, default=1280,
                    help="inference size for --no-sahi mode")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--stride", type=int, default=3, help="process every Nth frame")
    ap.add_argument("--start", type=float, default=0.0, help="start second")
    ap.add_argument("--duration", type=float, default=0.0, help="seconds (0 = to end)")
    args = ap.parse_args()

    import cv2
    vpath = Path(args.video)
    if not vpath.is_file():
        vpath = DRONE_DIR / args.video
    if not vpath.is_file():
        raise SystemExit(f"video not found: {args.video}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mode = "plain" if args.no_sahi else ("sahi" if args.no_tta else "sahi_tta")
    out_path = OUT_DIR / f"{vpath.stem}_{args.model}_{mode}_conf{args.conf:.2f}.mp4"

    # --- model
    use_sahi = not args.no_sahi
    if use_sahi:
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
        det = AutoDetectionModel.from_pretrained(
            model_type="ultralytics", model_path=str(MODELS[args.model]),
            confidence_threshold=args.conf, device="cuda:0")
    else:
        from ultralytics import YOLO
        det = YOLO(str(MODELS[args.model]))
        det.to("cuda")

    def infer(frame_bgr):
        if use_sahi:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            ctx = SahiTTAPatch() if not args.no_tta else _null()
            with ctx:
                res = get_sliced_prediction(
                    rgb, det, slice_height=SAHI_SLICE, slice_width=SAHI_SLICE,
                    overlap_height_ratio=SAHI_OVERLAP, overlap_width_ratio=SAHI_OVERLAP,
                    verbose=0, **SAHI_POSTPROCESS)
            return [(o.bbox.minx, o.bbox.miny, o.bbox.maxx, o.bbox.maxy,
                     o.score.value) for o in res.object_prediction_list]
        r = det.predict(frame_bgr, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        b = r.boxes
        if not len(b):
            return []
        xyxy = b.xyxy.cpu().numpy()
        cf = b.conf.cpu().numpy()
        return [(x1, y1, x2, y2, c) for (x1, y1, x2, y2), c in zip(xyxy, cf)]

    class _null:
        def __enter__(self):
            return self

        def __exit__(self, *e):
            return False

    # --- video io
    cap = cv2.VideoCapture(str(vpath))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f0 = int(args.start * fps)
    f1 = min(total, f0 + int(args.duration * fps)) if args.duration > 0 else total
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    out_fps = max(fps / args.stride, 5.0)
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                         out_fps, (W, H))
    if not vw.isOpened():
        raise SystemExit("VideoWriter failed to open -- check codec support")

    n_proc = (f1 - f0 + args.stride - 1) // args.stride
    label = f"{MODEL_LABELS[args.model]}" + \
            (f" + SAHI({SAHI_SLICE}/{SAHI_OVERLAP:.2f})" if use_sahi else
             f" @ {args.imgsz}") + \
            ("" if (args.no_tta or not use_sahi) else "+TTA")
    th = max(2, W // 960)          # box thickness scales with resolution
    fs = W / 1920.0                # font scale
    print(f"{vpath.name}: {W}x{H} @ {fps:.1f} fps, frames {f0}..{f1}, "
          f"stride {args.stride} -> {n_proc} frames to process")
    print(f"config: {label}, conf >= {args.conf}")
    print(f"output: {out_path}  ({out_fps:.1f} fps)")

    t_start = time.perf_counter()
    done = 0
    fi = f0
    while fi < f1:
        ok, frame = cap.read()
        if not ok:
            break
        # skip stride-1 frames cheaply
        for _ in range(args.stride - 1):
            cap.grab()
        dets = infer(frame)
        for x1, y1, x2, y2, c in dets:
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(frame, p1, p2, (0, 220, 40), th)
        hud = f"{label} | conf>={args.conf} | persons: {len(dets)}"
        (tw, tht), _ = cv2.getTextSize(hud, cv2.FONT_HERSHEY_SIMPLEX,
                                       1.4 * fs, max(1, th - 1))
        cv2.rectangle(frame, (0, 0), (tw + 24, tht + 28), (0, 0, 0), -1)
        cv2.putText(frame, hud, (12, tht + 14), cv2.FONT_HERSHEY_SIMPLEX,
                    1.4 * fs, (255, 255, 255), max(1, th - 1), cv2.LINE_AA)
        vw.write(frame)
        done += 1
        fi += args.stride
        if done % 20 == 0 or done == n_proc:
            el = time.perf_counter() - t_start
            eta = el / done * (n_proc - done)
            print(f"  [{done}/{n_proc}]  {el / done:.2f}s/frame  ETA {eta / 60:.1f} min")
    cap.release()
    vw.release()
    print(f"DONE -> {out_path}")


if __name__ == "__main__":
    main()
