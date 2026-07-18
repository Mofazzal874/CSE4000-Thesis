"""
render_drone_frames.py -- burn predictions_drone.json boxes onto the raw drone
frames and SAVE them as annotated JPGs (+ one contact sheet per altitude).

The Gradio app draws these boxes live and never writes them to disk; this makes
standalone image files you can drop into slides or email to a supervisor.
Offline, no GPU, no model -- it only reads the cached JSON and the raw frames.

    python app/render_drone_frames.py                          # cbam_p2_sahi_tta @0.30
    python app/render_drone_frames.py --config enriched_sahi_tta --conf 0.25
    python app/render_drone_frames.py --all-configs

Output (into <demo>/results/drone_annotated/<config>/):
    <frame>.jpg          every frame with green detection boxes + a HUD label
    contact_<alt>.jpg    a 3x3 grid of sample frames per altitude (10m/30m/50m)
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

APP_DIR = Path(__file__).resolve().parent
DEMO_ROOT = APP_DIR.parent
RESULTS = DEMO_ROOT / "results"
FRAMES_DIR = RESULTS / "drone_frames"
JSON_PATH = RESULTS / "predictions_drone.json"
OUT_ROOT = RESULTS / "drone_annotated"

GREEN = (40, 220, 0)      # BGR, matches the app
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


def imread_u(path):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def imwrite_u(path, img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if ok:
        buf.tofile(str(path))


def annotate(img, boxes, conf_thr, label):
    h, w = img.shape[:2]
    th = max(2, round(min(w, h) / 350))
    n = 0
    for x1, y1, x2, y2, c in boxes:
        if c < conf_thr:
            continue
        n += 1
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), GREEN, th)
    # HUD bar
    hud = f"{label} | conf>={conf_thr:.2f} | persons: {n}"
    fs = w / 1920.0
    (tw, tht), _ = cv2.getTextSize(hud, cv2.FONT_HERSHEY_SIMPLEX, 1.2 * fs,
                                   max(1, th - 1))
    cv2.rectangle(img, (0, 0), (tw + 24, tht + 26), BLACK, -1)
    cv2.putText(img, hud, (12, tht + 12), cv2.FONT_HERSHEY_SIMPLEX, 1.2 * fs,
                WHITE, max(1, th - 1), cv2.LINE_AA)
    return img, n


def altitude_of(fname):
    return fname.split("_", 1)[0]          # "10m_t0000.00s.jpg" -> "10m"


def contact_sheet(entries, cols=3, rows=3, thumb_w=640):
    """entries: list of (annotated_bgr_img, caption). Returns a tiled sheet."""
    pick = entries[:cols * rows]
    thumbs = []
    for img, cap in pick:
        h, w = img.shape[:2]
        tw, th = thumb_w, round(thumb_w * h / w)
        t = cv2.resize(img, (tw, th))
        cv2.rectangle(t, (0, th - 22), (tw, th), BLACK, -1)
        cv2.putText(t, cap, (6, th - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE,
                    1, cv2.LINE_AA)
        thumbs.append(t)
    if not thumbs:
        return None
    cell_h = max(t.shape[0] for t in thumbs)
    cell_w = thumb_w
    sheet = np.full((cell_h * rows, cell_w * cols, 3), 250, np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet[r * cell_h:r * cell_h + t.shape[0],
              c * cell_w:c * cell_w + t.shape[1]] = t
    return sheet


def render_config(data, cfg, conf):
    label = data["meta"]["configs"].get(cfg, cfg)
    out_dir = OUT_ROOT / cfg
    out_dir.mkdir(parents=True, exist_ok=True)
    by_alt = {}
    total_dets = 0
    frames = sorted(data["images"])
    for fname in frames:
        rec = data["images"][fname]
        entry = rec.get(cfg)
        if entry is None:
            continue
        img = imread_u(FRAMES_DIR / fname)
        if img is None:
            print(f"  !! missing raw frame: {fname}")
            continue
        img, n = annotate(img, entry["boxes"], conf, label)
        total_dets += n
        imwrite_u(out_dir / fname, img)
        by_alt.setdefault(altitude_of(fname), []).append(
            (img, f"{fname.split('_', 1)[1]}  ({n} persons)"))
    # per-altitude contact sheets: 9 evenly spaced frames
    for alt, items in by_alt.items():
        idx = np.linspace(0, len(items) - 1, min(9, len(items))).astype(int)
        sheet = contact_sheet([items[i] for i in idx])
        if sheet is not None:
            imwrite_u(out_dir / f"contact_{alt}.jpg", sheet)
    print(f"  {cfg}: {len(frames)} frames, {total_dets} boxes @conf>={conf} "
          f"-> {out_dir.relative_to(DEMO_ROOT)}  (+{len(by_alt)} contact sheets)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cbam_p2_sahi_tta",
                    help="config key in predictions_drone.json")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--all-configs", action="store_true",
                    help="render every config in the JSON")
    args = ap.parse_args()

    if not JSON_PATH.is_file():
        raise SystemExit(f"missing {JSON_PATH}")
    data = json.loads(JSON_PATH.read_text())
    available = list(data["meta"]["configs"])
    print(f"configs in JSON: {available}")

    targets = available if args.all_configs else [args.config]
    for cfg in targets:
        if cfg not in available:
            print(f"  !! '{cfg}' not in JSON -- skipping "
                  f"(choose from {available})")
            continue
        render_config(data, cfg, args.conf)
    print("DONE.")


if __name__ == "__main__":
    main()
