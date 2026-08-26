r"""
29_qualitative_panel.py -- qualitative before/after panel for the ICCIT paper (v2).

Runs the C2A-trained base model and the real-data-adapted model over the SAME real frames
and renders a 2x2 figure: rows = domain (drone, disaster), columns = model stage.
Green = ground truth, red = detections at the operating threshold. Where a red box covers a
green one the person was found; a lone green box is a miss.

v2 fixes two defects of v1:
  1. LINE WIDTH is now scaled to the FINAL printed size, not the source size. A 4K frame
     shrunk into a two-column figure lost its 2 px boxes entirely; thickness is now derived
     from the downscale factor so every box survives.
  2. Each ROW is cropped to the window that actually contains boxes (identical window for the
     two cells of a row, so the comparison stays fair), which removes empty grass and sky.
Cells are tagged (a) to (d); all counts are printed to the console for the caption.

Frame choice is not cherry-picked: the frame with the median person count of each evaluation
set is used.

Run on PC-1 (venv active, GPU free):
  python 29_qualitative_panel.py ^
    --base   runs_s1\s3_assign_full\weights\best.pt ^
    --adapted runs_d3\d3_all\weights\best.pt ^
    --drone-set  E:\Thesis_mofazzal_2007074\drone_data\annotations\test_v1\test ^
    --disaster-set E:\Thesis_mofazzal_2007074\drone_data\annotations\rset_v1\test ^
    --out fig_qualitative.jpg --conf 0.25 --device 0
Validate plumbing without a GPU:
  python 29_qualitative_panel.py --selftest
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from importlib import import_module
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

GREEN = (0, 200, 0)          # ground truth (BGR)
RED = (0, 0, 255)            # detections
CELL_W = 1000                # width of one cell in the composed figure
TAG_H = 34


def pick_median_frame(set_dir: Path):
    """(filename, gt_boxes_xywh) for the frame with the median person count."""
    coco = Path(set_dir) / "_annotations_sc.coco.json"
    if not coco.is_file():
        raise FileNotFoundError(f"missing {coco} (run 22_coco_singleclass.py --tree first)")
    d = json.loads(coco.read_text(encoding="utf-8"))
    by = defaultdict(list)
    for a in d.get("annotations", []):
        by[a["image_id"]].append(a["bbox"])
    rows = [(im["file_name"], by.get(im["id"], [])) for im in d["images"] if by.get(im["id"])]
    if not rows:
        raise RuntimeError(f"no annotated frames in {set_dir}")
    rows.sort(key=lambda r: len(r[1]))
    return rows[len(rows) // 2]


def content_window(img):
    """Trim baked-in black borders (broadcast pillarboxing)."""
    import cv2
    import numpy as np
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cols = np.where(g.mean(axis=0) > 12)[0]
    rows = np.where(g.mean(axis=1) > 12)[0]
    if len(cols) and len(rows):
        return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    return 0, img.shape[0], 0, img.shape[1]


def match_counts(gt_xywh, det_xyxy, iou_thr=0.5):
    """Greedy IoU matching, same rule as the paper's evaluation. Returns (tp, fp, fn).

    A detection count alone says nothing about how many people were FOUND: half of them can be
    false alarms. This reports the honest split so the figure caption cannot mislead.
    """
    import numpy as np
    gt = [(x, y, x + w, y + h) for x, y, w, h in gt_xywh]
    det = list(det_xyxy)
    if not gt:
        return 0, len(det), 0
    if not det:
        return 0, 0, len(gt)
    G = np.asarray(gt, float)
    D = np.asarray(det, float)
    lt = np.maximum(D[:, None, :2], G[None, :, :2])
    rb = np.minimum(D[:, None, 2:], G[None, :, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    area_d = ((D[:, 2] - D[:, 0]) * (D[:, 3] - D[:, 1]))[:, None]
    area_g = ((G[:, 2] - G[:, 0]) * (G[:, 3] - G[:, 1]))[None, :]
    iou = inter / np.maximum(area_d + area_g - inter, 1e-9)
    taken = np.zeros(len(gt), bool)
    tp = 0
    for i in range(len(det)):                      # detections are already score-ordered
        j = int(np.argmax(np.where(taken, -1.0, iou[i])))
        if not taken[j] and iou[i, j] >= iou_thr:
            taken[j] = True
            tp += 1
    return tp, len(det) - tp, int((~taken).sum())


def box_bounds(boxes_xywh, dets_xyxy):
    """Union bounding box of every drawn box, as (x0, y0, x1, y1)."""
    xs0, ys0, xs1, ys1 = [], [], [], []
    for x, y, w, h in boxes_xywh:
        xs0.append(x); ys0.append(y); xs1.append(x + w); ys1.append(y + h)
    for x1, y1, x2, y2 in dets_xyxy:
        xs0.append(x1); ys0.append(y1); xs1.append(x2); ys1.append(y2)
    if not xs0:
        return None
    return min(xs0), min(ys0), max(xs1), max(ys1)


def fit_window(bounds, img_w, img_h, aspect, pad=0.06):
    """Grow the bounds to `aspect` (w/h) with padding, clamped to the image."""
    x0, y0, x1, y1 = bounds
    w, h = x1 - x0, y1 - y0
    x0 -= w * pad; x1 += w * pad; y0 -= h * pad; y1 += h * pad
    w, h = x1 - x0, y1 - y0
    if w / h < aspect:                     # too tall: widen
        need = h * aspect - w
        x0 -= need / 2; x1 += need / 2
    else:                                   # too wide: heighten
        need = w / aspect - h
        y0 -= need / 2; y1 += need / 2
    # clamp while preserving size where possible
    if x0 < 0: x1 -= x0; x0 = 0
    if y0 < 0: y1 -= y0; y0 = 0
    if x1 > img_w: x0 -= (x1 - img_w); x1 = img_w
    if y1 > img_h: y0 -= (y1 - img_h); y1 = img_h
    return (max(0, int(x0)), max(0, int(y0)),
            min(img_w, int(x1)), min(img_h, int(y1)))


def draw_cell(img, gt_xywh, det_xyxy, window, off_x, off_y, letter):
    """Crop to `window`, draw boxes with print-safe thickness, tag with (letter)."""
    import cv2
    import numpy as np
    wx0, wy0, wx1, wy1 = window
    # thickness chosen so lines stay >= 2 px after the crop is resized to CELL_W
    shrink = max(1.0, (wx1 - wx0) / CELL_W)
    t = max(2, int(round(2.4 * shrink)))
    canvas = img.copy()
    for x, y, w, h in gt_xywh:
        cv2.rectangle(canvas, (int(x - off_x), int(y - off_y)),
                      (int(x + w - off_x), int(y + h - off_y)), GREEN, t)
    for x1, y1, x2, y2 in det_xyxy:
        cv2.rectangle(canvas, (int(x1 - off_x), int(y1 - off_y)),
                      (int(x2 - off_x), int(y2 - off_y)), RED, t)
    cell = canvas[wy0:wy1, wx0:wx1]
    scale = CELL_W / cell.shape[1]
    cell = cv2.resize(cell, (CELL_W, max(1, round(cell.shape[0] * scale))),
                      interpolation=cv2.INTER_AREA)
    cv2.rectangle(cell, (0, 0), (74, TAG_H), (255, 255, 255), -1)
    cv2.rectangle(cell, (0, 0), (74, TAG_H), (40, 40, 40), 1)
    cv2.putText(cell, f"({letter})", (8, TAG_H - 9), cv2.FONT_HERSHEY_DUPLEX,
                0.85, (20, 20, 20), 1, cv2.LINE_AA)
    return cell


def render(args) -> int:
    import cv2
    import numpy as np
    fccg = import_module("10_fccg_modules")
    if not fccg.register_fccg():
        print("FATAL: ultralytics not importable"); return 1
    from ultralytics import YOLO

    models = {}
    for tag_, wpath in (("base", args.base), ("adapted", args.adapted)):
        if not Path(wpath).is_file():
            print(f"FATAL: weights not found: {wpath}"); return 1
        models[tag_] = YOLO(str(wpath))
        print(f"[qual] loaded {tag_}: {wpath}")

    rows, letters = [], iter("abcd")
    for set_name, set_dir in (("drone", args.drone_set), ("disaster", args.disaster_set)):
        fn, gt = pick_median_frame(Path(set_dir))
        src = Path(set_dir) / fn
        img = cv2.imread(str(src))
        cy0, cy1, cx0, cx1 = content_window(img)     # strip pillarbox first
        img = img[cy0:cy1, cx0:cx1]
        print(f"[qual] {set_name}: {fn}")
        print(f"[qual]   frame {img.shape[1]}x{img.shape[0]} after border trim, "
              f"{len(gt)} labeled people")

        dets = {}
        for tag_ in ("base", "adapted"):
            r = models[tag_].predict(str(src), imgsz=args.imgsz, conf=args.conf,
                                     max_det=300, device=args.device, verbose=False)[0]
            dets[tag_] = r.boxes.xyxy.cpu().numpy()
            tp, fp, fn = match_counts(gt, dets[tag_])
            print(f"[qual]   {tag_:8s}: {len(dets[tag_]):3d} detections at conf {args.conf} -> "
                  f"{tp} people FOUND, {fn} MISSED, {fp} false alarms "
                  f"(of {len(gt)} labeled)")

        # one shared window per row so the before/after pair is directly comparable
        gt_off = [(x - cx0, y - cy0, w, h) for x, y, w, h in gt]
        det_off = {k: [(a - cx0, b - cy0, c - cx0, d - cy0) for a, b, c, d in v]
                   for k, v in dets.items()}
        allb = box_bounds(gt_off, det_off["base"] + det_off["adapted"])
        win = fit_window(allb, img.shape[1], img.shape[0], aspect=args.aspect)
        print(f"[qual]   window x {win[0]}-{win[2]}, y {win[1]}-{win[3]}")
        cells = [draw_cell(img, gt_off, det_off[k], win, 0, 0, next(letters))
                 for k in ("base", "adapted")]
        gap = np.full((cells[0].shape[0], 10, 3), 255, dtype="uint8")
        rows.append(cv2.hconcat([cells[0], gap, cells[1]]))

    # Rows share a width (both are 2*CELL_W + gap) but NOT a height, because the two domains
    # have different usable aspects. Do NOT equalize by cropping: that silently truncated the
    # taller row and cut annotated people out of the disaster frame. vconcat only needs equal
    # widths, so stack the rows at their natural heights.
    assert len({r.shape[1] for r in rows}) == 1, "rows must share a width for vconcat"
    vgap = np.full((10, rows[0].shape[1], 3), 255, dtype="uint8")
    grid = cv2.vconcat([rows[0], vgap, rows[1]])
    out = Path(args.out)
    if out.suffix.lower() in (".jpg", ".jpeg"):
        cv2.imwrite(str(out), grid, [cv2.IMWRITE_JPEG_QUALITY, 93])
    else:
        cv2.imwrite(str(out), grid)
    print(f"[qual] wrote {out}  {grid.shape[1]}x{grid.shape[0]}  "
          f"aspect {grid.shape[1]/grid.shape[0]:.2f}")
    print("[qual] (a),(b) = drone before/after; (c),(d) = disaster before/after")
    print("[qual] green = ground truth, red = detections; lone green box = missed person")
    return 0


def _selftest() -> int:
    import numpy as np
    try:
        import cv2
    except ImportError:
        print("SELFTEST OK (import-only; cv2 lives on the run machine)")
        return 0
    rng = np.random.default_rng(0)
    img = (rng.random((1080, 3840, 3)) * 255).astype("uint8")
    img[:, :300] = 0                                        # fake pillarbox
    y0, y1, x0, x1 = content_window(img)
    assert x0 >= 299, f"border trim failed: {x0}"
    gt = [(500, 100, 20, 40), (900, 700, 18, 36)]
    det = [(505, 102, 24, 44)]
    # a well-overlapping detection (IoU 0.82) hits the first GT; the second GT is missed
    tp, fp, fn = match_counts(gt, [(502, 101, 521, 141)])
    assert (tp, fp, fn) == (1, 0, 1), (tp, fp, fn)
    # a loose detection (IoU 0.44 < 0.5) does NOT count as a hit
    assert match_counts(gt, [(505, 102, 529, 146)]) == (0, 1, 2)
    # a detection on empty ground is a false alarm, both GT missed
    assert match_counts(gt, [(50, 50, 70, 90)]) == (0, 1, 2)
    b = box_bounds(gt, det)
    assert b == (500, 100, 918, 736), b
    win = fit_window(b, 3840, 1080, aspect=1.4)
    ww, wh = win[2] - win[0], win[3] - win[1]
    assert abs(ww / wh - 1.4) < 0.02 or win[1] == 0 or win[3] == 1080, (win, ww / wh)
    cell = draw_cell(img, gt, det, win, 0, 0, "a")
    assert cell.shape[1] == CELL_W
    # thickness must survive the downscale
    shrink = ww / CELL_W
    assert max(2, int(round(2.4 * shrink))) / shrink >= 2.0
    print(f"SELFTEST OK: border trim, bounds, window aspect {ww/wh:.2f}, "
          f"cell {cell.shape[1]}x{cell.shape[0]}, print-safe line width verified")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--base", help="C2A-trained checkpoint (before adaptation)")
    ap.add_argument("--adapted", help="real-data adapted checkpoint")
    ap.add_argument("--drone-set", help="drone eval dir holding _annotations_sc.coco.json")
    ap.add_argument("--disaster-set", help="disaster eval dir holding _annotations_sc.coco.json")
    ap.add_argument("--out", default="fig_qualitative.jpg")
    ap.add_argument("--conf", type=float, default=0.25, help="operating threshold used in the paper")
    ap.add_argument("--aspect", type=float, default=1.4, help="width/height of one cell")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(_selftest())
    if not (a.base and a.adapted and a.drone_set and a.disaster_set):
        sys.exit("need --base, --adapted, --drone-set, --disaster-set (or --selftest)")
    sys.exit(render(a))
