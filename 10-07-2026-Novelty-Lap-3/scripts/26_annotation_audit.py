r"""
26_annotation_audit.py -- QC pass over the drone COCO annotations before we train on them.

Two jobs:
 (1) PROGRAMMATIC audit -- flag suspicious boxes at scale (no eyeballing needed to catch these):
     degenerate (w/h < 1px), out-of-bounds, extreme aspect ratio (>8 or <0.125), oversized
     (>40% of frame), tiny (<4px = below detectability), and duplicate boxes (IoU>0.85 = probable
     double-annotation). Plus the per-size histogram (is this really a tiny-aerial-person set?).
 (2) VISUAL render -- draws every box on a downscaled copy of each frame (green=ok, red=flagged)
     and a per-set MONTAGE grid, so a human (or Claude) can confirm the boxes actually land on
     people. Renders go to a gitignored _qc folder next to the annotations.

  python 26_annotation_audit.py --tree "d:\...\Drone Shoot\extracted_v1\annotations" --out "d:\...\_qc"
  python 26_annotation_audit.py --coco <set>\_annotations_sc.coco.json --images <set> --out <qc> --tag test_v1
  python 26_annotation_audit.py --selftest
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

IMG_EXT = (".jpg", ".jpeg", ".png")
SIZE_BINS = (("very_tiny", 0, 8), ("tiny", 8, 16), ("small", 16, 32),
             ("medium", 32, 96), ("large", 96, 1e9))


def _iou(a, b):
    ax1, ay1, aw, ah = a; bx1, by1, bw, bh = b
    ax2, ay2, bx2, by2 = ax1 + aw, ay1 + ah, bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    u = aw * ah + bw * bh - inter
    return inter / u if u > 0 else 0.0


def audit_image(W, H, boxes):
    """boxes = list of [x,y,w,h] abs. Returns (per_box_flags, dup_pairs)."""
    flags = []
    img_area = W * H
    for (x, y, w, h) in boxes:
        f = []
        if w < 1 or h < 1:
            f.append("degenerate")
        if x < -2 or y < -2 or x + w > W + 2 or y + h > H + 2:
            f.append("out_of_bounds")
        ar = (w / h) if h > 0 else 999
        if ar > 8 or ar < 0.125:
            f.append("extreme_ar")
        if w * h > 0.40 * img_area:
            f.append("oversized")
        if (w * h) ** 0.5 < 4:
            f.append("tiny<4px")
        flags.append(f)
    dup = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _iou(boxes[i], boxes[j]) > 0.85:
                dup.append((i, j))
    return flags, dup


def _render(img, boxes, flags, dup_idx, max_side=1280):
    import cv2
    H, W = img.shape[:2]
    scale = min(1.0, max_side / max(W, H))
    vis = cv2.resize(img, (int(W * scale), int(H * scale))) if scale < 1.0 else img.copy()
    for k, (x, y, w, h) in enumerate(boxes):
        bad = bool(flags[k]) or any(k in p for p in dup_idx)
        color = (0, 0, 255) if bad else (0, 220, 0)
        p1 = (int(x * scale), int(y * scale)); p2 = (int((x + w) * scale), int((y + h) * scale))
        cv2.rectangle(vis, p1, p2, color, 1)
    return vis


def _montage(cells, cols=4, cell=320):
    import cv2
    if not cells:
        return None
    rows = (len(cells) + cols - 1) // cols
    canvas = np.full((rows * cell, cols * cell, 3), 30, np.uint8)
    for i, c in enumerate(cells):
        h, w = c.shape[:2]
        s = min(cell / w, cell / h)
        r = cv2.resize(c, (int(w * s), int(h * s)))
        rr, cc = divmod(i, cols)
        y0, x0 = rr * cell, cc * cell
        canvas[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return canvas


def audit_set(coco_json, images_dir, out_dir, tag, n_montage=16, render_all=False):
    import cv2
    d = json.loads(Path(coco_json).read_text(encoding="utf-8"))
    by_img = defaultdict(list)
    for a in d.get("annotations", []):
        by_img[a["image_id"]].append(a["bbox"])
    out = Path(out_dir) / tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "flagged").mkdir(exist_ok=True)

    size_hist = {b[0]: 0 for b in SIZE_BINS}
    tallies = defaultdict(int)
    flagged_items = []
    n_img = n_box = n_zero = 0
    montage_cells = []
    rng = np.random.default_rng(0)
    imgs = d["images"]
    sample_ids = set(rng.choice(len(imgs), min(n_montage, len(imgs)), replace=False).tolist())

    for idx, im in enumerate(imgs):
        W, H = im["width"], im["height"]
        boxes = by_img.get(im["id"], [])
        n_img += 1
        n_box += len(boxes)
        if not boxes:
            n_zero += 1
        for (x, y, w, h) in boxes:
            s = (max(w, 0) * max(h, 0)) ** 0.5
            for name, lo, hi in SIZE_BINS:
                if lo <= s < hi:
                    size_hist[name] += 1
                    break
        flags, dup = audit_image(W, H, boxes)
        for k, f in enumerate(flags):
            for tag_ in f:
                tallies[tag_] += 1
                flagged_items.append({"image": im["file_name"], "box": k, "issue": tag_,
                                      "bbox": [round(v, 1) for v in boxes[k]]})
        if dup:
            tallies["duplicate"] += len(dup)
            flagged_items.append({"image": im["file_name"], "issue": "duplicate_pairs", "pairs": dup})

        need_render = render_all or idx in sample_ids or any(flags) or dup
        if need_render:
            ip = Path(images_dir) / im["file_name"]
            img = cv2.imread(str(ip)) if ip.is_file() else None
            if img is not None:
                vis = _render(img, boxes, flags, dup)
                if any(flags) or dup:
                    cv2.imwrite(str(out / "flagged" / f"{Path(im['file_name']).stem}.jpg"), vis)
                if idx in sample_ids:
                    montage_cells.append(vis)

    mont = _montage(montage_cells)
    if mont is not None:
        cv2.imwrite(str(out / f"montage_{tag}.jpg"), mont)
    report = {"set": tag, "n_images": n_img, "n_boxes": n_box, "zero_gt_images": n_zero,
              "size_hist": size_hist, "flags": dict(tallies),
              "median_box_px": None, "flagged_examples": flagged_items[:40]}
    # median box size
    allsz = []
    for im in imgs:
        for (x, y, w, h) in by_img.get(im["id"], []):
            allsz.append((max(w, 0) * max(h, 0)) ** 0.5)
    if allsz:
        report["median_box_px"] = round(float(np.median(allsz)), 1)
    (out / f"report_{tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _print_report(r):
    print(f"\n=== {r['set']} : {r['n_images']} imgs, {r['n_boxes']} boxes, "
          f"{r['zero_gt_images']} empty, median box {r['median_box_px']}px ===")
    h = r["size_hist"]; tot = max(sum(h.values()), 1)
    print("  size: " + "  ".join(f"{k}={h[k]}({100*h[k]//tot}%)" for k in
                                  ("very_tiny", "tiny", "small", "medium", "large")))
    if r["flags"]:
        print("  FLAGS: " + ", ".join(f"{k}={v}" for k, v in sorted(r["flags"].items())))
    else:
        print("  FLAGS: none")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree")
    ap.add_argument("--coco")
    ap.add_argument("--images")
    ap.add_argument("--out")
    ap.add_argument("--tag", default="set")
    ap.add_argument("--n-montage", type=int, default=16)
    ap.add_argument("--render-all", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if a.tree:
        root = Path(a.tree)
        out = Path(a.out or (root.parent / "_qc"))
        for coco in sorted(root.rglob("_annotations_sc.coco.json")):
            tag = "_".join(coco.parent.relative_to(root).parts)
            r = audit_set(coco, coco.parent, out, tag, a.n_montage, a.render_all)
            _print_report(r)
        print(f"\n[audit] renders + reports -> {out}")
        return 0
    if a.coco and a.images and a.out:
        r = audit_set(Path(a.coco), Path(a.images), Path(a.out), a.tag, a.n_montage, a.render_all)
        _print_report(r)
        return 0
    ap.error("need --tree DIR --out DIR, or --coco --images --out, or --selftest")


def _selftest() -> int:
    # geometry-only checks (no cv2 needed)
    flags, dup = audit_image(100, 100, [[10, 10, 20, 40], [10, 10, 21, 40], [0, 0, 90, 90], [50, 50, 0.5, 0.5]])
    assert "duplicate" not in flags  # dup is separate
    assert dup == [(0, 1)], dup                     # near-identical boxes flagged
    assert "oversized" in flags[2], flags[2]        # 90x90 on 100x100 = 81% area
    assert "degenerate" in flags[3] and "tiny<4px" in flags[3], flags[3]
    assert flags[0] == [], flags[0]                 # a clean 20x40 box -> no flags
    assert abs(_iou([0, 0, 10, 10], [0, 0, 10, 10]) - 1.0) < 1e-9
    assert _iou([0, 0, 10, 10], [100, 100, 10, 10]) == 0.0
    print("SELFTEST OK: flag logic (degenerate/oversized/tiny/duplicate) + IoU verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
