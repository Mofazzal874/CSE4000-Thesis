"""
15_coco_validate.py — deep geometry validation of a COCO annotation file.
Goes beyond counts: checks every bbox's actual coordinates are valid and usable.

Checks per annotation:
  - bbox has exactly 4 numeric values [x,y,w,h], none NaN/inf
  - w > 0 and h > 0            (no zero/negative-area boxes)
  - x >= 0, y >= 0             (no negative origin; tol 1px)
  - x+w <= W+tol, y+h <= H+tol (box stays inside the image)
  - category_id exists in categories
  - image_id references a real image entry
Plus: images with no file on disk, images with 0 boxes, and a box-size summary.

Usage:
  python 15_coco_validate.py --coco "<...>\_annotations.coco.json" [more.json ...]
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path

TOL = 1.5  # px tolerance for bounds (rounding during export)


def validate(coco_path: Path) -> dict:
    d = json.loads(coco_path.read_text(encoding="utf-8"))
    imgs = {im["id"]: im for im in d["images"]}
    cat_ids = {c["id"] for c in d["categories"]}
    img_dir = coco_path.parent
    problems = {"bad_bbox_len": 0, "nan_inf": 0, "nonpos_wh": 0, "neg_xy": 0,
                "out_of_bounds": 0, "bad_cat": 0, "orphan_img": 0}
    examples = []
    ws, hs = [], []
    boxed_ids = set()
    for a in d["annotations"]:
        bb = a.get("bbox")
        iid = a.get("image_id")
        boxed_ids.add(iid)
        if not (isinstance(bb, list) and len(bb) == 4):
            problems["bad_bbox_len"] += 1; continue
        x, y, w, h = bb
        if any((not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v)) for v in bb):
            problems["nan_inf"] += 1; continue
        if w <= 0 or h <= 0:
            problems["nonpos_wh"] += 1
            if len(examples) < 6: examples.append(f"nonpos wh img{iid} bbox={bb}")
        if x < -TOL or y < -TOL:
            problems["neg_xy"] += 1
            if len(examples) < 6: examples.append(f"neg xy img{iid} bbox={bb}")
        im = imgs.get(iid)
        if im is None:
            problems["orphan_img"] += 1
        else:
            W, H = im["width"], im["height"]
            if x + w > W + TOL or y + h > H + TOL:
                problems["out_of_bounds"] += 1
                if len(examples) < 6: examples.append(f"OOB img{iid} bbox={bb} imgWH=({W},{H})")
        if a.get("category_id") not in cat_ids:
            problems["bad_cat"] += 1
        ws.append(w); hs.append(h)
    # disk + empty-image checks
    missing_files = sum(1 for im in d["images"] if not (img_dir / im["file_name"]).exists())
    empty_imgs = [im["id"] for im in d["images"] if im["id"] not in boxed_ids]
    total_problems = sum(problems.values())
    ok = total_problems == 0 and missing_files == 0

    print(f"\n=== {coco_path.parent.parent.name}/{coco_path.parent.name}/{coco_path.name} ===")
    print(f"images={len(d['images'])} annotations={len(d['annotations'])} "
          f"files_missing_on_disk={missing_files} images_with_0_boxes={len(empty_imgs)}")
    if ws:
        ws_s, hs_s = sorted(ws), sorted(hs)
        print(f"box w: min={ws_s[0]:.1f} med={ws_s[len(ws_s)//2]:.1f} max={ws_s[-1]:.1f} | "
              f"box h: min={hs_s[0]:.1f} med={hs_s[len(hs_s)//2]:.1f} max={hs_s[-1]:.1f}")
    print("geometry problems: " + (", ".join(f"{k}={v}" for k, v in problems.items() if v) or "NONE"))
    if examples:
        print("  examples: " + " | ".join(examples))
    print(("  >>> PASS — all bounding boxes valid" if ok
           else f"  >>> ISSUES FOUND ({total_problems} geom + {missing_files} missing files)"))
    return {"ok": ok, "problems": total_problems, "missing": missing_files}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", nargs="+", required=True)
    a = ap.parse_args()
    results = [validate(Path(c)) for c in a.coco]
    n_ok = sum(1 for r in results if r["ok"])
    print(f"\n===== SUMMARY: {n_ok}/{len(results)} files fully valid =====")
