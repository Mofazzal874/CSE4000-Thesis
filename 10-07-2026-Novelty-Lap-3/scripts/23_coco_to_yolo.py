r"""
23_coco_to_yolo.py -- export a (single-class) COCO set to Ultralytics YOLO structure.

Ultralytics trains from YOLO txt labels found by replacing '/images/' -> '/labels/' in each
image path. Roboflow drone exports keep images flat with a COCO json, so we materialize a clean
  <out>/images/*.jpg   <out>/labels/*.txt
copying the images and writing one txt per image. IMAGES WITH NO ANNOTATIONS GET AN EMPTY txt
= a background/negative frame -- this is exactly the void-FP hard-negative signal we want the
model to learn (a dark void that is correctly NOT a person).

  python 23_coco_to_yolo.py --coco <set>\_annotations_sc.coco.json --images <set> --out <drone_yolo>\train
  python 23_coco_to_yolo.py --selftest
"""
from __future__ import annotations
import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

IMG_EXT = (".jpg", ".jpeg", ".png")


def _iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    u = aw * ah + bw * bh - inter
    return inter / u if u > 0 else 0.0


def export_split(coco_json: Path, src_images: Path, out_dir: Path, copy_images: bool = True,
                 min_box_px: float = 3.0, dedup_iou: float = 0.9, img_max_side: int = 0):
    """COCO(_sc) -> YOLO images/+labels/. AUTO-CLEANS the annotation noise the audit found:
    drops sub-min_box_px boxes (accidental micro-clicks) and near-duplicate boxes (IoU>dedup_iou,
    same person labelled twice) so training never ingests them."""
    d = json.loads(Path(coco_json).read_text(encoding="utf-8"))
    by_img = defaultdict(list)
    for a in d.get("annotations", []):
        by_img[a["image_id"]].append(a)
    img_out = Path(out_dir) / "images"
    lab_out = Path(out_dir) / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lab_out.mkdir(parents=True, exist_ok=True)
    n_img = n_box = n_bg = n_miss = 0
    drop = {"degenerate": 0, "tiny": 0, "duplicate": 0}
    for im in d["images"]:
        W, H = float(im["width"]), float(im["height"])
        fn = im["file_name"]
        stem = Path(fn).stem
        kept = []
        for a in by_img.get(im["id"], []):
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                drop["degenerate"] += 1
                continue
            if (w * h) ** 0.5 < min_box_px:
                drop["tiny"] += 1
                continue
            if any(_iou_xywh((x, y, w, h), k) > dedup_iou for k in kept):
                drop["duplicate"] += 1
                continue
            kept.append((x, y, w, h))
        lines = []
        for (x, y, w, h) in kept:
            xc = min(max((x + w / 2) / W, 0.0), 1.0)
            yc = min(max((y + h / 2) / H, 0.0), 1.0)
            ww = min(max(w / W, 0.0), 1.0)
            hh = min(max(h / H, 0.0), 1.0)
            lines.append(f"0 {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")
            n_box += 1
        (lab_out / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        n_img += 1
        if not lines:
            n_bg += 1
        if copy_images:
            src = Path(src_images) / fn
            if src.is_file():
                dst = img_out / fn
                if not dst.exists():
                    if img_max_side and img_max_side > 0:      # downscale (labels are relative -> unaffected)
                        import cv2
                        arr = cv2.imread(str(src))
                        if arr is not None:
                            ih, iw = arr.shape[:2]
                            if max(iw, ih) > img_max_side:
                                s = img_max_side / max(iw, ih)
                                arr = cv2.resize(arr, (int(iw * s), int(ih * s)))
                            cv2.imwrite(str(dst), arr)
                        else:
                            shutil.copy2(src, dst)
                    else:
                        shutil.copy2(src, dst)
            else:
                n_miss += 1
    return {"images": n_img, "boxes": n_box, "backgrounds": n_bg, "missing_src": n_miss,
            "dropped": drop, "images_dir": str(img_out), "labels_dir": str(lab_out)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco")
    ap.add_argument("--images", help="dir holding the source images (Roboflow: same folder as the coco json)")
    ap.add_argument("--out", help="output dir; creates <out>/images + <out>/labels")
    ap.add_argument("--no-copy", action="store_true", help="write labels only (images already in place)")
    ap.add_argument("--img-max-side", type=int, default=0, help="downscale copied images to this max side (0=off; ~1280 for training to kill 4K RAM/decode blowup)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if not (a.coco and a.images and a.out):
        ap.error("--coco, --images, --out required (or --selftest)")
    stats = export_split(Path(a.coco), Path(a.images), Path(a.out), copy_images=not a.no_copy,
                         img_max_side=a.img_max_side)
    print(f"[c2y] {stats}")
    if stats["missing_src"]:
        print(f"[c2y] WARNING: {stats['missing_src']} images referenced in COCO not found in {a.images}")
    return 0


def _selftest() -> int:
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    (tmp / "src").mkdir()
    # two 100x50 images; one has a box, one is a pure background (void negative)
    for fn in ("a.jpg", "b.jpg"):
        (tmp / "src" / fn).write_bytes(b"\xff\xd8\xff\xd9")  # dummy jpg bytes
    coco = {"images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 50},
                       {"id": 2, "file_name": "b.jpg", "width": 100, "height": 50}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [10, 10, 20, 20]},
                            {"id": 3, "image_id": 1, "category_id": 0, "bbox": [10, 10, 20, 20]},  # duplicate
                            {"id": 4, "image_id": 1, "category_id": 0, "bbox": [50, 25, 1, 1]}],    # micro-click
            "categories": [{"id": 0, "name": "person"}]}
    cj = tmp / "coco.json"
    cj.write_text(json.dumps(coco))
    st = export_split(cj, tmp / "src", tmp / "out")
    assert st["images"] == 2 and st["boxes"] == 1 and st["backgrounds"] == 1, st
    assert st["dropped"]["duplicate"] == 1 and st["dropped"]["tiny"] == 1, st["dropped"]
    a_txt = (tmp / "out" / "labels" / "a.txt").read_text().strip().split()
    assert a_txt[0] == "0" and abs(float(a_txt[1]) - 0.20) < 1e-4 and abs(float(a_txt[3]) - 0.20) < 1e-4, a_txt
    assert (tmp / "out" / "labels" / "b.txt").read_text() == "", "background must be an EMPTY txt"
    assert (tmp / "out" / "images" / "a.jpg").is_file(), "image not copied"
    shutil.rmtree(tmp, ignore_errors=True)
    print("SELFTEST OK: box normalized correctly + empty-txt background (void-negative) preserved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
