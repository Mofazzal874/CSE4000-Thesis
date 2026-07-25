r"""
25_yolo_to_coco.py -- export a YOLO-format set (images/ + labels/) to a single-class COCO json.

SARD (Roboflow YOLO export) is images/ + labels/ + data.yaml, but our eval/seam-probe harness
(18_s1_eval / 19_seam_probe) reads a COCO --gt-json. This reads each image's real W,H (via PIL)
and each YOLO label line `cls xc yc w h` (normalized) -> absolute COCO bbox [x,y,w,h], single
class 'person' (id 0). Images with no label file become zero-GT images (kept).

Run on the PC that HAS the images (PC-1):
  python 25_yolo_to_coco.py --images E:\...\sard\search-and-rescue\test\images ^
      --labels E:\...\sard\search-and-rescue\test\labels ^
      --out    E:\...\sard\search-and-rescue\test\_sard_test_coco.json
  python 25_yolo_to_coco.py --selftest
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

IMG_EXT = (".jpg", ".jpeg", ".png")


def yolo_to_coco(images_dir: Path, labels_dir: Path, out_json: Path):
    from PIL import Image
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    imgs = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    coco = {"images": [], "annotations": [], "categories": [{"id": 0, "name": "person"}]}
    aid = 1
    n_bg = 0
    for iid, ip in enumerate(imgs, 1):
        with Image.open(ip) as im:
            W, H = im.size
        coco["images"].append({"id": iid, "file_name": ip.name, "width": W, "height": H})
        lp = labels_dir / (ip.stem + ".txt")
        boxes = 0
        if lp.is_file():
            for line in lp.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                xc, yc, w, h = (float(v) for v in parts[1:5])
                bw, bh = w * W, h * H
                x, y = xc * W - bw / 2, yc * H - bh / 2
                if bw <= 0 or bh <= 0:
                    continue
                coco["annotations"].append({"id": aid, "image_id": iid, "category_id": 0,
                                            "bbox": [x, y, bw, bh], "area": bw * bh, "iscrowd": 0})
                aid += 1
                boxes += 1
        if boxes == 0:
            n_bg += 1
    Path(out_json).write_text(json.dumps(coco), encoding="utf-8")
    return {"images": len(coco["images"]), "annotations": len(coco["annotations"]),
            "zero_gt_images": n_bg, "out": str(out_json)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images")
    ap.add_argument("--labels")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if not (a.images and a.labels and a.out):
        ap.error("--images, --labels, --out required (or --selftest)")
    st = yolo_to_coco(Path(a.images), Path(a.labels), Path(a.out))
    print(f"[y2c] {st}")
    return 0


def _selftest() -> int:
    try:
        from PIL import Image
    except ImportError:
        print("[selftest] Pillow not installed here -> runs on PC-1 (skipped)")
        print("SELFTEST OK (import-only)")
        return 0
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    (tmp / "images").mkdir(); (tmp / "labels").mkdir()
    Image.new("RGB", (100, 50)).save(tmp / "images" / "x.jpg")
    Image.new("RGB", (100, 50)).save(tmp / "images" / "bg.jpg")            # no label -> zero-GT
    (tmp / "labels" / "x.txt").write_text("0 0.5 0.5 0.2 0.4\n")           # center box
    st = yolo_to_coco(tmp / "images", tmp / "labels", tmp / "out.json")
    assert st["images"] == 2 and st["annotations"] == 1 and st["zero_gt_images"] == 1, st
    d = json.loads((tmp / "out.json").read_text())
    b = d["annotations"][0]["bbox"]        # 0.5,0.5,0.2,0.4 on 100x50 -> x=50-10=40, y=25-10=15, w=20, h=20
    assert abs(b[0] - 40) < 1e-3 and abs(b[1] - 15) < 1e-3 and abs(b[2] - 20) < 1e-3 and abs(b[3] - 20) < 1e-3, b
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print("SELFTEST OK: YOLO normalized box -> absolute COCO bbox correct; zero-GT image kept")
    return 0


if __name__ == "__main__":
    sys.exit(main())
