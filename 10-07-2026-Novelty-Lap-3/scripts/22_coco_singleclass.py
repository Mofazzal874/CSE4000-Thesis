r"""
22_coco_singleclass.py -- normalize Roboflow COCO exports to a single 'person' class.

Roboflow ships two categories (id 0 placeholder + id 1 'person'), which breaks single-class
COCO AP scoring (model predicts class 0; GT anns may use id 1 -> zero matches). This rewrites
every annotation to category_id 0 with a single category [{id:0, name:'person'}], writing
`_annotations_sc.coco.json` next to each `_annotations.coco.json` (originals untouched).

Then eval/train point at the `_sc` json:
  python 18_s1_eval.py ... --gt-json <set>\_annotations_sc.coco.json --images-dir <set>

  python 22_coco_singleclass.py --tree "d:\...\Drone Shoot\extracted_v1\annotations"   # all sets
  python 22_coco_singleclass.py --in <one.json> --out <one_sc.json>
  python 22_coco_singleclass.py --selftest
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path


def normalize(coco: dict) -> dict:
    out = dict(coco)
    out["categories"] = [{"id": 0, "name": "person", "supercategory": "none"}]
    anns = []
    for a in coco.get("annotations", []):
        b = dict(a)
        b["category_id"] = 0
        anns.append(b)
    out["annotations"] = anns
    return out


def _do_file(src: Path, dst: Path) -> tuple[int, int, int]:
    coco = json.loads(src.read_text(encoding="utf-8"))
    n_cats_before = len(coco.get("categories", []))
    fixed = normalize(coco)
    dst.write_text(json.dumps(fixed), encoding="utf-8")
    return len(fixed["images"]), len(fixed["annotations"]), n_cats_before


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", help="dir to recurse; normalize every _annotations.coco.json")
    ap.add_argument("--in", dest="inp")
    ap.add_argument("--out", dest="out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return _selftest()

    if a.tree:
        root = Path(a.tree)
        srcs = sorted(root.rglob("_annotations.coco.json"))
        if not srcs:
            print(f"[sc] no _annotations.coco.json under {root}")
            return 1
        for src in srcs:
            dst = src.with_name("_annotations_sc.coco.json")
            imgs, anns, nc = _do_file(src, dst)
            rel = src.parent.relative_to(root)
            print(f"[sc] {rel}: {imgs} imgs, {anns} anns, {nc} cats -> 1  -> {dst.name}")
        print(f"[sc] normalized {len(srcs)} set(s). Point evals at the _sc json.")
        return 0

    if a.inp and a.out:
        imgs, anns, nc = _do_file(Path(a.inp), Path(a.out))
        print(f"[sc] {a.inp}: {imgs} imgs, {anns} anns, {nc} cats -> 1  -> {a.out}")
        return 0

    ap.error("need --tree DIR, or --in FILE --out FILE, or --selftest")


def _selftest() -> int:
    coco = {"images": [{"id": 1, "file_name": "a.jpg", "width": 10, "height": 10}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 4, 4], "area": 16, "iscrowd": 0},
                            {"id": 2, "image_id": 1, "category_id": 0, "bbox": [5, 5, 3, 3], "area": 9, "iscrowd": 0}],
            "categories": [{"id": 0, "name": "person"}, {"id": 1, "name": "person"}]}
    out = normalize(coco)
    assert len(out["categories"]) == 1 and out["categories"][0]["id"] == 0
    assert all(x["category_id"] == 0 for x in out["annotations"])
    assert len(out["annotations"]) == 2 and len(out["images"]) == 1
    # bbox/area/image_id preserved
    assert out["annotations"][0]["bbox"] == [0, 0, 4, 4] and out["annotations"][0]["area"] == 16
    print("SELFTEST OK: 2 cats -> 1 'person' (id 0), all anns remapped, boxes preserved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
