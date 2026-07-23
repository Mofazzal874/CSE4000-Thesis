"""
14_gt_overlap_stats.py — GT crowding / box-overlap statistics for a COCO annotation file.
Answers "are my boxes too overlapped?" with numbers instead of vibes, and tells us what NMS
setting the eval harness must use so crowds aren't unfairly suppressed.

Usage:
  python 14_gt_overlap_stats.py --coco "<path>\_annotations.coco.json"
  python 14_gt_overlap_stats.py --coco a.json b.json c.json     # compare splits
Reports: box count, boxes/image, box-size bins (VT<8 / tiny8-16 / small16-32 / med / large px,
on sqrt(area)), pairwise-IoU crowding (pairs over 0.3/0.5/0.7/0.8, % boxes with a >=0.5 neighbour),
and the single most-overlapped pair. Pure stdlib + json.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path


def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    u = aw * ah + bw * bh - inter
    return inter / u if u > 0 else 0.0


def analyze(coco_path: Path):
    d = json.loads(coco_path.read_text(encoding="utf-8"))
    by_img: dict[int, list] = {}
    for a in d["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a["bbox"])
    n_imgs = len(d["images"]); n_box = len(d["annotations"])
    sizes = [math.sqrt(max(w * h, 0)) for a in d["annotations"] for (_, _, w, h) in [a["bbox"]]]
    bins = {"VT<8": 0, "tiny8-16": 0, "small16-32": 0, "med32-96": 0, "large>=96": 0}
    for s in sizes:
        bins["VT<8" if s < 8 else "tiny8-16" if s < 16 else "small16-32" if s < 32
             else "med32-96" if s < 96 else "large>=96"] += 1
    thr = {0.3: 0, 0.5: 0, 0.7: 0, 0.8: 0}
    boxes_with_neighbor = 0; total_considered = 0; worst = (0.0, None)
    per_img_counts = []
    for iid, boxes in by_img.items():
        per_img_counts.append(len(boxes))
        has_nb = [False] * len(boxes)
        for i in range(len(boxes)):
            total_considered += 1
            for j in range(i + 1, len(boxes)):
                v = iou(boxes[i], boxes[j])
                for t in thr:
                    if v >= t: thr[t] += 1
                if v >= 0.5: has_nb[i] = has_nb[j] = True
                if v > worst[0]: worst = (v, (coco_path.parent.name, iid))
        boxes_with_neighbor += sum(has_nb)
    mx = max(per_img_counts) if per_img_counts else 0
    avg = (n_box / n_imgs) if n_imgs else 0
    print(f"\n=== {coco_path.parent.name}/{coco_path.name} ===")
    print(f"images={n_imgs}  boxes={n_box}  boxes/img avg={avg:.1f} max={mx}")
    print("size bins (sqrt area px): " + " | ".join(f"{k}:{v} ({100*v/max(n_box,1):.0f}%)" for k, v in bins.items()))
    print(f"pairwise IoU pairs  >0.3:{thr[0.3]}  >0.5:{thr[0.5]}  >0.7:{thr[0.7]}  >0.8:{thr[0.8]}")
    print(f"boxes with a >=0.5-IoU neighbour: {boxes_with_neighbor}/{n_box} "
          f"({100*boxes_with_neighbor/max(n_box,1):.1f}%)")
    print(f"worst single pair IoU: {worst[0]:.3f}  (image_id {worst[1][1] if worst[1] else '-'})")
    return {"boxes": n_box, "pct_crowded": 100 * boxes_with_neighbor / max(n_box, 1),
            "pairs_gt08": thr[0.8]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", nargs="+", required=True)
    a = ap.parse_args()
    for c in a.coco:
        analyze(Path(c))
