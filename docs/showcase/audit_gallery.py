"""Count the cached boxes actually shown in the selected archived overlays."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "pendrive/THESIS_2007074_MSA-YOLO"
GT = ROOT / "c2a/C2A_Dataset/new_dataset3/test/test_annotations.json"
CASES = {"collapsed_building_image0509_0.png": "improved",
         "collapsed_building_image0124_3.png": "regressed",
         "flood_image0087_1.png": "missed-people"}


def iou(a, b):
    x = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    y = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = x * y
    return inter / (a[2] * a[3] + b[2] * b[3] - inter) if inter else 0


def main():
    gt = json.loads(GT.read_text(encoding="utf-8"))
    ids = {im["id"]: im["file_name"] for im in gt["images"] if im["file_name"] in CASES}
    boxes = {name: [] for name in CASES}
    for ann in gt["annotations"]:
        if ann["image_id"] in ids:
            boxes[ids[ann["image_id"]]].append(ann["bbox"])
    rows = []
    for model, conf in (("baseline_yolo11m", .19), ("cbam_p2_RECOMMENDED", .16)):
        path = ARCHIVE / "05_NUMERICAL_RESULTS/01_ablation_official_split" / model / "test_coco_dets.json"
        preds = {name: [] for name in CASES}
        for d in json.loads(path.read_text(encoding="utf-8")):
            if d["image_id"] in ids and d["score"] >= conf:
                preds[ids[d["image_id"]]].append(d)
        for name, ground_truth in boxes.items():
            if name == "flood_image0087_1.png" and model == "baseline_yolo11m":
                continue
            detections = sorted(preds[name], key=lambda x: -x["score"])
            used = set()
            for pred in detections:
                candidates = [(iou(pred["bbox"], box), i) for i, box in enumerate(ground_truth) if i not in used]
                if candidates:
                    best, index = max(candidates, key=lambda x: x[0])
                    if best >= .5:
                        used.add(index)
            tp = len(used); fp = len(detections) - tp; fn = len(ground_truth) - tp
            precision = tp / (tp + fp) if tp + fp else 0
            recall = tp / len(ground_truth)
            f2 = 5 * tp / (5 * tp + 4 * fn + fp) if tp + fn + fp else 0
            rows.append(dict(case=CASES[name], image=name, model=model, confidence=conf,
                             gt=len(ground_truth), TP=tp, FP=fp, FN=fn,
                             precision=round(precision, 6), recall=round(recall, 6), F2=round(f2, 6)))
    out = ROOT / "public-showcase/results/gallery_overlay_counts.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
