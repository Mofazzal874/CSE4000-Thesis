"""
00_leakage_evidence.py -- Quantify + visually document C2A scene leakage.

Produces the evidence pack for the paper's motivation section:
  evidence/leakage_stats.json      -- all headline numbers
  evidence/sibling_pairs/*.png     -- side-by-side train-vs-test same-background montages
  evidence/sard_dup_audit.json     -- SARD test duplicate-base-name audit

Runs anywhere (needs: numpy, opencv-python). Read-only w.r.t. the dataset.

Usage:
  python 00_leakage_evidence.py --root "<...>/C2A_Dataset/new_dataset3" --out "<...>/evidence"
  (defaults below auto-probe the known roots on this laptop / the lab PCs)
"""
from __future__ import annotations
import argparse, json, os, sys, random
from collections import Counter, defaultdict
from pathlib import Path

CANDIDATE_ROOTS = [
    r"d:/Academics/thesis folder/c2a/C2A_Dataset/new_dataset3",
    r"D:/thesis_2007074/common/c2a/C2A_Dataset/new_dataset3",
    r"D:/student_2k20/2007074/common/c2a/C2A_Dataset/new_dataset3",
    r"D:/2007074/common/c2a/C2A_Dataset/new_dataset3",
    r"E:/Thesis_mofazzal_2007074/common/c2a/C2A_Dataset/new_dataset3",
]
SARD_GT_CANDIDATES = [
    r"d:/Academics/thesis folder/Last Month/cross_dataset_SARD/sard_test_coco_gt.json",
    r"D:/thesis_2007074/cross_dataset_SARD/sard_test_coco_gt.json",
]

def find_root(cli: str | None) -> Path:
    cands = [cli] if cli else []
    cands += [os.environ.get("C2A_ROOT")]
    cands += CANDIDATE_ROOTS
    for c in cands:
        if c and (Path(c) / "train" / "images").is_dir():
            return Path(c)
    sys.exit("[FATAL] C2A root not found. Pass --root or set C2A_ROOT.")

def scene_of(fn: str) -> str:
    """collapsed_building_image0031_2.png -> collapsed_building_image0031"""
    stem = Path(fn).stem
    return stem.rsplit("_", 1)[0]

def variant_of(fn: str) -> str:
    return Path(fn).stem.rsplit("_", 1)[1]

def category_of(scene: str) -> str:
    return scene.split("_image")[0]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=None, help="evidence output dir")
    ap.add_argument("--pairs", type=int, default=6, help="montage pairs to export")
    ap.add_argument("--seed", type=int, default=20260705)
    args = ap.parse_args()

    root = find_root(args.root)
    out = Path(args.out) if args.out else Path(__file__).resolve().parent.parent / "evidence"
    (out / "sibling_pairs").mkdir(parents=True, exist_ok=True)
    print(f"[paths] C2A root = {root}")
    print(f"[paths] evidence out = {out}")

    # ---- load the three official split JSONs --------------------------------
    files: dict[str, set[str]] = {}
    for sp in ("train", "val", "test"):
        jp = root / sp / f"{sp}_annotations.json"
        if jp.is_file():
            with open(jp, encoding="utf-8") as f:
                d = json.load(f)
            files[sp] = {im["file_name"] for im in d["images"]}
        else:  # fall back to listing images dir
            files[sp] = {p.name for p in (root / sp / "images").iterdir()}
        print(f"[load] {sp}: {len(files[sp])} images")

    scenes = {sp: {scene_of(f) for f in files[sp]} for sp in files}
    all_scenes = set().union(*scenes.values())
    all_files = set().union(*files.values())

    # variants-per-scene histogram
    vc = Counter(scene_of(f) for f in all_files)
    var_hist = Counter(vc.values())

    # scene -> split memberships
    test_scene_in_train = scenes["test"] & scenes["train"]
    test_scene_in_val_only = (scenes["test"] & scenes["val"]) - scenes["train"]
    test_scene_unseen = scenes["test"] - scenes["train"] - scenes["val"]
    test_img_scene_in_train = [f for f in files["test"] if scene_of(f) in scenes["train"]]
    test_img_unseen = [f for f in files["test"] if scene_of(f) in test_scene_unseen]

    stats = {
        "root": str(root),
        "n_images": {sp: len(files[sp]) for sp in files},
        "n_scenes": {sp: len(scenes[sp]) for sp in scenes},
        "n_scenes_total_dataset": len(all_scenes),
        "paper_claim_backgrounds": 1345,
        "variants_per_scene_histogram": dict(sorted(var_hist.items())),
        "file_level_overlap": {
            "train^test": len(files["train"] & files["test"]),
            "train^val": len(files["train"] & files["val"]),
            "val^test": len(files["val"] & files["test"]),
        },
        "scene_level": {
            "test_scenes_also_in_train": len(test_scene_in_train),
            "test_scenes_also_in_train_pct": round(100 * len(test_scene_in_train) / max(len(scenes["test"]), 1), 2),
            "test_scenes_in_val_only": len(test_scene_in_val_only),
            "test_scenes_fully_unseen": len(test_scene_unseen),
            "test_images_sharing_scene_with_train": len(test_img_scene_in_train),
            "test_images_sharing_scene_with_train_pct": round(100 * len(test_img_scene_in_train) / max(len(files["test"]), 1), 2),
            "test_images_fully_unseen_scene": len(test_img_unseen),
        },
        "scenes_per_category": dict(Counter(category_of(s) for s in all_scenes)),
    }

    # ---- export visual sibling pairs (train vs test, same scene) ------------
    try:
        import cv2
        import numpy as np
        rng = random.Random(args.seed)
        candidates = sorted(test_scene_in_train)
        rng.shuffle(candidates)
        made = 0
        pair_index = []
        for sc in candidates:
            if made >= args.pairs:
                break
            t_img = next((f for f in sorted(files["test"]) if scene_of(f) == sc), None)
            r_img = next((f for f in sorted(files["train"]) if scene_of(f) == sc), None)
            if not t_img or not r_img:
                continue
            a = cv2.imread(str(root / "test" / "images" / t_img))
            b = cv2.imread(str(root / "train" / "images" / r_img))
            if a is None or b is None:
                continue
            h = max(a.shape[0], b.shape[0])
            def pad(x):
                return cv2.copyMakeBorder(x, 0, h - x.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
            m = np.hstack([pad(a), np.full((h, 8, 3), 255, np.uint8), pad(b)])
            cv2.putText(m, f"TEST {t_img}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            cv2.putText(m, f"TRAIN {r_img}", (a.shape[1] + 14, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            op = out / "sibling_pairs" / f"pair_{made:02d}_{sc}.png"
            cv2.imwrite(str(op), m)
            pair_index.append({"scene": sc, "test": t_img, "train": r_img, "file": op.name})
            made += 1
        stats["sibling_pairs_exported"] = pair_index
        print(f"[pairs] exported {made} montages -> {out/'sibling_pairs'}")
    except Exception as e:  # cv2 missing on some box: stats still valid
        stats["sibling_pairs_exported"] = f"SKIPPED ({e})"
        print(f"[pairs] SKIPPED: {e}")

    # ---- SARD duplicate-base audit ------------------------------------------
    sard = next((p for p in ([] if not SARD_GT_CANDIDATES else SARD_GT_CANDIDATES) if Path(p).is_file()), None)
    if sard:
        with open(sard, encoding="utf-8") as f:
            sd = json.load(f)
        def sard_base(fn: str) -> str:
            # gss1006_jpg.rf.<hash>.jpg -> gss1006
            return fn.split("_jpg.rf.")[0].split(".rf.")[0]
        bases = Counter(sard_base(im["file_name"]) for im in sd["images"])
        dups = {b: c for b, c in bases.items() if c > 1}
        stats["sard_test_audit"] = {
            "gt_json": sard,
            "n_images": len(sd["images"]),
            "n_unique_bases": len(bases),
            "bases_with_multiple_copies": len(dups),
            "images_in_duplicated_bases": sum(dups.values()),
            "worst_offenders": dict(sorted(dups.items(), key=lambda kv: -kv[1])[:10]),
        }
        print(f"[sard] {len(sd['images'])} imgs, {len(bases)} unique bases, "
              f"{len(dups)} bases duplicated ({sum(dups.values())} images affected)")

    with open(out / "leakage_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[done] evidence -> {out/'leakage_stats.json'}")

    s = stats["scene_level"]
    print("\n===== HEADLINE =====")
    print(f"test scenes also in train : {s['test_scenes_also_in_train']} / {stats['n_scenes']['test']}"
          f" ({s['test_scenes_also_in_train_pct']}%)")
    print(f"test images sharing background with train: {s['test_images_sharing_scene_with_train']}"
          f" / {stats['n_images']['test']} ({s['test_images_sharing_scene_with_train_pct']}%)")
    print(f"test images from fully-unseen scenes     : {s['test_images_fully_unseen_scene']}")
    print(f"total scenes in dataset: {stats['n_scenes_total_dataset']} (paper says 1345 backgrounds)")

if __name__ == "__main__":
    main()
