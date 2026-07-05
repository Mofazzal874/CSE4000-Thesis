"""
01_make_scene_split.py -- Build the scene-disjoint C2A split (c2a_scenesplit_v1).

Why not just split by filename prefix? The paper says 1,345 backgrounds but the
dataset has 2,043 filename prefixes -- prefixes UNDER-count shared backgrounds.
So we cluster backgrounds perceptually:

  cluster = connected component of the graph over all 10,215 images with edges
            (a) same filename prefix ("scene"), OR
            (b) perceptual-hash distance <= --hash-thr (near-identical pixels).

Splits are then assigned at the CLUSTER level (stratified by disaster category,
~60/20/20 by image count, deterministic seed) so no background can appear in two
splits, even across different prefixes.

Outputs (default: sibling of --root, named new_dataset3_scenesplit_v1):
  <out>/{train,val,test}/{images,labels}/...       (hardlinks; falls back to copy)
  <out>/{split}/{split}_annotations.json           (regenerated COCO GT per split)
  <out>/scene_assignment.csv                       (cluster,scene,image,split) -- canonical
  <out>/validation_report.json                     (all integrity checks)
  <out>/data.yaml                                  (ready for direct Ultralytics use)

Reproducibility: run with the same --seed/--hash-thr anywhere and you get the
same assignment. To FORCE the exact assignment produced on another machine,
pass --assignment scene_assignment.csv (then no hashing/clustering happens).

Usage (laptop):   python 01_make_scene_split.py
Usage (lab PC):   python 01_make_scene_split.py --root "D:/thesis_2007074/common/c2a/C2A_Dataset/new_dataset3"
       or:        python 01_make_scene_split.py --root ... --assignment scene_assignment.csv
"""
from __future__ import annotations
import argparse, csv, json, os, random, shutil, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

CANDIDATE_ROOTS = [
    r"d:/Academics/thesis folder/c2a/C2A_Dataset/new_dataset3",
    r"D:/thesis_2007074/common/c2a/C2A_Dataset/new_dataset3",
    r"D:/student_2k20/2007074/common/c2a/C2A_Dataset/new_dataset3",
    r"D:/2007074/common/c2a/C2A_Dataset/new_dataset3",
    r"E:/Thesis_mofazzal_2007074/common/c2a/C2A_Dataset/new_dataset3",
]
SPLITS = ("train", "val", "test")
TARGET = {"train": 0.60, "val": 0.20, "test": 0.20}

_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def find_root(cli: str | None) -> Path:
    for c in ([cli] if cli else []) + [os.environ.get("C2A_ROOT")] + CANDIDATE_ROOTS:
        if c and (Path(c) / "train" / "images").is_dir():
            return Path(c)
    sys.exit("[FATAL] C2A root not found. Pass --root or set C2A_ROOT.")


def scene_of(fn: str) -> str:
    return Path(fn).stem.rsplit("_", 1)[0]


def category_of(scene: str) -> str:
    return scene.split("_image")[0]


def phash64(img) -> np.uint64:
    """64-bit DCT perceptual hash (background-dominated for sprite images)."""
    import cv2
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(g)[:8, :8].flatten()
    med = np.median(d[1:])  # skip DC term
    bits = (d > med).astype(np.uint64)
    h = np.uint64(0)
    for b in bits:
        h = (h << np.uint64(1)) | b
    return h


class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def hamming_pairs(hashes: np.ndarray, thr: int, block: int = 512):
    """Yield (i, j) index pairs with hamming distance <= thr. Blockwise, O(n^2/8) bytes peak."""
    n = len(hashes)
    h8 = hashes.view(np.uint8).reshape(n, 8)  # little-endian bytes of each uint64
    for i0 in range(0, n, block):
        a = h8[i0:i0 + block]                      # (bi, 8)
        # compare against j >= i0 only (upper triangle)
        x = a[:, None, :] ^ h8[None, i0:, :]       # (bi, n-i0, 8)
        d = _POP8[x].sum(axis=2, dtype=np.uint16)  # (bi, n-i0)
        ii, jj = np.nonzero(d <= thr)
        for bi, rj in zip(ii.tolist(), jj.tolist()):
            i, j = i0 + bi, i0 + rj
            if i < j:
                yield i, j


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=20260705)
    ap.add_argument("--hash-thr", type=int, default=6, help="phash hamming threshold for same-background")
    ap.add_argument("--assignment", default=None, help="reuse an existing scene_assignment.csv (skips clustering)")
    ap.add_argument("--no-materialize", action="store_true", help="only compute assignment + report")
    args = ap.parse_args()

    root = find_root(args.root)
    out = Path(args.out) if args.out else root.parent / "new_dataset3_scenesplit_v1"
    print(f"[paths] root = {root}")
    print(f"[paths] out  = {out}")

    # ---------- inventory: every image, its original split, its annotations ----------
    images: list[str] = []           # file_name
    orig_split: dict[str, str] = {}  # file_name -> split
    coco: dict[str, dict] = {}
    anns_by_img: dict[str, list] = defaultdict(list)
    categories = None
    for sp in SPLITS:
        with open(root / sp / f"{sp}_annotations.json", encoding="utf-8") as f:
            d = json.load(f)
        coco[sp] = d
        categories = categories or d.get("categories")
        id2fn = {im["id"]: im["file_name"] for im in d["images"]}
        for im in d["images"]:
            images.append(im["file_name"])
            orig_split[im["file_name"]] = sp
        for a in d.get("annotations", []):
            anns_by_img[id2fn[a["image_id"]]].append(a)
    images = sorted(set(images))
    n = len(images)
    im_meta = {}
    for sp in SPLITS:
        for im in coco[sp]["images"]:
            im_meta[im["file_name"]] = {"width": im["width"], "height": im["height"]}
    print(f"[inv] {n} images, {sum(len(v) for v in anns_by_img.values())} annotations")

    # ---------- assignment ----------
    if args.assignment:
        assign: dict[str, str] = {}
        cluster_of: dict[str, str] = {}
        with open(args.assignment, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                assign[row["image"]] = row["split"]
                cluster_of[row["image"]] = row["cluster"]
        missing = [f for f in images if f not in assign]
        if missing:
            sys.exit(f"[FATAL] assignment file missing {len(missing)} images, e.g. {missing[:3]}")
        n_clusters = len(set(cluster_of.values()))
        cross_prefix_merges = None
        print(f"[assign] reused {args.assignment}: {n_clusters} clusters")
    else:
        # ---- perceptual hash every image (cached) ----
        import cv2
        cache_p = out.parent / f"{out.name}_phash_cache.json"
        cache = {}
        if cache_p.is_file():
            cache = json.load(open(cache_p, encoding="utf-8"))
        hashes = np.zeros(n, dtype=np.uint64)
        dirty = False
        for k, fn in enumerate(images):
            if fn in cache:
                hashes[k] = np.uint64(int(cache[fn]))
                continue
            p = root / orig_split[fn] / "images" / fn
            img = cv2.imread(str(p))
            if img is None:
                sys.exit(f"[FATAL] unreadable image {p}")
            hashes[k] = phash64(img)
            cache[fn] = str(int(hashes[k]))
            dirty = True
            if (k + 1) % 1000 == 0:
                print(f"[hash] {k+1}/{n}")
        if dirty:
            out.parent.mkdir(parents=True, exist_ok=True)
            json.dump(cache, open(cache_p, "w", encoding="utf-8"))
        print(f"[hash] done ({'cached' if not dirty else 'computed'})")

        # ---- union: same prefix ----
        dsu = DSU(n)
        idx = {fn: k for k, fn in enumerate(images)}
        by_scene = defaultdict(list)
        for fn in images:
            by_scene[scene_of(fn)].append(idx[fn])
        for members in by_scene.values():
            for m in members[1:]:
                dsu.union(members[0], m)

        # ---- union: near-identical hash (cross-prefix background sharing) ----
        cross_prefix_merges = []
        for i, j in hamming_pairs(hashes, args.hash_thr):
            if scene_of(images[i]) != scene_of(images[j]) and dsu.find(i) != dsu.find(j):
                cross_prefix_merges.append((images[i], images[j]))
            dsu.union(i, j)
        print(f"[cluster] cross-prefix background merges: {len(cross_prefix_merges)}")

        clusters = defaultdict(list)
        for k, fn in enumerate(images):
            clusters[dsu.find(k)].append(fn)
        clist = sorted(clusters.values(), key=lambda ms: sorted(ms)[0])
        n_clusters = len(clist)
        print(f"[cluster] {n_clusters} background clusters "
              f"(from {len(by_scene)} prefixes / paper claims 1345 backgrounds)")

        # ---- stratified cluster-level split, greedy fill by image count ----
        rng = random.Random(args.seed)
        by_cat = defaultdict(list)
        for ms in clist:
            by_cat[category_of(scene_of(ms[0]))].append(ms)
        assign, cluster_of = {}, {}
        for cat in sorted(by_cat):
            cl = by_cat[cat]
            rng.shuffle(cl)
            total = sum(len(ms) for ms in cl)
            filled = {sp: 0 for sp in SPLITS}
            for ci, ms in enumerate(cl):
                # assign to split with largest remaining relative deficit
                sp = max(SPLITS, key=lambda s: TARGET[s] - filled[s] / total)
                for fn in ms:
                    assign[fn] = sp
                    cluster_of[fn] = f"{cat}_c{ci:04d}"
                filled[sp] += len(ms)

    # ---------- validation ----------
    print("[validate] running integrity checks ...")
    report: dict = {"seed": args.seed, "hash_thr": args.hash_thr, "n_images": n,
                    "n_clusters": n_clusters}
    # V1: cluster disjointness (every cluster entirely inside one split)
    cl2sp = defaultdict(set)
    for fn in images:
        cl2sp[cluster_of[fn]].add(assign[fn])
    bad = {c: sorted(s) for c, s in cl2sp.items() if len(s) > 1}
    assert not bad, f"cluster split violation: {list(bad.items())[:3]}"
    # V2: prefix disjointness (implied by V1 but check independently)
    sc2sp = defaultdict(set)
    for fn in images:
        sc2sp[scene_of(fn)].add(assign[fn])
    bad2 = {s: v for s, v in sc2sp.items() if len(v) > 1}
    assert not bad2, f"scene split violation: {list(bad2.items())[:3]}"
    # V3: ratios + category balance + size distribution
    cnt = Counter(assign.values())
    report["images_per_split"] = dict(cnt)
    report["ratios"] = {sp: round(cnt[sp] / n, 4) for sp in SPLITS}
    cat_tab = defaultdict(Counter)
    for fn in images:
        cat_tab[category_of(scene_of(fn))][assign[fn]] += 1
    report["per_category_images"] = {c: dict(v) for c, v in cat_tab.items()}
    def size_bins(fns):
        import math
        b = Counter()
        for fn in fns:
            for a in anns_by_img[fn]:
                s = math.sqrt(max(a["bbox"][2], 0) * max(a["bbox"][3], 0))
                b["<8" if s < 8 else "8-16" if s < 16 else "16-32" if s < 32 else ">=32"] += 1
        t = sum(b.values()) or 1
        return {k: round(100 * v / t, 1) for k, v in sorted(b.items())}
    report["size_distribution_pct"] = {sp: size_bins([f for f in images if assign[f] == sp])
                                       for sp in SPLITS}
    report["instances_per_split"] = {sp: sum(len(anns_by_img[f]) for f in images if assign[f] == sp)
                                     for sp in SPLITS}
    if cross_prefix_merges is not None:
        report["cross_prefix_merges_count"] = len(cross_prefix_merges)
        report["cross_prefix_merge_examples"] = cross_prefix_merges[:12]

    # V4: every image's label file exists in the source tree
    missing_lbl = []
    for fn in images:
        lp = root / orig_split[fn] / "labels" / (Path(fn).stem + ".txt")
        if not lp.is_file():
            missing_lbl.append(str(lp))
    report["missing_label_files"] = missing_lbl[:10]
    assert not missing_lbl, f"{len(missing_lbl)} label files missing, e.g. {missing_lbl[:3]}"

    # ---------- materialize ----------
    out.mkdir(parents=True, exist_ok=True)
    if not args.no_materialize:
        def place(src: Path, dst: Path):
            if dst.exists():
                return
            try:
                os.link(src, dst)          # hardlink: instant, zero extra space
            except OSError:
                shutil.copy2(src, dst)     # cross-volume fallback
        for sp in SPLITS:
            (out / sp / "images").mkdir(parents=True, exist_ok=True)
            (out / sp / "labels").mkdir(parents=True, exist_ok=True)
        for k, fn in enumerate(images):
            sp, osp = assign[fn], orig_split[fn]
            stem = Path(fn).stem
            place(root / osp / "images" / fn, out / sp / "images" / fn)
            place(root / osp / "labels" / f"{stem}.txt", out / sp / "labels" / f"{stem}.txt")
            if (k + 1) % 2000 == 0:
                print(f"[link] {k+1}/{n}")
        # regenerated per-split COCO GT
        for sp in SPLITS:
            ims, anns = [], []
            iid = aid = 1
            for fn in sorted(f for f in images if assign[f] == sp):
                m = im_meta[fn]
                ims.append({"id": iid, "file_name": fn, "width": m["width"], "height": m["height"]})
                for a in anns_by_img[fn]:
                    na = dict(a)
                    na["id"], na["image_id"] = aid, iid
                    anns.append(na)
                    aid += 1
                iid += 1
            with open(out / sp / f"{sp}_annotations.json", "w", encoding="utf-8") as f:
                json.dump({"images": ims, "annotations": anns, "categories": categories}, f)
            print(f"[coco] {sp}: {len(ims)} images / {len(anns)} anns")
        # V5: materialized tree matches the assignment exactly
        for sp in SPLITS:
            have = {p.name for p in (out / sp / "images").iterdir()}
            want = {f for f in images if assign[f] == sp}
            assert have == want, f"{sp}: tree/assignment mismatch ({len(have ^ want)} diffs)"
        with open(out / "data.yaml", "w", encoding="utf-8") as f:
            f.write(f"path: {str(out).replace(os.sep, '/')}\n"
                    "train: train/images\nval: val/images\ntest: test/images\n"
                    "nc: 1\nnames:\n  0: person\n")

    # canonical assignment artifact (commit this to git)
    with open(out / "scene_assignment.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cluster", "scene", "image", "split"])
        for fn in images:
            w.writerow([cluster_of[fn], scene_of(fn), fn, assign[fn]])
    with open(out / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n===== SPLIT SUMMARY =====")
    print("images/split :", report["images_per_split"], "ratios:", report["ratios"])
    print("instances    :", report["instances_per_split"])
    print("size dist %  :", json.dumps(report["size_distribution_pct"]))
    if cross_prefix_merges is not None:
        print("cross-prefix background merges:", report["cross_prefix_merges_count"])
    print("ALL VALIDATION CHECKS PASSED")
    print(f"[done] -> {out}")

if __name__ == "__main__":
    main()
