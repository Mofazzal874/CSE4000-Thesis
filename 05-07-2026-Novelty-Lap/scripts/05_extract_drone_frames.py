"""
05_extract_drone_frames.py -- Deterministic frame extraction from the DJI shoot
(10m / 30m / 50m 4K videos) into two temporally-DISJOINT pools:

  test_frames/<alt>/       -> the frames YOU hand-label (Roboflow/labelImg).
                              These become the frozen REAL test set. Never
                              trained on, never pseudo-labeled.
  selftrain_frames/<alt>/  -> unlabeled pool for SAHI-guided self-training.

Separation scheme (the videos are short, ~2000 frames @60fps): each video is cut
into --segments contiguous blocks; EVEN blocks feed the test pool, ODD blocks
feed the self-train pool, with a --margin skipped at every block edge. Block-
level separation guarantees no near-identical neighbouring frames across pools
regardless of counts. (Both pools still come from the same flight -- the paper
frames this honestly as same-site deployment adaptation.)

Everything is deterministic (even spacing, no RNG). manifest.json is the
canonical record -- commit it to git.

Usage:
  python 05_extract_drone_frames.py                       # full extraction
  python 05_extract_drone_frames.py --smoke               # 2+2 frames/video
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

DEFAULT_VIDEOS = r"d:/Academics/thesis folder/Drone Shoot"


def spread_in(lo: int, hi: int, k: int) -> list[int]:
    """k deterministic evenly spaced ints in [lo, hi]."""
    if k <= 0 or hi < lo:
        return []
    if k == 1:
        return [(lo + hi) // 2]
    step = (hi - lo) / (k - 1)
    return sorted({int(round(lo + i * step)) for i in range(k)})


def pool_indices(n_frames: int, segments: int, margin: int, n_test: int, n_st: int):
    """Even blocks -> test, odd blocks -> selftrain, margins skipped."""
    seg = n_frames // segments
    test_blocks = [(i * seg + margin, (i + 1) * seg - margin)
                   for i in range(segments) if i % 2 == 0]
    st_blocks = [(i * seg + margin, (i + 1) * seg - margin)
                 for i in range(segments) if i % 2 == 1]
    def fill(blocks, k):
        out, per = [], max(1, round(k / max(len(blocks), 1)))
        for lo, hi in blocks:
            out += spread_in(lo, hi, per)
        return sorted(set(out))[:k] if len(out) > k else sorted(set(out))
    return fill(test_blocks, n_test), fill(st_blocks, n_st), seg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", default=DEFAULT_VIDEOS)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-test", type=int, default=20, help="frames to hand-label per video")
    ap.add_argument("--n-selftrain", type=int, default=80, help="pseudo-label pool per video")
    ap.add_argument("--segments", type=int, default=10, help="alternating temporal blocks")
    ap.add_argument("--margin", type=int, default=15, help="frames skipped at block edges")
    ap.add_argument("--smoke", action="store_true", help="2+2 frames per video (pipeline check)")
    args = ap.parse_args()

    import cv2
    vdir = Path(args.videos_dir)
    vids = sorted({p.resolve() for ext in ("*.MP4", "*.mp4", "*.mov", "*.MOV")
                   for p in vdir.glob(ext)})
    if not vids:
        sys.exit(f"[FATAL] no videos in {vdir}")
    out = Path(args.out) if args.out else (vdir / "extracted_v1")
    n_test, n_st = (2, 2) if args.smoke else (args.n_test, args.n_selftrain)

    manifest = {"videos": {}, "n_test": n_test, "n_selftrain": n_st,
                "segments": args.segments, "margin": args.margin}
    for vp in vids:
        alt = vp.stem.lower()
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            sys.exit(f"[FATAL] cannot open {vp}")
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        test_idx, st_idx, seg = pool_indices(n_frames, args.segments, args.margin, n_test, n_st)
        print(f"[{alt}] {n_frames} frames @{fps:.1f}fps, block={seg} -> "
              f"test {len(test_idx)}, selftrain {len(st_idx)}")

        for pool, idxs in (("test_frames", test_idx), ("selftrain_frames", st_idx)):
            d = out / pool / alt
            d.mkdir(parents=True, exist_ok=True)
            for fi in idxs:
                fp = d / f"{alt}_f{fi:06d}.jpg"
                if fp.exists():
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ok, frame = cap.read()
                if not ok:
                    print(f"  [warn] frame {fi} unreadable, skipped")
                    continue
                cv2.imwrite(str(fp), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cap.release()
        manifest["videos"][alt] = {"file": vp.name, "n_frames": n_frames, "fps": fps,
                                   "block": seg, "test_frames": test_idx,
                                   "selftrain_frames": st_idx}

    # hard invariants: pools disjoint; no frame of one pool inside the other's block
    for alt, m in manifest["videos"].items():
        seg = m["block"]
        assert not set(m["test_frames"]) & set(m["selftrain_frames"]), f"overlap in {alt}"
        for fi in m["test_frames"]:
            assert (fi // seg) % 2 == 0, f"{alt} test frame {fi} in odd block"
        for fi in m["selftrain_frames"]:
            assert (fi // seg) % 2 == 1, f"{alt} selftrain frame {fi} in even block"
    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[done] -> {out}")
    print("""
NEXT (manual, ~2-3 h total):
  1. Upload test_frames/* to Roboflow (or labelImg), draw a box on EVERY person,
     export YOLO format. One class: person. Label ALL altitudes.
  2. Put the exported labels next to the images as <name>.txt.
  3. NEVER label or train on selftrain_frames -- that pool is for pseudo-labels.
  4. Commit manifest.json to git (it freezes the test set).""")


if __name__ == "__main__":
    main()
