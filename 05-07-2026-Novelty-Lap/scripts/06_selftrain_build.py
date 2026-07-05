"""
06_selftrain_build.py -- Turn SAHI/model pseudo-labels on the self-train frame
pool into a YOLO dataset fragment, with hard anti-leakage guards.

Inputs:
  --frames-dir  extracted_v1/selftrain_frames        (from 05, all altitudes)
  --manifest    extracted_v1/manifest.json           (from 05)
  ONE OF:
  --preds-dir   directory of YOLO-format predictions <name>.txt with lines
                "cls cx cy w h conf"  (ultralytics predict save_txt=True,
                save_conf=True writes exactly this under .../labels/)
  --preds-json  predictions file saved by 04_eval_fusion_ablation.py
                (--save-preds) on the selftrain frames; boxes are fused here
                with --fuse-mode (default cwbf, optionally --cal) -- this is
                the "SAHI-guided" pseudo-labeling path and the RECOMMENDED one
                (tiny humans only surface in the sliced+fused predictions).

Guards (each ABORTS the build, they do not warn):
  G-a  every frame must be listed in the manifest's selftrain pool -- a test
       frame in the input is a protocol violation;
  G-b  pseudo-boxes below --conf-thr are dropped (confirmation-bias control;
       sweep 0.5-0.7 per the plan);
  G-c  boxes smaller than --min-size-px in EITHER dimension are dropped
       (sub-resolvable pseudo-labels are noise);
  G-d  frames whose detections were ALL dropped are kept as EMPTY negatives
       only when --keep-empty is set (recommended: the drone footage is mostly
       person-free field, exactly the negatives C2A lacks).

Output:
  <out>/images/*.jpg + <out>/labels/*.txt + <out>/train_list.txt
  <out>/selftrain_stats.json      (kept/dropped accounting per altitude)

Then fine-tune exactly like the enriched run (joint C2A + SARD + this fragment,
low LR) -- see RUNNING_GUIDE.md Phase 4.

Self-test: python 06_selftrain_build.py --selftest
"""
from __future__ import annotations
import argparse, json, os, shutil, sys, tempfile
from collections import defaultdict
from pathlib import Path


def parse_pred_txt(p: Path):
    rows = []
    if not p.is_file():
        return rows
    for ln in p.read_text().strip().splitlines():
        f = ln.split()
        if len(f) == 6:
            rows.append((int(float(f[0])), *map(float, f[1:])))
        elif len(f) == 5:  # no conf column -- refuse silently-unscored labels
            raise ValueError(f"{p.name}: prediction line has no confidence column; "
                             f"re-run predict with save_conf=True")
    return rows


def _fused_lookup(args):
    """Load 04's preds json and return {stem: [(cls,cx,cy,w,h,conf), ...]} with
    boxes fused by 03's fuse_sources (the SAHI-guided path)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from importlib import import_module
    _f = import_module("03_cwbf_fusion")
    preds = json.loads(Path(args.preds_json).read_text())
    scalers = None
    if args.cal:
        cal = json.loads(Path(args.cal).read_text())
        scalers = {k: _f.TempScaler(v["T"]) for k, v in cal.items()}
    out = {}
    for fn, p in preds.items():
        W, H = p["wh"]
        fb, fs = _f.fuse_sources(p["sources"], mode=args.fuse_mode,
                                 iou_thr=args.iou_fuse, scalers=scalers)
        rows = []
        for b, s in zip(fb, fs):
            w, h = (b[2] - b[0]) / W, (b[3] - b[1]) / H
            cx, cy = (b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H
            rows.append((0, float(cx), float(cy), float(w), float(h), float(s)))
        out[Path(fn).stem] = rows
    return out


def build(args) -> dict:
    frames_dir, out = Path(args.frames_dir), Path(args.out)
    manifest = json.loads(Path(args.manifest).read_text())
    fused = _fused_lookup(args) if args.preds_json else None
    preds_dir = Path(args.preds_dir) if args.preds_dir else None
    allowed = {f"{alt}_f{fi:06d}"
               for alt, m in manifest["videos"].items() for fi in m["selftrain_frames"]}
    test_names = {f"{alt}_f{fi:06d}"
                  for alt, m in manifest["videos"].items() for fi in m["test_frames"]}

    imgs = sorted(frames_dir.rglob("*.jpg"))
    if not imgs:
        sys.exit(f"[FATAL] no jpgs under {frames_dir}")
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    import cv2
    stats = defaultdict(lambda: {"frames": 0, "kept": 0, "drop_conf": 0,
                                 "drop_size": 0, "empty_kept": 0, "empty_skipped": 0})
    listed = []
    for ip in imgs:
        stem = ip.stem
        if stem in test_names:
            sys.exit(f"[FATAL G-a] {stem} is a TEST frame -- it must never enter "
                     f"the self-training pool. Aborting with no output trust.")
        if stem not in allowed:
            sys.exit(f"[FATAL G-a] {stem} not in manifest selftrain pool. Aborting.")
        alt = stem.split("_f")[0]
        st = stats[alt]
        st["frames"] += 1
        im = cv2.imread(str(ip))
        H, W = im.shape[:2]
        keep_lines = []
        rows = fused.get(stem, []) if fused is not None else parse_pred_txt(preds_dir / f"{stem}.txt")
        for cls, cx, cy, w, h, conf in rows:
            if conf < args.conf_thr:
                st["drop_conf"] += 1
                continue
            if w * W < args.min_size_px or h * H < args.min_size_px:
                st["drop_size"] += 1
                continue
            keep_lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            st["kept"] += 1
        if not keep_lines and not args.keep_empty:
            st["empty_skipped"] += 1
            continue
        if not keep_lines:
            st["empty_kept"] += 1
        dst_i = out / "images" / ip.name
        if not dst_i.exists():
            try:
                os.link(ip, dst_i)
            except OSError:
                shutil.copy2(ip, dst_i)
        (out / "labels" / f"{stem}.txt").write_text("\n".join(keep_lines) + ("\n" if keep_lines else ""))
        listed.append(str(dst_i.resolve()))

    (out / "train_list.txt").write_text("\n".join(listed))
    report = {"conf_thr": args.conf_thr, "min_size_px": args.min_size_px,
              "keep_empty": bool(args.keep_empty), "n_images_out": len(listed),
              "per_altitude": dict(stats)}
    (out / "selftrain_stats.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"[done] -> {out}  ({len(listed)} images)")
    return report


# ------------------------------------------------------------------ selftest
def _selftest() -> None:
    import numpy as np
    import cv2
    td = Path(tempfile.mkdtemp(prefix="st_selftest_"))
    frames = td / "selftrain_frames" / "10m"
    frames.mkdir(parents=True)
    preds = td / "preds"
    preds.mkdir()
    img = np.zeros((64, 64, 3), np.uint8)
    for fi in (100, 200):
        cv2.imwrite(str(frames / f"10m_f{fi:06d}.jpg"), img)
    manifest = {"videos": {"10m": {"test_frames": [300],
                                   "selftrain_frames": [100, 200]}}}
    (td / "manifest.json").write_text(json.dumps(manifest))
    # frame 100: one confident box, one low-conf, one too-small
    (preds / "10m_f000100.txt").write_text(
        "0 0.5 0.5 0.20 0.20 0.90\n0 0.2 0.2 0.20 0.20 0.30\n0 0.8 0.8 0.01 0.01 0.95\n")
    # frame 200: nothing detected -> negative

    class A:  # args stub
        frames_dir = str(td / "selftrain_frames")
        preds_dir = str(preds)
        preds_json = None
        cal = None
        fuse_mode = "cwbf"
        iou_fuse = 0.55
        manifest = str(td / "manifest.json")
        out = str(td / "out")
        conf_thr = 0.6
        min_size_px = 4.0
        keep_empty = True
    r = build(A)
    st = r["per_altitude"]["10m"]
    assert st["kept"] == 1 and st["drop_conf"] == 1 and st["drop_size"] == 1, st
    assert st["empty_kept"] == 1 and r["n_images_out"] == 2, st
    lbl = (td / "out" / "labels" / "10m_f000100.txt").read_text().strip()
    assert lbl == "0 0.500000 0.500000 0.200000 0.200000", lbl

    # preds-json (SAHI-guided) path: two overlapping tile boxes fuse to one label
    pj = {"10m_f000100.jpg": {"wh": [64, 64], "sources": [
        {"name": "tile256", "boxes": [[16, 16, 32, 32]], "scores": [0.9]},
        {"name": "whole", "boxes": [[17, 16, 33, 32]], "scores": [0.7]}]},
          "10m_f000200.jpg": {"wh": [64, 64], "sources": []}}
    (td / "preds.json").write_text(json.dumps(pj))

    class B(A):
        preds_dir = None
        preds_json = str(td / "preds.json")
        out = str(td / "out2")
        fuse_mode = "wbf"
    r2 = build(B)
    assert r2["per_altitude"]["10m"]["kept"] == 1, r2
    lbl2 = (td / "out2" / "labels" / "10m_f000100.txt").read_text().strip()
    assert lbl2.startswith("0 0.38"), f"fused label wrong: {lbl2}"  # cx ~ 24.4/64

    # G-a abort: a test frame smuggled into the pool must kill the build
    cv2.imwrite(str(frames / "10m_f000300.jpg"), img)
    aborted = False
    try:
        build(A)
    except SystemExit as e:
        aborted = "G-a" in str(e)
    assert aborted, "test-frame guard did not fire"
    shutil.rmtree(td, ignore_errors=True)
    print("SELFTEST PASSED (filtering, negatives, guard-abort)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--frames-dir")
    ap.add_argument("--preds-dir")
    ap.add_argument("--preds-json", help="04's --save-preds file (SAHI-guided path)")
    ap.add_argument("--fuse-mode", default="cwbf", choices=["nms", "wbf", "cwbf"])
    ap.add_argument("--iou-fuse", type=float, default=0.55)
    ap.add_argument("--cal", help="calibration json from 04 --fit-cal (for cwbf)")
    ap.add_argument("--manifest")
    ap.add_argument("--out")
    ap.add_argument("--conf-thr", type=float, default=0.60)
    ap.add_argument("--min-size-px", type=float, default=4.0)
    ap.add_argument("--keep-empty", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    else:
        if not (a.frames_dir and a.manifest and a.out and (a.preds_dir or a.preds_json)):
            sys.exit(__doc__)
        build(a)
