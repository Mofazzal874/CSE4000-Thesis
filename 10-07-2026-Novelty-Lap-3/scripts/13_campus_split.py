"""
13_campus_split.py — temporal train/eval split for the KUET campus-road frames.

WHY temporal: there is only ONE clip per altitude, so a clip-level split is impossible.
The drone flies ALONG the road, so a different timestamp = a different road section
(different trees, buildings, people) => a middle time-window held out is quasi-scene-disjoint.
Guard gaps on both sides of the eval window prevent near-duplicate frames leaking across.

Layout (per source video, timestamps parsed from the filename `..._t00042s.jpg`):
    [ train ][ guard ][      EVAL      ][ guard ][ train ]
      0-35%     5%        40-60%           5%      65-100%

Usage:
  python 13_campus_split.py --frames "..\\..\\Drone Shoot\\campus_v1\\frames_v1" \
      --out "..\\..\\Drone Shoot\\campus_v1"
  python 13_campus_split.py --selftest
Outputs `train_frames\\`, `eval_frames\\` (copies, originals untouched) + split_manifest.json.
Idempotent: re-running overwrites the same assignment (pure function of timestamps).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

TS_RE = re.compile(r"_t(\d+)s$", re.IGNORECASE)
EVAL_LO, EVAL_HI = 0.40, 0.60      # eval window (fraction of the clip's sampled span)
GUARD = 0.05                        # dead-zone on each side of the eval window


def parse(stem: str) -> tuple[str, int] | None:
    m = TS_RE.search(stem)
    if not m:
        return None
    return TS_RE.sub("", stem), int(m.group(1))


def assign(frames: list[Path]) -> dict[str, str]:
    """-> {filename: 'train'|'eval'|'guard'} using per-source min/max timestamps."""
    groups: dict[str, list[tuple[Path, int]]] = {}
    for p in frames:
        parsed = parse(p.stem)
        if parsed is None:
            continue
        src, ts = parsed
        groups.setdefault(src, []).append((p, ts))
    out: dict[str, str] = {}
    for src, items in groups.items():
        ts_vals = [t for _, t in items]
        lo, hi = min(ts_vals), max(ts_vals)
        span = max(hi - lo, 1)
        for p, ts in items:
            f = (ts - lo) / span
            if EVAL_LO <= f <= EVAL_HI:
                out[p.name] = "eval"
            elif (EVAL_LO - GUARD) <= f < EVAL_LO or EVAL_HI < f <= (EVAL_HI + GUARD):
                out[p.name] = "guard"
            else:
                out[p.name] = "train"
    return out


def run(frames_dir: Path, out_dir: Path) -> int:
    frames = sorted(frames_dir.glob("*.jpg"))
    if not frames:
        print(f"[split] no .jpg in {frames_dir}")
        return 1
    labels = assign(frames)
    tr_dir, ev_dir = out_dir / "train_frames", out_dir / "eval_frames"
    for d in (tr_dir, ev_dir):
        d.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "eval": 0, "guard": 0}
    for p in frames:
        lab = labels.get(p.name)
        if lab is None:
            continue
        counts[lab] += 1
        if lab == "train":
            shutil.copy2(p, tr_dir / p.name)
        elif lab == "eval":
            shutil.copy2(p, ev_dir / p.name)
        # guard frames are intentionally dropped (leak buffer)
    (out_dir / "split_manifest.json").write_text(json.dumps(
        {"rule": {"eval_window": [EVAL_LO, EVAL_HI], "guard": GUARD},
         "counts": counts, "assignment": labels}, indent=2), encoding="utf-8")
    print(f"[split] train={counts['train']} eval={counts['eval']} "
          f"guard-dropped={counts['guard']} -> {out_dir}")
    print("[split] NEXT: train_frames -> Roboflow 'campus-train-v1' (assist OK); "
          "eval_frames -> 'campus-eval-v1' (MANUAL, Label Assist OFF).")
    return 0


def selftest() -> int:
    names = [Path(f"KUET_ROAD_30m_t{ts:05d}s.jpg") for ts in range(0, 101, 10)]
    lab = assign(names)
    ok = (lab["KUET_ROAD_30m_t00000s.jpg"] == "train"
          and lab["KUET_ROAD_30m_t00050s.jpg"] == "eval"
          and lab["KUET_ROAD_30m_t00100s.jpg"] == "train"
          and lab["KUET_ROAD_30m_t00030s.jpg"] == "train"     # 30% -> train
          and lab["KUET_ROAD_30m_t00070s.jpg"] == "train")    # 70% -> train
    ev = sum(1 for v in lab.values() if v == "eval")
    print(f"[{'PASS' if ok else 'FAIL'}] selftest: eval={ev}/11 frames, boundaries correct")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not a.frames or not a.out:
        ap.error("--frames and --out required (or --selftest)")
    sys.exit(run(a.frames, a.out))
