r"""
28_bootstrap_sig.py -- paired image-level BOOTSTRAP significance for any two saved evals.

Answers "is model B's gain over model A bigger than TEST-SET sampling noise?" (the cheap,
no-GPU, no-retrain half of the rigor story; training-seed variance is the separate multi-seed run).

Method: resample the test images WITH REPLACEMENT (same resample applied to BOTH models = paired),
recompute each metric, take delta = B - A. Repeat --n-boot times -> mean delta, 95% percentile CI,
and a two-sided bootstrap p (fraction of resamples whose delta crosses 0, doubled). Reuses the exact
metric functions from 18_s1_eval (iou_matrix / greedy_match / SIZE_BINS) so numbers match the evals.

Inputs are the per-image prediction caches every eval already writes: runs_s1/s1_eval_<tag>_preds.json
+ the same GT coco json. Thresholds (OptThr_F1 for per-size recall, OptThr_F2 for F2) are read from
each tag's s1_eval_<tag>.json so the operating points match what was reported.

  python 28_bootstrap_sig.py --selftest
  # D2 pair (run on PC-1 where the preds live):
  $T="E:\Thesis_mofazzal_2007074\common\c2a\C2A_Dataset\new_dataset3_scenesplit_v1\test"
  python 28_bootstrap_sig.py --tag-a s3_control_full --tag-b s3_assign_full --gt-json "$T\test_annotations.json"
  # cross-event (disaster fine-tune vs base):
  python 28_bootstrap_sig.py --tag-a s3_xevent --tag-b d3all_xevent --gt-json <xevent test _sc json>
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from importlib import import_module
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
RUNS = HERE / "runs_s1"
_ev = import_module("18_s1_eval")          # iou_matrix, greedy_match, SIZE_BINS, IOU_MATCH, CONF_FLOOR, load_gt


def _ap50_allpoint(scores, correct, n_gt):
    """All-point AP50 from a pool of (score, correct) + total GT count. Matches 18_s1_eval."""
    if n_gt <= 0 or len(scores) == 0:
        return 0.0
    order = np.argsort(-scores)
    c = correct[order]
    tp = np.cumsum(c)
    fp = np.cumsum(1.0 - c)
    rec = tp / max(n_gt, 1)
    prec = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.concatenate([[0.0], rec, [rec[-1]]])
    mpre = np.concatenate([[1.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    return float(np.sum(np.diff(mrec) * mpre[1:]))


def _per_image_stats(preds, gt, thr_f1, thr_f2):
    """Precompute each image's fixed contributions as NUMPY ARRAYS so a bootstrap resample is a
    single vectorized arr[idx].sum() instead of a Python loop. Returns a dict of arrays ('S')."""
    scores_list, correct_list = [], []
    n_gt_a = []
    binnames = [b[0] for b in _ev.SIZE_BINS]
    hit = {n: [] for n in binnames}; tot = {n: [] for n in binnames}
    f2tp, f2fp, optp, opfp = [], [], [], []
    for fn, g in gt.items():
        g = np.asarray(g, float).reshape(-1, 4)
        n_gt = len(g)
        p = preds.get(fn)
        b = np.asarray(p["boxes"], float).reshape(-1, 4) if p else np.zeros((0, 4))
        s = np.asarray(p["scores"], float).reshape(-1) if p else np.zeros(0)
        correct, _ = _ev.greedy_match(b, s, g) if len(b) else (np.zeros(0), n_gt)
        scores_list.append(s.astype(np.float32)); correct_list.append(correct.astype(np.float32))
        n_gt_a.append(n_gt)
        sh = {n: 0 for n in binnames}; st = {n: 0 for n in binnames}
        if n_gt:
            sz = np.sqrt((g[:, 2] - g[:, 0]).clip(0) * (g[:, 3] - g[:, 1]).clip(0))
            keep = s >= thr_f1
            taken = np.zeros(n_gt, bool)
            if keep.any():
                M = _ev.iou_matrix(b[keep], g)
                for i in np.argsort(-s[keep]):
                    j = int(np.argmax(np.where(taken, -1.0, M[i])))
                    if not taken[j] and M[i, j] >= _ev.IOU_MATCH:
                        taken[j] = True
            for name, lo, hi in _ev.SIZE_BINS:
                mm = (sz >= lo) & (sz < hi)
                st[name] = int(mm.sum()); sh[name] = int(taken[mm].sum())
        for n in binnames:
            hit[n].append(sh[n]); tot[n].append(st[n])
        f2tp.append(float(correct[s >= thr_f2].sum()) if len(s) else 0.0)
        f2fp.append(float((s >= thr_f2).sum() - (correct[s >= thr_f2].sum() if len(s) else 0.0)))
        optp.append(float(correct[s >= 0.25].sum()) if len(s) else 0.0)
        opfp.append(float((s >= 0.25).sum() - (correct[s >= 0.25].sum() if len(s) else 0.0)))
    return {"n": len(n_gt_a), "scores_list": scores_list, "correct_list": correct_list,
            "n_gt": np.asarray(n_gt_a, float),
            "hit": {n: np.asarray(hit[n], float) for n in binnames},
            "tot": {n: np.asarray(tot[n], float) for n in binnames},
            "f2tp": np.asarray(f2tp), "f2fp": np.asarray(f2fp),
            "optp": np.asarray(optp), "opfp": np.asarray(opfp), "binnames": binnames}


def _metrics_on(S, idx):
    """Aggregate over a bootstrap index array (vectorized sums; only AP50 needs a concat)."""
    n_gt = float(S["n_gt"][idx].sum())
    sc = np.concatenate([S["scores_list"][i] for i in idx])
    co = np.concatenate([S["correct_list"][i] for i in idx])
    out = {"AP50": _ap50_allpoint(sc, co, n_gt)}
    for name in S["binnames"]:
        t = S["tot"][name][idx].sum()
        out[f"recall_{name}"] = float(S["hit"][name][idx].sum() / t) if t else float("nan")
    tp = S["f2tp"][idx].sum(); fp = S["f2fp"][idx].sum()
    P = tp / max(tp + fp, 1e-9); R = tp / max(n_gt, 1e-9)
    out["F2"] = float(5 * P * R / max(4 * P + R, 1e-9))
    otp = S["optp"][idx].sum(); ofp = S["opfp"][idx].sum()
    out["precision@25"] = float(otp / max(otp + ofp, 1e-9))
    out["recall@25"] = float(otp / max(n_gt, 1e-9))
    return out


def bootstrap(rows_a, rows_b, n_boot=1000, seed=0):
    """Paired bootstrap: same resampled image indices for A and B each iteration."""
    n = rows_a["n"]
    assert n == rows_b["n"], "A and B must cover the same images"
    rng = np.random.default_rng(seed)
    allidx = np.arange(n)
    base_a, base_b = _metrics_on(rows_a, allidx), _metrics_on(rows_b, allidx)
    keys = [k for k in base_a if not (isinstance(base_a[k], float) and np.isnan(base_a[k]))]
    deltas = {k: [] for k in keys}
    for it in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ma, mb = _metrics_on(rows_a, idx), _metrics_on(rows_b, idx)
        for k in keys:
            va, vb = ma[k], mb[k]
            if not (np.isnan(va) or np.isnan(vb)):
                deltas[k].append(vb - va)
        if (it + 1) % 250 == 0:
            print(f"[boot] {it + 1}/{n_boot}")
    res = {}
    for k in keys:
        d = np.asarray(deltas[k])
        if len(d) == 0:
            continue
        lo, hi = np.percentile(d, [2.5, 97.5])
        frac_pos = float((d > 0).mean()); frac_neg = float((d < 0).mean())
        p = min(1.0, 2.0 * min(frac_pos, frac_neg))          # two-sided bootstrap p
        res[k] = {"A": round(float(base_a[k]), 4), "B": round(float(base_b[k]), 4),
                  "delta_pp": round(float(base_b[k] - base_a[k]) * 100, 2),
                  "ci95_pp": [round(float(lo) * 100, 2), round(float(hi) * 100, 2)],
                  "p": round(float(p), 4), "sig": bool(lo > 0 or hi < 0)}
    return res


def _read_thr(tag):
    j = RUNS / f"s1_eval_{tag}.json"
    if j.is_file():
        d = json.loads(j.read_text(encoding="utf-8"))
        return float(d.get("OptThr_F1", 0.25)), float(d.get("OptThr_F2", 0.25))
    print(f"[boot] WARN: {j} not found -> thresholds default to 0.25")
    return 0.25, 0.25


def _run(a):
    gt, gt_raw = _ev.load_gt(a.gt_json)
    pa = json.loads((RUNS / f"s1_eval_{a.tag_a}_preds.json").read_text(encoding="utf-8"))
    pb = json.loads((RUNS / f"s1_eval_{a.tag_b}_preds.json").read_text(encoding="utf-8"))
    ta1, ta2 = _read_thr(a.tag_a)
    tb1, tb2 = _read_thr(a.tag_b)
    rows_a = _per_image_stats(pa, gt, ta1, ta2)
    rows_b = _per_image_stats(pb, gt, tb1, tb2)
    res = bootstrap(rows_a, rows_b, n_boot=a.n_boot, seed=a.seed)
    print(f"\n=== paired bootstrap  {a.tag_b} (B) vs {a.tag_a} (A)   "
          f"n_img={len(gt)}  n_boot={a.n_boot} ===")
    print(f"{'metric':14} {'A':>8} {'B':>8} {'delta_pp':>9} {'95% CI (pp)':>18} {'p':>8}  sig")
    order = ["AP50", "F2", "recall_very_tiny", "recall_tiny", "recall_small",
             "recall_medium", "precision@25", "recall@25"]
    for k in order:
        if k not in res:
            continue
        r = res[k]
        star = "  <-- SIGNIFICANT" if r["sig"] else ""
        print(f"{k:14} {r['A']:>8} {r['B']:>8} {r['delta_pp']:>+9} "
              f"{str(r['ci95_pp']):>18} {r['p']:>8}{star}")
    outp = RUNS / f"bootstrap_{a.tag_a}_vs_{a.tag_b}.json"
    outp.write_text(json.dumps({"A": a.tag_a, "B": a.tag_b, "n_img": len(gt),
                                "n_boot": a.n_boot, "results": res}, indent=2), encoding="utf-8")
    print(f"[boot] saved -> {outp}")
    return 0


def _selftest():
    rng = np.random.default_rng(0)
    gt, pa, pb = {}, {}, {}
    gt_raw = {"images": [], "annotations": [], "categories": [{"id": 0, "name": "person"}]}
    aid = 0
    for i in range(60):
        fn = f"img{i}"
        n = int(rng.integers(4, 12))
        sz = rng.uniform(5, 50, n)
        xy = rng.uniform(0, 600, (n, 2))
        g = np.concatenate([xy, xy + sz[:, None]], 1)
        gt[fn] = g
        gt_raw["images"].append({"id": i, "file_name": fn, "width": 640, "height": 640})
        for bb in g:
            gt_raw["annotations"].append({"id": aid, "image_id": i, "category_id": 0,
                                          "bbox": [float(bb[0]), float(bb[1]),
                                                   float(bb[2] - bb[0]), float(bb[3] - bb[1])], "iscrowd": 0})
            aid += 1
        # A = decent detector; B = A but consistently a bit better recall -> delta should be > 0
        ba = g + rng.normal(0, 1.0, g.shape); sa = np.clip(rng.beta(5, 2, n), .05, .99)
        pa[fn] = {"boxes": ba.tolist(), "scores": sa.tolist()}
        bb2 = g + rng.normal(0, 0.7, g.shape); sb = np.clip(rng.beta(7, 2, n) + 0.03, .05, .99)
        pb[fn] = {"boxes": bb2.tolist(), "scores": sb.tolist()}
    ra = _per_image_stats(pa, gt, 0.3, 0.2)
    rb = _per_image_stats(pb, gt, 0.3, 0.2)
    res = bootstrap(ra, rb, n_boot=300, seed=0)
    assert "AP50" in res and "ci95_pp" in res["AP50"], res
    assert isinstance(res["AP50"]["sig"], bool)
    # CI must bracket the point delta
    lo, hi = res["AP50"]["ci95_pp"]
    assert lo <= res["AP50"]["delta_pp"] <= hi, (lo, res["AP50"]["delta_pp"], hi)
    print("SELFTEST OK: paired bootstrap runs; AP50", res["AP50"])
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-a", help="baseline eval tag (reads runs_s1/s1_eval_<tag>_preds.json)")
    ap.add_argument("--tag-b", help="comparison eval tag; delta = B - A")
    ap.add_argument("--gt-json", help="the COCO GT both evals used")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(_selftest())
    if not (a.tag_a and a.tag_b and a.gt_json):
        ap.error("--tag-a, --tag-b, --gt-json required (or --selftest)")
    sys.exit(_run(a))
