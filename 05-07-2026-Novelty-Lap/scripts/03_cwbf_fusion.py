"""
03_cwbf_fusion.py -- Calibrated Weighted Boxes Fusion for sliced inference.

Pure numpy. No ensemble-boxes dependency (we need the calibration-weighted
variant anyway, and lab PCs should not need new pip installs).

What it provides (imported by 04_eval_fusion_ablation.py and the drone script):
  nms(boxes, scores, iou_thr)                     -- baseline greedy NMS
  wbf(boxes, scores, iou_thr, score_mode)         -- Weighted Boxes Fusion
                                                     (Solovyev et al., Image and
                                                     Vision Computing 2021)
  TempScaler().fit(conf, correct) / .apply(conf)  -- confidence temperature
                                                     calibration (fit on VAL only)
  fuse_sources(sources, mode, iou_thr, scalers)   -- merge detections from several
                                                     sources (tiles / whole / TTA):
                                                     mode in {nms, wbf, cwbf}
  match_greedy(det, det_scores, gt, iou_thr)      -- det<->GT matcher (calibration
                                                     targets + PR sweeps)

Boxes are ALWAYS xyxy in absolute global-image pixels. Single class (person).

Self-test (run on any machine, takes <5 s):
  python 03_cwbf_fusion.py --selftest
"""
from __future__ import annotations
import argparse
import numpy as np


# ----------------------------------------------------------------------------- IoU
def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between (N,4) and (M,4) xyxy -> (N,M)."""
    a = np.asarray(a, np.float64).reshape(-1, 4)
    b = np.asarray(b, np.float64).reshape(-1, 4)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(br - tl, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    ar_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    ar_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = ar_a[:, None] + ar_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


# ----------------------------------------------------------------------------- NMS
def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.5):
    """Greedy NMS. Returns (kept_boxes, kept_scores)."""
    boxes = np.asarray(boxes, np.float64).reshape(-1, 4)
    scores = np.asarray(scores, np.float64).reshape(-1)
    order = np.argsort(-scores)
    keep = []
    supp = np.zeros(len(boxes), bool)
    for i in order:
        if supp[i]:
            continue
        keep.append(i)
        ious = iou_matrix(boxes[i], boxes[order]).ravel()
        supp[order[ious > iou_thr]] = True
        supp[i] = False if i in keep else supp[i]
        supp[i] = True  # processed
    keep = np.array(keep, int)
    return boxes[keep], scores[keep]


# ----------------------------------------------------------------------------- WBF
def wbf(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.55,
        score_mode: str = "avg", n_sources: int | None = None):
    """
    Weighted Boxes Fusion over a single pooled detection list.

    Faithful to Solovyev et al.: process boxes by descending score; a box joins
    the first cluster whose FUSED box has IoU > iou_thr, else starts a cluster.
    A cluster's fused box is the score-weighted average of its members; its
    score is the members' mean.  If n_sources is given (ensembling S sources),
    scores are rescaled by min(n_members, S)/S, rewarding cross-source
    agreement (paper Eq. 6). For single-model tile merging leave it None.

    Returns (fused_boxes, fused_scores) sorted by descending score.
    """
    boxes = np.asarray(boxes, np.float64).reshape(-1, 4)
    scores = np.asarray(scores, np.float64).reshape(-1)
    if len(boxes) == 0:
        return boxes.reshape(0, 4), scores
    order = np.argsort(-scores)
    fused: list[np.ndarray] = []       # current fused box per cluster
    members: list[list[int]] = []      # member indices per cluster
    for i in order:
        placed = False
        if fused:
            ious = iou_matrix(boxes[i], np.stack(fused)).ravel()
            j = int(np.argmax(ious))
            if ious[j] > iou_thr:
                members[j].append(i)
                w = scores[members[j]]
                fused[j] = (boxes[members[j]] * w[:, None]).sum(0) / w.sum()
                placed = True
        if not placed:
            fused.append(boxes[i].copy())
            members.append([i])
    fb = np.stack(fused)
    if score_mode == "max":
        fs = np.array([scores[m].max() for m in members])
    else:
        fs = np.array([scores[m].mean() for m in members])
    if n_sources is not None and n_sources > 1:
        fs = fs * np.minimum(np.array([len(m) for m in members]), n_sources) / n_sources
    o = np.argsort(-fs)
    return fb[o], fs[o]


# --------------------------------------------------------------- calibration
class TempScaler:
    """Temperature scaling on detection confidences.

    fit(conf, correct): conf in (0,1), correct in {0,1} (matched-to-GT at
    IoU 0.5 on the VALIDATION split -- never fit on test).  Minimises NLL of
    sigmoid(logit(conf)/T) over T in [0.05, 20] by golden-section search.
    """
    def __init__(self, T: float = 1.0):
        self.T = float(T)

    @staticmethod
    def _nll(conf, correct, T):
        z = np.log(conf / (1 - conf)) / T
        p = 1.0 / (1.0 + np.exp(-z))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return -(correct * np.log(p) + (1 - correct) * np.log(1 - p)).mean()

    def fit(self, conf: np.ndarray, correct: np.ndarray) -> "TempScaler":
        conf = np.clip(np.asarray(conf, np.float64), 1e-6, 1 - 1e-6)
        correct = np.asarray(correct, np.float64)
        lo, hi = 0.05, 20.0
        gr = (np.sqrt(5) - 1) / 2
        for _ in range(80):
            m1 = hi - gr * (hi - lo)
            m2 = lo + gr * (hi - lo)
            if self._nll(conf, correct, m1) < self._nll(conf, correct, m2):
                hi = m2
            else:
                lo = m1
        self.T = float((lo + hi) / 2)
        return self

    def apply(self, conf: np.ndarray) -> np.ndarray:
        conf = np.clip(np.asarray(conf, np.float64), 1e-6, 1 - 1e-6)
        z = np.log(conf / (1 - conf)) / self.T
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def ece(conf, correct, bins: int = 15) -> float:
        conf = np.asarray(conf, np.float64)
        correct = np.asarray(correct, np.float64)
        edges = np.linspace(0, 1, bins + 1)
        e = 0.0
        for i in range(bins):
            m = (conf > edges[i]) & (conf <= edges[i + 1])
            if m.any():
                e += m.mean() * abs(correct[m].mean() - conf[m].mean())
        return float(e)


# --------------------------------------------------------------- source fusion
def fuse_sources(sources: list[dict], mode: str = "cwbf", iou_thr: float = 0.55,
                 scalers: dict[str, TempScaler] | None = None,
                 count_sources: bool = False, score_mode: str = "max"):
    """
    sources: [{"name": "tile_0_0"| "whole" | "tta1280", "boxes": (N,4) GLOBAL
               xyxy, "scores": (N,)}, ...]
    mode:    nms | wbf | cwbf   (cwbf = per-source-family temperature scaling
             via `scalers` keyed by name prefix before WBF)
    count_sources: pass len(sources) to WBF's agreement rescale -- use ONLY for
             genuinely redundant sources (whole+TTA ensemble), NOT for tiles
             (a tiny object lives in one tile; agreement rescale would crush it).
    score_mode: 'max' (default) for tile merging -- coordinates are fused as the
             score-weighted average (localization win) but the cluster keeps its
             strongest member's confidence, preserving score ranking. 'avg' is
             the paper-faithful ensemble variant; it drags confident detections
             toward their weak duplicates and hurts AP when source coverage is
             uneven (verified by the synthetic selftest in 04).
    """
    bs, ss = [], []
    for s in sources:
        b = np.asarray(s["boxes"], np.float64).reshape(-1, 4)
        c = np.asarray(s["scores"], np.float64).reshape(-1)
        if mode == "cwbf" and scalers:
            key = next((k for k in scalers if s["name"].startswith(k)), None)
            if key is not None:
                c = scalers[key].apply(c)
        bs.append(b)
        ss.append(c)
    boxes = np.concatenate(bs, 0) if bs else np.zeros((0, 4))
    scores = np.concatenate(ss, 0) if ss else np.zeros((0,))
    if mode == "nms":
        return nms(boxes, scores, iou_thr)
    return wbf(boxes, scores, iou_thr, score_mode=score_mode,
               n_sources=len(sources) if count_sources else None)


# --------------------------------------------------------------- GT matching
def match_greedy(det_boxes, det_scores, gt_boxes, iou_thr: float = 0.5):
    """Greedy score-ordered matching. Returns correct (N,) in {0,1} per det,
    and n_missed_gt."""
    det_boxes = np.asarray(det_boxes, np.float64).reshape(-1, 4)
    det_scores = np.asarray(det_scores, np.float64).reshape(-1)
    gt_boxes = np.asarray(gt_boxes, np.float64).reshape(-1, 4)
    correct = np.zeros(len(det_boxes))
    taken = np.zeros(len(gt_boxes), bool)
    if len(gt_boxes):
        M = iou_matrix(det_boxes, gt_boxes)
        for i in np.argsort(-det_scores):
            j = int(np.argmax(np.where(taken, -1.0, M[i])))
            if not taken[j] and M[i, j] >= iou_thr:
                correct[i] = 1
                taken[j] = True
    return correct, int((~taken).sum())


# ------------------------------------------------------------------- selftest
def _selftest() -> None:
    ok = 0

    # T1: exact hand-computed fusion
    b = [[0, 0, 10, 10], [2, 0, 12, 10], [30, 30, 40, 40]]
    s = [0.9, 0.6, 0.8]
    fb, fs = wbf(b, s, iou_thr=0.55)
    assert len(fb) == 2, f"T1 clusters {len(fb)}"
    # cluster of A+B: weighted avg -> [0.8, 0, 10.8, 10], score mean 0.75
    i = int(np.argmin(fs))  # 0.75 < 0.8
    assert np.allclose(fb[i], [0.8, 0, 10.8, 10], atol=1e-9), f"T1 fused {fb[i]}"
    assert abs(fs[i] - 0.75) < 1e-12 and abs(fs.max() - 0.8) < 1e-12
    ok += 1

    # T2: single box passthrough + permutation invariance
    fb1, fs1 = wbf([[5, 5, 9, 9]], [0.3])
    assert np.allclose(fb1, [[5, 5, 9, 9]]) and np.allclose(fs1, [0.3])
    rng = np.random.default_rng(0)
    B = rng.uniform(0, 100, (30, 2))
    B = np.concatenate([B, B + rng.uniform(4, 10, (30, 2))], 1)
    S = rng.uniform(0.05, 0.95, 30)
    p = rng.permutation(30)
    fa = wbf(B, S)[0]
    fbp = wbf(B[p], S[p])[0]
    assert np.allclose(np.sort(fa, 0), np.sort(fbp, 0), atol=1e-9), "T2 permutation"
    ok += 1

    # T3: duplicate-recovery -- WBF localises better than NMS on symmetric noise
    gt = np.array([10, 10, 22, 22], float)
    d1, d2 = gt + [-1.5, 0, -1.5, 0], gt + [1.5, 0, 1.5, 0]
    det_b, det_s = np.stack([d1, d2]), np.array([0.62, 0.60])
    nb, _ = nms(det_b, det_s, 0.5)
    wb, _ = wbf(det_b, det_s, 0.5)
    iou_n = iou_matrix(nb[0], gt).item()
    iou_w = iou_matrix(wb[0], gt).item()
    assert iou_w > iou_n, f"T3 wbf {iou_w:.3f} !> nms {iou_n:.3f}"
    ok += 1

    # T4: temperature scaling fixes overconfidence (and ECE drops)
    rng = np.random.default_rng(1)
    true_p = rng.uniform(0.05, 0.95, 4000)
    correct = (rng.uniform(size=4000) < true_p).astype(float)
    overconf = 1 / (1 + np.exp(-2.5 * np.log(true_p / (1 - true_p))))  # sharpened
    ts = TempScaler().fit(overconf, correct)
    assert ts.T > 1.5, f"T4 expected T>1.5, got {ts.T:.2f}"
    assert TempScaler.ece(ts.apply(overconf), correct) < TempScaler.ece(overconf, correct)
    ok += 1

    # T5: fuse_sources end-to-end with calibration (cwbf runs, shapes sane)
    srcs = [
        {"name": "tile", "boxes": [[0, 0, 10, 10]], "scores": [0.9]},
        {"name": "whole", "boxes": [[1, 0, 11, 10], [50, 50, 60, 60]], "scores": [0.5, 0.7]},
    ]
    scalers = {"tile": TempScaler(2.0), "whole": TempScaler(1.0)}
    fb5, fs5 = fuse_sources(srcs, "cwbf", 0.55, scalers)
    assert fb5.shape == (2, 4) and len(fs5) == 2
    ok += 1

    # T6: matcher sanity
    corr, missed = match_greedy([[0, 0, 10, 10], [100, 100, 110, 110]], [0.9, 0.8],
                                [[1, 1, 10, 10]], 0.5)
    assert corr.tolist() == [1.0, 0.0] and missed == 0
    ok += 1

    print(f"SELFTEST PASSED ({ok}/6 groups)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    else:
        print(__doc__)
