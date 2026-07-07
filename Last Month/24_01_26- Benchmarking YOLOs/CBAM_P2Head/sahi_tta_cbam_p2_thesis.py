"""
sahi_tta_cbam_p2_thesis.py
==========================
SAHI + TTA inference-time ablation of the C2A-trained **CBAM+P2** model (yolo11m_cbam_p2head)
per system_spec_thesis.md Sec 5 + Sec 6. Successor of the March run (31-03-26(Mamba-ViT-CNN)/
sahi_tta_eval_noaug.py) which evaluated the Mamba model on Kaggle -- this one targets CBAM+P2,
runs on the local PCs (PC-1 4070TiS / PC-2 A6000 / PC-4 4070), and is failproof + resumable.

WHAT IT PRODUCES (the report's SAHI/TTA ablation, Sec 5: "separate, clearly-labeled
inference-time ablations -- never mixed into the primary mAP column"):
  Row 0  Baseline  (640, no SAHI, no TTA)      <- anchors to the primary ablation table
  Rows   SAHI sweep (slice 256/320/512/640, GREEDYNMM+IOS, perform_standard_pred=True)
  Rows   TTA (ultralytics augment=True at 640/832/1280/1920, official mAP)
  Row    SAHI + TTA combined (best SAHI config, TTA per tile)

METRICS per configuration (spec Sec 6 MUST-haves):
  P / R / F1 / **F2** (operational, conf=0.25, IoU=0.5) ... F2 primary (SAR favors recall)
  COCO AP / AP50 / AP75 / AP_small / AP_medium / AP_large / AR@1 / AR@10 / AR@100
  per-size recall: very_tiny(<8px) tiny(8-16) small(16-32) medium(32-96) large(>=96)
  optimal-threshold F1/F2 (+ the threshold), PR curve, F1/F2-vs-conf, confidence
  histogram, calibration (ECE/MCE/Brier), confusion TP/FP/FN, latency mean/p50/p95.
  Curves saved as CSV + PNG (DPI 300) and indexed in plots/PLOTS_INDEX.md (spec Sec 7).

FOOTNOTES THAT GO IN THE REPORT (spec Sec 5):
  * SAHI rows: detections are collected once per config at a low score floor
    (FLOOR_CONF) and all thresholds are applied offline. COCO AP for SAHI rows is
    therefore lower-bounded by that floor -- footnote it. Ultralytics val() mAP is
    not computable through SAHI, hence the per-image protocol (same as the chain).
  * TTA rows: official ultralytics val(augment=True) mAP -- directly comparable to
    the primary table's val() protocol.

FAILPROOF (spec Sec 9; load-shedding country):
  * auto-SMOKE first (10 images through EVERY stage, then continues to full run)
  * per-image JSONL detection cache -> a power cut mid-config loses NOTHING done;
    re-running the same command resumes exactly where it stopped
  * per-config completion markers -> completed configs are never recomputed
  * OOM guard on every TTA size (skips + logs instead of crashing)
  * every stage prints incremental results (monitor from the terminal)
  * GPU auto-pick on shared boxes (nvidia-smi BEFORE torch import)

RUN (from this folder, venv active; works on PC-1 / PC-2 / PC-4 unchanged):
    python sahi_tta_cbam_p2_thesis.py
  Re-run the identical command after any crash/power-cut to resume.

OUTPUT: <this folder>/runs_sahi_tta/<run_id>/
    metrics/   per-config JSON + CSV curves + grand_summary.(json|xlsx|md)
    plots/     PNGs (DPI 300) + PLOTS_INDEX.md
    cache/     per-config JSONL detection caches (the resume state)
    qualitative/  annotated baseline-vs-SAHI-vs-SAHI+TTA sample frames
    env.json, manifest.json, skipped_metrics.txt
"""

# =============================================================================
# 0. CONFIG
# =============================================================================
MODEL_TAG   = "yolo11m_cbam_p2head"          # the model being enhanced (primary-table anchor)
RUN_TAG     = "sahi_tta_cbam_p2"

# Weights: leave None to auto-discover the newest completed yolo11m_cbam_p2head run
# (runs/*/weights/best.pt with matching summary.json) under SEARCH_ROOTS; or pin a path.
EXPLICIT_BEST_PT = None
# Known copies (checked before run-dir discovery, first hit wins). #1 = the canonical
# 20260602_063759 s0 run (C2A-test mAP50 0.8533) -- the primary-table anchor.
KNOWN_BEST_PT = [
    r"D:\Academics\thesis folder\Last Month\24_01_26- Benchmarking YOLOs\CBAM_P2Head\runs\20260602_063759_yolo11m_cbam_p2head_s0_nogit\weights\best.pt",
    r"E:\Thesis_mofazzal_2007074\Benchmarking YOLOs\CBAM_P2Head\runs\20260602_063759_yolo11m_cbam_p2head_s0_nogit\weights\best.pt",
    r"E:\Thesis_mofazzal_2007074\c2a_cbam_p2head_best.pt",
    r"D:\thesis_2007074\c2a_cbam_p2head_best.pt",
    r"D:\student_2k20\2007074\c2a_cbam_p2head_best.pt",
    r"D:\Academics\thesis folder\Last Month\deployable_model\c2a_cbam_p2head_best.pt",
]

# --- protocol thresholds (system_spec_thesis.md Sec 5 -- DO NOT change) ---
CONF_OP, IOU_OP = 0.25, 0.5      # operational P/R/F1/F2 + per-size recall + confusion
CONF_AP, IOU_AP = 0.001, 0.7     # AP-style (ultralytics val) for baseline + TTA rows
FLOOR_CONF      = 0.10           # SAHI collection floor; all SAHI thresholds applied offline
                                 # (COCO AP for SAHI rows lower-bounded by this -- footnote)

# --- SAHI sweep (same 4 configs as the March run, for continuity) ---
SAHI_CONFIGS = [
    {"name": "sahi_slice256_ov30", "slice": 256, "overlap": 0.30},
    {"name": "sahi_slice320_ov25", "slice": 320, "overlap": 0.25},
    {"name": "sahi_slice512_ov25", "slice": 512, "overlap": 0.25},
    {"name": "sahi_slice640_ov30", "slice": 640, "overlap": 0.30},
]
SAHI_POSTPROCESS = dict(perform_standard_pred=True, postprocess_type="GREEDYNMM",
                        postprocess_match_metric="IOS", postprocess_match_threshold=0.5)

# --- TTA (official ultralytics val augment=True) ---
TTA_IMG_SIZES   = [640, 832, 1280, 1920]   # 1920 auto-skips on OOM (12 GB cards)
TTA_VAL_BATCH   = {640: 8, 832: 4, 1280: 1, 1920: 1}

# --- run scope ---
SPLIT       = "test"             # the report table uses the frozen C2A test split
SMOKE_N     = 10                 # auto-smoke image count (runs first, every time marker is stale)
QUALITATIVE_N = 12               # annotated sample frames (deterministic pick)
GPU_ID_FALLBACK = 0              # PC-1/PC-4 single-GPU; PC-2 auto-pick prefers a free GPU

CLASS_NAMES = ["person"]; NC = 1

# =============================================================================
# 1. GPU PICK (before torch import) + IMPORTS + PACKAGE GUARD
# =============================================================================
import os, sys, json, time, gc, shutil, math, subprocess, hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

def _pick_gpu() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL, timeout=15
        ).decode("utf-8", "ignore")
        gpus = []
        for line in out.strip().splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) >= 3:
                gpus.append({"i": int(p[0]), "mem": float(p[1]), "util": float(p[2])})
        if not gpus:
            return GPU_ID_FALLBACK
        free = [g["i"] for g in gpus if g["mem"] < 1024 and g["util"] < 10]
        snap = ", ".join(f"GPU{g['i']}:{int(g['mem'])}MB/{int(g['util'])}%" for g in gpus)
        if free:
            chosen = GPU_ID_FALLBACK if GPU_ID_FALLBACK in free else min(free)
            print(f"[gpu] usage [{snap}] -> free {free} -> GPU {chosen}")
            return chosen
        print(f"[gpu] WARNING no fully-free GPU [{snap}] -> GPU {GPU_ID_FALLBACK}")
        return GPU_ID_FALLBACK
    except Exception:
        return GPU_ID_FALLBACK

if os.environ.get("_SAHI_GPU_PINNED") != "1":         # workers must not re-pick
    _gid = _pick_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_gid)
    os.environ["_SAHI_GPU_PINNED"] = "1"
    print(f"[gpu] CUDA_VISIBLE_DEVICES={_gid}")

def _ensure(pkgs):
    import importlib
    namemap = {"opencv-python": "cv2", "PyYAML": "yaml", "scikit-learn": "sklearn"}
    for name in pkgs:
        mod = namemap.get(name, name.replace("-", "_"))
        try:
            importlib.import_module(mod)
        except ImportError:
            print(f"[deps] installing {name} ...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", name])
            except Exception as e:
                print(f"[deps] WARN could not install {name}: {e}")
_ensure(["numpy", "pandas", "PyYAML", "matplotlib", "opencv-python", "pycocotools",
         "openpyxl", "tqdm", "sahi", "ultralytics", "tabulate"])

import numpy as np
import pandas as pd
import yaml
import cv2
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    HAS_PYCOCO = True
except Exception:
    HAS_PYCOCO = False

from ultralytics import YOLO

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"[env] device={DEVICE}"
      + (f" ({torch.cuda.get_device_name(0)}, "
         f"{round(torch.cuda.get_device_properties(0).total_memory/1024**3,1)} GB)"
         if torch.cuda.is_available() else ""))

# =============================================================================
# 2. CBAM MODULES + REGISTRATION (verbatim from the chain -- deserializes the ckpt;
#    the best.pt pickles these under __main__, and THIS script is __main__)
# =============================================================================
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1); self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, reduced, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        a = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        m = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return x * self.sigmoid(a + m)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7)
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=3 if kernel_size == 7 else 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))

class CBAM(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.reduction = 16; self.kernel_size = 7
        if len(args) == 1 and isinstance(args[0], int) and args[0] <= 32:
            self.reduction = args[0]
        elif len(args) == 2 and isinstance(args[0], int) and args[0] <= 32:
            self.reduction = args[0]; self.kernel_size = args[1] if isinstance(args[1], int) else 7
        elif len(args) >= 4:
            self.reduction = args[2] if isinstance(args[2], int) else 16
            self.kernel_size = args[3] if isinstance(args[3], int) else 7
        self.reduction = kwargs.get("reduction", self.reduction)
        self.kernel_size = kwargs.get("kernel_size", self.kernel_size)
        if self.kernel_size not in (3, 7):
            self.kernel_size = 7
        self._initialized = False
        self.channel_attention = None; self.spatial_attention = None; self._channels = None
    def _lazy_init(self, channels, device, dtype):
        self._channels = channels
        self.channel_attention = ChannelAttention(channels, self.reduction).to(device=device, dtype=dtype)
        self.spatial_attention = SpatialAttention(self.kernel_size).to(device=device, dtype=dtype)
        self._initialized = True
    def forward(self, x):
        if not self._initialized:
            self._lazy_init(x.size(1), x.device, x.dtype)
        return self.spatial_attention(self.channel_attention(x))

def register_cbam():
    import ultralytics.nn.modules as _m
    import ultralytics.nn.tasks as _t
    for ns in (_m, _t):
        ns.CBAM = CBAM; ns.ChannelAttention = ChannelAttention; ns.SpatialAttention = SpatialAttention
register_cbam()
print("[cbam] registered in ultralytics namespaces")

# =============================================================================
# 3. PATHS + DISCOVERY
# =============================================================================
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path(os.getcwd()).resolve()
RUNS_DIR = SCRIPT_DIR / "runs_sahi_tta"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

SEARCH_ROOTS = [p for p in [
    Path(os.environ["PROJECT_ROOT"]) if os.environ.get("PROJECT_ROOT") else None,
    SCRIPT_DIR.parent, SCRIPT_DIR.parent.parent, SCRIPT_DIR.parent.parent.parent,
    Path(r"E:\Thesis_mofazzal_2007074"),
    Path(r"D:\student_2k20\2007074"),
    Path(r"D:\thesis_2007074"),
    Path(r"D:\Academics\thesis folder\Last Month"),
    Path(r"D:\Academics\thesis folder"),          # laptop: c2a lives at <root>\c2a\...
] if p is not None]

def find_c2a_root() -> Path:
    env = os.environ.get("C2A_ROOT")
    cands = [Path(env)] if env else []
    for r in SEARCH_ROOTS:
        cands += [r / "common" / "c2a" / "C2A_Dataset" / "new_dataset3",
                  r / "common" / "c2a" / "new_dataset3",
                  r / "c2a" / "C2A_Dataset" / "new_dataset3"]
    for c in cands:
        if c and (c / "train" / "images").is_dir() and (c / SPLIT / "images").is_dir():
            return c
    raise FileNotFoundError("C2A dataset not found. Set $env:C2A_ROOT.")

def find_best_pt() -> Path:
    if EXPLICIT_BEST_PT and Path(EXPLICIT_BEST_PT).is_file():
        return Path(EXPLICIT_BEST_PT)
    for k in KNOWN_BEST_PT:
        if Path(k).is_file():
            return Path(k)
    found, found_run = None, ""
    for r in SEARCH_ROOTS:
        if not r.is_dir():
            continue
        for best in r.glob("**/runs/*/weights/best.pt"):
            run = best.parent.parent
            summ = run / "metrics" / "summary.json"
            if not summ.is_file():
                continue
            try:
                tag = json.loads(summ.read_text()).get("model_tag")
            except Exception:
                continue
            if tag == MODEL_TAG and run.name > found_run:
                found, found_run = best, run.name
    if found is None:
        raise FileNotFoundError(
            f"CBAM+P2 best.pt not found (KNOWN_BEST_PT missing + no completed {MODEL_TAG} run "
            f"under {[str(r) for r in SEARCH_ROOTS]}). Set EXPLICIT_BEST_PT.")
    return found

def _md5(p: Path, cap_mb=64) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        read = 0
        while chunk := f.read(1 << 20):
            h.update(chunk); read += 1
            if read >= cap_mb:
                break
    return h.hexdigest() + ("_partial" if read >= cap_mb else "")

# =============================================================================
# 4. SHARED EVAL HELPERS (chain-identical protocol)
# =============================================================================
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")

def list_images(d: Path) -> List[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)

def yolo_label_to_xyxy(lbl: Path, w: int, h: int) -> np.ndarray:
    if not lbl.is_file():
        return np.zeros((0, 4), np.float32)
    out = []
    for ln in lbl.read_text().splitlines():
        p = ln.strip().split()
        if len(p) < 5:
            continue
        _, cx, cy, bw, bh = (float(x) for x in p[:5])
        out.append([(cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h])
    return np.asarray(out, np.float32) if out else np.zeros((0, 4), np.float32)

def box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), np.float32)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2-ix1, 0, None) * np.clip(iy2-iy1, 0, None)
    aa = (a[:, 2]-a[:, 0])*(a[:, 3]-a[:, 1]); ab = (b[:, 2]-b[:, 0])*(b[:, 3]-b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-12)

SIZE_BINS = [("very_tiny", 0, 8), ("tiny", 8, 16), ("small", 16, 32),
             ("medium", 32, 96), ("large", 96, 10**9)]

def greedy_match(pb: np.ndarray, pc: np.ndarray, gt: np.ndarray, iou_thr: float):
    """Confidence-ordered greedy matching -> (tp, fp, fn, gt_matched_mask, pred_correct_mask)."""
    gtm = np.zeros(gt.shape[0], bool); pok = np.zeros(pb.shape[0], bool)
    if pb.shape[0] == 0:
        return 0, 0, gt.shape[0], gtm, pok
    if gt.shape[0] == 0:
        return 0, pb.shape[0], 0, gtm, pok
    iou = box_iou_xyxy(pb, gt)
    for pi in np.argsort(-pc):
        j = int(np.argmax(iou[pi]))
        if iou[pi, j] >= iou_thr and not gtm[j]:
            gtm[j] = True; pok[pi] = True
    tp = int(pok.sum())
    return tp, pb.shape[0]-tp, gt.shape[0]-tp, gtm, pok

# =============================================================================
# 5. DETECTION CACHE (JSONL per config -- THE resume state)
# =============================================================================
class DetCache:
    """cache/<config>.jsonl : one line per image {img, w, h, boxes[[x1,y1,x2,y2],..], scores[..], ms}."""
    def __init__(self, path: Path):
        self.path = path; self.done: Dict[str, dict] = {}
        if path.is_file():
            for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                try:
                    r = json.loads(ln); self.done[r["img"]] = r
                except Exception:
                    continue   # torn last line from a power cut -> that image redoes
    def has(self, img: str) -> bool:
        return img in self.done
    def add(self, rec: dict):
        self.done[rec["img"]] = rec
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

def cached_sweep(name: str, images: List[Path], lbl_dir: Path, cache_dir: Path,
                 predict_fn, desc: str) -> DetCache:
    """Run predict_fn(img_path)->(boxes Nx4, scores N, ms) over images, cached + resumable."""
    cache = DetCache(cache_dir / f"{name}.jsonl")
    todo = [p for p in images if not cache.has(p.name)]
    if not todo:
        print(f"[{name}] all {len(images)} images cached -- skipping inference")
        return cache
    print(f"[{name}] {len(images)-len(todo)} cached, {len(todo)} to run")
    for ip in tqdm(todo, desc=desc, ncols=88):
        im = cv2.imread(str(ip))
        if im is None:
            cache.add({"img": ip.name, "w": 0, "h": 0, "boxes": [], "scores": [], "ms": 0.0})
            continue
        h, w = im.shape[:2]
        try:
            boxes, scores, ms = predict_fn(str(ip))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); gc.collect()
            boxes, scores, ms = predict_fn(str(ip))   # one retry after cache clear
        cache.add({"img": ip.name, "w": w, "h": h,
                   "boxes": np.round(np.asarray(boxes, np.float32), 2).tolist(),
                   "scores": np.round(np.asarray(scores, np.float32), 4).tolist(),
                   "ms": round(float(ms), 2)})
    return cache

# =============================================================================
# 6. METRICS FROM CACHE (all offline -> thresholds applied uniformly)
# =============================================================================
def metrics_from_cache(cache: DetCache, images: List[Path], lbl_dir: Path,
                       name: str, mdir: Path, pdir: Path, coco_gt_json: Optional[Path],
                       plots_index: List[str]) -> Dict[str, Any]:
    recs = [cache.done[p.name] for p in images if p.name in cache.done]
    GT, PB, PC = [], [], []
    for p in images:
        r = cache.done.get(p.name)
        if r is None or r["w"] == 0:
            continue
        GT.append(yolo_label_to_xyxy(lbl_dir / (p.stem + ".txt"), r["w"], r["h"]))
        PB.append(np.asarray(r["boxes"], np.float32).reshape(-1, 4))
        PC.append(np.asarray(r["scores"], np.float32))
    head: Dict[str, Any] = {"config": name, "n_images": len(GT)}

    # --- operational block at CONF_OP ---
    tTP = tFP = tFN = 0
    f1s, f2s = [], []
    size_stats = {n: {"tp": 0, "tot": 0} for n, *_ in SIZE_BINS}
    for gt, pb, pc in zip(GT, PB, PC):
        k = pc >= CONF_OP
        tp, fp, fn, gtm, _ = greedy_match(pb[k], pc[k], gt, IOU_OP)
        tTP += tp; tFP += fp; tFN += fn
        p_ = tp / max(tp + fp, 1); r_ = tp / max(tp + fn, 1)
        f1s.append(2*p_*r_ / max(p_+r_, 1e-12)); f2s.append(5*p_*r_ / max(4*p_+r_, 1e-12))
        if gt.shape[0]:
            side = np.sqrt((gt[:, 2]-gt[:, 0]) * (gt[:, 3]-gt[:, 1]))
            for n, lo, hi in SIZE_BINS:
                m = (side >= lo) & (side < hi)
                size_stats[n]["tot"] += int(m.sum()); size_stats[n]["tp"] += int(gtm[m].sum())
    P = tTP / max(tTP + tFP, 1); R = tTP / max(tTP + tFN, 1)
    head.update({
        "precision": round(P, 4), "recall": round(R, 4),
        "F1": round(2*P*R / max(P+R, 1e-12), 4), "F2": round(5*P*R / max(4*P+R, 1e-12), 4),
        "F1_mean_per_image": round(float(np.mean(f1s)), 4) if f1s else None,
        "F2_mean_per_image": round(float(np.mean(f2s)), 4) if f2s else None,
        "confusion_TP_FP_FN": [tTP, tFP, tFN],
        "per_size_recall": {n: (round(s["tp"]/s["tot"], 4) if s["tot"] else None)
                            for n, s in size_stats.items()},
        "per_size_gt_count": {n: s["tot"] for n, s in size_stats.items()},
    })

    # --- latency ---
    ms = [r["ms"] for r in recs if r.get("ms", 0) > 0]
    if ms:
        a = np.asarray(ms)
        head["latency_ms"] = {"mean": round(float(a.mean()), 1),
                              "p50": round(float(np.percentile(a, 50)), 1),
                              "p95": round(float(np.percentile(a, 95)), 1)}

    # --- PR / F1 / F2 vs confidence (grid over the cached score range) ---
    grid = np.round(np.arange(max(FLOOR_CONF, 0.05), 0.96, 0.01), 3)
    rows = []
    for c in grid:
        TP = FP = FN = 0
        for gt, pb, pc in zip(GT, PB, PC):
            k = pc >= c
            tp, fp, fn, _, _ = greedy_match(pb[k], pc[k], gt, IOU_OP)
            TP += tp; FP += fp; FN += fn
        p_ = TP / max(TP+FP, 1); r_ = TP / max(TP+FN, 1)
        rows.append({"conf": float(c), "precision": p_, "recall": r_,
                     "F1": 2*p_*r_ / max(p_+r_, 1e-12), "F2": 5*p_*r_ / max(4*p_+r_, 1e-12),
                     "TP": TP, "FP": FP, "FN": FN})
    pr = pd.DataFrame(rows)
    pr.to_csv(mdir / f"pr_f1_conf_{name}.csv", index=False)
    i1, i2 = int(pr["F1"].idxmax()), int(pr["F2"].idxmax())
    head["opt_thresholds"] = {"OptThr_F1": float(pr.loc[i1, "conf"]), "Best_F1": round(float(pr.loc[i1, "F1"]), 4),
                              "OptThr_F2": float(pr.loc[i2, "conf"]), "Best_F2": round(float(pr.loc[i2, "F2"]), 4)}
    try:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(pr["recall"], pr["precision"]); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title(f"PR ({name})")
        fig.tight_layout(); fig.savefig(pdir / f"pr_curve_{name}.png", dpi=300); plt.close(fig)
        plots_index.append(f"pr_curve_{name}.png <- metrics/pr_f1_conf_{name}.csv (precision,recall) -- {Path(__file__).name}")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(pr["conf"], pr["F1"], label="F1"); ax.plot(pr["conf"], pr["F2"], label="F2")
        ax.set_xlabel("confidence"); ax.set_title(f"F1/F2 vs conf ({name})"); ax.legend()
        fig.tight_layout(); fig.savefig(pdir / f"f1f2_vs_conf_{name}.png", dpi=300); plt.close(fig)
        plots_index.append(f"f1f2_vs_conf_{name}.png <- metrics/pr_f1_conf_{name}.csv (F1,F2) -- {Path(__file__).name}")
    except Exception as e:
        print(f"[plots] {name} curve plots failed: {e}")

    # --- confidence histogram ---
    allc = np.concatenate(PC) if PC else np.array([])
    hist, edges = np.histogram(allc, bins=50, range=(0, 1))
    pd.DataFrame({"bin_left": edges[:-1], "bin_right": edges[1:], "count": hist}) \
        .to_csv(mdir / f"conf_hist_{name}.csv", index=False)

    # --- calibration (ECE / MCE / Brier) at IOU_OP over all cached dets ---
    cf, ok = [], []
    for gt, pb, pc in zip(GT, PB, PC):
        if pb.shape[0] == 0:
            continue
        _, _, _, _, pok = greedy_match(pb, pc, gt, IOU_OP)
        cf += list(pc); ok += list(pok.astype(float))
    if cf:
        cfa, oka = np.asarray(cf), np.asarray(ok)
        edges = np.linspace(0, 1, 11); ece = mce = 0.0; rows = []
        for i in range(10):
            m = (cfa >= edges[i]) & (cfa < edges[i+1] + (1e-9 if i == 9 else 0))
            n = int(m.sum())
            if n == 0:
                rows.append({"bin_lo": edges[i], "bin_hi": edges[i+1], "count": 0}); continue
            gap = abs(float(cfa[m].mean()) - float(oka[m].mean()))
            ece += (n/len(cfa))*gap; mce = max(mce, gap)
            rows.append({"bin_lo": edges[i], "bin_hi": edges[i+1], "count": n,
                         "mean_conf": float(cfa[m].mean()), "mean_acc": float(oka[m].mean()), "gap": gap})
        pd.DataFrame(rows).to_csv(mdir / f"calibration_{name}.csv", index=False)
        head["calibration"] = {"ECE": round(ece, 4), "MCE": round(mce, 4),
                               "Brier": round(float(np.mean((cfa-oka)**2)), 4)}

    # --- COCO AP (+AR@1/10/100, AP_small/med/large) from the cached detections ---
    if HAS_PYCOCO and coco_gt_json and coco_gt_json.is_file():
        try:
            gtc = COCO(str(coco_gt_json))
            f2i = {im["file_name"]: im["id"] for im in gtc.loadImgs(gtc.getImgIds())}
            dets = []
            for p in images:
                r = cache.done.get(p.name)
                if r is None or p.name not in f2i:
                    continue
                for (x1, y1, x2, y2), s in zip(r["boxes"], r["scores"]):
                    dets.append({"image_id": int(f2i[p.name]), "category_id": 1,
                                 "bbox": [x1, y1, x2-x1, y2-y1], "score": float(s)})
            if dets:
                dp = mdir / f"coco_dets_{name}.json"; dp.write_text(json.dumps(dets))
                ev = COCOeval(gtc, gtc.loadRes(str(dp)), "bbox")
                ev.evaluate(); ev.accumulate(); ev.summarize()
                s = ev.stats
                head["coco"] = {"AP": round(float(s[0]), 4), "AP50": round(float(s[1]), 4),
                                "AP75": round(float(s[2]), 4), "AP_small": round(float(s[3]), 4),
                                "AP_medium": round(float(s[4]), 4), "AP_large": round(float(s[5]), 4),
                                "AR@1": round(float(s[6]), 4), "AR@10": round(float(s[7]), 4),
                                "AR@100": round(float(s[8]), 4)}
        except Exception as e:
            print(f"[coco] {name} failed: {e}")

    (mdir / f"metrics_{name}.json").write_text(json.dumps(head, indent=2))
    print(f"[{name}] P={head['precision']} R={head['recall']} F1={head['F1']} F2={head['F2']} "
          f"| vt_recall={head['per_size_recall'].get('very_tiny')} "
          f"| AP={head.get('coco', {}).get('AP', 'n/a')} | lat={head.get('latency_ms', {}).get('mean', '?')}ms")
    return head

def build_coco_gt(images: List[Path], lbl_dir: Path, out_json: Path) -> Optional[Path]:
    if out_json.is_file():
        return out_json
    ims, anns, aid = [], [], 1
    for iid, ip in enumerate(images, 1):
        im = cv2.imread(str(ip))
        if im is None:
            continue
        h, w = im.shape[:2]
        ims.append({"id": iid, "file_name": ip.name, "width": w, "height": h})
        for (x1, y1, x2, y2) in yolo_label_to_xyxy(lbl_dir / (ip.stem + ".txt"), w, h):
            anns.append({"id": aid, "image_id": iid, "category_id": 1, "iscrowd": 0,
                         "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                         "area": float((x2-x1)*(y2-y1))}); aid += 1
    if not ims:
        return None
    out_json.write_text(json.dumps({"images": ims, "annotations": anns,
                                    "categories": [{"id": 1, "name": "person"}]}))
    return out_json

# =============================================================================
# 7. PREDICTORS (baseline / SAHI / TTA / SAHI+TTA)
# =============================================================================
def make_baseline_predictor(model):
    def f(img_path):
        t0 = time.perf_counter()
        r = model.predict(img_path, conf=FLOOR_CONF, imgsz=640, verbose=False)[0].boxes
        ms = (time.perf_counter()-t0)*1000
        if r is None or len(r) == 0:
            return np.zeros((0, 4)), np.zeros((0,)), ms
        return r.xyxy.cpu().numpy(), r.conf.cpu().numpy(), ms
    return f

def make_tta_predictor(model, imgsz):
    def f(img_path):
        t0 = time.perf_counter()
        r = model.predict(img_path, conf=FLOOR_CONF, imgsz=imgsz, augment=True, verbose=False)[0].boxes
        ms = (time.perf_counter()-t0)*1000
        if r is None or len(r) == 0:
            return np.zeros((0, 4)), np.zeros((0,)), ms
        return r.xyxy.cpu().numpy(), r.conf.cpu().numpy(), ms
    return f

def make_sahi_predictor(sahi_model, slice_px, overlap):
    from sahi.predict import get_sliced_prediction
    def f(img_path):
        t0 = time.perf_counter()
        res = get_sliced_prediction(img_path, sahi_model,
                                    slice_height=slice_px, slice_width=slice_px,
                                    overlap_height_ratio=overlap, overlap_width_ratio=overlap,
                                    verbose=0, **SAHI_POSTPROCESS)
        ms = (time.perf_counter()-t0)*1000
        boxes, scores = [], []
        for o in res.object_prediction_list:
            bb = o.bbox
            boxes.append([bb.minx, bb.miny, bb.maxx, bb.maxy]); scores.append(o.score.value)
        return (np.asarray(boxes, np.float32).reshape(-1, 4),
                np.asarray(scores, np.float32), ms)
    return f

class SahiTTAPatch:
    """Context manager: injects augment=True into every prediction SAHI makes (per-tile TTA)."""
    def __enter__(self):
        from sahi.models.ultralytics import UltralyticsDetectionModel
        self.cls = UltralyticsDetectionModel
        self.orig = UltralyticsDetectionModel.perform_inference
        orig = self.orig
        def patched(mself, image):
            oc = mself.model.__call__
            def aug_call(*a, **k):
                k["augment"] = True
                return oc(*a, **k)
            mself.model.__call__ = aug_call
            try:
                orig(mself, image)
            finally:
                mself.model.__call__ = oc
        UltralyticsDetectionModel.perform_inference = patched
        return self
    def __exit__(self, *exc):
        self.cls.perform_inference = self.orig
        return False

# =============================================================================
# 8. OFFICIAL ULTRALYTICS VAL (baseline + TTA rows -- spec-comparable mAP)
# =============================================================================
def official_val(best_pt: str, data_yaml: str, imgsz: int, augment: bool, batch: int,
                 label: str) -> Optional[Dict[str, float]]:
    try:
        m = YOLO(best_pt)
        r = m.val(data=data_yaml, split="test", imgsz=imgsz, augment=augment,
                  batch=batch, conf=CONF_AP, iou=IOU_AP, device=0, verbose=False, plots=False)
        out = {"mAP50": round(float(r.box.map50), 4), "mAP50-95": round(float(r.box.map), 4),
               "precision": round(float(r.box.mp), 4), "recall": round(float(r.box.mr), 4)}
        del m; gc.collect(); torch.cuda.empty_cache()
        print(f"[val:{label}] mAP50={out['mAP50']} mAP50-95={out['mAP50-95']}")
        return out
    except torch.cuda.OutOfMemoryError:
        print(f"[val:{label}] OOM -- skipped"); gc.collect(); torch.cuda.empty_cache()
        return None
    except Exception as e:
        print(f"[val:{label}] failed: {e}")
        return None

# =============================================================================
# 9. QUALITATIVE SAMPLES (spec Sec 6 -- and raw material for the report figures)
# =============================================================================
def save_qualitative(images: List[Path], lbl_dir: Path, caches: Dict[str, DetCache],
                     qdir: Path, n: int):
    # deterministic pick: densest, sparsest and evenly-spaced median-GT images
    counts = []
    for p in images:
        im = cv2.imread(str(p))
        if im is None:
            continue
        h, w = im.shape[:2]
        counts.append((p, yolo_label_to_xyxy(lbl_dir / (p.stem + ".txt"), w, h).shape[0]))
    counts.sort(key=lambda t: -t[1])
    picks = [c[0] for c in counts[:max(2, n//3)]]                       # densest
    picks += [c[0] for c in counts[len(counts)//2: len(counts)//2 + max(2, n//3)]]  # median
    picks += [c[0] for c in counts[-max(2, n - len(picks)):]]           # sparsest
    picks = picks[:n]
    colors = {"baseline_640": (255, 160, 0), "GT": (0, 220, 0)}
    for p in picks:
        im = cv2.imread(str(p))
        if im is None:
            continue
        h, w = im.shape[:2]
        gt = yolo_label_to_xyxy(lbl_dir / (p.stem + ".txt"), w, h)
        panels = []
        gtp = im.copy()
        for (x1, y1, x2, y2) in gt.astype(int):
            cv2.rectangle(gtp, (x1, y1), (x2, y2), colors["GT"], 2)
        cv2.putText(gtp, f"GT ({gt.shape[0]})", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors["GT"], 2)
        panels.append(gtp)
        for cname, cache in caches.items():
            r = cache.done.get(p.name)
            if r is None:
                continue
            pb = np.asarray(r["boxes"], np.float32).reshape(-1, 4)
            pc = np.asarray(r["scores"], np.float32)
            k = pc >= CONF_OP
            pan = im.copy()
            for (x1, y1, x2, y2) in pb[k].astype(int):
                cv2.rectangle(pan, (x1, y1), (x2, y2), (0, 128, 255), 2)
            cv2.putText(pan, f"{cname} ({int(k.sum())})", (8, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 255), 2)
            panels.append(pan)
            cv2.imwrite(str(qdir / f"{p.stem}_{cname}.jpg"), pan, [cv2.IMWRITE_JPEG_QUALITY, 92])
        strip = cv2.hconcat([cv2.resize(x, (min(640, w), int(min(640, w)*h/w))) for x in panels])
        cv2.imwrite(str(qdir / f"{p.stem}_COMPARE.jpg"), strip, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[qual] wrote {len(picks)} sample sets -> {qdir}")

# =============================================================================
# 10. MAIN
# =============================================================================
def main():
    print("=" * 78 + f"\nSAHI + TTA ABLATION -- {MODEL_TAG} (spec Sec 5/6)\n" + "=" * 78)
    c2a = find_c2a_root()
    best_pt = find_best_pt()
    img_dir = c2a / SPLIT / "images"; lbl_dir = c2a / SPLIT / "labels"
    all_images = list_images(img_dir)
    print(f"[data]  C2A={c2a}  split={SPLIT}  n={len(all_images)}")
    print(f"[model] {best_pt}  ({round(best_pt.stat().st_size/1024**2,1)} MB)")

    # ---- resume-aware run dir (newest without DONE marker) ----
    rdir = None
    for d in sorted([p for p in RUNS_DIR.iterdir() if p.is_dir() and RUN_TAG in p.name],
                    key=lambda x: x.name, reverse=True):
        if not (d / "metrics" / "grand_summary.json").is_file():
            rdir = d; print(f"[resume] continuing incomplete run {d.name}"); break
    if rdir is None:
        rdir = RUNS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{RUN_TAG}"
    mdir = rdir / "metrics"; pdir = rdir / "plots"; cdir = rdir / "cache"; qdir = rdir / "qualitative"
    for d in (mdir, pdir, cdir, qdir):
        d.mkdir(parents=True, exist_ok=True)
    plots_index: List[str] = []
    skipped: List[str] = []

    # ---- env.json (spec Sec 6 reproducibility) ----
    import platform
    def _ver(m):
        try:
            import importlib; return getattr(importlib.import_module(m), "__version__", None)
        except Exception:
            return None
    (rdir / "env.json").write_text(json.dumps({
        "run_tag": RUN_TAG, "model_tag": MODEL_TAG, "weights": str(best_pt),
        "weights_md5": _md5(best_pt), "python": platform.python_version(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "ultralytics": _ver("ultralytics"), "sahi": _ver("sahi"), "numpy": _ver("numpy"),
        "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
        "protocol": {"CONF_OP": CONF_OP, "IOU_OP": IOU_OP, "CONF_AP": CONF_AP, "IOU_AP": IOU_AP,
                     "FLOOR_CONF": FLOOR_CONF, "split": SPLIT, "sahi_postprocess": SAHI_POSTPROCESS},
        "c2a_root": str(c2a), "timestamp": datetime.now().isoformat(timespec="seconds")},
        indent=2, default=str))

    # ---- data yaml for official val() ----
    data_yaml = rdir / "c2a_eval.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(c2a).replace("\\", "/"), "train": "train/images",
        "val": "val/images", "test": f"{SPLIT}/images",
        "names": {0: "person"}, "nc": 1}, sort_keys=False))

    # ---- COCO GT (built once) ----
    coco_gt = build_coco_gt(all_images, lbl_dir, rdir / f"c2a_{SPLIT}_coco_gt.json") if HAS_PYCOCO \
        else None
    if not HAS_PYCOCO:
        skipped.append("[SKIPPED] COCO AP/AR -- pycocotools unavailable")

    # =========================================================================
    # SMOKE (spec Sec 9): 10 images through EVERY stage, then continue to full.
    # =========================================================================
    smoke_marker = rdir / ".smoke_passed"
    if not smoke_marker.is_file():
        print("\n" + "-" * 70 + "\n[smoke] 10-image dry pass through every stage\n" + "-" * 70)
        sm_imgs = all_images[:SMOKE_N]
        sm_cache_dir = rdir / "cache_smoke"; sm_cache_dir.mkdir(exist_ok=True)
        model = YOLO(str(best_pt)); model.to(DEVICE)
        cached_sweep("smoke_baseline", sm_imgs, lbl_dir, sm_cache_dir,
                     make_baseline_predictor(model), "smoke-baseline")
        del model; gc.collect(); torch.cuda.empty_cache()
        from sahi import AutoDetectionModel
        sm_sahi = AutoDetectionModel.from_pretrained(model_type="ultralytics",
                                                     model_path=str(best_pt),
                                                     confidence_threshold=FLOOR_CONF, device=DEVICE)
        cached_sweep("smoke_sahi", sm_imgs, lbl_dir, sm_cache_dir,
                     make_sahi_predictor(sm_sahi, 320, 0.25), "smoke-sahi")
        with SahiTTAPatch():
            cached_sweep("smoke_sahi_tta", sm_imgs, lbl_dir, sm_cache_dir,
                         make_sahi_predictor(sm_sahi, 320, 0.25), "smoke-sahi+tta")
        del sm_sahi; gc.collect(); torch.cuda.empty_cache()
        ov = official_val(str(best_pt), str(data_yaml), 640, False, 8, "smoke-val")
        if ov is None:
            raise RuntimeError("[smoke] official val failed -- aborting before the full run")
        shutil.rmtree(sm_cache_dir, ignore_errors=True)
        smoke_marker.write_text(datetime.now().isoformat())
        print("[smoke] PASSED -- continuing to the full run\n")
    else:
        print("[smoke] marker present -- already passed for this run dir")

    grand: Dict[str, Any] = {}
    partial = mdir / "grand_summary.partial.json"
    if partial.is_file():          # resume: restores _tta_official so val() reruns are skipped
        try:
            grand = json.loads(partial.read_text())
            print(f"[resume] restored partial summary ({len(grand)} entries)")
        except Exception:
            grand = {}
    caches_for_qual: Dict[str, DetCache] = {}

    # =========================================================================
    # ROW 0 -- BASELINE (640, no SAHI, no TTA)
    # =========================================================================
    print("\n" + "=" * 70 + "\nROW 0: BASELINE (640, standard)\n" + "=" * 70)
    if not (mdir / "metrics_baseline_640.json").is_file():
        model = YOLO(str(best_pt)); model.to(DEVICE)
        cache = cached_sweep("baseline_640", all_images, lbl_dir, cdir,
                             make_baseline_predictor(model), "baseline")
        del model; gc.collect(); torch.cuda.empty_cache()
        head = metrics_from_cache(cache, all_images, lbl_dir, "baseline_640",
                                  mdir, pdir, coco_gt, plots_index)
    else:
        head = json.loads((mdir / "metrics_baseline_640.json").read_text())
        cache = DetCache(cdir / "baseline_640.jsonl")
        print("[baseline_640] metrics exist -- skipping")
    ov = official_val(str(best_pt), str(data_yaml), 640, False, 8, "baseline_640") \
        if "official_val" not in head else head["official_val"]
    if ov:
        head["official_val"] = ov
        (mdir / "metrics_baseline_640.json").write_text(json.dumps(head, indent=2))
    grand["baseline_640"] = head
    caches_for_qual["baseline_640"] = cache

    # =========================================================================
    # SAHI SWEEP
    # =========================================================================
    print("\n" + "=" * 70 + "\nSAHI SWEEP (GREEDYNMM + IOS + standard-pred)\n" + "=" * 70)
    from sahi import AutoDetectionModel
    sahi_model = None
    for cfg in SAHI_CONFIGS:
        name = cfg["name"]
        if (mdir / f"metrics_{name}.json").is_file():
            grand[name] = json.loads((mdir / f"metrics_{name}.json").read_text())
            print(f"[{name}] metrics exist -- skipping")
            continue
        if sahi_model is None:
            sahi_model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics", model_path=str(best_pt),
                confidence_threshold=FLOOR_CONF, device=DEVICE)
        cache = cached_sweep(name, all_images, lbl_dir, cdir,
                             make_sahi_predictor(sahi_model, cfg["slice"], cfg["overlap"]), name)
        head = metrics_from_cache(cache, all_images, lbl_dir, name, mdir, pdir, coco_gt, plots_index)
        head["sahi_config"] = {**cfg, **SAHI_POSTPROCESS, "floor_conf": FLOOR_CONF}
        (mdir / f"metrics_{name}.json").write_text(json.dumps(head, indent=2))
        grand[name] = head
        _write_partial(grand, mdir)      # <- intermediate grand summary after every config

    best_sahi_name = max([c["name"] for c in SAHI_CONFIGS],
                         key=lambda n: grand[n]["per_size_recall"].get("very_tiny") or 0)
    best_cfg = next(c for c in SAHI_CONFIGS if c["name"] == best_sahi_name)
    print(f"\n[sahi] BEST config by very-tiny recall: {best_sahi_name} "
          f"(vt={grand[best_sahi_name]['per_size_recall'].get('very_tiny')})")
    caches_for_qual[best_sahi_name] = DetCache(cdir / f"{best_sahi_name}.jsonl")

    # =========================================================================
    # TTA (official val at each size; per-image custom pass at the best size)
    # =========================================================================
    print("\n" + "=" * 70 + "\nTTA (ultralytics augment=True)\n" + "=" * 70)
    tta_official = grand.get("_tta_official", {})
    for sz in TTA_IMG_SIZES:
        key = f"tta_{sz}"
        if key in tta_official:
            continue
        ov = official_val(str(best_pt), str(data_yaml), sz, True, TTA_VAL_BATCH.get(sz, 1), key)
        if ov is None:
            skipped.append(f"[SKIPPED] {key} official mAP -- OOM/failure on this GPU")
        tta_official[key] = ov
        grand["_tta_official"] = tta_official
        _write_partial(grand, mdir)
    ok_tta = {k: v for k, v in tta_official.items() if v}
    best_tta_key = max(ok_tta, key=lambda k: ok_tta[k]["mAP50-95"]) if ok_tta else None
    if best_tta_key:
        best_tta_sz = int(best_tta_key.split("_")[1])
        print(f"[tta] BEST size by mAP50-95: {best_tta_sz}")
        name = f"tta_{best_tta_sz}_custom"
        if not (mdir / f"metrics_{name}.json").is_file():
            model = YOLO(str(best_pt)); model.to(DEVICE)
            cache = cached_sweep(name, all_images, lbl_dir, cdir,
                                 make_tta_predictor(model, best_tta_sz), name)
            del model; gc.collect(); torch.cuda.empty_cache()
            head = metrics_from_cache(cache, all_images, lbl_dir, name, mdir, pdir, coco_gt, plots_index)
            head["official_val"] = ok_tta[best_tta_key]
            (mdir / f"metrics_{name}.json").write_text(json.dumps(head, indent=2))
            grand[name] = head
        else:
            grand[name] = json.loads((mdir / f"metrics_{name}.json").read_text())
        _write_partial(grand, mdir)

    # =========================================================================
    # SAHI + TTA (best SAHI config, TTA per tile)
    # =========================================================================
    print("\n" + "=" * 70 + f"\nSAHI + TTA COMBINED ({best_sahi_name} + augment)\n" + "=" * 70)
    combo_name = f"sahi_tta_{best_cfg['slice']}"
    if not (mdir / f"metrics_{combo_name}.json").is_file():
        if sahi_model is None:
            sahi_model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics", model_path=str(best_pt),
                confidence_threshold=FLOOR_CONF, device=DEVICE)
        with SahiTTAPatch():
            cache = cached_sweep(combo_name, all_images, lbl_dir, cdir,
                                 make_sahi_predictor(sahi_model, best_cfg["slice"],
                                                     best_cfg["overlap"]), combo_name)
        head = metrics_from_cache(cache, all_images, lbl_dir, combo_name, mdir, pdir,
                                  coco_gt, plots_index)
        head["sahi_config"] = {**best_cfg, **SAHI_POSTPROCESS, "tta": True, "floor_conf": FLOOR_CONF}
        (mdir / f"metrics_{combo_name}.json").write_text(json.dumps(head, indent=2))
        grand[combo_name] = head
    else:
        grand[combo_name] = json.loads((mdir / f"metrics_{combo_name}.json").read_text())
    caches_for_qual[combo_name] = DetCache(cdir / f"{combo_name}.jsonl")
    if sahi_model is not None:
        del sahi_model; gc.collect(); torch.cuda.empty_cache()

    # =========================================================================
    # QUALITATIVE + CROSS-CONFIG PLOTS + GRAND SUMMARY
    # =========================================================================
    try:
        save_qualitative(all_images, lbl_dir, caches_for_qual, qdir, QUALITATIVE_N)
    except Exception as e:
        print(f"[qual] failed (non-fatal): {e}")
        skipped.append(f"[SKIPPED] qualitative grids -- {e}")

    # per-size recall grouped bars across configs (the report's SAHI money-plot)
    try:
        cfgs = [k for k in grand if not k.startswith("_")]
        fig, ax = plt.subplots(figsize=(14, 6))
        cats = [n for n, *_ in SIZE_BINS][:4]
        x = np.arange(len(cats)); bw = 0.8 / len(cfgs)
        for i, k in enumerate(cfgs):
            vals = [grand[k]["per_size_recall"].get(c) or 0 for c in cats]
            ax.bar(x + i*bw, vals, bw, label=k)
        ax.set_xticks(x + bw*(len(cfgs)-1)/2); ax.set_xticklabels(cats)
        ax.set_ylabel("Recall"); ax.set_ylim(0, 1.05)
        ax.set_title(f"Per-size recall -- {MODEL_TAG}: baseline vs SAHI vs TTA vs SAHI+TTA ({SPLIT})")
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(pdir / "per_size_recall_all_configs.png", dpi=300)
        plt.close(fig)
        plots_index.append("per_size_recall_all_configs.png <- metrics/metrics_*.json (per_size_recall) -- "
                           + Path(__file__).name)
    except Exception as e:
        print(f"[plots] cross-config bar failed: {e}")

    # grand summary: json + xlsx + markdown
    _write_partial(grand, mdir)                     # final json
    rows = []
    for k, v in grand.items():
        if k.startswith("_"):
            continue
        rows.append({
            "Configuration": k,
            "mAP50 (val())": (v.get("official_val") or {}).get("mAP50", "-"),
            "mAP50-95 (val())": (v.get("official_val") or {}).get("mAP50-95", "-"),
            "COCO_AP": (v.get("coco") or {}).get("AP", "-"),
            "COCO_AP50": (v.get("coco") or {}).get("AP50", "-"),
            "AP_small": (v.get("coco") or {}).get("AP_small", "-"),
            "AR@100": (v.get("coco") or {}).get("AR@100", "-"),
            "P": v.get("precision"), "R": v.get("recall"),
            "F1": v.get("F1"), "F2": v.get("F2"),
            "vt_recall": v["per_size_recall"].get("very_tiny"),
            "tiny_recall": v["per_size_recall"].get("tiny"),
            "small_recall": v["per_size_recall"].get("small"),
            "medium_recall": v["per_size_recall"].get("medium"),
            "OptThr_F2": (v.get("opt_thresholds") or {}).get("OptThr_F2", "-"),
            "Best_F2": (v.get("opt_thresholds") or {}).get("Best_F2", "-"),
            "lat_mean_ms": (v.get("latency_ms") or {}).get("mean", "-"),
            "lat_p95_ms": (v.get("latency_ms") or {}).get("p95", "-"),
        })
    for k, v in (grand.get("_tta_official") or {}).items():
        if v and not any(r["Configuration"].startswith(k) for r in rows):
            rows.append({"Configuration": f"{k} (val() only)", "mAP50 (val())": v["mAP50"],
                         "mAP50-95 (val())": v["mAP50-95"], "P": v["precision"], "R": v["recall"]})
    df = pd.DataFrame(rows)
    df.to_excel(mdir / "grand_summary.xlsx", index=False)
    md = ["# SAHI + TTA ablation -- " + MODEL_TAG,
          f"weights: `{best_pt}` | split: {SPLIT} | {datetime.now().isoformat(timespec='seconds')}",
          "", "> Sec 5 footnotes: SAHI rows use the per-image protocol (val() mAP not computable "
          f"through SAHI); SAHI COCO AP lower-bounded by the {FLOOR_CONF} collection floor; "
          "TTA rows are official ultralytics val(augment=True).",
          "", _md_table(df)]
    (mdir / "grand_summary.md").write_text("\n".join(md), encoding="utf-8")
    (pdir / "PLOTS_INDEX.md").write_text("\n".join(plots_index) + "\n", encoding="utf-8")
    (rdir / "skipped_metrics.txt").write_text("\n".join(skipped) + "\n" if skipped else "none\n")
    (rdir / "manifest.json").write_text(json.dumps({
        "run_tag": RUN_TAG, "model_tag": MODEL_TAG, "weights": str(best_pt),
        "summary": "metrics/grand_summary.json", "table": "metrics/grand_summary.xlsx",
        "completed": datetime.now().isoformat(timespec="seconds")}, indent=2))

    print("\n" + "=" * 78 + "\nGRAND SUMMARY\n" + "=" * 78)
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(df.to_string(index=False))
    print(f"\n[done] everything under: {rdir}")
    print("       table: metrics/grand_summary.xlsx | md: metrics/grand_summary.md "
          "| plots: plots/ | samples: qualitative/")

def _md_table(df) -> str:
    """GitHub-markdown table if 'tabulate' is available, else a fenced plain table (never fails)."""
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```\n" + df.to_string(index=False) + "\n```"

def _write_partial(grand: dict, mdir: Path):
    (mdir / "grand_summary.partial.json").write_text(json.dumps(grand, indent=2, default=str))

def _finalize_marker(mdir: Path, grand: dict):
    (mdir / "grand_summary.json").write_text(json.dumps(grand, indent=2, default=str))

if __name__ == "__main__":
    try:
        main()
        # promote the partial to the DONE marker
        p = None
        for d in sorted([x for x in RUNS_DIR.iterdir() if x.is_dir()], reverse=True):
            if (d / "metrics" / "grand_summary.partial.json").is_file():
                p = d / "metrics"; break
        if p is not None and not (p / "grand_summary.json").is_file():
            shutil.copy2(p / "grand_summary.partial.json", p / "grand_summary.json")
    except KeyboardInterrupt:
        print("\n[interrupt] state saved in cache/ -- re-run the same command to resume.")
