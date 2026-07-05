"""
02_nwd_loss_patch.py -- Normalized Gaussian Wasserstein Distance hybrid loss
for Ultralytics YOLO (anchor-free v8/v11 head), version-defensive.

Loss becomes:   L_box = (1 - alpha) * (1 - CIoU) + alpha * (1 - NWD)
  NWD(a,b) = exp( -W2(a,b) / C ),
  W2^2     = (dcx)^2 + (dcy)^2 + ((w1-w2)^2 + (h1-h2)^2) / 4      (Wang et al.,
             arXiv 2110.13389; C is a dataset-scale constant, ~12.8 for AI-TOD.
             C2A median box is 12.0 px -> default C=12.8 is the right regime.)

Why a monkey-patch and not a fork: the thesis protocol scripts rebuild models
internally; a patch at `ultralytics.utils.loss.bbox_iou` touches EXACTLY the
box-loss call and nothing else (the assigner imports its own copy from
ultralytics.utils.metrics, which we deliberately do NOT touch in v1).

HARD LESSON FROM THE MAMBA BUG: silent no-ops are the enemy. This module
therefore (1) inspects BboxLoss.forward source to confirm the patched symbol is
actually used, (2) counts real invocations at train time, and (3) ships a
callback that ABORTS training after batch 1 if the patch never fired.

Modes:
  --selftest          pure-math + patch-mechanics tests (any machine, CPU, <10 s)
  --train ...         thin fine-tune launcher with the patch applied + verified
                      (for the NWD pilot; full-protocol retrains keep using the
                      thesis script -- see RUNNING_GUIDE.md)

Example (PC-4 pilot):
  python 02_nwd_loss_patch.py --train --weights D:/thesis_2007074/epoch125.pt ^
      --data <scenesplit>/data.yaml --alpha 0.5 --epochs 25 --lr0 0.0003 ^
      --batch 4 --no-amp --project runs_nwd --name pilot_a05
"""
from __future__ import annotations
import argparse, importlib.util, inspect, sys
from pathlib import Path

_STATE = {"applied": False, "calls": 0, "alpha": None, "C": None, "orig": None}


# ------------------------------------------------------- shared-GPU pinning
# Ported from the PC-2 joint script (TC-02/TC-08 tested there): the A6000 box is
# SHARED -- GPU 0 belongs to the other user, GPU 1 is ours. Pick a FREE GPU
# (<1 GB used AND <10% util) BEFORE torch import and pin it via
# CUDA_VISIBLE_DEVICES so device "0" in-process == the picked physical GPU and
# the other user's GPU is invisible. Never auto-grab a busy GPU. On single-GPU
# boxes this resolves to GPU 0. Manual override: set CUDA_VISIBLE_DEVICES
# yourself before launching (then this is a no-op). Spawned DataLoader workers
# inherit the sentinel and skip re-picking.
def pin_free_gpu(free_mem_mb: float = 1024, free_util_pct: float = 10) -> None:
    import os, subprocess
    if os.environ.get("_NLAP_GPU_PINNED") == "1" or os.environ.get("CUDA_VISIBLE_DEVICES"):
        return
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL, timeout=15
        ).decode("utf-8", "ignore")
        gpus = []
        for line in out.strip().splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) >= 4:
                gpus.append({"index": int(p[0]), "mem_used": float(p[1]), "util": float(p[3])})
    except Exception:
        print("[gpu] nvidia-smi unavailable -> leaving device selection to torch defaults")
        return
    if not gpus:
        return
    assigned = 1 if len(gpus) > 1 else 0          # PC-2 convention: GPU 1 is ours
    snap = ", ".join(f"GPU{g['index']}:{int(g['mem_used'])}MB/{int(g['util'])}%" for g in gpus)
    free = [g["index"] for g in gpus if g["mem_used"] < free_mem_mb and g["util"] < free_util_pct]
    if free:
        chosen = assigned if assigned in free else min(free)
        print(f"[gpu] usage [{snap}] -> free: {free} -> picking GPU {chosen}")
    else:
        chosen = assigned
        print(f"[gpu] WARNING: no fully-free GPU [{snap}] -> staying on assigned GPU {chosen} "
              f"(shared right now; consider waiting).")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(chosen)
    os.environ["_NLAP_GPU_PINNED"] = "1"
    print(f"[gpu] CUDA_VISIBLE_DEVICES={chosen} (in-process this is cuda:0; other GPUs invisible)")



# ------------------------------------------------------- checkpoint compat
# epoch125.pt / c2a_cbam_p2head_best.pt pickle the thesis' custom CBAM classes
# (saved from a script running as __main__). torch.load can only deserialize
# them if identically-named classes are resolvable from BOTH __main__ and the
# ultralytics namespaces. These are VERBATIM copies from
# `Last Month/deployable_model/joint_c2a_sard_train.py` (the same trick
# finetune_enriched.py uses). Do not "improve" them -- attribute names must
# match the pickled instances exactly.
def ensure_cbam_compat(compat_file: str | None = None) -> None:
    import torch
    import torch.nn as nn

    if compat_file:  # user-supplied module wins (e.g. future Mamba ckpts)
        spec = importlib.util.spec_from_file_location("ckpt_compat", compat_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        classes = [getattr(mod, n) for n in ("CBAM", "ChannelAttention", "SpatialAttention")
                   if hasattr(mod, n)]
    else:
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
        classes = [CBAM, ChannelAttention, SpatialAttention]

    import ultralytics.nn.modules as _m
    import ultralytics.nn.tasks as _t
    main_mod = sys.modules.get("__main__")
    for cls in classes:
        setattr(_m, cls.__name__, cls)
        setattr(_t, cls.__name__, cls)
        if main_mod is not None:
            setattr(main_mod, cls.__name__, cls)   # pickle looks in __main__
        globals()[cls.__name__] = cls
    try:  # torch>=2.6 weights_only allowlist (harmless if unavailable)
        torch.serialization.add_safe_globals(classes)
    except Exception:
        pass
    print(f"[compat] registered {[c.__name__ for c in classes]} into ultralytics + __main__")


# ----------------------------------------------------------------- NWD math
def nwd_similarity(box1, box2, xywh: bool = True, C: float = 12.8, eps: float = 1e-7):
    """NWD similarity in (0,1]. Shapes follow ultralytics bbox_iou conventions:
    box1/box2 (..., 4); returns (..., 1) to match bbox_iou's keepdim output."""
    import torch
    b1 = box1 if isinstance(box1, torch.Tensor) else torch.as_tensor(box1, dtype=torch.float32)
    b2 = box2 if isinstance(box2, torch.Tensor) else torch.as_tensor(box2, dtype=torch.float32)
    if xywh:
        cx1, cy1, w1, h1 = b1.split(1, -1)
        cx2, cy2, w2, h2 = b2.split(1, -1)
    else:
        x11, y11, x12, y12 = b1.split(1, -1)
        x21, y21, x22, y22 = b2.split(1, -1)
        w1, h1 = (x12 - x11), (y12 - y11)
        w2, h2 = (x22 - x21), (y22 - y21)
        cx1, cy1 = x11 + w1 / 2, y11 + h1 / 2
        cx2, cy2 = x21 + w2 / 2, y21 + h2 / 2
    w2sq = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2 + ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4.0
    return torch.exp(-torch.sqrt(w2sq.clamp(min=0) + eps) / C)


# ------------------------------------------------------------- patch machinery
def apply_nwd_patch(alpha: float = 0.5, C: float = 12.8) -> None:
    """Blend NWD into the CIoU call inside ultralytics' BboxLoss. Idempotent."""
    if _STATE["applied"]:
        _STATE["alpha"], _STATE["C"] = alpha, C
        return
    import ultralytics
    import ultralytics.utils.loss as UL

    if not hasattr(UL, "bbox_iou"):
        raise RuntimeError(
            f"[nwd] ultralytics {ultralytics.__version__}: ultralytics.utils.loss "
            f"has no `bbox_iou` symbol -- loss internals changed; DO NOT TRAIN. "
            f"Adapt the patch to this version first.")
    src = inspect.getsource(UL.BboxLoss.forward)
    if "bbox_iou(" not in src:
        raise RuntimeError(
            f"[nwd] ultralytics {ultralytics.__version__}: BboxLoss.forward does "
            f"not call bbox_iou() -- patching the symbol would be a SILENT NO-OP. "
            f"DO NOT TRAIN. Adapt the patch first.")

    _STATE["orig"] = UL.bbox_iou
    _STATE["alpha"], _STATE["C"] = float(alpha), float(C)

    def bbox_iou_nwd(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, **kw):
        base = _STATE["orig"](box1, box2, xywh=xywh, GIoU=GIoU, DIoU=DIoU, CIoU=CIoU, **kw)
        if not CIoU:                    # blend only the box-loss CIoU path
            return base
        nwd = nwd_similarity(box1, box2, xywh=xywh, C=_STATE["C"])
        while nwd.dim() < base.dim():
            nwd = nwd.unsqueeze(-1)
        while nwd.dim() > base.dim():
            nwd = nwd.squeeze(-1)
        _STATE["calls"] += 1
        a = _STATE["alpha"]
        return (1.0 - a) * base + a * nwd

    UL.bbox_iou = bbox_iou_nwd
    _STATE["applied"] = True
    print(f"[nwd-patch] ACTIVE  alpha={alpha}  C={C}  "
          f"ultralytics={ultralytics.__version__}  (L = {1-alpha:.2f}*(1-CIoU) + {alpha:.2f}*(1-NWD))")


def remove_nwd_patch() -> None:
    if _STATE["applied"]:
        import ultralytics.utils.loss as UL
        UL.bbox_iou = _STATE["orig"]
        _STATE["applied"] = False
        print("[nwd-patch] removed")


def make_verify_callback():
    """on_train_batch_end callback: ABORT if the patch never fired (mamba-bug
    insurance). Attach with: model.add_callback('on_train_batch_end', cb)."""
    def cb(trainer):
        if getattr(cb, "done", False):
            return
        cb.done = True
        if _STATE["calls"] == 0:
            raise RuntimeError(
                "[nwd-verify] FATAL: first training batch completed but the NWD-"
                "blended bbox_iou was NEVER called. The patch is not in the loss "
                "path (version drift?). Training aborted -- do not trust this run.")
        print(f"[nwd-verify] OK -- NWD blend active in loss path "
              f"(calls after batch 1: {_STATE['calls']})")
    return cb


# ------------------------------------------------------------------ selftest
def _selftest() -> None:
    import torch
    ok = 0
    # T1 identity -> 1 (up to the sqrt-eps used for gradient stability)
    b = torch.tensor([[10., 10., 4., 4.]])
    assert torch.allclose(nwd_similarity(b, b, xywh=True), torch.ones(1, 1), atol=1e-4)
    ok += 1
    # T2 monotonic decay with center distance; positive where IoU is dead-zero
    base = torch.tensor([[0., 0., 4., 4.]])
    d = [nwd_similarity(base, torch.tensor([[dx, 0., 4., 4.]]), xywh=True).item()
         for dx in (0.0, 2.0, 6.0, 20.0)]
    assert all(x > y for x, y in zip(d, d[1:])), f"T2 not monotone {d}"
    assert d[2] > 0.0, "T2 disjoint boxes must keep signal"
    ok += 1
    # T3 gradient flows for fully-disjoint tiny boxes (the IoU-plateau case)
    p = torch.tensor([[8., 0., 12., 4.]], requires_grad=True)     # xyxy
    t = torch.tensor([[0., 0., 4., 4.]])
    nwd_similarity(p, t, xywh=False).sum().backward()
    assert p.grad is not None and p.grad.abs().sum() > 0, "T3 no gradient"
    ok += 1
    # T4 C scaling: larger C -> more tolerant (higher similarity)
    a4 = nwd_similarity(base, torch.tensor([[6., 0., 4., 4.]]), xywh=True, C=6.0).item()
    b4 = nwd_similarity(base, torch.tensor([[6., 0., 4., 4.]]), xywh=True, C=26.0).item()
    assert b4 > a4, "T4 C-monotonicity"
    ok += 1
    # T5 compat registration: classes land in ultralytics namespaces + __main__,
    # and a lazy CBAM instance forwards correctly (deserialization prerequisite)
    try:
        ensure_cbam_compat(None)
        import ultralytics.nn.modules as _m
        import ultralytics.nn.tasks as _t
        assert hasattr(_m, "CBAM") and hasattr(_t, "CBAM") and hasattr(sys.modules["__main__"], "CBAM")
        cb = _m.CBAM(16, 7)
        y = cb(torch.randn(1, 32, 8, 8))
        assert y.shape == (1, 32, 8, 8) and cb._initialized
        ok += 1
    except ImportError:
        pass

    # T6 patch mechanics on the installed ultralytics (whatever version)
    try:
        import ultralytics.utils.loss as UL
        apply_nwd_patch(alpha=0.5, C=12.8)
        before = _STATE["calls"]
        p5 = torch.tensor([[0., 0., 4., 4.], [10., 10., 14., 14.]])
        t5 = torch.tensor([[1., 0., 5., 4.], [10., 10., 14., 14.]])
        out = UL.bbox_iou(p5, t5, xywh=False, CIoU=True)
        assert _STATE["calls"] == before + 1, "T5 patched fn not invoked"
        # blended value must sit between pure CIoU and pure NWD for row 0
        ciou = _STATE["orig"](p5, t5, xywh=False, CIoU=True)
        nwd = nwd_similarity(p5, t5, xywh=False)
        lo = torch.minimum(ciou, nwd)
        hi = torch.maximum(ciou, nwd)
        assert bool(((out >= lo - 1e-6) & (out <= hi + 1e-6)).all()), "T5 blend out of range"
        remove_nwd_patch()
        out2 = UL.bbox_iou(p5, t5, xywh=False, CIoU=True)
        assert torch.allclose(out2, ciou, atol=1e-6), "T6 unpatch failed"
        ok += 1
        print(f"SELFTEST PASSED ({ok}/6 groups, incl. compat-registration and "
              f"live-patch on installed ultralytics)")
    except ImportError:
        print(f"SELFTEST PASSED ({ok}/4 math groups; ultralytics not installed here, "
              f"patch/compat mechanics will be verified by --check-load + the training callback)")


# ------------------------------------------------------------------ launcher
def _check_load(args) -> None:
    """De-risk step: verify the checkpoint deserializes on THIS machine
    before spending any training time. Run once per PC."""
    pin_free_gpu()
    ensure_cbam_compat(args.compat)
    from ultralytics import YOLO
    model = YOLO(args.weights)
    n = sum(p.numel() for p in model.model.parameters())
    assert n > 1e6, f"suspiciously few params: {n}"
    has_cbam = any(type(m).__name__ == "CBAM" for m in model.model.modules())
    print(f"[check-load] OK: {args.weights}")
    print(f"[check-load] params={n/1e6:.2f}M  CBAM_present={has_cbam}  "
          f"layers={sum(1 for _ in model.model.modules())}")
    if not has_cbam:
        print("[check-load] WARN: no CBAM module found -- is this the right checkpoint?")


def _train(args) -> None:
    pin_free_gpu()
    ensure_cbam_compat(args.compat)
    apply_nwd_patch(alpha=args.alpha, C=args.C)
    from ultralytics import YOLO
    last = Path(args.project) / args.name / "weights" / "last.pt"
    resume = args.resume and last.is_file()
    model = YOLO(str(last) if resume else args.weights)
    if resume:
        print(f"[nwd-train] RESUMING from {last}")
    model.add_callback("on_train_batch_end", make_verify_callback())
    model.train(
        resume=resume,
        data=args.data, epochs=args.epochs, patience=args.patience,
        batch=args.batch, imgsz=args.imgsz, optimizer="AdamW",
        lr0=args.lr0, lrf=0.01, cos_lr=True, seed=args.seed, deterministic=True,
        amp=not args.no_amp, workers=args.workers, cache=args.cache,
        project=args.project, name=args.name, device=args.device, exist_ok=True,
    )
    print("[nwd-train] done. Evaluate with 04_eval_fusion_ablation.py / thesis harness.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--check-load", action="store_true",
                    help="only verify the checkpoint deserializes here, then exit")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--compat", default=None,
                    help="optional .py file defining the ckpt's custom classes "
                         "(default: built-in verbatim CBAM classes)")
    ap.add_argument("--weights")
    ap.add_argument("--data")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--C", type=float, default=12.8)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--lr0", type=float, default=0.0003)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cache", default="disk")
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="runs_nwd")
    ap.add_argument("--name", default="nwd_run")
    ap.add_argument("--no-amp", action="store_true", help="REQUIRED on PC-4 (fp16 NaNs)")
    ap.add_argument("--resume", action="store_true", help="resume from <project>/<name>/weights/last.pt if present (power-cut recovery); safe to always pass")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    elif a.check_load:
        if not a.weights:
            sys.exit("--check-load needs --weights")
        _check_load(a)
    elif a.train:
        if not (a.weights and a.data):
            sys.exit("--train needs --weights and --data")
        _train(a)
    else:
        print(__doc__)
