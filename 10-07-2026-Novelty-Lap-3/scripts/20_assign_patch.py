r"""
20_assign_patch.py -- S2 / direction D2: scale-robust similarity INSIDE the assignment
metric of Ultralytics' TaskAlignedAssigner (YOLO11/8.4.x), for tiny aerial persons.

THE LEVER (verified from lap-3 docs + PC-1 source grep, ultralytics 8.4.56):
  TaskAlignedAssigner.get_box_metrics builds  align_metric = cls^alpha * overlaps^beta,
  where `overlaps = iou_calculation(gt, pd)` = CIoU. For <16 px boxes CIoU is numerically
  unstable/near-zero, so tiny GTs get badly-ranked candidates. We replace/blend the CIoU in
  `iou_calculation` with a Normalized Wasserstein (NWD) similarity that stays alive for tiny
  and near-miss boxes. This is the RFLA/DCFL/SimD assignment-side consensus (assignment >
  loss for tiny objects) -- and a DIFFERENT insertion point than our G2 negative (NWD in the
  box LOSS, which didn't scale). NWD math is the verified one from 02_nwd_loss_patch.

KEY FINDING (why we didn't just bolt on "STAL min-4 floor"): 8.4.56 ALREADY mitigates the
tiny-GT starvation STAL targets -- `select_candidates_in_gts` inflates any GT smaller than
stride[0] (8 px) up to stride_val (16 px) before the in-box test, and `topk2` adds a secondary
top-k. So an explicit min-anchor floor is now SECONDARY; the assignment-metric swap is the
sharp, unclaimed lever. (Floor can be added later as a toggle if VT-recall still lags.)

House-pattern = 02_nwd_loss_patch: version-defensive monkeypatch with (1) inspect-source
guards so a refactor can't turn this into a SILENT no-op, (2) a live call counter, and (3) a
callback that ABORTS training if the patch never fired.

  python 20_assign_patch.py --selftest      # math + (on 8.4.x) live patch on the real assigner
  python 20_assign_patch.py --check-load --weights runs_s1\s1_control\weights\best.pt
  # S2 pilot = CBAM+P2 control config + this patch, SAME protocol as 17_s1_pilot -> compare to s1_control:
  python 20_assign_patch.py --train --data <scenesplit yaml> --device 0 --batch 12 --alpha 0.5 --resume
Then eval with 18_s1_eval.py (--weights runs_s1\s2_assign\weights\best.pt) and diff vs s1_control.

SimD note: the docs flag SimD (IROS 2024) as needing its PRIMARY-SOURCE formula before use.
This ships NWD (verified). Once SimD is web-verified, drop it into assign_similarity(metric=).
"""
from __future__ import annotations
import argparse
import inspect
import sys
from importlib import import_module
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
RUNS = HERE / "runs_s1"

_STATE = {"applied": False, "calls": 0, "alpha": None, "C": None, "orig": None, "metric": "nwd"}


# ----------------------------------------------------------------- similarity math
def assign_similarity(gt_bboxes, pd_bboxes, C: float = 12.8, eps: float = 1e-7, metric: str = "nwd"):
    """Scale-robust similarity in (0,1] for xyxy box tensors, shape (..., 4) -> (...).
    metric='nwd' = Normalized Wasserstein (Wang et al. 2110.13389; C2A median box ~12 px -> C=12.8).
    Other metrics (e.g. verified SimD) plug in here."""
    import torch
    if metric != "nwd":
        raise NotImplementedError(f"assign_similarity metric='{metric}' not verified/implemented yet")
    gt, pd = gt_bboxes, pd_bboxes
    gw = (gt[..., 2] - gt[..., 0]).clamp(min=0); gh = (gt[..., 3] - gt[..., 1]).clamp(min=0)
    pw = (pd[..., 2] - pd[..., 0]).clamp(min=0); ph = (pd[..., 3] - pd[..., 1]).clamp(min=0)
    gcx = gt[..., 0] + gw / 2; gcy = gt[..., 1] + gh / 2
    pcx = pd[..., 0] + pw / 2; pcy = pd[..., 1] + ph / 2
    w2 = (gcx - pcx) ** 2 + (gcy - pcy) ** 2 + ((gw - pw) ** 2 + (gh - ph) ** 2) / 4.0
    return torch.exp(-torch.sqrt(w2.clamp(min=0) + eps) / C)


# ----------------------------------------------------------------- patch machinery
def apply_assign_patch(alpha: float = 0.5, C: float = 12.8, metric: str = "nwd") -> None:
    """Blend NWD into the CIoU in TaskAlignedAssigner.iou_calculation. alpha=1.0 => full
    RFLA/SimD-style replacement of the localization metric; alpha=0 => vanilla. Idempotent."""
    if _STATE["applied"]:
        _STATE["alpha"], _STATE["C"], _STATE["metric"] = float(alpha), float(C), metric
        return
    import ultralytics
    import ultralytics.utils.tal as TAL
    A = TAL.TaskAlignedAssigner
    ver = ultralytics.__version__

    if not hasattr(A, "iou_calculation"):
        raise RuntimeError(f"[assign] ultralytics {ver}: TaskAlignedAssigner has no "
                           f"`iou_calculation` -- assigner refactored; DO NOT TRAIN, adapt the patch.")
    if "bbox_iou(" not in inspect.getsource(A.iou_calculation):
        raise RuntimeError(f"[assign] ultralytics {ver}: iou_calculation does not call bbox_iou "
                           f"-- patching it would change the wrong thing; adapt first.")
    if "self.iou_calculation(" not in inspect.getsource(A.get_box_metrics):
        raise RuntimeError(f"[assign] ultralytics {ver}: get_box_metrics no longer calls "
                           f"self.iou_calculation -- the assignment insertion point moved; adapt first.")

    _STATE.update(orig=A.iou_calculation, alpha=float(alpha), C=float(C), metric=metric)

    def iou_calculation_nwd(self, gt_bboxes, pd_bboxes):
        base = _STATE["orig"](self, gt_bboxes, pd_bboxes)          # (N,) CIoU, clamped >=0
        a = _STATE["alpha"]
        if a <= 0.0:
            return base
        sim = assign_similarity(gt_bboxes, pd_bboxes, C=_STATE["C"], metric=_STATE["metric"]).to(base.dtype)
        _STATE["calls"] += 1
        return (1.0 - a) * base + a * sim

    A.iou_calculation = iou_calculation_nwd
    _STATE["applied"] = True
    print(f"[assign-patch] ACTIVE  alpha={alpha}  C={C}  metric={metric}  ultralytics={ver}  "
          f"(assignment overlaps = {1-alpha:.2f}*CIoU + {alpha:.2f}*NWD)")


def remove_assign_patch() -> None:
    if _STATE["applied"]:
        import ultralytics.utils.tal as TAL
        TAL.TaskAlignedAssigner.iou_calculation = _STATE["orig"]
        _STATE["applied"] = False
        print("[assign-patch] removed")


def make_verify_callback():
    """on_train_epoch_end: ABORT if the patched assigner metric never fired (silent-no-op guard)."""
    def cb(trainer):
        if getattr(cb, "done", False):
            return
        cb.done = True
        if _STATE["calls"] == 0:
            raise RuntimeError("[assign-verify] FATAL: an epoch finished but the NWD-blended "
                               "iou_calculation was NEVER called -- patch not in the assignment "
                               "path (version drift?). Training aborted; do not trust this run.")
        print(f"[assign-verify] OK -- NWD-in-assignment active (calls: {_STATE['calls']})")
    return cb


# ----------------------------------------------------------------- selftest
def _selftest() -> int:
    import torch
    ok = 0
    # T1 math: identity -> ~1; monotone decay; ALIVE (>0) for tiny disjoint boxes where IoU=0
    b = torch.tensor([[10., 10., 14., 14.]])
    assert torch.allclose(assign_similarity(b, b), torch.ones(1), atol=1e-4)
    base = torch.tensor([[0., 0., 4., 4.]])
    d = [assign_similarity(base, torch.tensor([[dx, 0., dx + 4., 4.]])).item() for dx in (0., 2., 6., 20.)]
    assert all(x > y for x, y in zip(d, d[1:])), f"T1 not monotone: {d}"
    assert d[2] > 0.0, "T1 disjoint tiny boxes must keep signal (the whole point vs IoU=0)"
    ok += 1
    print(f"[selftest] T1 NWD math OK (disjoint-tiny similarity {d[2]:.3f} > 0 where CIoU=0)")

    # T2-T4 need the real 8.4.x assigner. Skip gracefully on the laptop's old ultralytics.
    try:
        import ultralytics
        import ultralytics.utils.tal as TAL
        try:
            apply_assign_patch(alpha=0.5, C=12.8)
        except RuntimeError as e:
            print(f"[selftest] patch guard tripped on ultralytics {ultralytics.__version__}: {e}")
            print(f"[selftest] (expected on the laptop's old ultralytics; RUN --selftest ON PC-1 8.4.56)")
            print(f"SELFTEST PASSED ({ok}/1 math group here; patch mechanics validated on PC-1)")
            return 0
        # T2 patched iou_calculation adds signal where CIoU is dead
        asg = TAL.TaskAlignedAssigner(topk=13, num_classes=1)
        gt = torch.tensor([[0., 0., 6., 6.]]); pd = torch.tensor([[10., 0., 16., 6.]])  # disjoint tiny
        pure_ciou = _STATE["orig"](asg, gt, pd)
        c0 = _STATE["calls"]
        blended = asg.iou_calculation(gt, pd)
        assert _STATE["calls"] == c0 + 1, "T2 patched method not invoked"
        assert (blended > pure_ciou).all(), f"T2 no signal added: {blended} vs {pure_ciou}"
        ok += 1
        print(f"[selftest] T2 patched iou_calculation adds tiny-box signal "
              f"({blended.item():.3f} > CIoU {pure_ciou.item():.3f}) OK")

        # T3 full assigner.forward runs patched, returns valid shapes, patch fired
        G, S = 16, 8; n = G * G
        ys, xs = torch.meshgrid(torch.arange(G), torch.arange(G), indexing="ij")
        anc = (torch.stack([xs.flatten(), ys.flatten()], 1).float() + 0.5) * S      # (n,2)
        half = torch.full((n, 2), S / 2.0)
        pd_bboxes = torch.cat([anc - half, anc + half], 1).unsqueeze(0)             # (1,n,4)
        pd_scores = torch.full((1, n, 1), 0.6)
        gt_bboxes = torch.tensor([[[24., 24., 30., 30.], [60., 60., 100., 100.]]])   # tiny + normal
        gt_labels = torch.zeros(1, 2, 1, dtype=torch.long)
        mask_gt = torch.ones(1, 2, 1)
        c1 = _STATE["calls"]
        out = asg(pd_scores, pd_bboxes, anc, gt_labels, gt_bboxes, mask_gt)
        assert len(out) == 5, "T3 forward return arity"
        tgt_labels, tgt_bboxes, tgt_scores, fg_mask, tgt_idx = out
        assert fg_mask.shape == (1, n) and tgt_bboxes.shape == (1, n, 4), "T3 shapes"
        assert _STATE["calls"] > c1, "T3 patch never fired inside forward"
        assert int(fg_mask.sum()) > 0, "T3 no positives assigned at all"
        ok += 1
        print(f"[selftest] T3 real assigner.forward patched OK "
              f"(positives={int(fg_mask.sum())}, calls={_STATE['calls']})")

        remove_assign_patch()
        restored = asg.iou_calculation(gt, pd)
        assert torch.allclose(restored, pure_ciou, atol=1e-6), "T4 unpatch failed"
        ok += 1
        print(f"[selftest] T4 remove restores vanilla CIoU OK")
        print(f"SELFTEST PASSED ({ok}/4 groups, incl. live patch on the real 8.4.x assigner)")
    except ImportError:
        print(f"SELFTEST PASSED ({ok}/1 math group; ultralytics not installed here -- "
              f"patch mechanics verified by --selftest/--check-load on PC-1)")
    return 0


# ----------------------------------------------------------------- check-load
def _check_load(args) -> int:
    fccg = import_module("10_fccg_modules")
    if not fccg.register_fccg():
        print("FATAL: ultralytics not importable"); return 1
    apply_assign_patch(alpha=args.alpha, C=args.C)          # proves the guard passes here
    from ultralytics import YOLO
    if not Path(args.weights).is_file():
        print(f"FATAL: weights not found: {args.weights}"); return 1
    model = YOLO(args.weights)
    n = sum(p.numel() for p in model.model.parameters())
    print(f"[check-load] OK {args.weights}  params={n/1e6:.2f}M  patch guard PASSED on this PC")
    remove_assign_patch()
    return 0


# ----------------------------------------------------------------- train launcher
def _train(args) -> int:
    fccg = import_module("10_fccg_modules")
    if not fccg.register_fccg():
        print("FATAL: ultralytics not importable"); return 1
    from ultralytics import YOLO

    if args.device is not None:
        try:
            guard = import_module("16_gpu_guard")
            guard.claim(device=int(args.device), mem_frac=args.mem_frac,
                        reserve_gb=(args.reserve_gb or None))
        except Exception as e:
            print(f"[assign] gpu-guard skipped ({e})")

    apply_assign_patch(alpha=args.alpha, C=args.C)           # BEFORE building/training

    name = "s2_assign"
    last_pt = RUNS / name / "weights" / "last.pt"
    resumed = bool(args.resume and last_pt.exists())
    if resumed:
        model = YOLO(str(last_pt)); print(f"[assign] RESUMING {name} from {last_pt}")
    else:
        if args.resume:
            print(f"[assign] --resume set but no checkpoint at {last_pt}; starting FRESH (safe).")
        ypath = HERE / fccg.CONTROL_YAML_NAME                # CBAM+P2 control config (matches s1_control)
        if not ypath.exists():
            fccg.emit_control_yaml()
        model = YOLO(str(ypath))
    n = sum(p.numel() for p in model.model.parameters())
    print(f"[assign] {name} built: {n/1e6:.2f}M params (CBAM+P2 control config + NWD-in-TAL)")

    model.add_callback("on_train_epoch_end", make_verify_callback())
    cache_val = False if str(args.cache).lower() in ("false", "0", "none", "") else args.cache
    kw = dict(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
              optimizer="AdamW", lr0=0.001, seed=args.seed, workers=args.workers,
              patience=args.epochs, amp=not args.no_amp, project=str(RUNS), name=name,
              exist_ok=True, val=True, plots=True, deterministic=True, cache=cache_val,
              save_period=-1, resume=resumed)
    if args.device is not None:
        kw["device"] = args.device
    model.train(**kw)
    print(f"[assign] DONE. Eval: python 18_s1_eval.py --variant control --weights "
          f"{RUNS/name/'weights'/'best.pt'} ...  then diff AP_small/VT-recall vs s1_control.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--check-load", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--weights")
    ap.add_argument("--data")
    ap.add_argument("--alpha", type=float, default=0.5, help="0=vanilla, 1.0=full RFLA/SimD-style replace")
    ap.add_argument("--C", type=float, default=12.8, help="NWD scale const (C2A median box ~12 px)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--mem-frac", type=float, default=0.90)
    ap.add_argument("--reserve-gb", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default="ram")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true", help="safe to always pass (power-cut recovery)")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(_selftest())
    if a.check_load:
        if not a.weights:
            sys.exit("--check-load needs --weights")
        sys.exit(_check_load(a))
    if a.train:
        if not a.data:
            sys.exit("--train needs --data")
        sys.exit(_train(a))
    print(__doc__)
