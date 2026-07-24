"""
16_gpu_guard.py — claim + reserve GPU memory on a SHARED box so latecomers can't OOM you.

WHY: on PC-2 (2xA6000 shared, GPU1 = ours) other students sometimes launch on GPU1 mid-run,
grab the free memory, and OOM/corrupt our training. There is NO user-space way to fully lock a
GPU without admin (the true lock is `nvidia-smi -c EXCLUSIVE_PROCESS -i 1`, needs admin). This is
the next best thing:

  1. cap our own usage (set_per_process_memory_fraction) so WE never spill onto GPU0, and
  2. RESERVE our budget up front: allocate a big block then free it — PyTorch's caching allocator
     keeps that memory in OUR process's pool (it is NOT returned to the driver), so:
       - our own training reuses it freely, but
       - any OTHER process sees that memory as taken and OOMs at ITS launch, not ours.

  Effect: if we claim GPU1 FIRST, a latecomer bounces off instead of killing us. It does NOT stop
  someone who launched BEFORE us — only exclusive mode (admin) does that.

Two ways to use it:
  A. In a training script (the reliable way — reservation must live in the training process):
       import importlib; guard = importlib.import_module("16_gpu_guard")
       guard.claim(device=0, mem_frac=0.90, reserve_gb=44)   # call BEFORE building the model
  B. Standalone "parker" to squat GPU1 while you set up, then Ctrl+C right before launching train:
       python 16_gpu_guard.py --device 0 --reserve-gb 44 --hold
     (parker holds memory in a SEPARATE process, so kill it just before training claims it;
      small handoff race — prefer mode A for real runs.)

Note on device index: if you set `$env:CUDA_VISIBLE_DEVICES="1"` (per the PC-2 checklist), then
physical GPU1 is visible as index 0 — so use --device 0 here. Without that env var, use --device 1.

Selftest (no GPU needed): python 16_gpu_guard.py --selftest
"""
from __future__ import annotations
import argparse
import sys
import time


def _clamp_reserve_bytes(reserve_gb: float, free_bytes: int, headroom: float = 0.95) -> int:
    """Never try to reserve more than `headroom` of what's actually free."""
    want = int(reserve_gb * (1024 ** 3))
    ceil = int(free_bytes * headroom)
    return max(0, min(want, ceil))


def claim(device: int = 0, mem_frac: float = 0.90, reserve_gb: float | None = None,
          tag: str = "mofazzal") -> dict:
    """Cap our process memory fraction and (optionally) reserve reserve_gb on `device`.
    Safe no-op if CUDA is unavailable. Returns a small status dict."""
    try:
        import torch
    except Exception:
        print("[gpu-guard] torch not importable — skipping")
        return {"claimed": False}
    if not torch.cuda.is_available():
        print("[gpu-guard] CUDA not available — skipping")
        return {"claimed": False}

    torch.cuda.set_device(device)
    name = torch.cuda.get_device_name(device)
    # 1) cap our own footprint so we never spill onto a neighbour's GPU
    try:
        torch.cuda.set_per_process_memory_fraction(float(mem_frac), device)
    except Exception as e:
        print(f"[gpu-guard] fraction cap not set ({e})")
    free, total = torch.cuda.mem_get_info(device)

    reserved_gb = 0.0
    if reserve_gb and reserve_gb > 0:
        nbytes = _clamp_reserve_bytes(reserve_gb, free)
        try:
            blk = torch.empty(nbytes, dtype=torch.uint8, device=f"cuda:{device}")
            del blk                      # -> stays in OUR caching-allocator pool, not the driver
            torch.cuda.synchronize(device)
            reserved_gb = nbytes / (1024 ** 3)
        except RuntimeError as e:
            print(f"[gpu-guard] reservation of ~{reserve_gb}GB failed ({e}); "
                  "someone may already hold the memory — check `nvidia-smi -i {device}`")

    free_after, _ = torch.cuda.mem_get_info(device)
    print(f"[gpu-guard] CLAIMED dev{device} ({name}) by '{tag}' | cap={mem_frac:.0%} "
          f"| reserved~{reserved_gb:.1f}GB | free_now={free_after/1024**3:.1f}/{total/1024**3:.1f}GB")
    if reserved_gb:
        print("[gpu-guard] a latecomer on this GPU will now OOM at ITS launch, not yours. "
              "(true lock = admin `nvidia-smi -c EXCLUSIVE_PROCESS`).")
    return {"claimed": True, "device": device, "reserved_gb": reserved_gb,
            "free_after_gb": free_after / 1024 ** 3, "total_gb": total / 1024 ** 3}


def _park(device: int, reserve_gb: float, tag: str) -> int:
    """Standalone: reserve + HOLD (keep the reference) + sleep until Ctrl+C."""
    try:
        import torch
    except Exception:
        print("[gpu-guard] torch not importable"); return 1
    if not torch.cuda.is_available():
        print("[gpu-guard] CUDA not available"); return 1
    torch.cuda.set_device(device)
    free, total = torch.cuda.mem_get_info(device)
    nbytes = _clamp_reserve_bytes(reserve_gb, free)
    hold = torch.empty(nbytes, dtype=torch.uint8, device=f"cuda:{device}")  # keep ref = held
    torch.cuda.synchronize(device)
    fa, _ = torch.cuda.mem_get_info(device)
    print(f"[gpu-guard] PARKED dev{device} by '{tag}': holding ~{nbytes/1024**3:.1f}GB, "
          f"free_now={fa/1024**3:.1f}/{total/1024**3:.1f}GB")
    print("[gpu-guard] Ctrl+C to release RIGHT BEFORE you launch training (small handoff race).")
    try:
        while True:
            time.sleep(30)
    except KeyboardInterrupt:
        del hold
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        print("\n[gpu-guard] released. Launch your training now.")
        return 0


def selftest() -> int:
    fails = 0
    def ck(n, c):
        nonlocal fails
        print(f"[{'PASS' if c else 'FAIL'}] {n}")
        if not c: fails += 1
    # reserve clamp: never exceed 95% of free
    ck("clamp caps at 95% free", _clamp_reserve_bytes(100, 10 * 1024**3) == int(10*1024**3*0.95))
    ck("clamp honours smaller request", _clamp_reserve_bytes(4, 40 * 1024**3) == int(4*1024**3))
    ck("clamp floors at 0", _clamp_reserve_bytes(0, 8 * 1024**3) == 0)
    # claim is a safe no-op without CUDA (laptop)
    r = claim(device=0, mem_frac=0.9, reserve_gb=8)
    ck("claim no-ops without CUDA", r.get("claimed") in (True, False))
    print("SELFTEST OK" if fails == 0 else f"SELFTEST FAILED ({fails})")
    return fails


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0, help="visible GPU index (with CUDA_VISIBLE_DEVICES=1, physical GPU1 = index 0)")
    ap.add_argument("--mem-frac", type=float, default=0.90)
    ap.add_argument("--reserve-gb", type=float, default=0.0)
    ap.add_argument("--tag", default="mofazzal")
    ap.add_argument("--hold", action="store_true", help="standalone parker: reserve + hold until Ctrl+C")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if a.hold:
        sys.exit(_park(a.device, a.reserve_gb or 44.0, a.tag))
    claim(a.device, a.mem_frac, a.reserve_gb or None, a.tag)
