# System Specification — Thesis Final Month

> ## 🔴 PRIME DIRECTIVE (read before everything else)
>
> 1. **Follow this file carefully and fully.** Every section is part of the contract — sections, sub-sections, tables, code snippets, file names, all of it. The agent does not get to pick which parts to honor.
> 2. **If something cannot be done — STOP and ASK.** If a requirement here conflicts with another, is technically infeasible on this machine, requires a tool that isn't installed, depends on data that isn't present, or would violate another section: do **not** silently work around it. Either tell the user clearly *what* cannot be done and *why*, or ask the user to decide between the available options. Never invent a substitute, never quietly skip a step, never produce partial output and label it complete.
> 3. **Auto-detect, don't hardcode.** The agent must auto-detect (a) where the code is running from, (b) where the dataset lives, and (c) which GPU is available. The **output directory must be derived from the location of the running script** (see Section 5.2 and Section 15.1) — never write outputs to a hardcoded absolute path. The dataset's absolute path is **not** guaranteed (the work happens on a remote AnyDesk PC and the drive layout there can change); only the dataset's internal folder structure (Section 5.1) is fixed.
> 4. **When in doubt, ask.** Asking the user one clarifying question is always preferable to producing wrong work that has to be re-run on a 16 GB GPU.

> **Purpose:** This file is the authoritative hardware/software profile of the local machine that will run all training, inference, ablations, and final experiments for the thesis. Any agent writing code for this project MUST read this first and tailor batch sizes, precision, parallelism, memory budgets, and data-loading choices to these constraints. Do not assume cloud/Kaggle limits — local runs have a different profile (more RAM, single GPU, no notebook timeout, but limited VRAM vs. dual-T4 etc.).

**Last updated:** 2026-05-28
**Owner:** Mofazzal (undergrad thesis — YOLO + Mamba aerial human detection)
**Status:** Living document — more sections (datasets, env, paths, run conventions) will be appended later.

---

## 1. CPU

| Field | Value |
|---|---|
| Model | Intel Core i7-14700K (Raptor Lake Refresh) |
| Base clock | 3.40 GHz |
| Observed boost (Task Manager) | ~5.45 GHz |
| Sockets | 1 |
| Physical cores | 20 (8 P-cores + 12 E-cores) |
| Logical processors (threads) | 28 |
| L1 cache | 1.8 MB |
| L2 cache | 28.0 MB |
| L3 cache | 33.0 MB |
| Virtualization | Enabled |

**Implications for code:**
- `num_workers` for `DataLoader`: safe upper bound **8–12** (leave headroom for P-cores running training + system). Do not set `num_workers=28` — it will thrash and starve the GPU.
- CPU is strong enough that **CPU-side augmentation** (Albumentations, copy-paste, SAHI tiling pre-compute) is NOT a bottleneck — use it freely instead of pushing aug to GPU.
- Heavy multi-process work (e.g., COCO eval, SAHI slicing, image preprocessing) can use `multiprocessing.Pool(processes=12)` comfortably.

---

## 2. RAM

| Field | Value |
|---|---|
| Total installed | 128 GB (4 × 32 GB DIMM, all 4 slots used) |
| Speed | 4000 MT/s |
| Form factor | DIMM (desktop) |
| In use (idle baseline) | ~19.5 GB |
| Available (idle baseline) | ~108 GB |
| Committed | 31 / 136 GB |
| Paged pool | 2.1 GB |
| Non-paged pool | 971 MB |

**Implications for code:**
- RAM is **not a constraint**. ~100 GB free is available for the training process.
- Safe to **cache the entire VisDrone / HIT-UAV train set in RAM** (`cache='ram'` in Ultralytics, or in-memory dataset). This will speed up training significantly vs. disk reads.
- Safe to keep large NumPy/Tensor buffers (SAHI slice indices, TTA prediction stacks, augmentation pools) in memory.
- Do NOT rely on swap/pagefile for speed — even with 136 GB committed available, pagefile will tank GPU utilization. Keep working set under ~100 GB.

---

## 3. GPU (Primary — Training & Inference)

| Field | Value |
|---|---|
| Model | NVIDIA GeForce RTX 4070 Ti SUPER |
| Architecture | Ada Lovelace (5 nm) |
| Silicon | AD103 |
| CUDA cores | 8,448 |
| Tensor cores | 4th generation (supports FP16, BF16, FP8, INT8, sparsity) |
| Ray Tracing cores | 3rd generation (irrelevant for ML) |
| Boost clock | ~2,610 – 2,685 MHz |
| **VRAM** | **16 GB GDDR6X** |
| Memory bus | 256-bit |
| Memory bandwidth | 21 Gbps, up to **672 GB/s** |
| Power (TGP) | 285 W |
| Bus interface | PCIe 4.0 x16 |
| Driver version | 32.0.101.7082 (12/1/2025) |
| DirectX | 12 (FL 12.1) |
| Idle VRAM use | ~3.1 / 16.0 GB |

**Implications for code (READ CAREFULLY — this is the binding constraint):**

- **VRAM ceiling: 16 GB total, ~12.5–13 GB usable** for training after driver/OS overhead (~3 GB baseline).
- **Mixed precision is mandatory.** Always use `amp=True` (Ultralytics) or `torch.cuda.amp.autocast()` + `GradScaler`. 4th-gen Tensor cores on Ada deliver large speedups on FP16/BF16. Prefer **BF16 on Ada** for stability (no loss scaling needed) when the framework supports it.
- **Single GPU only.** Do NOT use `DDP` / `nn.DataParallel` on this machine. Kaggle code paths that branch on `WORLD_SIZE > 1` must default to single-GPU here. (The Intel UHD 770 iGPU is for display only — never selected as a CUDA device.)
- **Batch size guidance (YOLO-class detectors @ 640 input, AMP on):**
  - YOLOv8n / YOLOv11n: batch 64–96 fits
  - YOLOv8s / YOLOv11s: batch 32–48 fits
  - YOLOv8m / YOLOv11m: batch 16–24 fits
  - YOLOv8l / YOLOv11l: batch 8–12 fits
  - YOLOv8x / YOLOv11x: batch 4–8 fits
  - With Mamba/SSM blocks added: expect 20–35 % more VRAM than vanilla YOLO of the same size → reduce batch accordingly.
  - SAHI tile inference @ 640 with overlap: peak VRAM driven by `tile_count × batch`; keep inference batch ≤ 16.
- **Always start with `batch=-1` (Ultralytics auto-batch) on a fresh model to measure**, then pin the value in config for reproducibility. Verify with `nvidia-smi` that peak usage stays below ~14.5 GB to leave headroom for validation passes.
- **Gradient accumulation** is the lever when a config OOMs: halve batch, set `accumulate=2`. Effective batch stays the same.
- **`torch.compile`** is supported on Ada and gives 10–25 % speedups for static-shape models. Worth enabling for final runs, but disable during debugging (compile times mask errors).
- **Set `torch.backends.cudnn.benchmark = True`** for fixed input sizes (training). Set `False` when input sizes vary (SAHI inference).
- **PCIe 4.0 x16** — host↔device transfer is not a bottleneck for 640×640 images. `pin_memory=True` in DataLoader is still recommended.
- **No NVLink, no multi-GPU.** Any pseudo-distributed code must be a no-op on this machine.

---

## 4. GPU (Secondary — Integrated, DO NOT USE for ML)

| Field | Value |
|---|---|
| Model | Intel UHD Graphics 770 (integrated in i7-14700K) |
| Shared GPU memory | 63.9 GB (system RAM, not VRAM) |
| Role | Display output only |

**Implications:**
- This device appears as GPU 0 in Task Manager but is **not a CUDA device** and must never be selected by PyTorch.
- Always pin training to the NVIDIA card explicitly:
  ```python
  import torch
  assert torch.cuda.is_available()
  device = torch.device("cuda:0")  # 4070 Ti SUPER
  ```
  or via env: `set CUDA_VISIBLE_DEVICES=0` (PowerShell: `$env:CUDA_VISIBLE_DEVICES = "0"`).
- If a future setup adds a second NVIDIA card, re-check device indices.

---

## 5. Storage, Datasets, and Path Auto-Detection

### 5.0 Where this code actually runs

All training/inference runs happen on a **remote high-config Windows PC accessed via AnyDesk**. The hardware in Sections 1–4 describes that remote PC, not the laptop the user composes commands from. Two consequences the agent MUST internalize:

- **Absolute dataset paths are not stable.** The user has shown one current location (`E:\Thesis_mofazzal_2007074\…`), but the drive letter, folder name, and parent path on the remote PC may change between sessions or when moved to a different machine. The agent must NEVER bake a single absolute dataset path into code. Use the discovery mechanism in Section 5.3 instead.
- **The output directory is not on a fixed drive either.** It is whichever directory the running script lives in (Section 5.2). Do not assume `D:\` or `E:\` or any specific letter.

The only things that ARE stable and can be relied on:
- The dataset's **internal folder structure** (Section 5.1) — `train/`, `val/`, `test/`, the pose label set, the `.png` image format.
- The dataset's **logical name** — C2A.

### 5.1 Primary dataset — C2A (Aerial human detection, SAR)

| Field | Value |
|---|---|
| Dataset | **C2A** (Nihal et al., ICPR 2024) |
| Location | Unfixed — discovered at runtime per Section 5.3. Currently on the remote AnyDesk PC; previously seen at `E:\Thesis_mofazzal_2007074\common\c2a\new_dataset3` but treat that as one possibility, not the canonical path. |
| Role | **Primary dataset** for every Phase A / B / C run in Section 9.6 and every row in the Section 25 ablation matrix |
| Other datasets | None yet — additional datasets may be added later; treat C2A as the canonical one until a new spec section is written for the new dataset |

**On-disk structure (FIXED — this is what the agent can rely on):**

```
<DATASET_ROOT>/                          # parent path varies; only the tree below is guaranteed
└── new_dataset3/
    ├── train/
    │   ├── images/                      # .png images  (YOLO inputs)
    │   ├── labels/                      # YOLO .txt files (cls cx cy w h)
    │   └── train_annotations.json       # COCO-style JSON (full annotations)
    ├── val/                             # presumed same structure as train/
    │   ├── images/
    │   ├── labels/
    │   └── val_annotations.json         # (verify at runtime, do not assume)
    ├── test/                            # presumed same structure as train/
    │   ├── images/
    │   ├── labels/
    │   └── test_annotations.json        # (verify at runtime, do not assume)
    └── All labels with Pose information/
        ├── labels/                      # YOLO .txt files WITH pose keypoints
        └── readme                       # describes the pose-extended format
```

Notes:
- The "All labels with Pose information" tree is an **alternate label set** with keypoints — useful for future pose-aware experiments. The mainline detection runs use `train/labels`, `val/labels`, `test/labels` (no keypoints).
- The agent MUST `os.path.isdir(...)` / `glob` each of these paths at script start and **fail loudly** if any required split is missing. Do NOT assume `val/` and `test/` look exactly like `train/` — confirm.
- Image format is `.png`. Do not write code that hard-codes `.jpg`.
- The classes are single-class (`person`); the `data.yaml` for Ultralytics must reflect `nc: 1, names: ['person']`. Always read the actual annotation file headers to confirm this rather than hardcoding it.

### 5.2 Output-directory auto-detection (HARD RULE)

The output / working directory MUST be derived at runtime from **the location of the running script**. Never hardcode any absolute path (no `D:\…`, no `E:\…`, no `/kaggle/working/…`) as the output root. The pattern below is mandatory for every script:

```python
from pathlib import Path
import os

# 1. Find the directory the running code lives in.
#    Works for .py scripts; for notebooks fall back to cwd.
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path(os.getcwd()).resolve()      # notebook / REPL

# 2. Output root = code directory. Every output (runs/, smoke/, logs/, plots/, docs/)
#    is created RELATIVE to SCRIPT_DIR — so if the user runs the script from any
#    folder on any machine, outputs land beside the code that produced them.
OUTPUT_ROOT = SCRIPT_DIR

# 3. The per-run directory still follows Section 16's run_id convention.
RUN_DIR = OUTPUT_ROOT / "runs" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)
```

Why this rule exists:
- The actual run host is a remote AnyDesk PC whose drive letters and paths the agent cannot predict. Hardcoding any absolute path breaks the moment the code is moved to a different machine, drive, or working copy.
- Outputs co-located with the code make it trivial to zip-up and share a single self-contained experiment folder.
- It removes the "where did my results go?" failure mode.

### 5.3 Dataset-path auto-detection

The dataset does not live next to the code, and its absolute path may change between machines / AnyDesk sessions. The agent must discover the dataset by probing an ordered list of candidates, AND must accept an explicit override via environment variable for the case where none of the candidates match.

```python
def find_dataset_root(env_var: str, structure_marker: str, candidates: list[Path]) -> Path:
    """Return the first existing directory among (env var) → candidates that
    contains `structure_marker` (e.g., 'train/images'). Raise with a clear
    message if nothing matches — the user can then set the env var."""
    # 1. Explicit override always wins.
    override = os.environ.get(env_var)
    if override:
        p = Path(override)
        if (p / structure_marker).is_dir():
            return p
        raise FileNotFoundError(f"{env_var}={override} does not contain {structure_marker}")
    # 2. Otherwise probe candidates in order.
    for p in candidates:
        if p and (p / structure_marker).is_dir():
            return p
    raise FileNotFoundError(
        f"Could not locate dataset (looking for {structure_marker}). "
        f"Set {env_var} to the directory that contains it."
    )

# Candidate list for C2A. The drive letter / parent may differ between runs;
# we list a few common shapes and rely on the env var for anything else.
C2A_CANDIDATES = [
    SCRIPT_DIR / "common" / "c2a" / "new_dataset3",                       # co-located with code
    SCRIPT_DIR.parent / "common" / "c2a" / "new_dataset3",                # one level up
    *[Path(f"{d}:/Thesis_mofazzal_2007074/common/c2a/new_dataset3")       # common drive letters
      for d in ("E", "D", "F", "G")],
]

DATASET_ROOT = find_dataset_root(
    env_var="C2A_ROOT",
    structure_marker="train/images",
    candidates=C2A_CANDIDATES,
)
```

Same pattern applies to any future dataset added later — give it its own env var and `*_CANDIDATES` list; never hardcode a single absolute path.

### 5.4 Auto-detect everything else, too

By default the agent should auto-detect, not ask the user, for:
- Which CUDA device is the NVIDIA one (skip the Intel UHD 770 — see Section 4).
- Whether the script is being run as a `.py` file or inside a notebook.
- Python version, torch version, CUDA build (log them to `env.json`, see Section 11.7).
- Whether `last.pt` exists in the run directory (→ enable resume).
- Whether git is available and the working tree is clean.

If a probe fails or returns an ambiguous result (e.g., two NVIDIA GPUs, or the dataset has unexpected subfolders), the agent MUST **stop and ask** — see the Prime Directive at the top of this file. Never pick one silently.

### 5.5 Drive layout note (remote AnyDesk PC — current snapshot, not contract)

The run host is the remote AnyDesk PC described in Sections 1–4. At the time of writing, its drives are roughly:

- `C:` — OS / Windows on the remote PC.
- `E:` — research drive on the remote PC where the dataset currently lives (e.g., `E:\Thesis_mofazzal_2007074\…`). Treat as **read-mostly** for datasets; do not write training outputs here unless the code's own location happens to be on `E:` (Section 5.2 still wins).
- Other drives may exist; do not assume.

This layout is a **current snapshot, not a contract** — the user may switch to a different AnyDesk machine, move the data to a different drive, or run the same code on their local laptop. All code paths must rely on Sections 5.2 / 5.3 discovery, not on this snapshot.

- Free-space check: every script must verify ≥ 20 GB free on the drive where `OUTPUT_ROOT` resolves (Section 24 pre-flight).

---

## 6. Operating System & Shell

| Field | Value |
|---|---|
| OS | Windows 11 Home Single Language |
| Build | 10.0.26200 |
| Primary shell | PowerShell (NOT bash) |
| Secondary | Bash available (e.g., via Git Bash / WSL — confirm before assuming) |

**Implications for code & commands:**
- **Path separators:** use forward slashes `/` in Python strings — `pathlib.Path` handles it. In shell commands, prefer `pathlib`-built paths over hardcoded `D:\...` strings.
- **Env vars in shell:** PowerShell uses `$env:NAME = "value"`, NOT `export NAME=value`. Do not generate bash-style commands for the user to paste.
- **Line endings:** keep files at LF (configure `.gitattributes` if needed); editors/IDEs on Windows may default to CRLF.
- **No `/dev/null`** — use `$null` in PowerShell or just don't redirect.
- **Long path support:** Windows defaults to 260-char path limit. Keep dataset paths short (e.g., `D:/data/visdrone/...` not nested 8 levels deep).
- **GPU monitoring:** `nvidia-smi` works natively in PowerShell. For continuous: `nvidia-smi --loop=2` or `nvidia-smi dmon`.

---

## 7. Practical Runbook Defaults (derived from the above)

When writing/generating training code for this machine, default to these unless explicitly overridden:

```python
# Hardware-pinned defaults for the 4070 Ti SUPER box
DEVICE          = "cuda:0"
AMP             = True
PRECISION       = "bf16"          # prefer BF16 on Ada; fall back to fp16 if a layer complains
NUM_WORKERS     = 8               # raise to 12 only if GPU util < 90 %
PIN_MEMORY      = True
PERSISTENT_WORKERS = True
CACHE           = "ram"           # we have 100+ GB free; cache the dataset
CUDNN_BENCHMARK = True            # for fixed-size training; False for SAHI inference
COMPILE         = False           # enable only for final reproducible runs
GRAD_ACCUM      = 1               # bump to 2/4 if a desired batch OOMs

# Ultralytics-style defaults
train_kwargs = dict(
    device=0,
    amp=True,
    workers=8,
    cache="ram",
    batch=-1,        # auto-batch on first run, then pin
    half=False,      # train in fp32+amp; only set half=True for inference export
)
```

**Anti-patterns to avoid (will cause OOM or wasted time):**
- `batch=128` for any model larger than YOLOv8n at 640 input.
- `device='cpu'` left over from local debug — kills throughput by 100×.
- `num_workers=0` in long runs — GPU will starve.
- DDP / `torch.distributed` initialization on this single-GPU box.
- Loading dataset from a slow external drive without `cache='ram'`.
- Forgetting to set `CUDA_VISIBLE_DEVICES` when a notebook from Kaggle expects `cuda:0` and `cuda:1`.

---

## 8. Sanity-check snippet (run once per environment)

```python
import torch, platform
print("Python  :", platform.python_version())
print("Torch   :", torch.__version__)
print("CUDA?   :", torch.cuda.is_available())
print("CUDA ver:", torch.version.cuda)
print("Device  :", torch.cuda.get_device_name(0))
props = torch.cuda.get_device_properties(0)
print(f"VRAM    : {props.total_memory / 1024**3:.1f} GB")
print(f"SMs     : {props.multi_processor_count}")
print(f"Capability: {props.major}.{props.minor}")  # Ada = 8.9
```

Expected output snippet:
```
Device  : NVIDIA GeForce RTX 4070 Ti SUPER
VRAM    : 16.0 GB
Capability: 8.9
```

If `Capability` is not `8.9` or VRAM is not `16.0`, the wrong device was selected — STOP and fix before training.

---

## 9. Models Already Explored & Run Order for the Final Month

This is the **inventory of architectures/configurations** that have been tried during the thesis so far (per the progress slides). Numbers are intentionally NOT included here — they will be regenerated under the full metric protocol of Sections 11–13 during the final-month re-runs. Treat this list as the menu of *what exists* and *what order to re-run it in*.

### 9.1 Generation 1 — Baseline family benchmarking on C2A

Goal: pick a single working baseline family from YOLO v8 / v9 / v10 / v11.

- YOLOv8 — `n`, `s`, `m`
- YOLOv9 — `s`, `m`, `e`
- YOLOv10 — `s`, `m`, `l`
- YOLOv11 — `s`, `m`, `l`

**Outcome of this generation (locked-in for everything below):** YOLOv11-m is the working baseline.

### 9.2 Generation 2 — Attention-module ablation on YOLO11m

Replace the original `C2PSA` attention block (×2 after the SPPF layer) with:
- `YOLO11m baseline` (C2PSA kept — reference)
- `YOLO11m + ECA` (Efficient Channel Attention; channel-only; 1 block at 1024 ch)
- `YOLO11m + CBAM` (Convolutional Block Attention; channel + spatial sequential; 1 block; channels auto-detected at runtime)

**Outcome of this generation:** CBAM is the locked-in attention block.

### 9.3 Generation 3 — Detection-head ablation (P2 small-object head)

P2 head adds a 4th detection scale at 160×160 (stride 4 — covers ~4 px objects).

- `YOLO11m + P2Head` (standard C2PSA, 4 scales P2/P3/P4/P5)
- `YOLO11m + CBAM + P2Head` (CBAM neck attention + 4-scale heads)

### 9.4 Generation 4 — State-Space-Model (SSM / Mamba) neck variants

Replace `C3k2` in the neck with custom SSM blocks. CBAM + P2 retained.

- `Mamba + CBAM + P2Head` — Bidirectional local-window SSM (forward + reverse), post-init injection at 6 neck layers, `d_state=4`.
- `AtrousMamba + CBAM + P2Head` — 3 parallel dilated SSM branches (`d=1, 2, 4`) + gated fusion, YAML-native at 3 neck layers, `d_state=4`. **Kept as a negative-result ablation** (per slides, slower and weaker than plain Mamba — publishable as such).

### 9.5 Generation 5 — Training-data and inference-time enhancements on `Mamba + CBAM + P2Head`

Training-data ablation (retraining required):
- `+ CopyPaste augmentation` (parameters used in the earlier run: `copy_paste=0.5, mixup=0.15, flipud=0.5, degrees=10, mosaic=1.0, close_mosaic=10`) — **kept as a negative-result ablation.**

Inference-time enhancements (NO retraining — applied to the trained `Mamba + CBAM + P2Head` weights):
- `+ SAHI` — slice sweep over `{256, 320, 512, 640}` with `GREEDYNMM` postprocess, `IOS` match metric, `perform_standard_pred=True`, slice confidence in `[0.15, 0.20]`, slice overlap in `[0.25, 0.30]`.
- `+ TTA` — multi-scale test-time augmentation at `imgsz ∈ {832, 1280, 1920}`.
- `+ SAHI + TTA combined` — TTA per slice on the best SAHI config.

⚠ For any SAHI-based row in the final results, standard Ultralytics `mAP50` / `mAP50-95` are NOT directly computable — SAHI uses a per-image box-matching protocol at IoU=0.5. This must be footnoted explicitly in the paper.

### 9.6 Final-month re-run order (the experiment flow the agent must follow)

Run **in this order**. Don't skip ahead — each step's output (especially seeds and per-image scores) is an input to later paired-significance tests (Section 12).

**Phase A — Trained models (P0, headline ablation chain):**
1. `yolo11m_baseline`
2. `yolo11m_cbam`
3. `yolo11m_p2head`
4. `yolo11m_cbam_p2head`
5. `mamba_cbam_p2head`  ← **headline model**

**Phase B — Trained models (P1, negative-result ablations — still required for the paper):**
6. `atrousmamba_cbam_p2head`
7. `mamba_cbam_p2head_copypaste`

**Phase C — Inference-time enhancements on the best Phase-A weights (no retraining):**
8. `mamba_cbam_p2head + SAHI` (sweep slice ∈ {256, 320, 512, 640})
9. `mamba_cbam_p2head + TTA` (sweep imgsz ∈ {832, 1280, 1920})
10. `mamba_cbam_p2head + SAHI + TTA` (combined, using each sweep's best slice and best imgsz)

**Phase D — Cross-run aggregation (after all of A + B + C are done):**
11. Compute paired significance tests for every pair in the ablation matrix (Section 12).
12. Build `ablation_master/` cross-run plots and LaTeX tables (Sections 13 + 15).

### 9.7 Comparison baselines (cited from prior published work — NOT re-trained by us)

The paper will compare our headline numbers against the published C2A results from Nihal et al. (ICPR 2024): Faster R-CNN, RetinaNet, RTMDet, Cascade R-CNN, DINO, YOLOv5, YOLOv9-c, YOLOv9-e. These numbers are quoted from their paper — we do not re-train them; we cite them.

### 9.8 What "running a model" means in the final month

For each item in Phases A–C, a single "run" comprises:

1. The Section 24 pre-flight checklist passes.
2. Section 17 smoke passes for that model.
3. Training (Phases A–B) or inference (Phase C) executes with the full metric protocol of Section 11 capturing every applicable metric.
4. Section 13 plots are produced and the Section 14 provenance manifest is populated.
5. For trained models, repeat 1–4 across **5 seeds** (Section 12.1).
6. Per-image scores are persisted so paired-significance tests in Phase D can run later.
7. `manifest.json` is finalized; the run directory is closed and the next model begins.

The agent does NOT need to babysit. After smoke passes and Phase A run 1 launches, the agent should let it complete, verify outputs, then start run 2. Do not run two trainings concurrently on this single GPU.

---

## 10. Metrics Already Being Logged (and gaps to close)

### 10.1 Currently captured (from progress slides + `mamba_cbam_p2_copypaste.py` evaluation cells)

- Per-epoch from Ultralytics `results.csv`: `train/box_loss`, `train/cls_loss`, `train/dfl_loss`, `val/box_loss`, `val/cls_loss`, `val/dfl_loss`, `metrics/precision(B)`, `metrics/recall(B)`, `metrics/mAP50(B)`, `metrics/mAP50-95(B)`, `lr/pg0..pg2` — plus computed `metrics/F1(B)` and `metrics/F2(B)`.
- Per-image (custom eval): `TP`, `FP`, `FN`, `Precision`, `Recall`, `F1`, `F2`, `Avg_Conf`, `Inference_ms`.
- Aggregate: overall P/R/F1/F2, per-size recall (very-tiny <8 px², tiny 8–16 px², small 16–32 px², medium 32–96 px², large ≥96 px²), `Avg_Inf_ms`, `Std_Inf_ms`, `P95_Inf_ms`.
- Model complexity: params (M), GFLOPs, layer count, `best.pt` size on disk.
- Calibration: ECE, high-FN/FP image counts.
- Confidence distribution histograms.
- Confusion matrix (person recall, miss rate, BG-FP).

### 10.2 Gaps — what is MISSING for a Q1/Q2 submission

(These will be added in Section 11. Flagged here so the gap is explicit.)

- ❌ No **multi-seed runs** → no variance estimate → no significance test.
- ❌ No **bootstrap confidence intervals** on any reported metric.
- ❌ No **paired statistical test** (paired t-test / paired bootstrap / sign-flip / McNemar) between ablation pairs.
- ❌ No **COCO-style AP_small / AP_medium / AP_large** breakdown (only recall-per-size).
- ❌ No **AR (Average Recall) AR@1, AR@10, AR@100**.
- ❌ No **per-class AP** (trivially 1-class here, but include for symmetry with multi-class extensions).
- ❌ No **PR curve** / **F1 vs confidence curve** / **per-class confusion matrix** exported as data.
- ❌ No **energy / wall-clock / GPU-hours / kWh / CO₂** logging.
- ❌ No **fixed train/val/test seed manifest** committed alongside the weights.
- ❌ No **deterministic re-run hash** for the test images used.
- ❌ No **memory peak (VRAM MB)** logged per epoch.
- ❌ No **throughput (img/s)** at training time, only `it/s`.
- ❌ No **per-image latency distribution plot** (only mean/std/P95 scalars).
- ❌ No **failure-case taxonomy** (occlusion / lighting / scale / pose) image grid.
- ❌ No `runs/` lineage manifest tying a checkpoint → exact code commit → exact data split.

Section 11 is the full corrected catalog.

---

## 11. Complete Metrics Catalog (MANDATORY for every final-month run)

This is the contract. Every training/inference run for the paper MUST log every applicable item below. If a metric cannot be computed for a given technique (e.g., mAP for raw SAHI), the run script MUST emit a line: `[SKIPPED] <metric_name> — <reason>` to `<run_dir>/skipped_metrics.txt` so reviewers see we did not silently drop it.

### 11.1 Detection-quality metrics (per-split: val + test)

| Metric | Definition | Why we need it |
|---|---|---|
| `precision` | TP / (TP+FP) at conf=0.001 | Standard |
| `recall` | TP / (TP+FN) at conf=0.001 | Standard |
| `F1` | 2PR/(P+R) | SAR cares about balance |
| `F2` | 5PR/(4P+R) | SAR favors recall — F2 is the right primary metric |
| `mAP@0.5` | mean AP at IoU=0.5 | Ultralytics-standard |
| `mAP@0.5:0.95` | COCO-mean over IoU∈{0.50,0.55,…,0.95} | Localization quality |
| `AP_small` (COCO) | mAP@0.5:0.95 for GT area < 32² px | Headline for tiny-person paper |
| `AP_medium` | 32² ≤ area < 96² | Sanity — should not regress |
| `AP_large` | area ≥ 96² | Sanity |
| `AR_1`, `AR_10`, `AR_100` | Average Recall at 1/10/100 dets per image | Required by COCO eval |
| `AR_small`, `AR_medium`, `AR_large` | Size-broken AR | C2A is small-heavy — VT recall matters |
| `per-size recall` (very-tiny / tiny / small / medium / large) | Keep our existing area bins (<8², 8–16², 16–32², 32–96², ≥96²) **in addition to** COCO bins | We've reported these in all progress slides — keep continuity |
| `mAP@0.75` | Strict-IoU mAP | Often requested by reviewers |
| `OptThr_F1`, `OptThr_F2` | Confidence threshold that maximizes F1 / F2 | Deployment-relevant |
| `Best_F1`, `Best_F2` (at opt threshold) | Peak F1 / F2 | |
| `PR_curve_points` | (precision, recall) pairs sweeping confidence ∈ [0, 1] step 0.01 | Save as CSV, plot as PNG |
| `F1_vs_conf_curve` | F1 at every confidence step | Save as CSV, plot as PNG |
| `confidence_histogram` | Bin counts of all prediction confidences | Save as CSV, plot as PNG |
| `TP`, `FP`, `FN`, `TN` (where applicable) | Confusion counts at default conf=0.25 AND at opt threshold | Required for McNemar + paired tests |
| `confusion_matrix` | 2×2 (person vs background) | PNG + CSV |
| **Bootstrap 95 % CI** on `mAP50`, `mAP50-95`, `F1`, `F2`, `AR_small`, `VT_recall` | Resampled over images, B=1000 | Reviewers ask for this in 2025 |

### 11.2 Training-dynamics metrics (logged every epoch)

| Metric | Source |
|---|---|
| `train/box_loss`, `train/cls_loss`, `train/dfl_loss`, `train/total_loss` | Ultralytics |
| `val/box_loss`, `val/cls_loss`, `val/dfl_loss`, `val/total_loss` | Ultralytics |
| `lr/pg0`, `lr/pg1`, `lr/pg2` | Ultralytics |
| `epoch_time_s` | Custom callback |
| `samples_per_sec` | `len(train_loader.dataset) / epoch_time_s` |
| `gpu_vram_peak_MB` | `torch.cuda.max_memory_allocated() / 1024²`, reset every epoch |
| `gpu_util_avg_%`, `gpu_util_max_%` | `nvidia-smi --query-gpu=utilization.gpu --format=csv -l 1` sampled in background |
| `gpu_temp_avg_C`, `gpu_temp_max_C` | `nvidia-smi --query-gpu=temperature.gpu` |
| `gpu_power_avg_W`, `gpu_power_max_W` | `nvidia-smi --query-gpu=power.draw` |
| `cpu_util_avg_%` | `psutil.cpu_percent(interval=None)` in background |
| `ram_used_GB` | `psutil.virtual_memory().used` |
| `grad_norm_total` | Custom callback — sum of L2 norms across all params |
| `nan_count_this_epoch` | NaNStop callback counter |
| Whether **early stop** was triggered (and by which criterion: built-in mAP patience vs custom F2 patience) | Custom callback |

### 11.3 Efficiency / deployment metrics (per model, run on test split)

| Metric | Definition |
|---|---|
| `params_total_M` | `sum(p.numel() for p in model.parameters()) / 1e6` |
| `params_trainable_M` | `sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6` |
| `gflops` | `thop.profile(model, inputs=(dummy,))` ÷ 1e9, at fixed 1×3×640×640 |
| `layers_total` | Count of `nn.Module` leaves |
| `weights_size_MB` | `os.path.getsize(best.pt) / 1024²` |
| `latency_mean_ms`, `latency_std_ms`, `latency_p50_ms`, `latency_p95_ms`, `latency_p99_ms` | Over ≥500 single-image forward passes after 50 warmup passes; `torch.cuda.synchronize()` between |
| `throughput_imgs_per_sec` | Batched, batch=1, batch=4, batch=8, batch=16 |
| `fps_at_resolutions` | Latency at 320, 480, 640, 800, 1280 input sizes |
| `vram_inference_MB` | Peak VRAM during inference |
| `cpu_inference_latency_ms` | Same model on CPU at batch=1 (for "is it deployable") |
| `export_onnx_OK` (bool) + `onnx_size_MB` | `model.export(format='onnx')` succeeded |

### 11.4 Calibration metrics (per-split)

| Metric | Definition |
|---|---|
| `ECE` (Expected Calibration Error) | Already computed; 10-bin |
| `MCE` (Maximum Calibration Error) | Worst bin |
| `Brier_score` | Mean squared error between confidence and correctness |
| `calibration_curve_points` | (mean_conf, mean_acc, count) per bin — CSV + PNG |
| `high_FN_image_count` | Images with FN > 5 |
| `high_FP_image_count` | Images with FP > 5 |

### 11.5 Per-image artifacts (kept for the paper's qualitative section)

- `detection_grid.png` — 8×8 grid of test images with GT (green) + Pred (red/orange by conf), labeled with image filename.
- `failure_grid.png` — 16 worst-FN-rate images.
- `success_grid.png` — 16 highest-F1 images.
- A **failure-case taxonomy** CSV with columns `image, FN_count, FP_count, primary_failure_mode` where mode ∈ {occlusion, scale, lighting, motion-blur, dense-crowd, edge-of-frame}. Tag manually for ≥30 representative failures — reviewers love this.

### 11.6 Architecture-specific metrics (REQUIRED — see Section 22 for *why*)

Some metrics only exist for certain architectures. Each variant MUST emit its own subset:

**Attention modules (ECA / CBAM):**
- `attention_map_examples.png` — overlay of attention heatmap on 6 representative test images (Grad-CAM++ on the post-attention conv).
- For CBAM only: `channel_attention_weights.csv` (mean channel-attention vector across val set) and `spatial_attention_entropy` (mean entropy of spatial-attention map).

**P2 detection head:**
- `per-stride_AP` — AP broken down per detection head (P2, P3, P4, P5). Use the head index of each prediction.
- `tiny_obj_recall_by_head` — which head actually catches the <8 px detections.

**Mamba / SSM blocks (LocalWindowSSM, C3K2Mamba, AtrousSSM):**
- `ssm_state_norm` — mean L2 norm of the SSM hidden state across a calibration batch (sanity: should be bounded, no explosion).
- `forward_vs_backward_scan_disagreement` — cosine distance between forward and reverse scan outputs (we already test this in the smoke test — log it as a metric for the final model).
- `window_size_per_layer` — record `_get_window_size(C)` per injected layer.
- `dilation_branch_contribution` (AtrousMamba only) — per-branch (d=1, d=2, d=4) magnitude after gated fusion — shows which dilation actually matters.
- `injection_layer_indices` — exact layer indices where SSM blocks replaced C3k2 (e.g., `[13, 16, 19, 22, 25, 28]`).

**SAHI runs:**
- `slice_size`, `overlap_ratio`, `match_metric` (IoU / IoS), `postprocess` (NMS / GREEDYNMM), `confidence_threshold`, `perform_standard_pred` (bool).
- `slices_per_image_mean`, `total_slices_processed`.
- `per-image SAHI vs no-SAHI delta` (paired CSV — one row per image).

**TTA runs:**
- `tta_imgsz`, `tta_scales`, `tta_flips` — exact config.
- `per-augmentation contribution` — disable each aug one at a time and rerun (mini-ablation).

### 11.7 Environment & reproducibility metrics (logged ONCE per run, into `env.json`)

```json
{
  "run_id": "<see Section 16>",
  "timestamp_utc": "...",
  "git_commit": "...",
  "git_dirty": true,
  "python_version": "...",
  "torch_version": "...",
  "cuda_version": "...",
  "cudnn_version": "...",
  "ultralytics_version": "...",
  "numpy_version": "...",
  "opencv_version": "...",
  "sahi_version": "...",
  "platform": "...",
  "os": "...",
  "gpu_name": "...",
  "gpu_vram_gb": ...,
  "gpu_driver": "...",
  "cpu_model": "...",
  "ram_gb": ...,
  "random_seed": ...,
  "torch_deterministic": true,
  "cudnn_deterministic": true,
  "dataset_split_md5": "<hash of sorted list of image filenames>",
  "model_yaml_md5": "<hash of the .yaml architecture file>",
  "weights_md5": "<sha256 of best.pt>",
  "cli_argv": "...",
  "hyperparameters": { ... full dict ... }
}
```

### 11.8 Energy / carbon (REQUIRED for a 2025-onwards paper)

Use **`codecarbon`** (`pip install codecarbon`) and `pynvml` for ground-truth wattage. Log:
- `training_energy_kWh`
- `training_co2_kg` (CodeCarbon takes country grid intensity automatically; pass `country_iso_code='BGD'` for Bangladesh)
- `training_wallclock_hours`
- `gpu_hours` (= wallclock × num_gpus, here = wallclock × 1)
- Per-epoch `energy_kWh` and `co2_g` snapshot.

A reviewer in 2026 *will* ask about this. Don't get caught.

---

## 12. Statistical Significance Testing Protocol (REQUIRED)

A 2025 study showed that the paired-bootstrap protocol routinely fails to declare significance for 0.6–2.0-point mAP improvements with only 3 seeds. Our ablation deltas (e.g., +1.29 % small-obj recall for CBAM, +2.12 % mAP50 for Mamba+CBAM+P2) are in exactly this danger zone. Every "improvement" claim in the paper MUST be backed by one of the tests below.

### 12.1 Multi-seed requirement

For each model in the final ablation table:
- Train with **≥ 5 seeds** (recommended 5; gold-standard 10). Seeds: `[0, 1, 2, 3, 4]` for the 5-seed protocol.
- Keep **identical** data split, hyperparameters, hardware, software versions across seeds. Only the seed changes.
- For inference-only methods (SAHI, TTA) seeds are unnecessary because they are deterministic given the weights — but run the **paired comparison** on per-image scores instead.

Pin the seed everywhere:
```python
import random, numpy as np, torch, os
def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark    = False
    os.environ["PYTHONHASHSEED"] = str(s)
```

### 12.2 Tests to run

For each *pair* (e.g., `Baseline` vs `+CBAM`, `+CBAM+P2` vs `+Mamba+CBAM+P2`):

| Test | When to use | What it tells you | Library |
|---|---|---|---|
| **Paired bootstrap** (resample images **with replacement**, B=10 000, paired per image) | Comparing two models on the **same** test images for any metric (mAP, F1, recall, etc.) | 95 % CI of (model_A − model_B); p-value (% of resamples where sign flips) | Custom (15 lines) — see snippet below |
| **Sign-flip / permutation test on per-seed deltas** | Comparing 5-seed runs (5 paired deltas) | Whether the median delta is reliably > 0 | `scipy.stats.permutation_test` |
| **Paired t-test on per-seed metric** | Same as above when ≥5 seeds and metric is ~normal | p-value, with t-statistic | `scipy.stats.ttest_rel` |
| **Wilcoxon signed-rank** | Non-parametric alternative to paired t-test (skewed metrics like latency) | Non-parametric p-value | `scipy.stats.wilcoxon` |
| **McNemar's test** | Per-image binary outcome (image classified correctly or not at conf=0.25) — discrete TP/FN swaps | p-value on whether two detectors disagree systematically | `statsmodels.stats.contingency_tables.mcnemar` |
| **BCa bootstrap CI** | Reporting **absolute** metric of a single model (not a comparison) | 95 % CI to put next to every mAP number in the paper | `scipy.stats.bootstrap(method='BCa')` |
| **DeLong test** | Comparing two AUCs / PR-AUCs | Whether two curves differ | `sklearn`-based community implementations |

### 12.3 Reporting convention (paper-ready)

For every comparison in the ablation table the row MUST show:
```
mAP50 = 0.877 [0.864, 0.889]   Δ vs baseline = +0.021 [+0.009, +0.034]   p = 0.003  (paired-bootstrap, B=10 000)
```
- Square brackets = 95 % CI.
- Δ is paired — same test images.
- `p < 0.05` → bold the delta. `p < 0.01` → bold + asterisk. `p ≥ 0.05` → italicize and say "not significant".
- Multiple-comparison correction: when reporting ≥ 5 ablation rows against the same baseline, apply **Holm-Bonferroni** correction (`statsmodels.stats.multitest.multipletests(p, method='holm')`).

### 12.4 Minimal paired-bootstrap snippet (drop into eval scripts)

```python
import numpy as np
from typing import Sequence

def paired_bootstrap(per_image_A: Sequence[float],
                     per_image_B: Sequence[float],
                     B: int = 10_000, seed: int = 0):
    """Returns (mean_delta, ci_low, ci_high, p_two_sided)."""
    rng   = np.random.default_rng(seed)
    a     = np.asarray(per_image_A, dtype=np.float64)
    b     = np.asarray(per_image_B, dtype=np.float64)
    n     = len(a)
    idx   = rng.integers(0, n, size=(B, n))
    deltas = (a[idx] - b[idx]).mean(axis=1)
    obs    = a.mean() - b.mean()
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    p     = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return float(obs), float(ci_lo), float(ci_hi), float(p)
```

Save the output as `significance/<modelA>_vs_<modelB>.json`.

### 12.5 What "per-image score" means for each metric

- **F1, F2, Precision, Recall** — already per-image (we compute them in `evaluate_model_comprehensive`).
- **mAP50, mAP50-95** — bootstrap on the per-image AP contributions; or use the COCOeval `eval_imgs` array (each entry is a single (img, cat) pair).
- **Latency** — paired per-image inference times → Wilcoxon signed-rank is appropriate.

---

## 13. Visualization Requirements — every metric must have a plot

**Rule:** if it lives in a CSV / Excel cell, it MUST also live in a PNG. Reviewers and the thesis committee read images, not numbers.

### 13.1 Mandatory plots (per run)

| Plot | Source data | Filename pattern |
|---|---|---|
| Loss curves (box / cls / dfl, train + val) | `results.csv` | `<run_id>_loss_curves.png` |
| Total loss train vs val | computed | `<run_id>_total_loss.png` |
| Metric panel (P, R, mAP50, mAP50-95, F1, F2 vs epoch — 2×3 grid) | `results.csv` | `<run_id>_metrics_6panel.png` |
| Learning-rate schedule | `results.csv` lr columns | `<run_id>_lr_schedule.png` |
| GPU VRAM / power / temperature vs epoch | system monitor | `<run_id>_resource_usage.png` |
| PR curve (test split) | `pr_curve.csv` | `<run_id>_pr_curve.png` |
| F1 vs confidence curve | `f1_vs_conf.csv` | `<run_id>_f1_conf.png` |
| Confidence histogram | `confidence_hist.csv` | `<run_id>_conf_hist.png` |
| Confusion matrix | `confusion.csv` | `<run_id>_confusion.png` |
| Calibration / reliability diagram | `calibration.csv` | `<run_id>_calibration.png` |
| Per-size recall bar chart (very-tiny → large) | `per_size.csv` | `<run_id>_per_size_recall.png` |
| Latency vs input resolution | `latency_by_res.csv` | `<run_id>_latency_resolution.png` |
| Latency per-image histogram | `per_image_latency.csv` | `<run_id>_latency_hist.png` |

### 13.2 Mandatory plots (cross-run / paper-table-style)

| Plot | Description | Filename |
|---|---|---|
| Multi-model PR overlay | All ablation models' PR curves on one axis | `compare_pr_overlay.png` |
| Multi-model metric bars (with error bars from bootstrap CIs) | mAP50, mAP50-95, F1, F2, VT recall | `compare_metric_bars.png` |
| Pareto plot: params (x) vs mAP50-95 (y), bubble size = latency | One bubble per model | `pareto_params_vs_map.png` |
| Speed-accuracy trade-off | latency (x) vs mAP50-95 (y) | `pareto_speed_vs_acc.png` |
| Significance heatmap | model × model matrix, cell color = p-value | `significance_heatmap.png` |
| Per-seed boxplot | mAP50-95 distribution per model across seeds | `seed_variance_boxplot.png` |
| Waterfall improvement plot | baseline → +CBAM → +P2 → +Mamba → +TTA | `improvement_waterfall.png` |

### 13.3 Style rules

- DPI ≥ 300 for any plot that may go in the paper; DPI 200 acceptable for diagnostic plots.
- Use a **single global style** (`matplotlib.style.use('seaborn-v0_8-whitegrid')` + a fixed color map). Save the style to `plots/_style.py`.
- Colors per model MUST be consistent across every plot in the run. Define once in `plots/_palette.py`.
- Always label axes with units (ms, %, MB, etc.). Never publish a default `matplotlib` axis label.
- Use `plt.tight_layout()` and `bbox_inches='tight'` on `savefig`.

---

## 14. Graph-Data Provenance Manifest (so you can recreate any plot later)

For every PNG saved, the script MUST also write **the exact CSV/JSON that produced it**, AND a one-line entry to `plots/PLOTS_INDEX.md`:

Example `PLOTS_INDEX.md` line:
```
- <run_id>_pr_curve.png  ←  excel_reports/<run_id>/pr_curve.csv  (cols: confidence, precision, recall) — produced by: scripts/plot_pr.py
```

Maintain this file religiously. When you're writing the paper and need to tweak a plot's font size, you must be able to grep one line and find the data + the producing script.

Also produce a sibling **`plots/PLOTS_INDEX.csv`** (machine-readable) with columns:
`plot_filename, data_file, data_columns, producing_script, run_id, timestamp`.

---

## 15. Output Folder Structure

All run outputs live UNDER `OUTPUT_ROOT` (Section 5.2 — the directory of the running script). No absolute paths anywhere in this layout: the tree below describes the structure *relative to* `OUTPUT_ROOT`, regardless of which machine, drive, or folder the code happens to be in.

The dataset lives **elsewhere** (discovered via `DATASET_ROOT`, Section 5.3) and is NOT inside this tree. If shared assets like pretrained weights, dataset YAMLs, and frozen splits need to live with the code, they go in a `common/` directory next to the script (still under `OUTPUT_ROOT`); the dataset images/labels themselves stay where Section 5.3 finds them.

```
<OUTPUT_ROOT>/                           # = SCRIPT_DIR (Section 5.2); drive/parent unknown by design
├── system_spec.md                       # this file (kept next to the code it governs)
├── common/                               # shared assets that travel with the code (NOT the dataset images)
│   ├── yolo11m.pt                        # pretrained imagenet weights
│   ├── c2a.yaml                          # Ultralytics data YAML — points at DATASET_ROOT via env / Section 5.3
│   └── splits/                           # frozen train/val/test image lists + md5
│       ├── train_images.txt
│       ├── val_images.txt
│       ├── test_images.txt
│       └── splits.md5
├── runs/                                # ALL training runs land here
│   └── <run_id>/                        # see Section 16 for run_id pattern
│       ├── code/                        # snapshot of the .py / .yaml / .ipynb at launch
│       ├── env.json                     # Section 11.7
│       ├── hyperparams.yaml
│       ├── weights/
│       │   ├── best.pt
│       │   ├── last.pt
│       │   └── epoch_XXX.pt             # periodic checkpoints (Section 20)
│       ├── results.csv                  # ultralytics
│       ├── metrics/                     # all CSVs from Section 11
│       │   ├── per_image_test.csv
│       │   ├── per_image_val.csv
│       │   ├── per_size.csv
│       │   ├── pr_curve.csv
│       │   ├── f1_vs_conf.csv
│       │   ├── confidence_hist.csv
│       │   ├── confusion.csv
│       │   ├── calibration.csv
│       │   ├── latency_by_res.csv
│       │   ├── per_image_latency.csv
│       │   ├── summary.json             # all headline numbers in one file
│       │   └── summary.xlsx             # human-readable
│       ├── plots/                       # all PNGs from Section 13
│       │   ├── PLOTS_INDEX.md
│       │   └── PLOTS_INDEX.csv
│       ├── architecture/                # Section 11.6 + Section 22
│       │   ├── model_summary.txt        # torchinfo
│       │   ├── module_table.csv
│       │   ├── flops_breakdown.csv
│       │   ├── attention_maps/*.png
│       │   ├── ssm_state_norms.csv
│       │   └── injection_layers.json
│       ├── significance/                # Section 12
│       │   ├── <runA>_vs_<runB>.json
│       │   └── ...
│       ├── energy/                      # Section 11.8
│       │   ├── emissions.csv            # codecarbon
│       │   └── nvml_log.csv
│       ├── logs/
│       │   ├── train.log                # stdout (tee'd)
│       │   ├── train.err
│       │   ├── nvidia_smi_loop.csv
│       │   ├── psutil_loop.csv
│       │   └── skipped_metrics.txt
│       └── manifest.json                # run_id, parent_run_id (for resumes), code commit, weights md5
├── ablation_master/                     # cross-run aggregation (built after all runs done)
│   ├── master_table.xlsx                # every run × every metric
│   ├── compare_pr_overlay.png
│   ├── pareto_params_vs_map.png
│   ├── significance_heatmap.png
│   └── paper_tables/                    # LaTeX-ready tables
│       ├── ablation_table.tex
│       ├── efficiency_table.tex
│       └── significance_table.tex
├── smoke/                               # smoke-test outputs (deleted after each successful smoke pass — see Section 17)
└── docs/                                # responses, decisions, notes (per feedback_save_responses memory)
```

### 15.1 Resolving paths at runtime

- `OUTPUT_ROOT` comes from Section 5.2 (derived from `SCRIPT_DIR`). Every output (`runs/`, `smoke/`, `logs/`, `plots/`, `docs/`, `ablation_master/`) is created **relative to `OUTPUT_ROOT`**.
- `DATASET_ROOT` comes from Section 5.3 (env-var override → ordered candidate probe). The dataset is **not** inside `OUTPUT_ROOT`.
- Per-run working directory:
  ```python
  from pathlib import Path

  # OUTPUT_ROOT → Section 5.2
  # DATASET_ROOT → Section 5.3
  # RUN_ID → Section 16
  WORK_ROOT = OUTPUT_ROOT / "runs" / RUN_ID
  WORK_ROOT.mkdir(parents=True, exist_ok=True)
  ```
- Do NOT hardcode any absolute path anywhere (no `D:\…`, no `E:\…`, no `/kaggle/working/…`). If the user moves the code to a new AnyDesk PC, a different drive, or a worktree, outputs follow the code and the dataset is rediscovered via Section 5.3.

---

## 16. File naming, timestamps, run IDs

### 16.1 Run ID format

```
<YYYYMMDD>_<HHMMSS>_<model_tag>_<seed>_<short_hash>
e.g.  20260603_143022_mamba_cbam_p2head_s0_e7a91c
```
- Timestamp = local time at launch.
- `model_tag` = snake_case (e.g. `yolo11m_baseline`, `yolo11m_cbam`, `mamba_cbam_p2head`, `atrousmamba_cbam_p2head`, `mamba_cbam_p2head_sahi512`).
- `seed` = `s<N>`.
- `short_hash` = first 6 chars of `git rev-parse HEAD` (or `"dirty"` if working tree is dirty; we still log dirty runs, just flag them).

### 16.2 Checkpoint filename

```
weights/epoch_<EEE>__<YYYYMMDD-HHMMSS>__map50_<0.XXXX>__map5095_<0.XXXX>.pt
```
Long names are fine. The point is you can `ls` and immediately know which checkpoint is which.

### 16.3 Every output file gets a `run_id` prefix

Inside `runs/<run_id>/plots/`, files are prefixed `<run_id>_…` so that if files ever get copy-pasted around they don't collide.

---

## 17. Smoke-Test Protocol (mandatory before every full run)

The existing `mamba_cbam_p2_copypaste.py` has a 7-step smoke test (shape, NaN, gradient, AMP, bidirectional asymmetry, VRAM estimate). **Keep that. Extend it.**

### 17.1 Smoke-test contract

Before any full-resource training run starts, the script MUST:

1. **Run on GPU** (NOT CPU) — smoke must use the same device and AMP setting as the full run. A CPU-only smoke is useless for catching CUDA shape / memory bugs.
2. **Use a tiny but real slice of the dataset.** Configurable: `SMOKE_FRACTION` (default 0.01 = 1 % of images), `SMOKE_EPOCHS` (default 2). For C2A's ~10 215 train images that's ~100 images.
3. **Run all metric-computation paths.** If the full run computes mAP, per-size recall, calibration, latency, and SAHI inference — the smoke must compute the same things (just on 100 images). This catches downstream-pipeline bugs that don't show up in the model-only smoke.
4. **Run all plot-saving paths.** Save the plots to `smoke/<run_id>_smoke/plots/`. Verify each expected PNG exists and is non-empty.
5. **Run all CSV/Excel/JSON saving paths.** Verify each expected file exists and is parseable.
6. **Verify the manifest** (Section 14) — every plot has a row, every row has an existing data file.
7. **Run the env.json snapshot** (Section 11.7) — confirms git, package versions, etc., are captured.
8. **Time it.** Smoke MUST finish in < 5 minutes on the 4070 Ti SUPER. If it takes longer, the test fraction is too large.

### 17.2 Cleanup mandate

On smoke success: print summary, **delete** the entire `smoke/<run_id>_smoke/` directory (or move to `smoke/_passed/` with a 24-h auto-purge). The user does not want smoke artifacts polluting their disk.

On smoke failure: **keep** the directory, print the failure cause + the location of every file produced, and **abort the full run**. No silent fallback.

### 17.3 Suggested control flags (every script must support)

```python
SMOKE_TEST     = False       # True = run smoke only, do not train full
SMOKE_FRACTION = 0.01        # fraction of train/val/test to use during smoke
SMOKE_EPOCHS   = 2
SMOKE_BATCH    = None        # None = use full-run BATCH_SIZE; override if needed
SMOKE_KEEP_OUTPUTS = False   # True = keep smoke/ folder after success (debug)
```

### 17.4 Hard rule
**No `SMOKE_TEST = False` ever ships without a passing smoke run in the same session.** The script must record a smoke timestamp in `manifest.json` and refuse to start the full run if the smoke timestamp is missing or older than 24 h.

---

## 18. Dynamic Epochs + Early Stopping (you have compute — use it)

On Kaggle you ran 25 / 50 / 120 epochs because of session caps. Locally there is no cap. Switch to **dynamic** training:

### 18.1 Strategy
- Set `NUM_EPOCHS = 300` as an *upper bound* (not a target).
- Rely on **two** stopping criteria simultaneously:
  1. **Ultralytics built-in `patience`** on `fitness` (= mAP50-95-dominated combo) — set `patience=30` (was 15 on Kaggle).
  2. **Custom F2 patience** — F2 is the SAR-relevant metric. Set `F2_PATIENCE=20`. (Already implemented as `F2EarlyStopCallback`.)
- Add a **plateau-aware LR schedule** as backup: `cos_lr=True` with `lrf=0.01` (Ultralytics default cosine) — gives the model many "second chances" before the patience timer fires.

### 18.2 Logging requirements
- Log which criterion stopped training, and at which epoch.
- Plot a vertical line on the metric-panel PNG marking the early-stop epoch.

### 18.3 Don't accidentally undertrain
- If smoke shows the loss is still decreasing at the end of `SMOKE_EPOCHS=2`, that's normal. Smoke is not for convergence — only for plumbing.
- For the full run: confirm that the final epoch is **at least** `patience + 10` away from the best-fitness epoch. If not, training stopped too early — re-launch with a longer patience.

---

## 19. GPU / Resource Utilization Maximization

Symptom you've reported: "sometimes GPU shows 0 %". That's a bottleneck somewhere else (CPU, disk, dataloader). Address it before each run.

### 19.1 Diagnostic
Run a background sampler during every training run:
```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv -l 2 > logs/nvidia_smi_loop.csv &
```
After the run, **plot the GPU-utilization timeline**. If the mean is < 80 %, the run is bottlenecked.

### 19.2 Fixes (apply in order)

1. `num_workers=8` and `persistent_workers=True`, `pin_memory=True`.
2. `cache='ram'` (we have 100 GB free — use it). For C2A train set (~2–3 GB on disk), full RAM cache is trivial.
3. `amp=True` (BF16 preferred on Ada).
4. Largest batch that fits — saturates Tensor cores.
5. `torch.backends.cudnn.benchmark = True` (training, fixed shapes).
6. Disable any heavy CPU augmentation that the GPU is waiting on; move what you can to GPU-side aug.
7. If still bottlenecked, profile with `torch.profiler` (one-epoch profile saved to `logs/profile.pt.trace.json`).

### 19.3 Target metrics for "fully utilized"
- GPU utilization avg ≥ 85 %.
- VRAM peak between 12 GB and 14.5 GB (leave 1.5 GB headroom).
- Power draw average ≥ 220 W (close to the 285 W TGP).
- Epoch time stable (std across epochs < 5 % of mean).

If any of these is off, FIX it before launching the final run — re-running for low-util data is exactly what we're trying to avoid.

---

## 20. OOM, Power-Failure, and Error Recovery

### 20.1 OOM (out-of-memory) — automatic retry

The existing copypaste script has `OOM_RETRY_BATCHES = [BATCH_SIZE, BATCH_SIZE//2, BATCH_SIZE//4]`. Keep this. Extend to:

```python
def train_with_oom_retry(model, kwargs):
    last_err = None
    for bs in OOM_RETRY_BATCHES:
        kwargs["batch"] = bs
        kwargs["accumulate"] = max(1, 16 // bs)
        torch.cuda.empty_cache(); gc.collect()
        try:
            return model.train(**kwargs)
        except torch.cuda.OutOfMemoryError as e:
            last_err = e
            print(f"OOM at batch={bs}. Trying smaller…")
            torch.cuda.empty_cache(); gc.collect()
    raise RuntimeError(f"OOM at every batch size. Last: {last_err}\n"
                       "Try: imgsz 640->512; close_mosaic earlier; disable cache='ram'.")
```

### 20.2 Power failure / unexpected reboot — resume from checkpoint

- `save_period=5` in Ultralytics → checkpoint every 5 epochs.
- A `SessionCheckpointManager` callback should also `shutil.copy2(last.pt, runs/<run_id>/weights/last_safety.pt)` every epoch — the Kaggle runs already did this; **keep it locally too**. Disk is cheap.
- On script restart: detect `runs/<run_id>/weights/last.pt`, set `resume=True` in Ultralytics. Log a `[RESUMED]` line to `train.log` and append a `resume_history` array to `manifest.json`.
- Bangladesh load-shedding is real. UPS is recommended. In the absence of UPS, accept that the worst case is losing 5 epochs of progress, not the whole run.

### 20.3 Other common errors — what to print + what to do

| Error | Likely cause | Action |
|---|---|---|
| `RuntimeError: CUDA out of memory` | Batch too large | Auto-retry chain above |
| `RuntimeError: CUDA error: device-side assert triggered` | Label index out of range / NaN in loss | Print last batch, set `CUDA_LAUNCH_BLOCKING=1`, re-run smoke |
| `RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED` | Driver / torch mismatch | Re-check Section 11.7 versions; reinstall correct torch wheel for the driver's CUDA |
| `ValueError: NaN in loss` | LR too high / AMP overflow / corrupt label | NaNStop callback aborts; re-run with `amp=False` once to diagnose |
| `DataLoader worker (pid …) exited unexpectedly` | num_workers too high / shared memory | Drop `num_workers` from 8 to 4 |
| `FileNotFoundError: dataset path` | Forgot to copy data / wrong env | Auto-detect Kaggle vs local (Section 15.1) |
| `OSError: [WinError 1455] paging file too small` | Windows-specific | Increase pagefile or drop `num_workers`/cache |
| `RuntimeError: Mamba shape mismatch` | Window size doesn't divide H | Smoke test 1 catches it before training |

Each handler MUST log the full traceback to `logs/train.err` AND print a short hint to stdout.

---

## 21. Package / Dependency Check Mandate

Every script must, at startup, **verify (not assume)** that every package it will import is installed, and at a compatible version. The existing `pip_install()` helper is the right idea — institutionalize it.

### 21.1 Standard preamble (paste at top of every script)

```python
REQUIRED = [
    ("ultralytics", ">=8.3.0"),
    ("torch",       ">=2.4.0"),
    ("timm",        ">=1.0.0"),
    ("sahi",        ">=0.11.0"),
    ("thop",        None),
    ("openpyxl",    None),
    ("pandas",      "<3.0"),
    ("matplotlib",  "<3.10"),
    ("scikit-learn",None),
    ("scipy",       None),         # significance tests
    ("statsmodels", None),         # McNemar, multipletests
    ("codecarbon",  None),         # energy
    ("pynvml",      None),         # GPU sampler
    ("psutil",      None),         # CPU/RAM sampler
    ("torchinfo",   None),         # model summary
    ("tqdm",        None),
    ("opencv-python", None),
    ("PyYAML",      None),
    ("seaborn",     None),         # significance heatmap
    ("tabulate",    None),         # pretty CLI tables
]

def ensure_packages():
    import importlib, subprocess, sys
    for name, ver in REQUIRED:
        try:
            importlib.import_module(name.replace("-", "_").split("[")[0])
        except ImportError:
            spec = name + (ver if ver else "")
            print(f"  Installing missing package: {spec}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", spec])
ensure_packages()
```

### 21.2 Don't shim, don't try/except-pass

If a package needed for a metric is missing AND cannot be installed automatically, the script MUST:
- Log `[METRIC SKIPPED] <metric> — package <pkg> unavailable` to `skipped_metrics.txt`,
- Continue with the rest,
- Exit code 0 only if the **core** metric set (Section 11.1 detection metrics) succeeded.

Never silently zero out a missing metric. Reviewer asking "why is CO₂ 0.000?" is the kind of question we are avoiding.

---

## 22. Model Parameter Reporting (for the paper)

Every run produces a `architecture/model_summary.txt` using **`torchinfo`** (not the Ultralytics minimal print). Captures:

```python
from torchinfo import summary
s = summary(model, input_size=(1, 3, 640, 640),
            col_names=("input_size","output_size","num_params","mult_adds","trainable"),
            depth=4, verbose=0)
with open(f"{ARCH_DIR}/model_summary.txt", "w") as f:
    f.write(str(s))
```

Plus a **module-level CSV** (`architecture/module_table.csv`) with columns:
`layer_idx, name, type, input_shape, output_shape, params, mult_adds, trainable`

And a **FLOPs breakdown CSV** (`architecture/flops_breakdown.csv`) split by `backbone / neck / head / attention / ssm` — paper readers want to see "the Mamba neck costs X GFLOPs".

Also write `architecture/architecture_overview.md` — a short markdown summarizing in plain language:
- What was replaced vs the YOLO11m baseline.
- Where the SSM / CBAM / P2 modules sit (layer indices).
- Total params, GFLOPs, layer count.
- Hyperparameters that affect the architecture (d_state, window_size, dilations, etc.).

---

## 23. Cross-Run Comparability Contract

To make Sections 11–12 meaningful, every run in the final ablation MUST share:

1. **The same dataset split.** Frozen once. MD5 of sorted filenames committed to `common/splits/splits.md5`. Any run whose split-md5 differs is excluded from the comparison.
2. **The same test images.** Per-image metrics depend on this — paired tests collapse if the images differ.
3. **The same image preprocessing** at inference (letterbox vs stretch, same `imgsz`).
4. **The same evaluation script.** Don't have one model evaluated with Ultralytics val() and another with the custom per-image loop and expect them to be paired-bootstrap comparable.
5. **The same conf / iou thresholds for "headline" numbers.** Stick to `conf=0.001, iou=0.7` for AP-style metrics (COCO standard) and `conf=0.25, iou=0.5` for "operational" F1/F2.
6. **The same hardware** (this machine). Re-run anything that was previously measured on T4×2 if you're going to claim a latency number.

If a metric cannot be computed under those constraints (e.g., SAHI doesn't produce a standard mAP), **say so in the table** with an explicit footnote — don't omit it silently.

---

## 24. Master Pre-Run Checklist (the agent's pre-flight)

Before launching ANY full training/eval run, an agent must satisfy ALL items. Auto-fail and refuse to start otherwise.

- [ ] System sanity (Section 8) — GPU is 4070 Ti SUPER, 16 GB, capability 8.9.
- [ ] `ensure_packages()` passed (Section 21).
- [ ] `env.json` snapshot written (Section 11.7).
- [ ] Random seed set everywhere (Section 12.1).
- [ ] Dataset split-md5 matches `common/splits/splits.md5` (Section 23).
- [ ] `SMOKE_TEST=True` run passed in last 24 h AND all smoke outputs verified (Section 17).
- [ ] Smoke artifacts cleaned up.
- [ ] `OOM_RETRY_BATCHES` configured (Section 20.1).
- [ ] Checkpoint cadence configured: `save_period=5`, last-safety copy every epoch (Section 20.2).
- [ ] Resource sampler launched in background (`nvidia-smi -l 2`, `psutil` sampler) → `logs/nvidia_smi_loop.csv` (Section 19.1).
- [ ] CodeCarbon emissions tracker started (Section 11.8).
- [ ] All metric-CSV writers initialized for the targets in Section 11.
- [ ] `PLOTS_INDEX.md` initialized empty (Section 14).
- [ ] `manifest.json` has `run_id`, `git_commit`, `parent_run_id` (if resume), `smoke_passed_at`.
- [ ] Disk free space ≥ 20 GB.
- [ ] Confirmed: GPU util target ≥ 85 % (Section 19.3) — re-check after first epoch.

---

## 25. Reproducibility & ablation matrix (the actual experiment plan)

Per Section 9.6, **9 runs are mandatory** for the paper. Each across **5 seeds** → 45 model trainings + inference-only variants. Keep this table in `ablation_master/plan.csv` and tick it off as runs complete.

| # | model_tag | seeds | training? | inference-time | priority |
|---|---|---|---|---|---|
| 1 | `yolo11m_baseline` | 0,1,2,3,4 | yes | — | P0 |
| 2 | `yolo11m_cbam` | 0,1,2,3,4 | yes | — | P0 |
| 3 | `yolo11m_p2head` | 0,1,2,3,4 | yes | — | P0 |
| 4 | `yolo11m_cbam_p2head` | 0,1,2,3,4 | yes | — | P0 |
| 5 | `mamba_cbam_p2head` | 0,1,2,3,4 | yes | — | **P0 (headline)** |
| 6 | `atrousmamba_cbam_p2head` | 0,1,2,3,4 | yes | — | P1 (negative result) |
| 7 | `mamba_cbam_p2head` + SAHI-512 | — | no | yes | P0 |
| 8 | `mamba_cbam_p2head` + TTA-1280 | — | no | yes | P0 |
| 9 | `mamba_cbam_p2head` + SAHI-256 + TTA | — | no | yes | P1 |

Wall-clock estimate on the 4070 Ti SUPER (rough, 100 epochs with early-stop, full C2A, batch 8): ~6–10 h per training run → 9 × 5 ≈ 45 × ~8 h ≈ **15 days of GPU time** if all P0 + P1 done. If time-constrained, drop seeds to 3 and re-evaluate the P1s. **Do this math before launching — don't be surprised.**

---

## 26. Things the user did NOT explicitly mention but should NOT skip

These were added based on the brainstorm in the request:

- **Frozen test-set hash.** Every paper says "we used the standard C2A split." Prove it — commit the test-image-filename md5 to git so reviewers can verify.
- **Code commit hash in every output.** No "we used the model from `last week`."
- **Negative results are publishable.** Keep AtrousMamba and CopyPaste in the paper as ablations even though they failed — this is what gives the paper depth and saves reviewers' "did you try X?" questions.
- **A model card.** Write `runs/<run_id>/MODEL_CARD.md` per Mitchell et al. (2019) format — Q1 reviewers increasingly ask for this. 1-page template: intended use, limitations, training data, evaluation, ethical considerations.
- **Dataset datasheet.** Same idea for C2A — a 1-page summary of license, composition, biases. Cite Nihal et al. ICPR 2024.
- **ONNX export sanity.** Export the headline model to ONNX and run one inference — proves the architecture isn't using non-exportable ops (Mamba SSM custom ops sometimes are). Done once per final model.
- **Per-resolution evaluation.** Already partly done in your progress slides — extend to a full sweep at 320 / 480 / 640 / 800 / 1280.
- **Failure-case grid.** 16 images where the model fails worst — paper readers ALWAYS flip to these images.
- **A blind comparison.** Pick 10 random test images, run all 4–5 main models on them, save side-by-side overlays. Reviewer-friendly.
- **License & ethics.** If C2A is CC-BY, cite it. If any human-image dataset, mention you are doing SAR research (defensive / humanitarian) — distinguishes from surveillance.

---

## 27. Sections still to be filled (TODO — append later)

- [x] ~~Storage layout (NVMe paths, dataset locations, free space) — Section 5.~~ → Filled 2026-05-28 (C2A on `E:\`, output auto-detected from script dir).
- [x] ~~Exact dataset paths (C2A, and any others) + sizes.~~ → C2A path captured in Section 5.1; sizes still TODO.
- [ ] C2A on-disk size (GB) — run `Get-ChildItem -Recurse | Measure-Object -Property Length -Sum` and paste.
- [ ] Python / CUDA / cuDNN / PyTorch versions actually installed (run Section 8 snippet, paste output).
- [ ] Conda/venv environment name + `pip freeze` snapshot.
- [ ] Additional datasets (when added) — append a new Section 5.x per dataset with on-disk layout + candidate paths.
- [ ] Kaggle CLI / W&B credentials handling (no secrets in this file).
- [ ] Known-good training commands for each row of the ablation matrix (Section 25).

---

## Agent reminder (final)

Before writing ANY training, inference, or evaluation code for this project, an agent must:

1. Read Sections 3, 7, 11, 12, 15, 17, 21, 24 in full.
2. Match the run against the Section 25 ablation matrix and the Section 9 history — do not re-invent variants.
3. Use the Section 16 run-ID convention. Place every output under the Section 15 folder layout.
4. Implement every Section 11 metric or explicitly skip it via `skipped_metrics.txt`.
5. Produce every Section 13 plot, and a Section 14 provenance manifest.
6. Run the Section 17 smoke before the Section 24 checklist before the full run.
7. When unsure, **ASK** — don't assume Kaggle dual-GPU or 30 GB VRAM. This is one 16 GB Ada card. Don't omit a metric and hope the reviewer doesn't notice. Don't skip seeds and claim significance. Don't save a plot without saving its data.

This file IS the contract. If the agent's output deviates from it, that's a bug in the agent, not in the spec.
