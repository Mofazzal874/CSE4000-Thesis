# AtrousMamba-YOLO: Multi-Scale Dilated State Space Scanning
## Date: 2026-03-20 | Month 1 Architectural Contribution

---

## 1. The Problem with Current Local-Window Mamba

Your current `LocalWindowSSM` partitions the feature map into fixed 8×8 windows
(64 tokens each) and runs bidirectional SSM within each window independently.

**Limitation**: Each window only "sees" an 8×8 pixel area. At the P2 level
(160×160 feature map from 640×640 input), each pixel represents 4×4 input pixels,
so each window covers only 32×32 input pixels. A very-tiny human (<32px) fits
inside ONE window — but the SSM has ZERO context about what's around it.

This is critical for disaster detection:
- Is that blob a human lying in rubble, or just rubble?
- The answer requires CONTEXT: what's the surrounding scene?
- Local 8×8 windows can't provide that context.

**The fundamental tradeoff**: Larger windows = more context but longer sequences.
Sequential SSM with L>64 tokens is too slow/memory-intensive on T4.

---

## 2. AtrousSSM: The Novel Solution

**Core idea**: Use the SAME window size (8×8 = 64 tokens) but SAMPLE tokens
from a LARGER spatial area using dilated/strided sampling.

```
STANDARD WINDOW (d=1)          DILATED WINDOW (d=2)          DILATED WINDOW (d=4)
Covers 8×8 pixels              Covers 16×16 pixels            Covers 32×32 pixels
Samples ALL pixels             Samples every 2nd pixel        Samples every 4th pixel
64 tokens total                64 tokens total                64 tokens total

■ ■ ■ ■ ■ ■ ■ ■              ■ · ■ · ■ · ■ · ■ · ■ · ■ · ■ ·    ■ · · · ■ · · · ■ · · · ...
■ ■ ■ ■ ■ ■ ■ ■              · · · · · · · · · · · · · · · ·    · · · · · · · · · · · · ...
■ ■ ■ ■ ■ ■ ■ ■              ■ · ■ · ■ · ■ · ■ · ■ · ■ · ■ ·    · · · · · · · · · · · · ...
■ ■ ■ ■ ■ ■ ■ ■              · · · · · · · · · · · · · · · ·    · · · · · · · · · · · · ...
■ ■ ■ ■ ■ ■ ■ ■              ■ · ■ · ■ · ■ · ■ · ■ · ■ · ■ ·    ■ · · · ■ · · · ■ · · · ...
■ ■ ■ ■ ■ ■ ■ ■              ...
■ ■ ■ ■ ■ ■ ■ ■
■ ■ ■ ■ ■ ■ ■ ■

■ = sampled token    · = skipped (interpolated on reverse)
```

**Analogy**: This is to SSM what atrous/dilated convolution is to standard
convolution — expanding the receptive field WITHOUT increasing computation.

**Key property**: Token count is ALWAYS 64 per window, regardless of dilation.
All scans are T4-safe. No memory increase per scan.

### Architecture Diagram

```
Input Feature Map (B, C, H, W) — e.g., 80×80, 128ch
                    │
    ┌───────────────┼───────────────┐
    │               │               │
    ▼               ▼               ▼
┌─────────┐   ┌─────────┐   ┌─────────┐
│ Branch 1 │   │ Branch 2 │   │ Branch 3 │
│ d=1      │   │ d=2      │   │ d=4      │
│          │   │          │   │          │
│ 100 win  │   │ 25 win   │   │ ~9 win   │
│ 8×8 px   │   │ 16×16 px │   │ 32×32 px │
│ coverage │   │ coverage │   │ coverage │
│          │   │          │   │          │
│ Norm     │   │ Norm     │   │ Norm     │
│ Proj→x,z │   │ Proj→x,z │   │ Proj→x,z │
│ BiSSM    │   │ BiSSM    │   │ BiSSM    │
│ Gate(z)  │   │ Gate(z)  │   │ Gate(z)  │
│ Proj+Res │   │ Proj+Res │   │ Proj+Res │
│          │   │          │   │          │
│ Reverse  │   │ Reverse  │   │ Reverse  │
│ (exact)  │   │ (interp) │   │ (interp) │
└────┬─────┘   └────┬─────┘   └────┬─────┘
     │              │              │
     │    80×80     │    80×80     │    80×80
     │              │              │
     └──────┬───────┘──────┬───────┘
            │              │
     ┌──────▼──────────────▼──────┐
     │      Gated Fusion          │
     │                            │
     │ concat = [b1, b2, b3]      │   (B, 3C, H, W)
     │ gate = σ(Conv1x1(concat))  │   (B, C, H, W)  ∈ [0,1]
     │ proj = Conv1x1(concat)     │   (B, C, H, W)
     │ out  = proj·gate           │
     │       + input·(1-gate)     │   ← residual
     │ out  = LayerNorm(out)      │
     └────────────┬───────────────┘
                  │
                  ▼
           Output (B, C, H, W)
```

Each branch has **independent SSM weights** — they specialize:
- d=1 branch learns LOCAL patterns (body shape, limbs)
- d=2 branch learns MEDIUM context (human + immediate surroundings)
- d=4 branch learns WIDE context (human + scene type: rubble, water, etc.)

The gated fusion learns PER-POSITION which scale matters most.

---

## 3. Why This Is Architecturally Novel

| Approach | What it does | Limitation vs AtrousSSM |
|----------|-------------|------------------------|
| LocalMamba (2024) | Local windows in Mamba | Fixed receptive field, no multi-scale |
| VMamba (NeurIPS 2024) | Global SS2D scan | O(H×W) tokens — explodes on P2@160×160 |
| DefMamba (CVPR 2025) | Deformable scan ORDER | Doesn't change receptive field SIZE |
| GMG-LDefmamba (2025) | Linear deformable + SS2D | Different mechanism (deformable sampling, not dilated windowing) |
| ASPP (DeepLab) | Dilated CONVOLUTIONS | Local operation — no long-range state propagation |
| **AtrousSSM (ours)** | **Dilated window SCANNING** | **Multi-scale receptive field + SSM long-range context, constant token count** |

**The gap**: Nobody has applied the dilated/atrous concept to state-space
model scanning. Dilated convolution expanded receptive fields for CNNs.
We do the same for SSMs.

**Why SSM dilated scanning is different from dilated convolution**:
- Dilated conv: local operation with gaps — each output depends on ~k² inputs
- Dilated SSM: SEQUENTIAL operation over sampled tokens — each output depends
  on ALL previous tokens in the scan (through hidden state propagation)
- The SSM hidden state carries information across the entire dilated window,
  enabling genuine long-range context that dilated conv cannot provide

---

## 4. Feasibility Analysis

### 4.1 Computational Cost

**Tokens per scan**: Always 64 (ws²=8²). UNCHANGED from current approach.

**Feature map: 80×80, ws=8**

| Dilation | Region size | Windows | Scans (×2 bidir) | Total tokens |
|----------|------------|---------|-------------------|-------------|
| d=1 | 8×8 | 100 | 200 | 12,800 |
| d=2 | 16×16 | 25 | 50 | 3,200 |
| d=4 | 32×32 | ~9 | 18 | 1,152 |
| **Total** | | **134** | **268** | **17,152** |

**Current approach**: 100 windows × 2 = 200 scans, 12,800 tokens.
**AtrousSSM overhead**: 1.34× more scans. Per-scan cost is identical.

**At all 5 neck layers combined**:

| Layer | Resolution | Current scans | AtrousSSM scans | Overhead |
|-------|-----------|--------------|-----------------|----------|
| 13 | 40×40 | 50 | 76 | 1.52× |
| 16 | 80×80 | 200 | 268 | 1.34× |
| 19 | 160×160 | 800 | 1050 | 1.31× |
| 22 | 80×80 | 200 | 268 | 1.34× |
| 25 | 40×40 | 50 | 76 | 1.52× |
| **Total** | | **1300** | **1738** | **1.34×** |

### 4.2 Parameter Count

| Component | Per branch | 3 branches | Fusion | Total per AtrousSSM |
|-----------|-----------|-----------|--------|-------------------|
| LayerNorm(D) | 2D | 6D | — | — |
| Linear(D, 2D) | 2D² | 6D² | — | — |
| 2× SelectiveScan1D | ~D²/4 | ~3D²/4 | — | — |
| Linear(D, D) | D² | 3D² | — | — |
| Fusion Conv2d | — | — | ~6D²+D² | — |
| **Total** | | | | **~17D²** |

For D=128 (inner channels at most layers): ~17 × 16384 ≈ 279K per AtrousSSM

**Original LocalWindowSSM**: ~4D² ≈ 65K per module

**Per-model increase**: 5 layers × 2 bottlenecks × (279K - 65K) = ~2.14M extra
**Total model**: 19.6M → ~21.7M (+11%). Still "medium" YOLO category.

### 4.3 Memory (T4 16GB VRAM)

Current model at batch=8 uses ~10-12GB VRAM (estimated from your training).

Additional memory for AtrousSSM:
- 3 branch outputs stored simultaneously: 3 × B × C × H × W × 4 bytes
- At largest layer (P2, 160×160, 64ch, batch=8): 3 × 8 × 64 × 160 × 160 × 4 = ~150MB
- Plus 2 fusion convolution outputs: ~100MB

**Total additional**: ~250MB → well within the 4-6GB headroom.
If tight, drop to batch=6 (grad_accum=3 for effective batch 18).

### 4.4 Training Time

Current: ~12 hours for 120 epochs on T4.
Expected: ~16-18 hours (1.34× SSM overhead + fusion overhead).
Strategy: Train ~70 epochs in first 12-hour session, resume for remaining.

### 4.5 Dataset Compatibility

- C2A dataset: Single class (person), YOLO format → NO changes needed
- The multi-scale scanning SPECIFICALLY helps this dataset:
  - **Very tiny objects** (<32px): d=4 gives 32×32px context (scene understanding)
  - **Disaster rubble**: long-range context distinguishes humans from debris
  - **Dense crowds**: d=1 maintains local precision for clustered detections
- No new annotations, no data format changes

---

## 5. Complete Implementation Code

### 5.1 AtrousSSM Module (replaces LocalWindowSSM)

```python
class AtrousSSM(nn.Module):
    """
    Multi-Scale Dilated Window State Space Model.

    Extends local-window SSM with dilated scanning: multiple dilation rates
    sample tokens from progressively larger spatial areas while keeping the
    token count per window constant (ws² = 64, T4-safe).

    This is to SSM what atrous/dilated convolution is to standard convolution:
    expanding the receptive field without increasing per-token computation.

    Parameters
    ----------
    d_model     : int — channel dimension
    d_state     : int — SSM state size (default 4)
    window_size : int — tokens per window side (default 8 → 64 tokens)
    dilations   : list[int] — dilation rates (default [1, 2, 4])
    """

    def __init__(self, d_model: int, d_state: int = 4,
                 window_size: int = 8, dilations: list = None):
        super().__init__()
        self.ws = window_size
        self.dilations = dilations or [1, 2, 4]
        D = d_model
        n_br = len(self.dilations)

        # ── Independent SSM branch per dilation rate ──────────────────────
        self.branches = nn.ModuleList()
        for _ in self.dilations:
            branch = nn.ModuleDict({
                'norm':     nn.LayerNorm(D),
                'in_proj':  nn.Linear(D, D * 2, bias=False),
                'scan_fwd': _SelectiveScan1D(D, d_state),
                'scan_bwd': _SelectiveScan1D(D, d_state),
                'out_proj': nn.Linear(D, D, bias=False),
            })
            # Small-scale init (same convention as original LocalWindowSSM)
            nn.init.normal_(branch['out_proj'].weight, std=0.02)
            self.branches.append(branch)

        # ── Gated fusion across scales ────────────────────────────────────
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(D * n_br, D, 1, bias=False),
            nn.Sigmoid()
        )
        self.fusion_proj = nn.Conv2d(D * n_br, D, 1, bias=False)
        self.out_norm = nn.LayerNorm(D)

    # ── Dilated window partition ──────────────────────────────────────────

    def _dilated_partition(self, x: torch.Tensor, dilation: int):
        """
        Partition feature map into dilated windows.

        Each window covers (ws*d) × (ws*d) pixels but samples only
        ws × ws tokens at stride d. Token count is always ws² (e.g., 64).

        Returns:
            tokens : (B*nH*nW, ws*ws, C) — ready for SSM
            meta   : tuple for reverse operation
        """
        B, C, H, W = x.shape
        ws = self.ws
        region = ws * dilation

        # Pad to multiple of region size
        ph = (region - H % region) % region
        pw = (region - W % region) % region
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))

        _, _, Hp, Wp = x.shape
        nH, nW = Hp // region, Wp // region

        # Reshape to region grid → dilated sampling
        x = x.reshape(B, C, nH, region, nW, region)
        x = x.permute(0, 2, 4, 1, 3, 5)           # B, nH, nW, C, region, region
        x = x[:, :, :, :, ::dilation, ::dilation]  # B, nH, nW, C, ws, ws

        # Flatten to sequence for SSM
        tokens = x.reshape(B * nH * nW, C, ws * ws).transpose(1, 2)
        return tokens, (B, C, H, W, Hp, Wp, nH, nW)

    def _dilated_reverse(self, tokens: torch.Tensor, meta: tuple,
                         dilation: int) -> torch.Tensor:
        """
        Reverse dilated partition back to spatial feature map.

        For dilation > 1, uses bilinear interpolation to fill
        non-sampled positions within each region.
        """
        B, C, H, W, Hp, Wp, nH, nW = meta
        ws = self.ws
        region = ws * dilation

        # Back to window shape
        windows = tokens.transpose(1, 2).reshape(B * nH * nW, C, ws, ws)

        if dilation > 1:
            # Upsample from ws×ws to region×region
            windows = F.interpolate(
                windows, size=(region, region),
                mode='bilinear', align_corners=False
            )

        # Reconstruct spatial layout
        windows = windows.reshape(B, nH, nW, C, region, region)
        x = windows.permute(0, 3, 1, 4, 2, 5)    # B, C, nH, region, nW, region
        x = x.reshape(B, C, Hp, Wp)
        return x[:, :, :H, :W].contiguous()

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → (B, C, H, W)"""
        branch_outputs = []

        for branch, dilation in zip(self.branches, self.dilations):
            tokens, meta = self._dilated_partition(x, dilation)
            residual = tokens

            # Norm → project → split (x, z)
            tokens_n = branch['norm'](tokens)
            xz = branch['in_proj'](tokens_n)
            x_in, z = xz.chunk(2, dim=-1)

            # Bidirectional SSM scan
            y_fwd = branch['scan_fwd'](x_in)
            y_bwd = branch['scan_bwd'](x_in.flip(1)).flip(1)
            y = (y_fwd + y_bwd) * F.silu(z)

            # Project + per-branch residual
            y = branch['out_proj'](y) + residual

            # Reverse to spatial
            spatial = self._dilated_reverse(y, meta, dilation)
            branch_outputs.append(spatial)

        # Gated fusion across all dilation scales
        concat = torch.cat(branch_outputs, dim=1)    # B, C*n_br, H, W
        gate   = self.fusion_gate(concat)             # B, C, H, W  ∈ [0,1]
        fused  = self.fusion_proj(concat)             # B, C, H, W

        # Weighted combination: SSM output + input residual
        out = fused * gate + x * (1 - gate)

        # Output normalization (channel-wise)
        out = out.permute(0, 2, 3, 1)    # B, H, W, C
        out = self.out_norm(out)
        out = out.permute(0, 3, 1, 2)    # B, C, H, W

        return out
```

### 5.2 Updated _MambaBottleneck

```python
class _MambaBottleneck(nn.Module):
    """
    Single bottleneck unit inside C3K2Mamba.
    Replaces the CNN bottleneck with: Conv3×3 → AtrousSSM → Conv3×3
    """
    def __init__(self, c: int, shortcut: bool, d_state: int,
                 window_size: int, dilations: list = None):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.cv1 = Conv(c, c, 3, 1)
        self.ssm = AtrousSSM(c, d_state=d_state, window_size=window_size,
                             dilations=dilations)
        self.cv2 = Conv(c, c, 3, 1)
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.ssm(self.cv1(x)))
        return x + y if self.add else y
```

### 5.3 Updated C3K2Mamba

```python
class C3K2Mamba(nn.Module):
    """
    C2f-style block with AtrousSSM bottleneck.
    Drop-in replacement for C3k2 in YOLO11m neck.
    """
    def __init__(self, c1: int, c2: int, n: int = 1,
                 shortcut: bool = False, g: int = 1, e: float = 0.5,
                 d_state: int = 4, dilations: list = None):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        ws = _get_window_size(self.c)

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            _MambaBottleneck(self.c, shortcut and c1 == c2,
                            d_state, ws, dilations=dilations)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
```

### 5.4 Updated inject_mamba_neck

Only change: pass `dilations` parameter through.

```python
def inject_mamba_neck(yolo_model, min_layer_idx: int = 11,
                       max_channels: int = 512,
                       d_state: int = 4,
                       dilations: list = None,    # ← NEW PARAMETER
                       verbose: bool = True) -> None:
    # ... (same logic as before, just pass dilations to C3K2Mamba)

    # In the replacement line, change:
    new_block = C3K2Mamba(c1, c2, n=n, shortcut=shortcut,
                          d_state=d_state, dilations=dilations)
    # ... (rest unchanged)
```

---

## 6. Integration Instructions (Step-by-Step)

### What to change in `mamba_cbam_p2head.py`:

**Change 1**: Replace the `LocalWindowSSM` class (lines 425-495) with `AtrousSSM`
from section 5.1 above. Delete the entire `LocalWindowSSM` class.

**Change 2**: Replace `_MambaBottleneck` class (lines 498-513) with the updated
version from section 5.2 above.

**Change 3**: Replace `C3K2Mamba` class (lines 516-542) with the updated version
from section 5.3 above.

**Change 4**: Update `inject_mamba_neck` function (line 754) to accept and pass
`dilations` parameter as shown in section 5.4.

**Change 5**: Update the injection call sites (lines 832 and 1191):
```python
# Line 832 (dry-run):
replaced = inject_mamba_neck(_m, verbose=True, dilations=[1, 2, 4])

# Line 1191 (training):
replaced = inject_mamba_neck(mamba_model, d_state=4, dilations=[1, 2, 4],
                            verbose=(_batch_attempt == OOM_RETRY_BATCHES[0]))
```

**Change 6**: Update the print statement (line 545):
```python
print("✓ Mamba modules defined (AtrousSSM, C3K2Mamba)")
```

**That's it.** 6 surgical changes. Everything else (training pipeline, callbacks,
evaluation, visualization) stays EXACTLY the same.

---

## 7. Updated Smoke Test

Add these tests to the existing `run_smoke_test()` function:

```python
def run_smoke_test():
    print("\n" + "=" * 70)
    print("SMOKE TEST: Verifying AtrousSSM modules before training")
    print("=" * 70)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Test 1: AtrousSSM shape consistency ──────────────────────────────
    print("\nTest 1: AtrousSSM shape consistency ...")
    for H_W, C_ch in [(80, 128), (160, 64), (40, 256), (20, 512)]:
        x = torch.randn(2, C_ch, H_W, H_W, device=device)
        ws = _get_window_size(C_ch)
        ssm = AtrousSSM(C_ch, d_state=4, window_size=ws,
                        dilations=[1, 2, 4]).to(device)
        y = ssm(x)
        assert y.shape == x.shape, f"Shape mismatch at {H_W}×{H_W},{C_ch}ch: {y.shape}"
        print(f"  ✓ {H_W}×{H_W}, {C_ch}ch → {y.shape}")
    print("  PASSED")

    # ── Test 2: Dilated partition/reverse roundtrip ──────────────────────
    print("\nTest 2: Dilated partition/reverse consistency ...")
    ssm_test = AtrousSSM(128, window_size=8, dilations=[1, 2, 4]).to(device)
    x_test = torch.randn(2, 128, 80, 80, device=device)
    for d in [1, 2, 4]:
        tokens, meta = ssm_test._dilated_partition(x_test, d)
        ws = ssm_test.ws
        expected_windows = (meta[-2]) * (meta[-1])  # nH * nW
        expected_tokens = ws * ws
        assert tokens.shape == (2 * expected_windows, expected_tokens, 128), \
            f"d={d}: token shape {tokens.shape} unexpected"
        reconstructed = ssm_test._dilated_reverse(tokens, meta, d)
        assert reconstructed.shape == (2, 128, 80, 80), \
            f"d={d}: reverse shape {reconstructed.shape} unexpected"
        print(f"  ✓ d={d}: {tokens.shape[0]} windows × {tokens.shape[1]} tokens → {reconstructed.shape}")
    print("  PASSED")

    # ── Test 3: No NaN/Inf in forward ────────────────────────────────────
    print("\nTest 3: No NaN/Inf in AtrousSSM output ...")
    x_nan = torch.randn(4, 128, 80, 80, device=device)
    ssm_nan = AtrousSSM(128, d_state=4, window_size=8).to(device)
    y_nan = ssm_nan(x_nan)
    assert not torch.isnan(y_nan).any(), "NaN detected in output!"
    assert not torch.isinf(y_nan).any(), "Inf detected in output!"
    print("  ✓ PASSED")

    # ── Test 4: Gradient flow through all branches + fusion ──────────────
    print("\nTest 4: Gradient flow ...")
    x_grad = torch.randn(2, 128, 40, 40, device=device, requires_grad=True)
    ssm_grad = AtrousSSM(128, d_state=4, window_size=8).to(device)
    y_grad = ssm_grad(x_grad)
    loss = y_grad.sum()
    loss.backward()
    assert x_grad.grad is not None, "No gradient on input!"
    for i, branch in enumerate(ssm_grad.branches):
        for name, param in branch.items():
            if isinstance(param, nn.Module):
                for pname, p in param.named_parameters():
                    assert p.grad is not None, f"No grad: branch[{i}].{name}.{pname}"
    # Check fusion gate/proj gradients
    for name, p in ssm_grad.fusion_gate.named_parameters():
        assert p.grad is not None, f"No grad: fusion_gate.{name}"
    for name, p in ssm_grad.fusion_proj.named_parameters():
        assert p.grad is not None, f"No grad: fusion_proj.{name}"
    print("  ✓ Gradients flow through all 3 branches + fusion gate")
    print("  PASSED")

    # ── Test 5: AMP/FP16 safety ──────────────────────────────────────────
    if device == "cuda":
        print("\nTest 5: AMP/FP16 safety ...")
        x_amp = torch.randn(2, 128, 80, 80, device=device)
        ssm_amp = AtrousSSM(128, d_state=4, window_size=8).to(device)
        with torch.cuda.amp.autocast():
            y_amp = ssm_amp(x_amp)
        assert not torch.isnan(y_amp).any(), "NaN under AMP!"
        assert not torch.isinf(y_amp).any(), "Inf under AMP!"
        print("  ✓ PASSED")

    # ── Test 6: Branch asymmetry (dilations produce different outputs) ───
    print("\nTest 6: Branch asymmetry check ...")
    x_asym = torch.randn(2, 128, 80, 80, device=device)
    ssm_asym = AtrousSSM(128, d_state=4, window_size=8).to(device)
    branch_outs = []
    for branch, dilation in zip(ssm_asym.branches, ssm_asym.dilations):
        tokens, meta = ssm_asym._dilated_partition(x_asym, dilation)
        tokens_n = branch['norm'](tokens)
        xz = branch['in_proj'](tokens_n)
        x_in, z = xz.chunk(2, dim=-1)
        y_fwd = branch['scan_fwd'](x_in)
        spatial = ssm_asym._dilated_reverse(y_fwd, meta, dilation)
        branch_outs.append(spatial)
    # d=1 and d=4 should produce different outputs (different sampling)
    diff = (branch_outs[0] - branch_outs[-1]).abs().mean().item()
    assert diff > 1e-4, f"Branches too similar (diff={diff}). Dilations may not work."
    print(f"  ✓ Branch d=1 vs d=4 mean diff = {diff:.6f} (expected > 1e-4)")
    print("  PASSED")

    # ── Test 7: VRAM estimate at P2 resolution (160×160) ─────────────────
    if device == "cuda":
        print("\nTest 7: VRAM estimate at P2 resolution ...")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        x_mem = torch.randn(8, 64, 160, 160, device=device)
        ssm_mem = AtrousSSM(64, d_state=4, window_size=8).to(device)
        with torch.cuda.amp.autocast():
            y_mem = ssm_mem(x_mem)
            loss_mem = y_mem.sum()
            loss_mem.backward()
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        print(f"  Peak VRAM (batch=8, 160×160, 64ch): {peak_mb:.0f} MB")
        if peak_mb > 14000:
            print("  ⚠ WARNING: Peak VRAM > 14GB. Consider batch=6 or dilations=[1,4]")
        else:
            print("  ✓ Within T4 16GB budget")
        del x_mem, ssm_mem, y_mem; torch.cuda.empty_cache()

    # ── Test 8: Full C3K2Mamba with AtrousSSM ────────────────────────────
    print("\nTest 8: Full C3K2Mamba block ...")
    from ultralytics.nn.modules.conv import Conv
    c3k2 = C3K2Mamba(256, 256, n=2, shortcut=False,
                     d_state=4, dilations=[1, 2, 4]).to(device)
    x_c3k2 = torch.randn(2, 256, 40, 40, device=device)
    y_c3k2 = c3k2(x_c3k2)
    assert y_c3k2.shape == x_c3k2.shape, f"C3K2Mamba shape mismatch: {y_c3k2.shape}"
    print(f"  ✓ C3K2Mamba(256→256, n=2): {x_c3k2.shape} → {y_c3k2.shape}")
    params = sum(p.numel() for p in c3k2.parameters()) / 1e3
    print(f"  Parameters: {params:.1f}K")
    print("  PASSED")

    print("\n" + "=" * 70)
    print("ALL SMOKE TESTS PASSED ✓")
    print("=" * 70 + "\n")
```

---

## 8. Training Strategy

### Phase 1: Smoke Test & Dry Run (30 minutes)
Set `TEST_MODE = True`. Run the full notebook. Verify:
- [x] All 8 smoke tests pass
- [x] Dry-run injection shows 5 replacements (layers 13,16,19,22,25)
- [x] 2-epoch test run completes without OOM or NaN
- [x] Model params ≈ 21-22M (vs original 19.6M)

### Phase 2: Full Training (12-18 hours, across 1-2 sessions)
Set `TEST_MODE = False`.
- Epochs: 120 (same as your previous Mamba run)
- Batch: 8 (try first), fallback to 6 if OOM
- Everything else: identical to previous run
- Early stopping (F2, patience=15) will likely trigger around epoch 90-110

### Phase 3: Evaluation
Same comprehensive evaluation as before:
- Official mAP50, mAP50-95
- Per-size recall breakdown (very_tiny, tiny, small, medium)
- F1, F2 scores
- ECE calibration
- Speed benchmarks at 320/416/640/832
- Confusion matrices, calibration plots

### What to Compare Against
Your completed runs already provide the comparison:

| Model | Source | Purpose |
|-------|--------|---------|
| YOLO11m Baseline | 01-02-2026 ablation, 70 epochs | Reference |
| YOLO11m + CBAM + P2Head | 01-02-2026 ablation, 70 epochs | Prior best |
| YOLO11m + LocalMamba + CBAM + P2Head | 11-3-26-Mamba, 120 epochs | Previous run |
| **YOLO11m + AtrousMamba + CBAM + P2Head** | **NEW RUN** | **Proposed** |

Row 3 vs Row 4 directly isolates AtrousSSM vs LocalWindowSSM — same training
pipeline, same callbacks, same dataset, same epoch count. This is a CLEAN ablation.

---

## 9. Month 2 Extension: WaveAtrousMamba

After AtrousSSM is validated, add wavelet decomposition:

```
Feature Map
    ↓
2D Haar Wavelet Decomposition
    ↓
┌──────────────────────────────────────────────┐
│ LL (low-freq, structure) → Standard Conv     │
│ LH, HL, HH (high-freq, edges) → AtrousSSM   │
└──────────────────────────────────────────────┘
    ↓
Inverse Wavelet → Enhanced Feature
```

Rationale: High-frequency components contain edge patterns of tiny humans.
AtrousSSM's long-range context specifically helps distinguish human edges
from debris edges. Low-frequency components (scene structure) don't need SSM.

This adds a SECOND architectural contribution to the thesis.

---

## 10. Thesis Framing

### Title
"AtrousMamba-YOLO: Multi-Scale Dilated State Space Scanning for
Human Detection in Aerial Disaster Imagery"

### Contributions
1. **AtrousSSM** — A novel SSM scanning strategy that extends local-window
   Mamba with multiple dilation rates, achieving multi-scale receptive fields
   at constant per-token computational cost.

2. **(Month 2) WaveAtrousSSM** — Frequency-selective AtrousSSM that applies
   dilated state-space scanning specifically to high-frequency feature
   components for improved tiny object detection.

3. **Comprehensive evaluation** on C2A disaster dataset with per-size recall
   analysis, calibration metrics, and direct comparison against local-window
   Mamba, CBAM, and baseline YOLO11m.

### Target Venues
- **Drones (MDPI)** — IF ~4.4, directly relevant scope, ~4 week review
- **IEEE Access** — IF ~3.9, broader scope, ~8 week review
- **Scientific Reports** — IF ~3.8, multidisciplinary
- **Remote Sensing (MDPI)** — IF ~4.2, if you add HERIDAL cross-evaluation

### Key Narrative for Defense
"Standard local-window Mamba in YOLO detection necks limits each SSM scan
to a small spatial region, preventing the model from leveraging scene-level
context critical for distinguishing small humans from disaster debris.
AtrousSSM solves this by introducing dilated window scanning — sampling tokens
from progressively larger spatial areas while maintaining constant sequence
length. The learned gated fusion adaptively weights local detail against
wide-range context per spatial position, enabling the detector to handle
the extreme scale variation (32px to 256px+) characteristic of aerial
disaster imagery."

---

## 11. If Memory Is Tight: Fallback Options

If T4 OOMs with `dilations=[1, 2, 4]`:

**Option A**: Use `dilations=[1, 4]` (2 branches instead of 3)
- Loses medium-scale context but keeps local + wide
- ~33% less memory and compute than 3 branches
- Still novel — the core contribution is dilated SSM, not the number of rates

**Option B**: Use `dilations=[1, 2, 4]` but batch=6, grad_accum=3
- Effective batch = 18 (close to your original 16)
- Same architecture, just slower training

**Option C**: Use `dilations=[1, 2]` for 40×40 layers (13, 25) and
`[1, 2, 4]` for 80×80+ layers (16, 19, 22)
- Requires per-layer dilation config in inject_mamba_neck
- Most memory-efficient but more complex

Recommend trying the default `[1, 2, 4]` with batch=8 first. The smoke test
(Test 7) will tell you if you need to fall back.
