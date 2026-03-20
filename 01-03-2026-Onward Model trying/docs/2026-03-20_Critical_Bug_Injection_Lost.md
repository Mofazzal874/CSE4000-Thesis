# CRITICAL BUG: Ultralytics Trainer Discards Post-Init Module Injection
## Date: 2026-03-20

---

## The Bug

**All post-init module injections (Mamba, AtrousMamba) were silently discarded by Ultralytics' internal trainer.**

When you call `model.train()`, Ultralytics internally:

1. Creates a `DetectionTrainer`
2. Calls `trainer.get_model(cfg=model.yaml, weights=model)`
3. `get_model()` **rebuilds a brand-new model from the YAML config**
4. Transfers only matching weights via `intersect_dicts()`
5. The YAML only has `C3k2` — so the rebuilt model has no `C3K2Mamba`
6. The injected AtrousSSM/Mamba modules are **completely lost**

### Evidence
- Dry run showed **25.03M params** (with injection)
- Training model summary showed **19.59M params** (vanilla C3k2)
- Saved `best.pt` had **19.592M params** (identical to base model)
- All 3 models in complexity comparison showed identical param counts

### Code Path (Ultralytics source)
```
Model.train()
  → trainer = DetectionTrainer(...)
  → trainer.model = trainer.get_model(cfg=self.model.yaml, weights=self.model)
    → model = DetectionModel(yaml_dict)     # FRESH from YAML — no custom modules
    → model.load(weights)                   # transfers matching keys only
  → trainer.train()                          # trains vanilla model
```

---

## Impact on Previous Results

### Previous Mamba Run (11-3-26-Mamba)
The previous Mamba+CBAM+P2 result was **NOT actually Mamba**:

| What We Thought | What Actually Happened |
|-----------------|----------------------|
| Mamba+CBAM+P2, 120 epochs, mAP50=0.877 | CBAM+P2, 120 epochs, mAP50=0.877 |
| +0.6% mAP50 from Mamba injection | +0.6% mAP50 from 50 more epochs |
| +2.4% mAP50-95 from Mamba SSM | +2.4% mAP50-95 from longer training |

The "Mamba improvement" was just **the effect of training 120 epochs vs 70 epochs** on the same CBAM+P2 architecture.

### AtrousMamba Test Runs (20-3-26)
Both test runs (2 epochs each) trained the vanilla CBAM+P2 model:
- First run: DDP + no injection = all-zero detections (DDP rebuild issue)
- Second run: Single GPU + no injection = P=0.528, R=0.406 (base model)

---

## The Fix

**Monkey-patch `DetectionTrainer.get_model()`** to re-inject C3K2Mamba after the model is rebuilt from YAML:

```python
from ultralytics.models.yolo.detect.train import DetectionTrainer

_orig_get_model = DetectionTrainer.get_model

def _patched_get_model(self, cfg=None, weights=None, verbose=True):
    model = _orig_get_model(self, cfg, weights, verbose)
    # Re-inject AtrousSSM into the freshly built model
    _inject_into_sequential(model.model, ...)
    return model

DetectionTrainer.get_model = _patched_get_model
```

This ensures:
1. The trainer's model contains C3K2Mamba during training
2. The optimizer includes AtrousSSM parameters
3. Saved checkpoints contain the full architecture
4. Loading checkpoints preserves custom modules (via torch.load unpickling)

Additionally:
- **cv1/cv2 weight transfer**: Pretrained conv weights from C3k2 are transferred to C3K2Mamba
- **Proper initialization**: Xavier init on in_proj, small init on fusion gate/proj
- **Verification callback**: `on_pretrain_routine_start` confirms C3K2Mamba presence + param count

---

## Thesis Implications

### The Good News
- The CBAM+P2 baseline (70 epochs, mAP50=0.871) is still valid
- The YOLO11m baseline (70 epochs, mAP50=0.854) is still valid
- With the fix, AtrousMamba will now ACTUALLY train — this is the first real test
- If it works, the improvement will be genuine and novel

### The Bad News
- The "Mamba+CBAM+P2 = 0.877" result needs to be re-evaluated
- The ablation table in the thesis needs correction
- We need to re-run the Mamba model with the fix for fair comparison

### Recommended Ablation Table (corrected)
| Model | Epochs | mAP50 | mAP50-95 | Status |
|-------|--------|-------|----------|--------|
| YOLO11m Baseline | 70 | 0.854 | 0.611 | Valid |
| YOLO11m + P2Head | 70 | 0.871 | 0.630 | Valid |
| YOLO11m + CBAM + P2Head | 70 | 0.871 | 0.630 | Valid |
| YOLO11m + CBAM + P2Head | 120 | 0.877 | 0.654 | Was labeled "Mamba" — actually just more epochs |
| YOLO11m + AtrousMamba + CBAM + P2Head | 120 | TBD | TBD | First real SSM test (with fix) |

### Recovery Strategy
1. Run AtrousMamba with the fix (this script) → first real SSM result
2. If time permits, also re-run original Mamba (LocalWindowSSM) with the fix for fair comparison
3. The thesis narrative remains valid: "we tried multiple approaches, AtrousSSM is the novel one"
4. The 0.877 result can be presented as "CBAM+P2 at 120 epochs" in the ablation table

---

## How to Verify the Fix Works

After running the updated script, check for these prints in the output:

1. **During model build**: `"Re-injecting AtrousSSM into trainer model (post-get_model patch)"`
2. **Param count**: Should show ~25.03M (NOT 19.59M)
3. **Verification callback**: `"INJECTION VERIFICATION"` → `"✓ AtrousSSM ACTIVE"` with `C3K2Mamba layers: 6`
4. **Complexity comparison**: AtrousMamba should show ~25M params, CBAM+P2 should show ~19.6M
