# ThunderCompute Platform Research & A100 Optimization Guide
**Date:** 2026-03-25

---

## TABLE OF CONTENTS

1. [What is Thunder Compute?](#1-what-is-thunder-compute)
2. [Pricing](#2-pricing)
3. [Hardware Specifications](#3-hardware-specifications)
4. [How it Works (GPU-over-TCP)](#4-how-it-works-gpu-over-tcp)
5. [Prototyping vs Production Mode](#5-prototyping-vs-production-mode)
6. [Getting Started - VSCode Extension](#6-getting-started---vscode-extension)
7. [CLI Reference (tnr)](#7-cli-reference-tnr)
8. [SSH Access](#8-ssh-access)
9. [Data Upload/Download](#9-data-uploaddownload)
10. [Storage & Snapshots](#10-storage--snapshots)
11. [Installing Dependencies](#11-installing-dependencies)
12. [Instance Templates](#12-instance-templates)
13. [Known Issues & Limitations](#13-known-issues--limitations)
14. [Troubleshooting](#14-troubleshooting)
15. [A100 80GB Optimization for YOLO + Mamba](#15-a100-80gb-optimization-for-yolo--mamba)
16. [Key URLs & Resources](#16-key-urls--resources)

---

## 1. What is Thunder Compute?

Thunder Compute is a **Y Combinator (S24)-backed** GPU cloud platform offering the cheapest on-demand GPU instances. Founded by Carl Peterson (CEO) and Brian Model (CTO) in February 2024.

**Key selling points:**
- Up to 80% cheaper than AWS
- Pay-as-you-go, per-minute billing, no contracts
- One-click GPU instances via VSCode, Cursor, or Windsurf extensions
- Pre-configured environments with CUDA + PyTorch
- Instances located in **Quebec, Canada**

**Support:** Discord community (founding team responds directly) or support@thundercompute.com

---

## 2. Pricing

### Prototyping Tier (GPU-over-TCP, cheaper)
| GPU | VRAM | Price |
|-----|------|-------|
| RTX A6000 | 48 GB | **$0.27/hr** |
| A100 | 80 GB | **$0.78/hr** |
| H100 PCIe | 80 GB | **$1.38/hr** |

### Production Tier (Bare-metal-like, NVLink)
| GPU | VRAM | Price | Multi-GPU |
|-----|------|-------|-----------|
| A100 NVLink | 80 GB | **$1.79/GPU/hr** | 1-8 GPUs |
| H100 PCIe NVLink | 80 GB | **$2.49/GPU/hr** | 1-8 GPUs |

### Add-On Costs
| Resource | Price | Included Free |
|----------|-------|---------------|
| Storage (disk) | $0.10/GB/month | First 100 GB during runtime |
| Snapshots | $0.05/GB/month | - |
| Extra vCPUs | $0.06/vCPU/month | 4 vCPUs (prototyping) |

### Billing Details
- **Per-minute billing** - no minimum commitment
- Two payment methods: auto-pay via credit card or preloaded account credit
- Billing alerts and instance reminders via email
- You MUST add a payment method before creating instances

### Cost Example for Your Use Case
- A100 prototyping: $0.78/hr = ~$18.72 for 24 hours continuous training
- A100 production: $1.79/hr = ~$42.96 for 24 hours continuous training

---

## 3. Hardware Specifications

### A100 Prototyping Instance
- **GPU:** NVIDIA A100 80 GB
- **vCPU:** 4 included (expandable)
- **RAM:** 32 GB included (expandable; 8 GB per vCPU)
- **Multi-GPU:** Up to 2 GPUs
- **Network:** 7 Gbps egress/ingress, dynamic IP

### A100 Production Instance
- **GPU:** NVIDIA A100 80 GB NVLink
- **vCPU:** 18 per GPU
- **RAM:** 90 GB per GPU
- **Multi-GPU:** Up to 8 GPUs
- **Network:** 7 Gbps egress/ingress, dynamic IP

### Pre-installed Software
- **CUDA Version:** 13.0
- **CUDA Driver Version:** 580
- **PyTorch Version:** 2.9.0+cu128
- **JupyterLab:** Pre-installed
- **Python libraries:** NumPy, Pandas, and scientific stack

---

## 4. How it Works (GPU-over-TCP)

**CRITICAL TO UNDERSTAND:** Thunder Compute's prototyping mode uses **network-attached GPUs**, NOT physically-attached GPUs.

### How GPU-over-TCP Works
- Your VM communicates with the GPU over TCP (same protocol browsers use)
- When you call `device="cuda"`, the instance translates those calls into network messages
- You still `pip install torch`, use `device="cuda"`, and run code normally
- The GPU is physically in a different machine, attached over the network

### Performance Impact (Prototyping Mode)
- **Initial connection latency:** ~10-20 milliseconds
- **Potential slowdown:** 20-50% for certain tasks compared to physically-attached GPU
- **Early tests:** Initially 100x slower, improved to ~2x slower for most AI workloads
- **Target:** Within 5% of native GPU performance (ongoing optimization)
- **Best for:** Data science workflows (most performant and stable)
- **Key insight:** Most ML jobs spend more time computing than waiting for data, which mitigates network overhead

### Production Mode
- Production mode provides **bare-metal-like** GPU access
- Stronger performance guarantees
- No GPU-over-TCP overhead
- But costs ~2.3x more ($1.79 vs $0.78 for A100)

---

## 5. Prototyping vs Production Mode

### Prototyping Mode ($0.78/hr for A100)
- CUDA-level optimizations enabled for cost efficiency
- **PyTorch:** Fully supported
- **Model Serving:** ComfyUI, Ollama, VLLM supported
- **Fine Tuning:** Unsloth supported
- **Custom CUDA Kernels:** UNPREDICTABLE behavior, errors and profiling issues
- **Performance may vary** - still in beta
- Up to 2 GPUs per instance
- 50% lower cost than production

### Production Mode ($1.79/hr for A100)
- Full CUDA compatibility (all features work)
- Stronger performance guarantees
- Higher uptime
- Multi-GPU nodes (up to 8 GPUs)
- Recommended for inference servers, larger configurations
- Use if prototyping mode has compatibility issues

### IMPORTANT for Your Workload (YOLO + Mamba)
- Standard PyTorch YOLO training should work fine in **prototyping mode**
- **Mamba SSM custom CUDA kernels** (selective_scan_cuda, causal-conv1d) may have issues in prototyping mode
- If you encounter "This function is not implemented" errors, switch to **production mode**
- Test in prototyping first; fall back to production if needed

---

## 6. Getting Started - VSCode Extension

### Prerequisites
1. VSCode installed
2. **Remote - SSH extension** installed in VSCode (REQUIRED)

### Step-by-Step Setup

**Step 1: Install the Thunder Compute Extension**
- Open VSCode
- Go to Extensions (Ctrl+Shift+X)
- Search "Thunder Compute" and install
- Or use: `vscode:extension/ThunderCompute.thunder-compute`
- Also works with **Cursor** and **Windsurf** editors

**Step 2: Login / Authenticate**
- Extension may auto-prompt you to log in
- If not: Open Command Palette (Ctrl+Shift+P) -> "Thunder Compute: Login"
- Browser opens for OAuth authentication
- Complete login in browser

**Step 3: Add Payment Method**
- Go to https://console.thundercompute.com
- Add a payment method (required before creating instances)

**Step 4: Create an Instance**
- Open Thunder Compute tab in VSCode sidebar
- Press the **Create Instance** button (plus icon)
- Select GPU type, mode, template, and disk size
- Instance spins up in ~60 seconds

**Step 5: Connect to Instance**
- Click the **Connect** button next to your instance (double-arrow icon)
- A fresh editor window launches with SSH connection to remote instance
- Every terminal, debugger, and extension now runs on the remote GPU

**Step 6: Work on Your Code**
- Drag-and-drop local files into the remote file explorer
- Use terminal as if it were local
- All extensions (Python, Jupyter, etc.) work on the remote instance

### VSCode File Operations
- **Upload:** Drag-and-drop files from local to remote file explorer
- **Download:** Right-click files in remote explorer -> Download
- Run notebooks, scripts, and terminal commands as if local

---

## 7. CLI Reference (tnr)

### Installation
```bash
pip install tnr
```

### Authentication
```bash
tnr login          # OAuth login via browser
tnr logout         # Remove stored credentials
```

For automated/programmatic access:
```bash
export TNR_API_TOKEN=<your-token>
```

### Instance Management
```bash
tnr create                              # Interactive instance creation menu
tnr create --gpu a100 --vcpus 4         # One-command creation with flags
tnr status                              # List instances with IDs and IPs
tnr modify <instance_id>                # Modify disk size (can only INCREASE, not decrease)
tnr delete <instance_id>                # Delete instance (STOPS BILLING)
```

### Connecting
```bash
tnr connect                             # Connect to default instance (ID 0)
tnr connect <instance_id>               # Connect to specific instance
tnr connect -t 8888                     # Connect with port forwarding (e.g., Jupyter)
tnr connect -t 8888 -t 6006            # Multiple port forwards (Jupyter + TensorBoard)
```

### File Transfer (SCP)
```bash
# Upload local -> remote
tnr scp ./local_file.txt 0:/remote/path/

# Download remote -> local
tnr scp 0:/remote/file.txt ./local_path/

# Upload entire directory
tnr scp ./my_dataset/ 0:/home/user/data/
```

**Notes:**
- SSH key setup, compression, and `~/` expansion handled automatically
- File transfers have a **60-second connection timeout**
- Instance ID is shown in `tnr status` (default is `0`)

### Port Forwarding
```bash
tnr connect -t <local_port>             # Forward single port
tnr connect -t 8888 -t 6006            # Forward multiple ports
```
- Public HTTPS URLs auto-generated: `https://<uuid>-<port>.thundercompute.net`
- DDoS protection included

---

## 8. SSH Access

### Recommended: Use CLI or VSCode Extension
For most users, `tnr connect` or the VSCode extension is easiest.

### Manual SSH Setup
1. Connect once with CLI: `tnr connect`
2. This auto-adds your instance as `tnr-0` in `~/.ssh/config`
3. Now you can use any SSH-compatible tool:
   ```bash
   ssh tnr-0                            # Direct SSH
   ```
4. Works with: VSCode Remote-SSH, JetBrains Gateway, or any SSH IDE

### Direct SSH (Advanced)
- Get instance IP from `tnr status`
- SSH port may NOT be 22 (dynamically assigned)
- SSH keys are managed at organization level
- Auto-configured after first `tnr connect`

---

## 9. Data Upload/Download

### Method 1: VSCode Drag-and-Drop (Easiest)
- **Upload:** Drag files from local file explorer into remote VSCode file explorer
- **Download:** Right-click files in remote explorer -> Download

### Method 2: CLI SCP
```bash
# Upload
tnr scp ./dataset.zip 0:/home/user/

# Download trained model
tnr scp 0:/home/user/runs/best.pt ./local_results/
```

### Method 3: git clone (Recommended for Code)
```bash
# On the remote instance terminal
git clone https://github.com/your-repo/your-project.git
```

### Method 4: wget/curl (For Datasets)
```bash
# On the remote instance terminal
wget https://url-to-your-dataset.zip
# or
curl -O https://url-to-your-dataset.zip
```

### Method 5: Cloud Storage (Recommended for Large Datasets)
Upload datasets to Google Drive, S3, or Hugging Face Hub, then download on instance.
- Cheaper than keeping data on Thunder Compute storage
- Prevents accidental billing from leaving instances running

### Network Speed
- 7 Gbps egress/ingress
- Fast enough for dataset downloads

---

## 10. Storage & Snapshots

### Disk Storage
- **First 100 GB included** during runtime (free)
- Additional storage: $0.10/GB/month
- Storage can only be **increased**, not decreased
- Use `tnr modify` to resize disk regardless of instance state

### CRITICAL: Storage is Ephemeral
- **When you DELETE an instance, ALL data is lost**
- Instances CANNOT be stopped or restarted - only deleted
- You MUST backup data before deleting

### Snapshots (Pause/Resume Workflow)
Snapshots preserve your complete instance state (files, installed packages, configs).

**Workflow:**
1. Work on your instance
2. Create a snapshot (via VSCode extension or console)
3. Delete the instance (stops billing)
4. When ready to resume: create new instance from snapshot
5. Restoring from snapshot takes ~10 minutes per 100 GB

**Snapshot Cost:** $0.05/GB/month

**Speed Up Snapshots with .thunderignore:**
- Create a `.thunderignore` file to exclude files from snapshots
- Exclude large datasets, cache files, etc.
- Dramatically speeds up snapshot creation/restoration

### Backup Best Practices
1. **Code:** Use GitHub/GitLab (push regularly!)
2. **Trained models:** Download best.pt via SCP or VSCode
3. **Datasets:** Store on cloud storage (Google Drive, HuggingFace, S3)
4. **Experiment logs:** Use Weights & Biases integration
5. **Environment:** Use snapshots for installed packages/configs

---

## 11. Installing Dependencies

### Pre-installed (Base Template)
- CUDA 13.0 + Driver 580
- PyTorch 2.9.0+cu128
- JupyterLab
- NumPy, Pandas, scientific Python stack

### Using pip
```bash
pip install ultralytics                  # YOLO
pip install mamba-ssm --no-build-isolation   # Mamba SSM
pip install causal-conv1d>=1.4.0 --no-build-isolation  # causal-conv1d for Mamba
pip install wandb                        # Weights & Biases
```

**IMPORTANT:** For mamba-ssm, ALWAYS use `--no-build-isolation` flag so pip uses the existing CUDA-enabled PyTorch instead of installing torch-cpu in an isolated build environment.

### Using Conda (Miniforge ONLY)
**WARNING:** Only use **Miniforge** on Thunder Compute. Anaconda and Miniconda may have compatibility issues with system libraries.

```bash
# Install Miniforge
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh

# Create environment
conda create -n myenv python=3.11
conda activate myenv

# Install packages with conda first, then pip
conda install pytorch torchvision -c pytorch
pip install ultralytics
pip install mamba-ssm --no-build-isolation
```

**Best Practices:**
1. Install as many packages as possible with conda FIRST
2. Then use pip for remaining packages
3. Use isolated conda environments
4. Do NOT use `pip --user` flag
5. Snapshot after installing all dependencies to save setup time

### Docker (Experimental)
```bash
# Must use this flag format (not --runtime=nvidia or --gpus=all)
docker run --device nvidia.com/gpu=all your-image
```

---

## 12. Instance Templates

| Template | Description |
|----------|-------------|
| **base** | Ubuntu + PyTorch + CUDA (use this for YOLO/Mamba) |
| **ollama** | Ollama server with OpenWebUI |
| **comfy-ui** | ComfyUI for AI image generation |
| **comfy-ui-wan** | ComfyUI with Wan2.1 pre-installed |
| **webui-forge** | WebUI Forge for Stable Diffusion |

**For your work:** Use the **base** template.

---

## 13. Known Issues & Limitations

### Critical Limitations
1. **Instances CANNOT be stopped or restarted** - only deleted and recreated
2. **Storage is ephemeral** - all data lost when instance deleted
3. **Custom CUDA kernels** have unpredictable behavior in prototyping mode
4. **Some CUDA API functions** not implemented in prototyping mode
5. **GPU library support** is limited but can be added on request (within days)
6. **Multi-GPU:** Max 2 GPUs in prototyping, max 8 in production

### Performance Concerns
7. **Prototyping mode performance varies** - GPU-over-TCP adds latency
8. **Potential 20-50% slowdown** for certain tasks vs physically-attached GPU
9. **Network dependency** - internet issues can affect GPU access
10. **7 Gbps network cap** - may bottleneck very large data transfers

### Operational Issues
11. **Snapshot restoration:** ~10 min per 100 GB (can be slow for large envs)
12. **SCP timeout:** 60-second connection timeout for file transfers
13. **SSH port is NOT always 22** - dynamically assigned
14. **Dynamic IP addresses** - IP changes between instances

### Compliance
15. **Crypto mining = immediate termination** and credit revocation
16. **Service unavailable** in 13 countries (China, Russia, North Korea, etc.)
17. **One account per person** - no self-referrals

### What This Means for Your YOLO + Mamba Project
- **YOLO training (PyTorch):** Should work fine in prototyping mode
- **Mamba SSM (custom CUDA kernels):** MAY have issues in prototyping mode
- **If Mamba fails:** Switch to production mode ($1.79/hr vs $0.78/hr)
- **Always backup:** Push code to GitHub, save models via SCP before deleting

---

## 14. Troubleshooting

### Connection Issues
```bash
# Disconnect and reconnect
ctrl + d
tnr connect <instance_id>

# Upgrade CLI
pip install tnr --upgrade

# Nuclear option: backup data, delete instance, recreate
```

### "This function is not implemented" Error
- Unsupported CUDA API in prototyping mode
- Solution: Switch to production mode
- Or contact support to request library support

### SSH Errors ("Error reading SSH protocol banner", permission denied)
1. Retry the command first
2. Check Remote-SSH extension is installed
3. Backup data and recreate instance if persistent

### GPU Not Detected
- Verify with `nvidia-smi` on the instance
- Check if instance is fully initialized (~60 sec after creation)

### Mamba SSM Installation Issues
```bash
# If standard install fails:
pip install mamba-ssm --no-build-isolation

# If version conflict:
pip install mamba-ssm==2.2.3.post2 --no-build-isolation

# Check PyTorch CUDA match:
python -c "import torch; print(torch.version.cuda)"
```

---

## 15. A100 80GB Optimization for YOLO + Mamba

### A100 80GB Key Specs
- **VRAM:** 80 GB HBM2e
- **Memory Bandwidth:** 2.0 TB/s
- **FP32:** 19.5 TFLOPS
- **TF32:** 312 TFLOPS (tensor cores)
- **FP16/BF16:** 312 TFLOPS (tensor cores)
- **CUDA Cores:** 6,912
- **Tensor Cores:** 432

### YOLO Training Optimization on A100

**Batch Size:**
- A100 80 GB can handle MUCH larger batches than Kaggle T4/P100
- **Recommended starting point:** batch=64 for YOLOv8/v11/v12 at imgsz=640
- **Use AutoBatch:** Set `batch=-1` to let Ultralytics auto-determine optimal batch
- AutoBatch typically recommends batch ~43 for A100 (targeting ~66% VRAM usage)
- With 80 GB VRAM: batch=64-128 is feasible for most YOLO model sizes
- **Test first:** Do a 2-3 epoch sanity run to verify batch size doesn't OOM

**Image Size:**
- A100 can handle imgsz=960-1280 (beyond standard 640)
- For VisDrone/aerial: imgsz=960 or 1280 can significantly improve small object detection
- Trade-off: larger imgsz = better accuracy but slower training
- Standard: imgsz=640 with 600 epochs (COCO standard)

**Mixed Precision (BF16 - BEST for A100):**
```python
# Ultralytics handles this automatically, but ensure it's enabled:
model.train(data='dataset.yaml', epochs=100, batch=64, amp=True)
```
- A100 has native BF16 support on tensor cores
- BF16 is as fast as FP16 but MORE numerically stable
- BF16 keeps 8-bit exponent (same as FP32), so no gradient scaling needed
- No NaN gradient issues like FP16
- Performance: 1.5-5.5x faster than FP32, additional 1.3-2.5x on A100 vs V100

**TF32 (Automatic on A100):**
- TF32 is the default for single-precision operations on A100
- 10x faster tensor math than V100 (convolutions and matrix multiplies)
- Practical speedup: 2-6x over V100
- No code changes needed - automatic
- Same accuracy as FP32

**DataLoader Optimization:**
```python
# For Ultralytics YOLO, set workers parameter:
model.train(
    data='dataset.yaml',
    epochs=100,
    batch=64,
    workers=8,        # Match to available vCPUs (4 in prototyping, 18 in production)
    amp=True,
)
```
- Set `num_workers` = number of available vCPUs (4 for prototyping, 18 for production)
- `pin_memory=True` (Ultralytics does this automatically)
- `persistent_workers=True` avoids worker respawn overhead between epochs

### Mamba SSM on A100

**Installation:**
```bash
pip install mamba-ssm --no-build-isolation
pip install causal-conv1d>=1.4.0 --no-build-isolation
```

**Version Compatibility (as of 2026):**
- mamba-ssm 2.2.5 confirmed working with CUDA 12.x
- Requires matching PyTorch + CUDA versions
- Thunder Compute has PyTorch 2.9.0+cu128 (CUDA 12.8)
- Check for pre-built wheels: look for versions matching your CUDA

**If Build Fails:**
```bash
# Try specific version with pre-built wheel
pip install mamba-ssm==2.2.3.post2 --no-build-isolation

# Or install from source
pip install mamba-ssm --no-build-isolation --no-cache-dir

# Verify CUDA match
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
```

**A100 Mamba Performance Tips:**
- Mamba's selective scan CUDA kernel is optimized for A100 tensor cores
- Use BF16 for Mamba layers where possible
- causal-conv1d CUDA kernels provide significant speedup over naive implementation
- The 80 GB VRAM allows larger sequence lengths and batch sizes

### Recommended Training Configuration

```python
# YOLO + Mamba training on Thunder Compute A100
from ultralytics import YOLO

model = YOLO('your_model.yaml')
model.train(
    data='your_dataset.yaml',
    epochs=100,               # Adjust based on budget
    batch=-1,                 # AutoBatch (or set 64-128 manually)
    imgsz=640,                # Or 960/1280 for aerial detection
    workers=8,                # Match available vCPUs
    amp=True,                 # Mixed precision (BF16 on A100)
    device=0,                 # Single GPU
    patience=50,              # Early stopping
    save_period=10,           # Save checkpoint every 10 epochs
    project='runs/train',
    name='atrous_mamba_exp',
)
```

### Time & Cost Estimates (A100 Prototyping at $0.78/hr)

| Training Scenario | Est. Time | Est. Cost |
|-------------------|-----------|-----------|
| 100 epochs, batch=64, imgsz=640, VisDrone | ~4-8 hrs | ~$3-6 |
| 300 epochs, batch=64, imgsz=640, VisDrone | ~12-24 hrs | ~$9-19 |
| 100 epochs, batch=32, imgsz=1280, VisDrone | ~8-16 hrs | ~$6-12 |

*Note: These are rough estimates. Actual time depends on model complexity, dataset size, and prototyping mode overhead.*

---

## 16. Key URLs & Resources

### Official Documentation
- **Docs Home:** https://www.thundercompute.com/docs
- **Full Docs (text):** https://www.thundercompute.com/docs/llms-full.txt
- **VSCode Quickstart:** https://www.thundercompute.com/docs/vscode/quickstart
- **CLI Reference:** https://www.thundercompute.com/docs/cli-reference
- **Troubleshooting:** https://www.thundercompute.com/docs/troubleshooting
- **Technical Specs:** https://www.thundercompute.com/docs/technical-specs
- **Prototyping Mode:** https://www.thundercompute.com/docs/prototyping-mode
- **Pricing:** https://www.thundercompute.com/pricing

### Guides
- **SSH Guide:** https://www.thundercompute.com/docs/guides/ssh-on-thunder-compute
- **Jupyter Notebooks:** https://www.thundercompute.com/docs/guides/running-jupyter-notebooks-on-thunder-compute
- **Installing Conda:** https://www.thundercompute.com/docs/guides/installing-conda
- **Ephemeral Storage:** https://www.thundercompute.com/docs/guides/using-ephemeral-storage
- **VSCode Integration:** https://www.thundercompute.com/docs/guides/vscode-integration-for-thunder-compute

### Console & Extensions
- **Console:** https://console.thundercompute.com
- **VSCode Extension:** https://marketplace.visualstudio.com/items?itemName=ThunderCompute.thunder-compute
- **GitHub:** https://github.com/Thunder-Compute
- **tnr CLI (PyPI):** https://pypi.org/project/tnr/

### Support
- **Discord:** Primary support channel (founding team responds)
- **Email:** support@thundercompute.com
- **Product Hunt:** https://www.producthunt.com/products/thunder-compute

### A100 & Training References
- **NVIDIA A100 Specs:** https://www.nvidia.com/en-us/data-center/a100/
- **PyTorch AMP Docs:** https://docs.pytorch.org/docs/stable/amp.html
- **Ultralytics Training Docs:** https://docs.ultralytics.com/modes/train/
- **Mamba SSM GitHub:** https://github.com/state-spaces/mamba
- **Mamba SSM PyPI:** https://pypi.org/project/mamba-ssm/

---

## Quick Start Checklist for Your YOLO+Mamba Project

1. [ ] Install VSCode + Remote-SSH extension
2. [ ] Install Thunder Compute extension in VSCode
3. [ ] Create account at console.thundercompute.com
4. [ ] Add payment method
5. [ ] Create A100 instance (prototyping mode, base template, 100GB disk)
6. [ ] Connect via VSCode extension
7. [ ] Clone your repo: `git clone <your-repo>`
8. [ ] Install dependencies:
   ```bash
   pip install ultralytics wandb
   pip install mamba-ssm --no-build-isolation
   pip install causal-conv1d>=1.4.0 --no-build-isolation
   ```
9. [ ] Upload/download dataset (wget, tnr scp, or cloud storage)
10. [ ] Run training with batch=-1, amp=True, workers=8
11. [ ] Monitor with wandb or TensorBoard (port forward 6006)
12. [ ] Download results: `tnr scp 0:/path/to/best.pt ./`
13. [ ] Create snapshot before deleting instance
14. [ ] Delete instance to stop billing
15. [ ] If Mamba CUDA kernels fail: recreate with production mode
