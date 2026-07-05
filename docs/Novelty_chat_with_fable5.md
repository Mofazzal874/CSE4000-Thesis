I now have a complete factual picture. Here is the structured audit report.

THESIS REPO AUDIT — Aerial SAR Human Detection (YOLO11m+CBAM+P2)
All paths below are absolute under d:/Academics/thesis folder/. Numbers are read directly from summary JSONs / CSVs / master-report txts / args.yaml. No recommendations, facts only.

(A) FINAL ABLATION NUMBERS
A1. Core C2A architecture ablation (single seed s0, one consistent protocol: AdamW lr0=0.001, ~300 ep chain, COCO eval + per-size recall)
Source table: Last Month/docs/2026-06-13_complete_ablation_table.md. Per-model metrics cross-checked against each run's metrics/summary.json.

Model	COCO AP (mAP50-95)	COCO AP50	AP_small	very-tiny(<8px) recall	tiny(8-16px) recall	params	GFLOPs	latency e2e
YOLO11m baseline	0.6151	0.8432	0.6147	0.7427	0.8688	20.03M	67.7	13.7 ms
+ CBAM	0.6161	0.8473	0.6160	0.7461	0.8730	19.10M	~68	~14 ms
+ CBAM + P2	0.6153	0.8533	0.6156	0.7575	0.8651	19.57M	86.7	14.6 ms
+ CBAM + P2 + Mamba (SSM neck)	0.6143	0.8521	0.6146	0.7567	0.8700	22.01M	98.4	41.1 ms
Dataset split: C2A test (2043 images), single seed. Multi-seed rollups exist per model (.../runs/<tag>_multi_seed_rollup/cross_seed_metrics.csv) but the headline table is single-seed s0.
CBAM+P2 full per-size recall (.../CBAM_P2Head/runs/20260602_063759_.../metrics/summary.json): very_tiny 0.7575, tiny 0.8651, small 0.8857, medium 0.8076.
Mamba per-size recall (.../02-06-26-Mamba_CBAM_P2Head/runs/20260609_205717_.../metrics/summary.json): very_tiny 0.7567, tiny 0.8700, small 0.8873, medium 0.8107.
Baseline/Mamba also carry Ultralytics mAP50 (0.8597 / 0.8678); CBAM & CBAM+P2 only have COCO-AP headline (their pre-fix .val crashed — noted in Last Month/cross_dataset_SARD/README.md).
Documented interpretation (same doc): mAP50-95 is saturated ~0.615 across all variants (attributed to C2A's composited/pasted boxes capping high-IoU localization); the P2 head is the only monotonic gain (+1.0 AP50, +1.5pt very-tiny recall at near-zero param cost); CBAM marginal; Mamba adds +2.4M params and ~2.8–3× latency for no gain.
Run dirs (all on RTX 4070 Ti SUPER 16GB, confirmed via env.json gpu_name):

baseline: Last Month/24_01_26- Benchmarking YOLOs/Yolo11m/runs/20260615_230315_yolo11m_baseline_s0_nogit/
cbam: Last Month/24_01_26- Benchmarking YOLOs/CBAM/runs/20260601_232929_yolo11m_cbam_s0_nogit/
cbam_p2: Last Month/24_01_26- Benchmarking YOLOs/CBAM_P2Head/runs/20260602_063759_yolo11m_cbam_p2head_s0_nogit/
mamba: Last Month/02-06-26-Mamba_CBAM_P2Head/runs/20260609_205717_mamba_cbam_p2head_s0_nogit/
A2. SAHI + TTA (inference-only; run on the "mamba_cbam_p2_pretrained" checkpoint, C2A 2043-test)
Source: 31-03-26(Mamba-ViT-CNN)/SAHI+TTA/benchmark_reports/MASTER_REPORT_SAHI_TTA_NOAUG.txt. (Note: that "mamba" pretrained model is the pre-fix build = CBAM+P2 architecture, 19.592M — see section C.)

Config	mAP50	mAP50-95	very-tiny recall	F1	latency
Baseline 640	0.8736	0.6443	0.7668	0.8500	45.2 ms
SAHI slice256_ov30	—	—	0.8292 (+0.0624)	0.8289	534 ms
SAHI slice320_ov25	—	—	0.8220	0.8421	365 ms
SAHI slice512_ov25	—	—	0.8081	0.8657	187 ms
SAHI slice640_ov30	—	—	0.8005	0.8635	147 ms
TTA 832	0.8800	0.6768	—	—	—
TTA 1280 (best TTA)	0.8900	0.6982	—	—	—
TTA 1920	0.8658	0.6824	—	—	—
SAHI(256_ov30)+TTA combined	—	—	0.8292	0.8289 (F2 0.8542)	530 ms
Best small-object win = SAHI slice256_ov30 (very-tiny recall +0.062); best aggregate = TTA@1280 (mAP50-95 +0.054). Design/analysis docs: 31-03-26(Mamba-ViT-CNN)/docs/06.sahi-tta-eval-design-2026-04-12.md, 07.sahi-tta-results-analysis-2026-04-12.md.

A3. Cross-dataset C2A→SARD (generalization)
Source: Last Month/cross_dataset_SARD/ablation_master/sard_generalization.csv and fewshot_curve.csv.

Zero-shot C2A→SARD collapses ~99% for every model: baseline mAP50 0.8597→0.0058 (−99.3%); CBAM 0.0147; CBAM+P2 0.0042; Mamba 0.8678→0.0004 (−99.96%).
Few-shot fine-tune curve (fine-tune on N SARD imgs, eval SARD-test), CBAM+P2 vs Mamba, key point N=200: CBAM+P2 mAP50 0.763 / COCO_AP 0.399; Mamba mAP50 0.743 / COCO_AP 0.356. (very-tiny/tiny recall = 0.0 at all N — SARD has almost no tiny objects.) Both COCO_AP@200 far exceed the Lee et al. rendered-synthetic anchor (17.73) cited in the README.
Few-shot run dirs: Last Month/cross_dataset_SARD/runs_fewshot/{mamba_cbam_p2head,yolo11m_cbam_p2head}_N{0,20,50,100,200}/.
(B) CURRENT BEST MODEL + WEIGHTS PATH + WHICH GPU/PC RAN THE RECENT FINETUNE
"deployable_model" folder contents (Last Month/deployable_model/):

c2a_cbam_p2head_best.pt (39.8 MB, Jun 2) — the C2A-only CBAM+P2 checkpoint (from run 20260602_063759); used as the joint-training init.
epoch125.pt (157.7 MB, Jun 28) — the joint C2A+SARD model, produced by an earlier joint run runs_joint/20260627_162506_cbam_p2head_joint_c2a_sard (referenced as the MODEL path inside A6000 run/run_on_drone_footage.py:31). Reference metrics (from finetune_summary): C2A-test mAP50 0.878, SARD-test mAP50 0.917. This is the strongest benchmark model.
joint_c2a_sard_train.py, README.md.
Most recent finetune — runs_joint/20260702_144117_finetune_enriched/ (this is the only run currently under A6000 run/runs_joint/):

Source: .../metrics/finetune_summary.json + .../ultra/args.yaml.
Init = epoch125.pt; trained on C2A+SARD plus enriched C2A (common/c2a/c2a_enriched, produced by enrich_c2a_for_robustness.py = "Novelty Angle B": hard-negative real-background tiles + multi-blend copy-paste to de-bias paste artifacts).
Hyperparams: 25 epochs, patience 25, lr0=0.0003 (AdamW, cos_lr), batch=4, imgsz 640, cache=disk, workers=4, amp=false, deterministic, seed 0.
Result: C2A-test mAP50 0.8741 / mAP50-95 0.6477; SARD-test mAP50 0.8976 / mAP50-95 0.5574 — i.e. slightly BELOW the epoch125 reference it started from (C2A 0.878→0.874, SARD 0.917→0.898). C2A AP_small 0.630, SARD AP_small 0.294. C2A per-size recall: very_tiny 0.766 / tiny 0.872 / small 0.897 / medium 0.893; SARD very_tiny/tiny = 0.0 (few such objects), small 0.648 / medium 0.916 / large 0.953.
Finetuned weights: Last Month/deployable_model/A6000 run/runs_joint/20260702_144117_finetune_enriched/ultra/weights/best.pt (39.75 MB, Jul 2 20:25). (The summary JSON's finetuned_weights field points to a weights/best.pt at the run root, but that dir is empty; the actual best.pt is under ultra/weights/. Also present: ultra/weights/epoch0.pt, and an epoch10.pt (917 MB) timestamped Jul 4 — i.e. a run was writing checkpoints during today's session.)
GPU / PC that ran it: the enriched finetune ran on the KUET Mobile Lab A6000 machine ("PC-2"), not the ablation PC. Evidence:

Paths in finetune_summary.json are D:\thesis_2007074\...; args.yaml device: '0' (in-process cuda:0 after the auto-GPU-pick sets CUDA_VISIBLE_DEVICES).
A6000 run/README.md: PC-2 = 2× NVIDIA RTX A6000, 47.5 GB each, Intel Xeon Gold 6346 (32c/64t), 256 GB RAM, Windows 11; script auto-pins to a free GPU ("You: GPU 1").
A6000 run/SETUP_PC2.md (root D:\student_2k20\2007074) and SETUP_PC3.md (root D:\2007074) document transfer to two lab PCs.
Machine credentials: PC Access.txt lists AnyDesk IDs for "Masud Sir pc", "Mobile Lab_PC_1", "Mobile_Lab_PC_2".
By contrast, the C2A architecture ablation chain (baseline/CBAM/CBAM+P2/Mamba) all ran on an RTX 4070 Ti SUPER 16GB ("PC-1"), confirmed by gpu_name in each run's env.json (torch 2.12.0+cu126, Windows-10.0.26200); e.g. baseline batch=16, Mamba VRAM peak 16.4 GB.
(C) COMPLETE LIST OF MAMBA ATTEMPTS + FAILURE MODES
Timeline (earliest→latest). Master reports and failure docs cited per item.

11-Mar-2026 — "Mamba+CBAM+P2" (LocalWindowSSM), 120 ep, Kaggle T4/P100.

01-03-2026-Onward Model trying/11-3-26-Mamba/mamba_run/benchmark_reports/MASTER_REPORT_MAMBA_CBAM_P2.txt: reported mAP50 0.877 / mAP50-95 0.6539, 19.592M params, 43.7 GFLOPs.
FAILURE MODE (not genuine Mamba): 01-03-2026-Onward Model trying/docs/2026-03-20_Critical_Bug_Injection_Lost.md — Ultralytics model.train() rebuilds the model fresh from YAML in DetectionTrainer.get_model(), discarding all post-init injected modules. Dry-run showed 25.03M params but the trained/saved model was 19.592M (= vanilla CBAM+P2). The "0.877" was CBAM+P2 trained 120 ep, not Mamba. Injection silently lost.
20-Mar-2026 — AtrousSSM Mamba (dilations [1,2,4], d_state=4, 3 parallel bidirectional branches, gated fusion). Dir: 01-03-2026-Onward Model trying/20-3-26-atrousSSM Mamba/ (single_gpu_atrous_mamba_1st_run, double_gpu_atrous_mamba, atrous_mamba_checkpoint).

Two initial 2-epoch tests: injection lost again — DDP run gave all-zero detections; single-GPU gave the base model (2026-03-20_Critical_Bug_Injection_Lost.md).
After the monkey-patch fix (re-inject C3K2Mamba inside a patched get_model), a genuine 80-ep 2×T4 run completed (24.156M params, 48.1 GFLOPs — injection confirmed). Result: .../double_gpu_atrous_mamba/benchmark_reports/MASTER_REPORT_ATROUSMAMBA_CBAM_P2.txt → mAP50 0.8228 / mAP50-95 0.5540, very-tiny recall 0.6831, latency 105 ms. FAILURE MODE: genuine but markedly WORSE than CBAM+P2 (0.874/0.645, VT 0.765) — ~10% mAP drop, ~2× latency, degraded tiny-object recall. Documented as a negative result in 2026-03-20_AtrousSSM_Honest_Assessment.md and 2026-03-29_Detailed_Training_Metrics.md (§4/§5).
31-Mar/11-12-Apr-2026 — SAHI+TTA + copy-paste on "mamba_cbam_p2_pretrained". Dir: 31-03-26(Mamba-ViT-CNN)/.

The "pretrained mamba" model used here is the pre-fix 19.592M build (= CBAM+P2) — used for the inference experiments in section A2.
Copy-paste augmentation training: mamba_cbam_p2_copypaste/benchmark_reports/MASTER_REPORT_COPYPASTE.txt → worse (mAP50 0.8690 vs 0.8736 no-aug). Git commit 5ec97cd: "fail: run the copypaste augmentation and model performance decreased."
09-Jun-2026 — Mamba+CBAM+P2, FIRST genuinely valid SSM run. Dir: Last Month/02-06-26-Mamba_CBAM_P2Head/runs/20260609_205717_.../. Verdict doc: .../docs/2026-06-10_mamba_run_complete_verdict.md.

Architecture = 6 LocalWindowSSM blocks injected at neck layers 13,16,19,22,25,28, d_state=4, per-layer window sizes 6–8 (metrics/summary.json → mamba_ssm block). Genuineness confirmed: 22.01M params / 98.4 GFLOPs / 183 layers; fwd-vs-reverse scan cosine distance 0.836; SSM state A≈2.5 bounded. Gated re-injection patch used to survive the train()-rebuild bug.
Result: NULL — mAP50-95 0.6143 vs CBAM+P2 0.6153, AP50 0.8521 vs 0.8533, very-tiny recall 0.7567 vs 0.7575; +2.4M params and ~2.8× latency (41.1 vs 14.6 ms) for no gain. Trained 29.5h, best epoch 149, no divergence. Declared a "credible NULL result"; CBAM+P2 kept as the recommended/deployable model.
Additional documented reasons Mamba was dropped (Last Month/docs/2026-06-29_novelty_research_findings.md §9): mamba-ssm/causal-conv1d CUDA kernels are Linux-only (won't build on the Windows box); independent SeaDronesSee test shows MambaYOLO < YOLO11m; LocalMamba shows zero small-object gain. Framing corrected away from mis-citing MambaOut.
Net: only LocalWindowSSM and AtrousSSM neck variants were ever actually trained. LocalWindowSSM = null (tie); AtrousSSM = genuine failure (−10% mAP). No Mamba backbone swap was implemented.

(D) OTHER ARCHITECTURES TRIED
ECA attention (alternative to CBAM/C2PSA): actually trained. 01-02-2026- ablation study/5-2-26/results/benchmark_reports/MASTER_SUMMARY.txt and 2026-03-29_Detailed_Training_Metrics.md → ECA ≈ baseline (mAP50 0.8489 vs 0.8490, mAP50-95 0.5995 vs 0.6020; −0.25%). "RECOMMENDATION: Baseline" — ECA gave no gain, abandoned.
YOLO family benchmarking (Phase 1, 25 ep, Kaggle): YOLOv9 s/m/e, YOLOv10 s/m/l, YOLO11 s/m/l — numbers in 2026-03-29_Detailed_Training_Metrics.md §1 (YOLO11m chosen: mAP50 0.8408, mAP50-95 0.5863). Last Month/24_01_26- Benchmarking YOLOs/Yolov9s/yolov9s_final_month.py present (script only; no local run dir).
P2 head and CBAM individually — part of the ablation (section A/A1).
ViT / CNN-hybrid / MambaVision / RT-DETR — PLANNED / DISCUSSED ONLY, NEVER IMPLEMENTED. Despite the folder name 31-03-26(Mamba-ViT-CNN), that folder contains only Mamba work (copypaste + SAHI/TTA + pretrained). No vit/swin/detr/transformer python files exist anywhere in the repo (grep returned none). Hybrid CNN+ViT / MambaVision / Mamba-YOLO / DETR were evaluated on paper and rejected in 31-03-26(Mamba-ViT-CNN)/docs/01.research-direction-analysis-2026-04-11.md (Direction D "HIGH RISK, NOT RECOMMENDED": ViT/DETR too heavy/slow, MambaVision 31.8M+ params non-trivial YOLO integration, CC-BY-NC license). BiFPN/AFPN/Gold-YOLO neck were also only suggested (Direction B), not built. RT-DETR appears only as reference notes in week-11/RT-DETR_Quick_Reference.md and RT-DETR_Numerical_Examples.md — no training.
(E) GREP RESULTS — fusion / loss / SR techniques (already-tried vs never-tried)
None of the requested techniques were ever implemented or experimented. No .py file in the entire repo contains wbf / weighted-boxes / wasserstein / nwd / rfla / soft-nms / wiou / wise-iou / shape-iou / inner-iou / esrgan / ensemble-boxes (grep over all *.py returned "No files found"). They appear only as text in proposal/citation docs:

Technique	Status	Where mentioned
WBF (weighted boxes fusion)	Never tried — suggested as a "bonus reliable booster" (ensemble-boxes pkg)	Last Month/docs/2026-06-29_novelty_research_findings.md:64
NWD / Wasserstein loss	Never tried — explicitly listed under "What I would NOT do" and as an optional-only idea; plus a FlyPose citation	Last Month/docs/2026-06-10_transfer_estimates_and_novelty_proposal.md:104,137; 2026-06-10_paper_worthiness_verdict.md:136
Super-resolution / ESRGAN / diffusion-SR	Never tried — a rejected "Option C" preprocessing idea (EDiffSR)	01-03-2026-Onward Model trying/claude suggestion-01.md:14, claude_response.md:41 (unrelated hit in Defense/IEEE Reference Style Guide...md)
RFLA	Never mentioned anywhere	—
Soft-NMS	Never mentioned	— (only standard torchvision NMS used to merge SAHI tiles in run_on_drone_footage.py)
WiseIoU / ShapeIoU / InnerIoU	Never mentioned	—
Adaptive slicing	Never tried — only fixed SAHI slice sweeps (256/320/512/640) in section A2	31-03-26(Mamba-ViT-CNN)/SAHI+TTA/
Note: the only IoU-loss code hits repo-wide (GIoU/DIoU/CIoU in Object-Detection-main/.../yolov5/utils/loss.py, metrics.py, segment/loss.py) are the stock Ultralytics/YOLOv5 framework functions in a cloned repo — not thesis experiments. NMS in the deployable scripts is plain torchvision.ops.nms for tile merging.

(F) DRONE SHOOT DATASET FACTS
Dir: d:/Academics/thesis folder/Drone Shoot/ — 3 raw video files, no labels, no extracted frames stored here:

10m.MP4 — 320,906,847 bytes (~306 MB)
30m.MP4 — 290,072,890 bytes (~277 MB)
50m.MP4 — 354,113,014 bytes (~338 MB)
Filenames = capture altitudes (10 m / 30 m / 50 m). Per A6000 run/run_on_drone_footage.py header, frames are 3840×2160 (4K); the model runs SAHI-style 640-px tiled inference (overlap, torchvision-NMS merge, CONF=0.35) plus a whole-frame imgsz=1280 baseline for comparison. No TTA/WBF in that script.

No label files exist. The self-training plan (Last Month/docs/2026-06-29_novelty_research_findings.md §8) states ~15–20 frames still need hand-labeling in Roboflow to build a real test set — i.e. the footage is currently unlabeled.
Inference output so far — only the 10 m video processed: Last Month/deployable_model/A6000 run/drone_inference_out/10m/ contains annotated sliced_fXXXXXX.jpg frames (sampled every 120 frames, 0→~1920) and counts.csv (frame_id, sliced_count, whole_count) showing sliced tiling consistently detects more people than whole-frame (e.g. frame 480: sliced 48 vs whole 61; frame 1680: sliced 21 vs whole 44 — counts vary per frame). No 30m/ or 50m/ output dirs exist yet.
Related capture guidance: Last Month/docs/2026-06-28_drone_capture_altitude_guide.md.
Cross-cutting note on provenance / naming caveats
The label "Mamba" on the pre-June checkpoints (mamba_cbam_p2_pretrained, the 11-Mar and copypaste/SAHI models, all 19.592M params) is misleading — those are architecturally CBAM+P2 due to the injection-loss bug. Only the two June-era runs (LocalWindowSSM 22.01M, and the March AtrousSSM 24.156M) are genuine SSM models.
Path roots differ per machine: ablation on PC-1 (RTX 4070 Ti SUPER); enrichment/joint/finetune/inference on the A6000 lab PC(s) under D:\thesis_2007074 / D:\student_2k20\2007074 (PC-2) or D:\2007074 (PC-3).
Web SearchRLRD-YOLO Drones 2025 VisDrone mAP results ablation cross-dataset
OUT
Web search results for query: "RLRD-YOLO Drones 2025 VisDrone mAP results ablation cross-dataset"

Links: [{"title":"SL-YOLO: A Stronger and Lighter Drone Target Detection Model","url":"https://arxiv.org/html/2411.11477v3"},{"title":"CF-YOLO for small target detection in drone imagery based on YOLOv11 algorithm - PMC","url":"https://pmc.ncbi.nlm.nih.gov/articles/PMC12078580/"},{"title":"RLRD-YOLO: An Improved YOLOv8 Algorithm for Small Object Detection from an Unmanned Aerial Vehicle (UAV) Perspective","url":"https://www.mdpi.com/2504-446X/9/4/293"},{"title":"IECA-YOLOv7: A Lightweight Model with Enhanced Attention and Loss for Aerial Wildlife Detection - PMC","url":"https://pmc.ncbi.nlm.nih.gov/articles/PMC12466753/"},{"title":"(PDF) RLRD-YOLO: An Improved YOLOv8 Algorithm for Small Object Detection from an Unmanned Aerial Vehicle (UAV) Perspective","url":"https://www.researchgate.net/publication/390692325_RLRD-YOLO_An_Improved_YOLOv8_Algorithm_for_Small_Object_Detection_from_an_Unmanned_Aerial_Vehicle_UAV_Perspective"},{"title":"LRDS-YOLO enhances small object detection in UAV aerial images with a lightweight and efficient design | Scientific Reports","url":"https://www.nature.com/articles/s41598-025-07021-6"},{"title":"CF-YOLO for small target detection in drone imagery based on YOLOv11 algorithm | Scientific Reports","url":"https://www.nature.com/articles/s41598-025-99634-0"},{"title":"MASF-YOLO: An Improved YOLOv11 Network for Small Object Detection on   Drone View","url":"https://arxiv.org/pdf/2504.18136"},{"title":"DroneScan-YOLO: Redundancy-Aware Lightweight Detection for Tiny Objects in UAV Imagery","url":"https://arxiv.org/pdf/2604.13278"}]

I found relevant search results about RLRD-YOLO. Here's what the research shows:

## RLRD-YOLO Performance Results

RLRD-YOLO outperformed YOLOv8 on the VisDrone2019 dataset, achieving improvements of 12.2% in mAP@0.5 and 8.4% in mAP@0.5:0.95.

## Key Components

The RLRD-YOLO model incorporates several important improvements:

- A Reparameterized Generalized Feature Pyramid Network (RepGFPN) replaces the original architecture in the neck network to enhance feature fusion and effectively integrate low-level details with high-level semantic information
- A small-target detection layer is added to enhance the model's ability to perceive small targets
- The detecting head is replaced with the Dynamic Head, designed to improve the localization accuracy of small targets in complex scenarios

## Ablation and Cross-Dataset Evaluation

The paper includes ablation experiments and cross-dataset evaluation. The research presents ablation experiments and comparisons of different algorithms on the HIT-UAV dataset, demonstrating the model's performance across multiple drone detection benchmarks beyond VisDrone2019.


REMINDER: You MUST include the sources above in your response to the user using markdown hyperlinks.
Web Fetchhttps://api.semanticscholar.org/graph/v1/paper/DOI:10.3390/ai6100272?fields=title,abstract,venue,year
Fetched from https://api.semanticscholar.org/graph/v1/paper/DOI:10.3390/ai6100272?fields=title,abstract,venue,year
Web Search"Mamba-YOLO" 2406.05835 accepted venue journal OR conference 2025
OUT
Web search results for query: ""Mamba-YOLO" 2406.05835 accepted venue journal OR conference 2025"

Links: [{"title":"REA-YOLO for small object detection in UAV aerial images | The Journal of Supercomputing | Springer Nature Link","url":"https://link.springer.com/article/10.1007/s11227-025-07836-0"},{"title":"YOLO-BSMamba: A YOLOv8s-Based Model for Tomato Leaf Disease Detection in Complex Backgrounds","url":"https://www.mdpi.com/2073-4395/15/4/870"},{"title":"FF-Mamba-YOLO: An SSM-Based Benchmark for Forest Fire Detection in UAV Remote Sensing Images","url":"https://www.mdpi.com/2313-433X/12/1/43"},{"title":"[2406.05835v1] Mamba YOLO: SSMs-Based YOLO For Object Detection","url":"https://arxiv.org/abs/2406.05835v1"},{"title":"MHS-VIT: Mamba hybrid self-attention vision transformers for traffic image detection | PLOS One","url":"https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0325962"},{"title":"[2406.05835] Mamba YOLO: A Simple Baseline for Object Detection with State Space Model","url":"https://arxiv.org/abs/2406.05835"},{"title":"GitHub - HZAI-ZJNU/Mamba-YOLO: the official pytorch implementation of “Mamba-YOLO：SSMs-based for Object Detection”","url":"https://github.com/HZAI-ZJNU/Mamba-YOLO"},{"title":"Paper page - Mamba YOLO: SSMs-Based YOLO For Object Detection","url":"https://huggingface.co/papers/2406.05835"},{"title":"Segmentation Strategies in Deep Learning for Prostate Cancer Diagnosis:   A Comparative Study of Mamba, SAM, and YOLO","url":"https://arxiv.org/pdf/2409.16205"}]

Based on the search results, the Mamba YOLO paper (arXiv 2024, arXiv:2406.05835) is titled "Mamba YOLO: A simple baseline for object detection with state space model".

However, the search results indicate that this paper was submitted to arXiv on June 9, 2024, and appears to remain as an arXiv preprint. While the search results show multiple 2025 publications that cite the Mamba YOLO paper (such as papers in *The Journal of Supercomputing*, MDPI journals, and *PLOS One*), the search results do not contain explicit information about the Mamba YOLO paper itself being accepted to a specific venue or journal for 2025.

The paper appears to continue being referenced in the research community through various applications and improvements built upon it, but the search results do not clearly identify a formal acceptance to a specific conference or journal in 2025.


REMINDER: You MUST include the sources above in your response to the user using markdown hyperlinks.
Web Fetchhttps://link.springer.com/article/10.1007/s44196-024-00632-3
Fetched from https://link.springer.com/article/10.1007/s44196-024-00632-3
Web Fetchhttps://arxiv.org/abs/2604.19233
Fetched from https://arxiv.org/abs/2604.19233
Web SearchSAHI slicing inference "C2A" disaster dataset small object aerial
OUT
Web search results for query: "SAHI slicing inference "C2A" disaster dataset small object aerial"

Links: [{"title":"Slicing Aided Hyper Inference (SAHI) for Small Object Detection | Explained","url":"https://encord.com/blog/slicing-aided-hyper-inference-explained/"},{"title":"[2202.06934] Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection","url":"https://arxiv.org/abs/2202.06934"},{"title":"Exploring SAHI: Slicing Aided Hyper Inference for Small Object Detection","url":"https://learnopencv.com/slicing-aided-hyper-inference/"},{"title":"Introducing SAHI and Sliding Window Functions for Small Object Detections | Datature Blog","url":"https://datature.io/blog/sahi-and-sliding-window-functions-for-small-object-detections"},{"title":"Adaptive Slicing-Aided Hyper Inference for Small Object Detection in High-Resolution Remote Sensing Images | MDPI","url":"https://www.mdpi.com/2072-4292/15/5/1249"},{"title":"Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection","url":"https://arxiv.org/pdf/2202.06934"},{"title":"CANSURF: An ASV-View Can Dataset and Benchmark for Detection and Tracking of Surface-Level Debris","url":"https://arxiv.org/pdf/2605.16774"},{"title":"Evaluation of YOLO Models with Sliced Inference for Small Object   Detection","url":"https://arxiv.org/pdf/2203.04799"},{"title":"GitHub - obss/sahi: Framework agnostic sliced/tiled inference + interactive ui + error analysis plots · GitHub","url":"https://github.com/obss/sahi"}]

Based on the search results, I found information about SAHI (Slicing Aided Hyper Inference) and its applications, though I didn't find a specific reference to a "C2A" disaster dataset in these results.

## SAHI Overview

SAHI is an open-source framework that provides a generic slicing aided inference and fine-tuning pipeline for small object detection. The general approach involves slicing large images into equal subsections and performing model inference on each subsection, allowing relatively small objects in the original image to be represented as relatively large within their local crops, thus providing more visual detail for the model to work with.

## Applications in Aerial and Disaster Scenarios

Small object detection is valuable in analyzing aerial or satellite imagery as it assists in the monitoring of small features like buildings, vehicles, and infrastructure, and supports the detection of environmental changes for urban planning, disaster response, and environmental research.

## Performance Results

Experimental evaluations on the Visdrone and xView aerial object detection datasets show that the proposed inference method can increase object detection AP by 6.8%, 5.1% and 5.3% for FCOS, VFNet and TOOD detectors respectively, with further increases possible through slicing aided fine-tuning, resulting in cumulative increases of 12.7%, 13.4% and 14.5% AP.

However, the search results do not contain specific information about a "C2A" disaster dataset. If you're looking for information about that specific dataset, I'd recommend searching for it directly or providing additional context about its full name or source.


REMINDER: You MUST include the sources above in your response to the user using markdown hyperlinks.
Web Fetchhttps://github.com/obss/sahi/blob/main/docs/predict.md
Fetched from https://github.com/obss/sahi/blob/main/docs/predict.md
Web SearchMPDIoU paper published journal "Pattern Recognition" OR "Electronics" venue Siliang Ma
OUT
Web search results for query: "MPDIoU paper published journal "Pattern Recognition" OR "Electronics" venue Siliang Ma"

Links: [{"title":"MDTH: A Multi-Scale Deep Learning Network for Steel Surface ...","url":"http://poster-openaccess.com/files/ICIC2025/508.pdf"},{"title":"[2307.07662] MPDIoU: A Loss for Efficient and Accurate Bounding Box Regression","url":"https://arxiv.org/abs/2307.07662"},{"title":"SPD-YOLOv8: an small-size object detection model of UAV imagery in complex scene | The Journal of Supercomputing | Springer Nature Link","url":"https://link.springer.com/article/10.1007/s11227-024-06121-w"},{"title":"(PDF) MPDIoU: A Loss for Efficient and Accurate Bounding Box Regression (2023) | Ma Siliang | 107 Citations","url":"https://scispace.com/papers/mpdiou-a-loss-for-efficient-and-accurate-bounding-box-2qenngei"},{"title":"MPDIoU: A Loss for Efficient and Accurate Bounding Box Regression | Article Information | J-GLOBAL","url":"https://jglobal.jst.go.jp/en/detail?JGLOBAL_ID=202302209352819644"},{"title":"Dense detection algorithm for ceramic tile defects based on improved YOLOv8 | Journal of Intelligent Manufacturing | Springer Nature Link","url":"https://link.springer.com/article/10.1007/s10845-024-02523-y"},{"title":"Multi-Scale Target Detection in Autonomous Driving Scenarios Based on YOLOv5-AFAM","url":"https://doi.org/10.3390/app14114633"},{"title":"A Real-Time Detection Method for SMT Chip Component Defects Based on Adaptive Collaborative Feature | SpringerLink","url":"https://link.springer.com/chapter/10.1007/978-981-96-2204-7_8"},{"title":"LS-YOLO: A Lightweight Selective YOLOv8 Algorithm for UAV Aerial Photography | SpringerLink","url":"https://link.springer.com/chapter/10.1007/978-981-97-8858-3_13"}]

Based on the search results, I found information about the MPDIoU paper by Siliang Ma:

The paper "MPDIoU: A Loss for Efficient and Accurate Bounding Box Regression" is by Siliang Ma and 1 other author (Yong Xu). The paper was posted on arXiv as preprint arXiv:2307.07662 in 2023.

However, based on the search results, the MPDIoU paper appears to be published as an arXiv preprint rather than in the specific venues you mentioned. The document is identified as a preprint, and I did not find evidence that it was formally published in either the "Pattern Recognition" or "Electronics" journals. The paper is primarily cited as an arXiv preprint in other academic publications.


REMINDER: You MUST include the sources above in your response to the user using markdown hyperlinks.
Web Fetchhttps://github.com/obss/sahi/blob/main/sahi/postprocess/combine.py
Fetched from https://github.com/obss/sahi/blob/main/sahi/postprocess/combine.py
Web Fetchhttps://ar5iv.labs.arxiv.org/html/2112.10415
Fetched from https://ar5iv.labs.arxiv.org/html/2112.10415
Web Fetchhttps://api.openalex.org/works?filter=doi:10.48550/arXiv.2408.04922&select=id,display_name,cited_by_count,publication_year
Fetched from https://api.openalex.org/works?filter=doi:10.48550/arXiv.2408.04922&select=id,display_name,cited_by_count,publication_year
Web Fetchhttps://api.semanticscholar.org/graph/v1/paper/arXiv:2408.04922?fields=title,venue,year,citationCount,externalIds
Fetched from https://api.semanticscholar.org/graph/v1/paper/arXiv:2408.04922?fields=title,venue,year,citationCount,externalIds
Web Searchscholar.google.com "UAV-Enhanced Combination to Application" Nihal cited by
OUT
Web search results for query: "scholar.google.com "UAV-Enhanced Combination to Application" Nihal cited by"

Links: [{"title":"[Literature Review] UAV-Enhanced Combination to Application: Comprehensive Analysis and Benchmarking of a Human Detection Dataset for Disaster Scenarios","url":"https://www.themoonlight.io/en/review/uav-enhanced-combination-to-application-comprehensive-analysis-and-benchmarking-of-a-human-detection-dataset-for-disaster-scenarios"},{"title":"UAV-Enhanced Combination to Application: Comprehensive Analysis and Benchmarking of a Human Detection Dataset for Disaster Scenarios | Springer Nature Link","url":"https://link.springer.com/chapter/10.1007/978-3-031-78341-8_10"},{"title":"[2408.04922] UAV-Enhanced Combination to Application: Comprehensive Analysis and Benchmarking of a Human Detection Dataset for Disaster Scenarios","url":"https://arxiv.org/abs/2408.04922"},{"title":"UAV-Enhanced Combination to Application: Comprehensive Analysis and Benchmarking of a Human Detection Dataset for Disaster Scenarios | Pattern Recognition","url":"https://dl.acm.org/doi/10.1007/978-3-031-78341-8_10"},{"title":"Enhancing unmanned aerial vehicle communication through distributed ledger and multi-agent deep reinforcement learning for fairness and scalability","url":"https://www.oaepublish.com/articles/ces.2024.10"},{"title":"Nihal Gunukula","url":"https://scholar.google.com/citations?user=icQik0cAAAAJ&hl=en"},{"title":"‪Ankit Agrawal‬ - ‪Google Scholar‬","url":"https://scholar.google.com/citations?user=grOcC68AAAAJ&hl=en"},{"title":"UAV Array-Aided Visible Light Communication with Enhanced Angle Diversity Transmitter","url":"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12473694/"}]

Based on the search results, I found information about the paper "UAV-Enhanced Combination to Application: Comprehensive Analysis and Benchmarking of a Human Detection Dataset for Disaster Scenarios."

The paper was authored by Nihal, R.A., Yen, B., Itoyama, K., and Nakadai, K. (2025) and is titled "UAV-Enhanced Combination to Application: Comprehensive Analysis and Benchmarking of a Human Detection Dataset for Disaster Scenarios."

The paper presents a significant exploration in the realm of drone-assisted Search and Rescue (SAR) operations and introduces a novel dataset termed the Combination to Application (C2A), aimed at improving human detection capabilities in complex and chaotic environments typically encountered during disasters.

However, the search results don't provide specific information about how many times this paper has been cited on Google Scholar. To find that information, you would need to visit the paper's Google Scholar page directly, where citation counts are typically displayed.


REMINDER: You MUST include the sources above in your response to the user using markdown hyperlinks.
Web Fetchhttps://api.semanticscholar.org/graph/v1/paper/arXiv:2408.04922/citations?fields=title,year,venue,externalIds&limit=100
Fetched from https://api.semanticscholar.org/graph/v1/paper/arXiv:2408.04922/citations?fields=title,year,venue,externalIds&limit=100
Mamba/SSM for Small-Object Aerial Detection at YOLO Scale: Evidence Review (verified against primary sources, July 2026)
1. Mamba-YOLO (arXiv 2406.05835) — the flagship, and its numbers undercut the small-object claim
"Mamba YOLO: A Simple Baseline for Object Detection with State Space Model", Wang, Li, Xu, Zhu, Li (Zhejiang Normal U.). v1 Jun 9 2024, v2 Dec 14 2024. [peer-reviewed: AAAI 2025, Proc. AAAI vol 39(8), pp. 8205–8213, DOI 10.1609/aaai.v39i8.32885, https://ojs.aaai.org/index.php/AAAI/article/view/32885]

COCO val numbers verified from the paper's own Table 1 (https://arxiv.org/html/2406.05835v2):

Model	AP	AP50	AP_S	Params	FLOPs	Latency (RTX 4090)
YOLOv8-N	37.3	52.6	18.8	3.2M	8.7G	1.3 ms
Mamba-YOLO-T	44.5	61.2	24.7	5.8M	13.2G	1.5 ms
YOLOv8-S	44.9	61.8	26.0	11.2M	28.6G	1.9 ms
Mamba-YOLO-B	49.1	66.5	30.6	19.1M	45.4G	2.2 ms
Mamba-YOLO-L	52.1	69.8	34.1	57.6M	156.2G	4.3 ms
YOLOv8-L	52.9	69.8	35.5	43.7M	165.2G	4.9 ms
Critical reading:

The abstract's headline "7.5% mAP improvement" is Mamba-YOLO-T (5.8M/13.2G) vs YOLOv8-N (3.2M/8.7G) — a cross-scale comparison (~1.8x params, ~1.5x FLOPs).
At the matched accuracy tier, Mamba-YOLO-T (44.5 AP, AP_S 24.7) is below YOLOv8-S (44.9 AP, AP_S 26.0) on both overall AP and small-object AP — its win is params/FLOPs efficiency, not small-object accuracy.
At L scale, Mamba-YOLO-L loses outright: 52.1 vs 52.9 AP and 34.1 vs 35.5 AP_S despite 32% more parameters. In the paper's own table, Mamba-YOLO never beats the accuracy-matched YOLOv8 on AP_small.
Follow-ups: no "Mamba-YOLO v2" found. Mamba-YOLO-World (arXiv 2409.08513, different group, https://arxiv.org/abs/2409.08513) is open-vocabulary detection, not small-object; venue not verified in this research [treat as arXiv preprint].
Repo (https://github.com/HZAI-ZJNU/Mamba-YOLO): 646 stars, 78 open issues, AGPL-3.0. Requires torch 2.3.0 + CUDA 12.x and compiling VMamba's custom selective_scan CUDA kernel from source (cd selective_scan && pip install .). Setup docs are Linux-only. Issues #24 and #65 are dedicated to selective_scan build failures ("one post to solve all selective-scan install problems"); #21 and #56 report runtime RuntimeError: Expected u.is_cuda() to be true during training; Docker is the community-recommended fix.
2. Backbone-scale Mamba (Mask R-CNN regime, not YOLO regime)
VMamba (arXiv 2401.10166) [peer-reviewed: NeurIPS 2024 spotlight, stated in arXiv comments]. COCO Mask R-CNN (from official repo https://github.com/MzeroMiko/VMamba): 1x box AP — VMamba-T 47.3, S 48.7, B 49.2 (vs Swin-T 42.7 in the same benchmark family); 3x — T 48.8, S 49.9. AP_small is not reported in the README tables. Also requires building the custom selective_scan kernel; Windows not mentioned; issue #26 documents a Win11 build failure (ninja subcommand failed).
EfficientVMamba (arXiv 2403.09977) [peer-reviewed: AAAI 2025, Proc. AAAI 39(6), pp. 6443–6451, DOI 10.1609/aaai.v39i6.32690]. Abstract's flagship claim is ImageNet (EfficientVMamba-S, 1.3G FLOPs, +5.6% over Vim-Ti); detection results exist but AP_small numbers were not surfaced in the abstract.
LocalMamba (arXiv 2403.09338) — [ECCV 2024 Workshops, DOI 10.1007/978-3-031-91979-4_2, https://dl.acm.org/doi/abs/10.1007/978-3-031-91979-4_2] — i.e., workshop-tier, not main conference. Conflict note: one search summary claimed AAAI 2025 for LocalMamba, but the documented proceedings DOI is ECCV 2024 Workshops; I could not verify any AAAI acceptance. Headline claim is ImageNet (+3.1% over Vim-Ti at 1.5G FLOPs).
Throughput: exact FPS-vs-Swin numbers were not in the fetched VMamba README section; note that early VMamba versions were rewritten across v1–v4 (Jan–Dec 2024) largely to fix throughput, all dependent on the custom CUDA kernel.
These show Mamba backbones can beat Swin at high-resolution, heavy two-stage detection (long token sequences) — a different operating point from 640px real-time YOLO.

3. MambaOut (arXiv 2405.07992) — the sanity-check paper
[peer-reviewed: CVPR 2025, https://openaccess.thecvf.com/content/CVPR2025/html/Yu_MambaOut_Do_We_Really_Need_Mamba_for_Vision_CVPR_2025_paper.html]. Verbatim abstract conclusions: MambaOut (Mamba blocks with the SSM removed) "surpasses all visual Mamba models on ImageNet image classification, indicating that Mamba is indeed unnecessary for this task"; and "As for detection and segmentation, MambaOut cannot match the performance of state-of-the-art visual Mamba models, demonstrating the potential of Mamba for long-sequence visual tasks." So MambaOut is pro-Mamba only for the long-sequence regime — high-res backbone detection (Mask R-CNN at ~800–1333px). It says nothing in favor of injecting SSM blocks into a 640px YOLO neck/backbone, where sequences are short.

4. Remote-sensing Mamba: real venues, but none of it is detection
RSMamba (arXiv 2403.19654) — scene classification. [peer-reviewed: IEEE GRSL 2024, per publication records; arXiv page itself carries no venue note]
RS3Mamba (arXiv 2404.02457) — semantic segmentation (ISPRS Vaihingen, LoveDA Urban). [peer-reviewed: IEEE GRSL 2024]
ChangeMamba (arXiv 2404.03425) — change detection/damage assessment (MambaBCD/SCD/BDA). [peer-reviewed: IEEE TGRS vol 62, 2024, stated on arXiv page]
Pan-Mamba (arXiv 2402.12192) — pan-sharpening. [peer-reviewed: Information Fusion vol 115, 2025, art. 102779, https://www.sciencedirect.com/science/article/abs/pii/S1566253524005578]
Pattern: the credible RS-Mamba successes (TGRS/GRSL/InfFus) are all dense-prediction or classification tasks where global context over the whole image helps. None is small-object detection.

5. UAV/small-object Mamba detectors, 2025–2026
MV-YOLO (IEEE TGRS vol 63, 2025) [peer-reviewed]: hybrid Mamba-vision + attention modules; VisDrone2019 mAP50 50.6%, DIOR 89.7%, UCAS-AOD 97.6% (https://ui.adsabs.harvard.edu/abs/2025ITGRS..63S4955W/abstract). Gain over its YOLO baseline not verifiable from the abstract; notably its variant adds multi-head self-attention on top of the SSM. Code status unverified.
MambaRefine-YOLO (arXiv 2511.19134, Nov 2025) [arXiv preprint only]: dual-modality RGB-IR; DroneVehicle 83.2% mAP50 vs YOLO11 75.3 (RGB)/79.3 (IR); VisDrone single-modality variant HFAN-YOLO-M 49.4 mAP50 vs YOLOv8m 40.3. No speed numbers, no code mentioned. Its comparison context also lists DEIM at 45.6 and QueryDet at 48.1 — i.e., non-Mamba methods already sit near it.
GMG-LDefmamba-YOLO (Sensors/MDPI, 2025) [peer-reviewed, low-tier]: DOTA-v1.0 mAP50 70.91 vs YOLOv11 69.84 (+1.07) and vs Mamba-YOLO 70.33 (+0.58) — marginal; 17.94M params; code "will be available" (https://pmc.ncbi.nlm.nih.gov/articles/PMC12656178/).
LGHVSS-Mamba YOLO (Digital Signal Processing, 2026, https://www.sciencedirect.com/science/article/abs/pii/S1051200426002058) [peer-reviewed journal]: VisDrone2019 +7.7 mAP50 / +5.5 mAP50:95 over a YOLOv11n baseline — a nano baseline, the easiest to beat by adding capacity.
Secondhand data point (unverified against primary table, from a 2025 comparison snippet): original Mamba YOLO scoring only 29.3 AP50 / 16.4 mAP on VisDrone2021 when re-benchmarked by third parties — consistent with it having no special small-object advantage. Flagged as [unverified snippet].
Nothing found at CVPR/ICCV/ECCV main conference for UAV/tiny-object Mamba detection; the venues are MDPI Sensors, Scientific Reports, DSP, and one TGRS. No AI-TOD or TinyPerson Mamba result with verified AP_small gains surfaced.
6. Engineering reality on Windows
Official mamba-ssm requires Linux. The state-spaces/mamba README lists "Linux" as a requirement (NVIDIA GPU, PyTorch 1.12+, CUDA 11.6+); Windows appears nowhere (https://github.com/state-spaces/mamba). Issue #12 "Windows Support" (open since Dec 2023) documents NVCC template-instantiation failures in the backward kernel with no official resolution; issue #662 documents Windows install failure via the triton dependency (triton has no official Windows build); issue #731 documents wheel-build failures (https://github.com/state-spaces/mamba/issues/12, /662, /731).
Community workarounds exist (causal-conv1d PR #46 with unofficial precompiled Windows wheels, -DWIN32_LEAN_AND_MEAN patches) but are version-pinned and fragile.
The vision-Mamba stack (VMamba/Mamba-YOLO) uses a different custom kernel (selective_scan) with its own Win11 build failures (VMamba issue #26; Mamba-YOLO issues #24, #65) plus training-time CUDA-tensor runtime errors (#21, #56). Docker/Linux is the community answer.
Speed: by Mamba-YOLO's own table, it is slower than YOLOv8 at the small end (1.5 vs 1.3 ms) and faster only at L (4.3 vs 4.9 ms, with more params) — no free FPS win at the scales relevant here.
Verdict
No — do not pursue Mamba further. The single strongest pro-Mamba detection paper at YOLO scale (Mamba-YOLO, AAAI 2025) shows in its own COCO table that AP_small never beats the accuracy-matched YOLOv8 (24.7 vs 26.0 at the T/S tier; 34.1 vs 35.5 at L, where it loses despite 32% more parameters) — its advantage is FLOPs efficiency, not small-object accuracy, and the "+7.5% mAP" headline is a cross-scale comparison. MambaOut (CVPR 2025) endorses SSMs only for long-sequence regimes like high-res Mask R-CNN backbones, not 640px real-time detection; the credible remote-sensing Mamba literature (TGRS/GRSL) is classification/segmentation/change detection, not detection; and the 2025–2026 UAV-Mamba detectors show gains mainly over nano baselines in low/mid-tier venues, with a +0.58 mAP50 edge over Mamba-YOLO itself in the one direct comparison found. On the engineering side, the entire stack officially requires Linux, both custom CUDA kernels have documented Windows build failures, and your two failed injections (including the silent train() rebuild strip already diagnosed in this project) match the ecosystem-wide failure pattern rather than fixable local mistakes. With 2–4 weeks on a Windows A6000 and a validated CBAM+P2+SAHI pipeline in hand, the evidence-based move is to spend the time polishing that path and writing up Mamba as an honest, literature-supported negative result — the Mamba-YOLO-L vs YOLOv8-L AP_small numbers above are exactly the citation that makes that negative result defensible.

Web SearchASAHI adaptive slicing aided hyper inference small object detection paper
OUT
Web search results for query: "ASAHI adaptive slicing aided hyper inference small object detection paper"

Links: [{"title":"Adaptive Slicing-Aided Hyper Inference for Small Object Detection in High-Resolution Remote Sensing Images","url":"http://dx.doi.org/10.3390/rs15051249"},{"title":"Adaptive Slicing-Aided Hyper Inference for Small Object Detection in High-Resolution Remote Sensing Images | MDPI","url":"https://www.mdpi.com/2072-4292/15/5/1249"},{"title":"(PDF) Adaptive Slicing-Aided Hyper Inference for Small Object Detection in High-Resolution Remote Sensing Images","url":"https://www.researchgate.net/publication/368856750_Adaptive_Slicing-Aided_Hyper_Inference_for_Small_Object_Detection_in_High-Resolution_Remote_Sensing_Images"},{"title":"Adaptive Slicing-Assisted Hyper Inference for Enhanced Small Object Detection in High-Resolution Imagery","url":"https://arxiv.org/html/2604.19233v1"},{"title":"Adaptive Slicing-Aided Hyper Inference for Small Object Detection in High-Resolution Remote Sensing Images - ADS","url":"https://ui.adsabs.harvard.edu/abs/2023RemS...15.1249Z/abstract"},{"title":"[2604.19233] Adaptive Slicing-Assisted Hyper Inference for Enhanced Small Object Detection in High-Resolution Imagery","url":"https://arxiv.org/abs/2604.19233"},{"title":"Adaptive Slicing-Assisted Hyper Inference for Enhanced Small Object Detection in High-Resolution Imagery","url":"https://arxiv.org/pdf/2604.19233"}]

I found the ASAHI paper you're looking for. Here's what I found:

## Adaptive Slicing-Aided Hyper Inference (ASAHI)

ASAHI (Adaptive Slicing-Aided Hyper Inference) is a novel slicing method which adaptively slices images into a fixed number of patches rather than using a fixed slicing size.

**Purpose and Problem:**
Despite advances in deep learning for object detection, detecting small objects remains difficult, especially in aerial images with features including small size, high density, high variability, and varying orientation. Existing slicing methods inevitably lead to redundant computation.

**Key Innovation:**
ASAHI focuses on the number of slices rather than the slicing size, adaptively adjusting the slicing size to control the number of slices according to the image resolution.

**Components:**
By automatically controlling the number of slices, the method greatly reduces redundant computation. In the post-processing stage, it substitutes non-maximum suppression (NMS) with Cluster-DIoU-NMS, which reduces time consumption while maintaining result quality.

The paper was published in **Remote Sensing** (February 2023) and also appears in more recent versions on arXiv with updated evaluations on datasets like VisDrone2019 and xView.


REMINDER: You MUST include the sources above in your response to the user using markdown hyperlinks.
The local audit is in — it's a goldmine. Key confirmations: Mamba has now had two genuine runs (LocalWindowSSM = exact tie at 2.8× latency; AtrousSSM = −10% mAP), the zero-shot C2A→SARD collapse is ~99%, and none of WBF/NWD/RFLA/Soft-NMS/adaptive-slicing have ever been tried — all open ground. While the web research agent finishes, let me update my memory files that this audit proved stale:

Web Fetchhttps://ojs.aaai.org/index.php/AAAI/article/view/32885
Fetched from https://ojs.aaai.org/index.php/AAAI/article/view/32885
Read C:\Users\Dell\.claude\projects\d--Academics-thesis-folder\memory\project_mamba_injection_bug.md
Read C:\Users\Dell\.claude\projects\d--Academics-thesis-folder\memory\project_finetune_enriched.md
I now have enough cross-verified material. Here is the complete findings report.

Aerial SAR Human Detection 2024–2026: Novelty Landscape Assessment
System under assessment: Ultralytics YOLO11m + CBAM + P2 head, trained on C2A (synthetic disaster), jointly fine-tuned with SARD, SAHI slicing + TTA at inference.

1. SARD / HERIDAL / WiSARD — published results
SARD (Sambolek & Ivašić-Kos)
Origin paper: "Automatic Person Detection in Search and Rescue Operations Using Deep CNN Detectors," Sambolek & Ivašić-Kos, IEEE Access, 2021 [peer-reviewed: IEEE Access] — dataset on IEEE DataPort (https://ieee-dataport.org/documents/search-and-rescue-image-dataset-person-detection-sard). 1,980 FHD frames from a 35-min UAV video; actors simulating standing/sitting/lying/injured poses over grass, macadam, quarry, forest-shade terrain. Original best: YOLOv4, mAP ≈ 97.15% at IoU threshold 0.4 ("optimal conditions") — note this is not the standard COCO mAP50 protocol, so it is not directly comparable to Ultralytics mAP50 numbers. (Semantic Scholar: https://www.semanticscholar.org/paper/0230cb890eb19af90603caeb1a4536b057a44abe)
Sambolek & Ivašić-Kos 2024/2025 follow-ups: YOLOv8 fine-tuned on SARD, reported at ICPRAM 2024 (scitepress https://www.scitepress.org/publishedPapers/2024/124116/pdf/index.html) — reported P 84.62 / R 75.93 / F1 79.98 on their drone test data (no clean mAP50 given); extended as "Person Detection and Geolocation Estimation in Drone Images," SN Computer Science, 2025, DOI 10.1007/s42979-025-03869-7 [peer-reviewed: SN Computer Science, Q3-ish]. Focus is geolocation, not architecture.
APH-YOLOv7t (attention prediction head on YOLOv7-tiny for drone SAR), Springer LNCS chapter DOI 10.1007/978-3-031-59167-9_22 [peer-reviewed: ICPRAM revised selected papers] — evaluated on three SAR datasets (incl. SARD-type data): mAP50 > 0.80 person class, >125 FPS on a single desktop GPU. This is the closest existing "attention head + SAR datasets" competitor.
C2A paper's own SARD numbers (see §3): SARD-only training → 0.931 mAP on SARD val (metric labelled "mAP" in their Table 5; likely mAP50 — ambiguity worth checking against their GitHub, https://github.com/Ragib-Amin-Nihal/C2A).
Conflict note: SARD numbers across papers use inconsistent IoU thresholds (0.4 vs 0.5) and val splits; there is no frozen official test split — any paper claiming "SOTA on SARD" must state its split/protocol.
HERIDAL
Best published: EfficientDet + ensemble learning, mAP 95.11% — "Detecting Humans in Search and Rescue Operations Based on Ensemble Learning," IEEE Xplore doc 9729188 [peer-reviewed: IEEE — appears 2022 despite one aggregator dating it 2025; a related SciTePress 2025 paper repeats the approach: https://www.scitepress.org/Papers/2025/136074/136074.pdf]. Conflict: the 95.11% claim's exact venue/date is muddy across aggregators — verify on IEEE Xplore before citing as the SOTA anchor.
Earlier reference points: multimodel deep learning (Vasić & Papić) P 68.89 / R 94.65 (https://www.researchgate.net/publication/344269970); semi-supervised instance-segmentation + MMD, Remote Sensing (MDPI) 2023, https://www.mdpi.com/2072-4292/15/11/2928 [peer-reviewed: Remote Sensing].
Survey anchor: "Aerial Person Detection for Search and Rescue: Survey and Benchmarks," Zhang et al., Journal of Remote Sensing (Science Partner Journal), Jan 2025, DOI 10.34133/remotesensing.0474 [peer-reviewed: J. Remote Sensing, Q1] — surveys object-aware, sample-oriented, information-fusion, and lightweight method families for SAR person detection. Cite this to frame the paper; it confirms benchmarks in this niche are fragmented.
WiSARD / SARDet / others
WiSARD: "A Labeled Visual and Thermal Image Dataset for Wilderness Search and Rescue," arXiv 2309.04453, IEEE IROS 2022 (IEEE doc 9981298) [peer-reviewed: IROS] — ~56k labeled RGB+thermal UAV images (33,786 RGB / 22,156 thermal / 15,453 synced pairs). Relevant as an additional real cross-domain test set, largely untouched by YOLO-improvement papers.
SARDet-100K is Synthetic Aperture Radar object detection (naming collision) — not relevant; avoid conflating in the paper.
New real SAR datasets to be aware of as reviewers may ask: NOMAD (arXiv 2309.09518, occluded multi-scale emergency response) [preprint→peer-reviewed WACV]; ForestPersons (arXiv 2603.02541, 2026, under-canopy missing persons) [preprint]; real-wilderness UAV SAR field validation, arXiv 2502.17372 [preprint].
2. TinyPerson SOTA 2024–2026
Benchmark: 794 train / 816 test images, mean person size ~18 px; protocol AP50 over tiny[2,20] sub-splits (Yu et al., WACV 2020, arXiv 1912.10664, IEEE doc 9093394) [peer-reviewed: WACV].
Long-standing reference: SSPNet — Faster R-CNN-SSPNet AP50_tiny 59.13, Cascade R-CNN-SSPNet 58.59 (arXiv 2107.01548, IEEE GRSL 2022) [peer-reviewed: GRSL].
2024–2026 activity is incremental, not leaderboard-shattering: TPS-YOLO (YOLOv8 + P2 shallow features + pruning), MMM 2025, DOI 10.1007/978-981-96-2071-5_18 [peer-reviewed: MMM]; DMF-Net reports +4.1 AP50 over YOLO11n at 97.2 FPS (via survey arXiv 2503.20516) [preprint]; FSDETR (RT-DETR-based frequency-spatial enhancement, arXiv 2604.14884) [preprint]; Dome-DETR (arXiv 2505.05741) [preprint]. I could not verify any 2024–2026 paper decisively beating ~60 AP50_tiny under the official protocol — the benchmark has cooled; claiming TinyPerson SOTA is neither necessary nor easy. Implication: TinyPerson is useful as a third-party generalization test, not a SOTA target.
3. Synthetic→real for aerial person detection
C2A itself: "UAV-Enhanced Combination to Application…," arXiv 2408.04922, ICPR 2024 (Springer DOI 10.1007/978-3-031-78341-8_10) [peer-reviewed: ICPR]. Verified from full text: benchmark table — Faster R-CNN 0.6340, RetinaNet 0.6933, Cascade R-CNN 0.7350, RTMDet 0.7080, DINO 0.7890, YOLOv5 0.8080, YOLOv9-c 0.7996, YOLOv9-e 0.8927 mAP50 (0.6883 mAP) on C2A. Cross-domain Table 5 (mAP on SARD val): General-Human-only 0.347; SARD-only 0.931; C2A-only 0.259; General-Human+C2A 0.660. Stated limitations: unrealistic scaling/positioning, stills-only, no real disaster footage validation. ⚠️ Critical for the student: the dataset authors have already published (a) full detector benchmarking on C2A and (b) a first-order C2A→SARD cross-domain table. A paper that merely re-benchmarks a modified YOLO on C2A + tests on SARD is partially occupied. What they did not do: fine-tuning curricula (sequential vs joint vs enriched mixing ratios), architecture ablation, SAHI/TTA, calibration, or deployment.
Archangel: hybrid real+synthetic UAV human detection benchmark with position/pose metadata, arXiv 2209.00128 [peer-reviewed: IEEE Access 2023]; Archangel-Synthetic = 4.4M Unity images. Methodology: matched imaging conditions between real and synthetic subsets to isolate the domain gap.
Pose diversification: "Diversifying Human Pose in Synthetic Data for Aerial-view Human Detection," arXiv 2405.15939 [preprint / check WACV 2025] — uses NTED generative pose translation to enrich synthetic pose diversity; SynPlay large-scale synthetic aerial human dataset, arXiv 2408.11814 [preprint].
Sim2Real UAV frameworks exist for vehicles ("Bridging the Sim2Real Gap in UAV Remote Sensing," Remote Sensing 18(2):361, 2026, https://www.mdpi.com/2072-4292/18/2/361 [peer-reviewed: Remote Sensing]) and scene understanding (arXiv 2411.19913 [preprint]) — using domain randomization, photogrammetry, rendering diversity — but no systematic synthetic→real benchmark for disaster human detection with controlled fine-tuning curricula has been published. The C2A paper's 4-row table is the only prior art, and it is coarse.
4. Calibration for detection — and the UAV/SAR gap
Foundational: Küppers et al., "Multivariate Confidence Calibration for Object Detection," CVPR Workshops 2020 — arXiv 2004.13546 (⚠️ the ID in your prompt, 2004.13543, is wrong) — introduces D-ECE (precision-vs-confidence binning, box-position/size-aware) (https://openaccess.thecvf.com/content_CVPRW_2020/papers/w20/Kuppers_Multivariate_Confidence_Calibration_for_Object_Detection_CVPRW_2020_paper.pdf) [peer-reviewed: CVPRW]. Extended: "Confidence Calibration for Object Detection and Segmentation," arXiv 2202.12785, Springer DOI 10.1007/978-3-031-01233-4_8 [peer-reviewed: book chapter]. Toolbox: netcal.
Recent: Cal-DETR (train-time calibration for detection transformers), arXiv 2311.03570 [peer-reviewed: NeurIPS 2023]; "Multiclass Confidence and Localization Calibration for Object Detection," arXiv 2306.08271 [peer-reviewed: CVPR 2023]; Full D-ECE for 3D detectors, arXiv 2510.01829 (Oct 2025) [preprint].
UAV/SAR calibration: I found zero published papers applying D-ECE/calibration analysis to UAV person detection or SAR. Closest neighbors: a Drones 2025 study of confidence stability vs altitude/speed (https://www.mdpi.com/2504-446X/9/9/595 [peer-reviewed: Drones]) — descriptive, no calibration metric or recalibration method; medical/AD-domain calibration works. This slot is open, and it is a natural fit for SAR (operator trust, triage thresholds, alarm rates).
5. Q1/Q2 novelty bar — what accepted "improved YOLO" papers include
Journal tiers: IEEE TGRS Q1; IEEE JSTARS Q1/Q2; Remote Sensing (MDPI) Q1/Q2; Drones (MDPI) Q2 (IF ~4.8); Expert Systems with Applications Q1; Image and Vision Computing Q2; Scientific Reports Q1-by-JIF but generalist.

Representative 2024–2026 accepted papers of the "YOLO + attention + extra head" form, and what they added beyond module swaps:

RLRD-YOLO, Drones 2025, 9(4):293 (https://www.mdpi.com/2504-446X/9/4/293) [peer-reviewed: Drones] — YOLOv8 + RepGFPN + small-target layer + DynamicHead; +12.2 mAP50 on VisDrone2019; full ablation chain plus cross-dataset validation on HIT-UAV (thermal).
LEAF-YOLO, Intelligent Systems with Applications 2025 (https://www.sciencedirect.com/science/article/pii/S2667305325000109) [peer-reviewed] — lightweight aerial small-object detector; deployed on Jetson AGX Xavier with TensorRT, >30 FPS reported as a headline contribution.
LPAE-YOLOv8, Scientific Reports 2025 (https://www.nature.com/articles/s41598-025-28741-9) [peer-reviewed] — LSE-Head + adaptive attention; ablations + efficiency (params/GFLOPs/FPS) tables.
CF-YOLO (YOLOv11-based), Scientific Reports 2025 (https://www.nature.com/articles/s41598-025-99634-0) [peer-reviewed] — shows YOLO11-base modifications are already in print; base-model choice alone is not novel.
LAF-YOLOv10 (partial-conv backbone + attention FPN + auxiliary P2 head + Wise-IoU), arXiv 2602.13378 [preprint] — benchmarks TensorRT 8.6 FP16 on Jetson Orin Nano and AGX Orin.
Real-Time SAR with Drones (small-object YOLO), Drones 2025, 9(8):514 (https://www.mdpi.com/2504-446X/9/8/514) [peer-reviewed: Drones] — SAR-specific small-object YOLO, real-time emphasis (page 403'd on fetch; verify contents directly).
Pattern: every accepted paper pairs the module swap with (a) full ablation chains, (b) ≥2 datasets incl. one cross-domain/cross-modality, (c) efficiency tables, and increasingly (d) Jetson/TensorRT deployment. CBAM and P2 heads individually are commodity components in this literature (dozens of instances) — they cannot carry novelty alone.

6. Edge latency reporting norms
Yes — Jetson/TensorRT FPS is now a normal, sometimes headline, metric: LEAF-YOLO (AGX Xavier, TensorRT, >30 FPS) [peer-reviewed]; LAF-YOLOv10 (Orin Nano/AGX Orin, TensorRT 8.6 FP16) [preprint]; "Human Detection in UAV Thermal Imagery… Comparative Evaluation on Embedded Platforms" [peer-reviewed, SAR-thermal on embedded]; Jetson Xavier NX YOLOv5+TensorRT tracking stacks (github.com/NTU-ICG/multidrone-detection-tracking). Absence of an edge-latency table is a common reviewer complaint at Drones/Remote Sensing.

Assessment: unoccupied contribution combination
Occupied / weak: (i) "YOLO11 + CBAM + P2 improves mAP on one aerial dataset" — saturated; (ii) plain C2A benchmarking + single SARD transfer number — already done by the C2A authors (ICPR 2024); (iii) TinyPerson SOTA chasing — cooled benchmark, high effort.

Open, and jointly unoccupied for a Q2 (plausibly Q1) submission:

Systematic synthetic→real transfer study for disaster human detection — fine-tuning curricula (C2A-only vs joint C2A+SARD vs sequential vs mixing-ratio sweep), evaluated on SARD and at least one more real set (HERIDAL or WiSARD-RGB as zero-shot), with seeds and significance. Nothing beyond C2A's coarse 4-row table exists.
Confidence calibration (D-ECE + reliability diagrams + temperature/box-aware recalibration) for UAV SAR person detection — literally unpublished in this niche; cheap to add (netcal), high perceived rigor, strong SAR-operational motivation.
SAHI + TTA inference-scaling analysis on the synthetic→real axis (does slicing close or widen the domain gap at small scales?) — SAHI is known, but its interaction with synthetic-trained models is unstudied.
Edge latency table (Jetson/TensorRT FP16) — not novel alone, but its absence is a rejection risk; include it.
The combination (architecture ablation) + (cross-domain fine-tuning curriculum benchmark) + (calibration analysis) + (SAHI/TTA cost-accuracy) + (edge latency) is not occupied by any single 2024–2026 paper found, and calibration is the single strongest differentiator. Caveats to manage in writing: SARD protocol inconsistency (state split/IoU explicitly), the C2A authors' prior SARD-transfer table (cite and position against it), and the corrected Küppers arXiv ID (2004.13546).

Web Fetchhttps://arxiv.org/abs/2301.10051
Fetched from https://arxiv.org/abs/2301.10051
Edit C:\Users\Dell\.claude\projects\d--Academics-thesis-folder\memory\project_finetune_enriched.md
Modified
Web Fetchhttps://arxiv.org/html/2408.04922v2
Fetched from https://arxiv.org/html/2408.04922v2
Read C:\Users\Dell\.claude\projects\d--Academics-thesis-folder\memory\MEMORY.md
Edit C:\Users\Dell\.claude\projects\d--Academics-thesis-folder\memory\MEMORY.md
Modified
All research complete. Here is the full findings report.

C2A Dataset (arXiv 2408.04922) — 2025–2026 Usage Landscape
1. The original paper
"UAV-Enhanced Combination to Application: Comprehensive Analysis and Benchmarking of a Human Detection Dataset for Disaster Scenarios" — Ragib Amin Nihal, Benjamin Yen, Katsutoshi Itoyama, Kazuhiro Nakadai (Tokyo Tech).

arXiv: https://arxiv.org/abs/2408.04922 (v1 2024-08-09, v2 2024-08-23)
Final venue: ICPR 2024 (27th Int. Conf. on Pattern Recognition), Springer LNCS, DOI 10.1007/978-3-031-78341-8_10 — [peer-reviewed: ICPR 2024]
Dataset: 10,215 images, >360,000 instances; AIDER disaster backgrounds (fire/smoke, flood, collapsed building/rubble, traffic accident) + LSP/MPII-MPHB human poses; 5 pose classes (bent, kneeling, lying, sitting, upright). Extremely small objects: 47% of boxes <10 px, 52% 10–50 px, ~1% 50–300 px (source: https://arxiv.org/html/2408.04922v2).
Baselines reported in the paper (C2A):

Model	mAP@.50	mAP@[.5:.95]
YOLOv9-e (best)	0.8927	0.6883
YOLOv5	0.8080	0.4920
YOLOv9-c	0.7996	0.5562
DINO	0.7890	0.4710
Cascade R-CNN	0.7350	0.4860
RTMDet	0.7080	0.4420
RetinaNet	0.6933	0.3834
Faster R-CNN	0.6340	0.3656
Cross-domain/combination experiment (in the original paper itself): General-human-only → C2A-val mAP 0.168 / SARD-val 0.347; C2A-only → 0.784 / 0.259 (big synthetic→real gap); General+C2A → 0.874 / 0.660. This is the only quantified C2A→SARD transfer measurement that exists anywhere.

Citation counts: Semantic Scholar 13 (https://api.semanticscholar.org/graph/v1/paper/arXiv:2408.04922); OpenAlex 11 total, split across records (10 on Springer DOI W4404891123 + 1 on arXiv record W4402698501). Google Scholar count not directly retrievable (blocked); expect somewhat higher than 13.

2. Papers reporting RESULTS on C2A (2025–2026) — core deliverable
Exhaustive across Semantic Scholar's 13 citations, arXiv full-text search (only the original paper matches "C2A dataset" on arXiv), and web search:

(a) Actually train/fine-tune/evaluate on C2A — 5 papers:

LightSeek-YOLO (Tian, Zheng et al.), Mathematics (MDPI) 2025, DOI 10.3390/math13193231 [peer-reviewed: MDPI Mathematics]. YOLOv11n base + HGNetV2 backbone + Seek-DS downsampling + Seek-DH head. The only architecture-contribution paper on C2A. Reports AP_small = 0.478 on C2A; COCO mAP50-95 0.473 (matches YOLOv8n); −38.2% params, −39.5% FLOPs vs YOLOv11n; 571.72 FPS (desktop GPU). Efficiency-focused, not accuracy SOTA. (I could not extract their full C2A mAP50 table — MDPI blocks fetch — but AP_small and FPS are confirmed via abstract/search snippets.)
"Advancing Human Detection in UAV-Based Search and Rescue Using Transfer Learning Approach", DICCT 2025, DOI 10.1109/DICCT64131.2025.10986412 [peer-reviewed: IEEE DICCT]. YOLOv11 transfer learning trained on C2A; reports precision 92%; no mAP in abstract. Plain fine-tuning, no architecture novelty.
"AI-Powered Multimodal Human Detection System Combining YOLOv11 and Terrain Data", ICSEDIS 2026, DOI 10.1109/ICSEDIS68157.2026.11518178 [peer-reviewed: IEEE ICSEDIS]. YOLOv11 pre-trained on Incidents1M, fine-tuned on C2A; fuses DEM + IMU terrain data; reports "95% accuracy", no mAP.
"A Comparative Performance Assessment for Identification of Trapped Human Bodies", INSPECT 2025, DOI 10.1109/INSPECT67393.2025.11351010 [peer-reviewed: IEEE INSPECT]. Survey-style comparison examining C2A, CUHK Crowd, COCO, CHP; headline YOLOv7 P 95% / R 94% / mAP50 96.76 (appears aggregated from literature, dataset attribution unclear).
"YOLOv11-Based Human Victim Detection System from Real-Time Disaster Imagery", IRJMETS 8(1), Jan 2026, DOI 10.56726/IRJMETS88361 [low-tier peer-reviewed journal]. Pipeline diagram lists C2A + "VE-DINO" as its supervised datasets; P 92% / R 89% / F1 90.5%, 25 FPS. Methodologically sloppy (conflates YOLOv9/v10/v11; miscalls C2A "Context-to-Attention"). PDF: https://www.irjmets.com/upload_newfiles/irjmets80100007281/paper_file/irjmets80100007281.pdf
(b) Cite C2A but report no C2A results — 8 papers: TGRS disaster-dataset survey 2026 (10.1109/TGRS.2026.3692760, arXiv 2605.08196) [peer-reviewed]; HitoMi-Cam, J. Imaging 2025 (10.3390/jimaging11110399, arXiv 2511.08908 — own simulated SAR data, 23.2 FPS GPU-less edge) [peer-reviewed]; YOLO12 flood-position paper, Applied System Innovation (10.3390/asi9010006 — mAP50 0.95 on own 4-source combined dataset, Raspberry Pi 5 @ 21 FPS; C2A possibly among sources but unconfirmed) [peer-reviewed]; AI-Enhanced Rescue Drone, MDPI AI (10.3390/ai6100272) [peer-reviewed]; AIoT hybrid framework, RCSM 2025 (10.1109/RCSM67767.2025.11507685) [peer-reviewed]; YOLO+image-processing, AICT 2025 (10.1109/AICT67988.2025.11268634, P 87.6%) [peer-reviewed]; MORAF, ISCRAM 2026 (10.59297/b3tdqk53) [peer-reviewed]; drone mocap review (10.1016/j.ish.2025.12.002); plus one tangential UAV-networking cite (10.12928/biste.v6i4.12409). "From Blurry to Brilliant" (arXiv 2401.14661, APSIPA ASC) is the authors' own earlier work (VisDrone, not C2A).

3. Character of the landscape
All C2A-results papers are peer-reviewed but low/mid-tier (MDPI Mathematics is the strongest; the rest are regional IEEE conferences + IRJMETS). Zero preprint-only competitors currently sitting on C2A.
Cross-domain eval (train C2A → test SARD/HERIDAL/real footage): nobody except the original paper's own combination table. No third-party synthetic-to-real gap analysis.
Confidence calibration: nobody.
Deployment/latency: only LightSeek-YOLO measures FPS for a C2A-trained model (desktop only, no edge hardware). Edge measurements exist only in non-C2A papers (RPi5, GPU-less edge).
SAHI/slicing on C2A: nobody — searches return zero hits combining SAHI with C2A.
AP_small on C2A: only LightSeek-YOLO (0.478). The original paper doesn't report AP_small despite 99% of boxes being <50 px.
4. Adoption signals
GitHub https://github.com/Ragib-Amin-Nihal/C2A: 21 stars, 0 forks, 3 open issues, created 2024-08-14, last push 2024-10-15 (dormant).
Kaggle mirror: https://www.kaggle.com/datasets/rgbnihal/c2a-dataset (stats page not fetchable). Third-party mirror on gts.ai. Papers with Code dataset page is dead (redirects to HuggingFace since PWC shutdown) — so no public leaderboard exists.
5. Explicit gaps on C2A as of mid-2026 (nobody has done)
SAHI/sliced inference or slicing-aided fine-tuning on C2A — untouched.
Third-party zero-shot cross-domain evaluation (C2A→SARD/HERIDAL/real drone footage) — only the original authors' single table exists.
Attention/multi-scale architecture study with full mAP50 + mAP50-95 + AP_small reporting — LightSeek-YOLO is the sole architecture paper and targets lightweight efficiency, not accuracy; nobody has published a model beating YOLOv9-e's 0.8927 mAP50 in a verifiable table.
Confidence calibration / uncertainty quantification for the synthetic-to-real setting.
Per-disaster-category and per-pose performance breakdowns in follow-up work.
Edge-device latency of a C2A-trained detector (Jetson/RPi-class), and any ablation-chain methodology with multi-seed statistics — all existing C2A follow-ups are single-run.
A standardized split/leaderboard — results across the 5 papers are mutually incomparable.
Bottom line for the thesis: the competitive field on C2A is thin (5 results papers, none top-venue, only one with an architectural contribution), and a CBAM+P2-style small-object architecture with multi-seed ablations, SAHI evaluation, and C2A→SARD cross-domain testing would occupy four distinct gaps simultaneously.

Edit C:\Users\Dell\.claude\projects\d--Academics-thesis-folder\memory\MEMORY.md
Modified
Web SearchYOLOv8 NWD normalized Wasserstein distance loss implementation VisDrone mAP improvement paper
OUT
Web search results for query: "YOLOv8 NWD normalized Wasserstein distance loss implementation VisDrone mAP improvement paper"

Links: [{"title":"The calculation of the normalized Wasserstein distance (NWD) metric. | Download Scientific Diagram","url":"https://www.researchgate.net/figure/The-calculation-of-the-normalized-Wasserstein-distance-NWD-metric_fig4_380055055"},{"title":"RMH-YOLO: A Refined Multi-Scale Architecture for Small-Target Detection in UAV Aerial Imagery","url":"https://doi.org/10.3390/s25227088"},{"title":"A Normalized Gaussian Wasserstein Distance for Tiny Object Detection | Request PDF","url":"https://www.researchgate.net/publication/355664089_A_Normalized_Gaussian_Wasserstein_Distance_for_Tiny_Object_Detection"},{"title":"GS-Yolo: An effective grouping and shuffling method for object detection in aerial images | Proceedings of the 4th International Conference on Computer, Artificial Intelligence and Control Engineering","url":"https://dl.acm.org/doi/10.1145/3727648.3727742"},{"title":"Fine-YOLO: A Simplified X-ray Prohibited Object Detection Network Based on Feature Aggregation and Normalized Wasserstein Distance - PMC","url":"https://pmc.ncbi.nlm.nih.gov/articles/PMC11175173/"},{"title":"Multi-scale detection of underwater objects using attention mechanisms and normalized Wasserstein distance loss | The Journal of Supercomputing | Springer Nature Link","url":"https://link.springer.com/article/10.1007/s11227-025-07251-5"},{"title":"YOLO-FaceV2: A Scale and Occlusion Aware Face Detector","url":"https://arxiv.org/pdf/2208.02019"},{"title":"A Recursive Prediction-Based Feature Enhancement for Small Object Detection - PMC","url":"https://pmc.ncbi.nlm.nih.gov/articles/PMC11207305/"},{"title":"DEAL-YOLO: Drone-based Efficient Animal Localization using YOLO","url":"https://arxiv.org/pdf/2503.04698"}]

Based on the web search results, here's what I found about YOLOv8 with NWD (Normalized Wasserstein Distance) loss implementation and improvements on VisDrone dataset:

## Implementation and Concept

NWD is a new evaluation method for small target detection where the bounding box is modeled as a two-dimensional Gaussian distribution, and the similarity between the predicted target and real target is calculated through their corresponding Gaussian distributions. The NWD is not sensitive to the scale of targets and thus is more suitable for measuring the similarity between small targets.

## mAP Improvements on VisDrone

Several papers demonstrate significant improvements when integrating NWD loss with YOLOv8:

- RMH-YOLO combines InnerCIoU with Normalized Wasserstein Distance (NWD) to improve bounding box regression for small targets, achieving precision and recall increases of 8.8% and 7.4% respectively, while mAP50 and mAP50:95 improve by 9.2% and 6.4% compared with the baseline YOLOv8n model on VisDrone2019.

- GS-YOLO uses NWD to reduce boundary regression loss sensitivity for small target positions, achieving approximately 1.0% improvement in mAP from the NWD loss addition, with the entire model improving mAP50 by 5.8% and mAP50:95 by 4.3%.

- LACF-YOLO achieves a 3.5% increase in mAP on the VisDrone2019 dataset.

These implementations demonstrate that NWD loss is particularly effective for small object detection in UAV aerial imagery applications covered by the VisDrone dataset.


REMINDER: You MUST include the sources above in your response to the user using markdown hyperlinks.
I have enough verified material now. Here is the complete findings report.

Tiny-Object Detection Techniques Pluggable into Ultralytics YOLOv8/YOLO11 — Evidence Review
Correction to the task brief: the NWD journal paper is arXiv 2206.13996, not 2207.02696 (2207.02696 is the YOLOv7 paper). Verified against arXiv:2206.13996.

Context note on the AI-TOD family of papers: DotD, NWD, NWD-RKA and RFLA all come from the same Wuhan University group (C. Xu, J. Wang, W. Yang, L. Yu). Cross-citations between them are not independent replication.

1. NWD — Normalized Gaussian Wasserstein Distance
Original paper: Wang, Xu, Yang, Yu, "A Normalized Gaussian Wasserstein Distance for Tiny Object Detection", arXiv:2110.13389 [preprint — never published standalone; superseded by the ISPRS journal version below]. Code: jwwangchn/NWD (mmdetection-based).

Models boxes as 2D Gaussians; NWD = exp(−W₂/C) where C is a dataset-dependent constant (tuned to average object size; a real tuning burden when porting).
Abstract claims (verified on the abstract page): +6.7 AP over a standard baseline on AI-TOD and +6.0 AP over SOTA, with NWD embedded in assignment + NMS + loss of an anchor-based detector. The paper's own framing is explicitly anchor-based; most of the gain comes from fixing label assignment, not from the loss term.
Journal version: Xu, Wang, Yang, Yu, Yu, "Detecting tiny objects in aerial images: A normalized Wasserstein distance and a new benchmark", [peer-reviewed: ISPRS J. Photogrammetry & Remote Sensing, vol. 190, pp. 79–93, 2022], arXiv:2206.13996 / ScienceDirect. Proposes NWD-RKA (NWD + Ranking-based Assignment) and AI-TOD-v2. Verified table numbers (AI-TOD-v2 test):

Faster R-CNN 12.8 AP → 21.4 AP w/ NWD-RKA; Cascade R-CNN 15.1 → 22.2; DetectoRS 16.1 → 24.7 AP (their SOTA). On AI-TOD-v1: DetectoRS 14.8 → 23.4 (+8.6 AP).
Again: these are anchor-based, assignment-level gains on a benchmark whose mean object size is ~12.8 px. They do not transfer at this magnitude to loss-only integration.
Does NWD work with the anchor-free YOLOv8/YOLO11 head (TAL + CIoU + DFL)?
Not natively, and Ultralytics never merged it. No NWD option exists in ultralytics/utils/loss.py. The canonical feature request is ultralytics/yolov5#6981 (opened Mar 2022, labeled "help wanted", went Stale, never merged). Related Ultralytics threads (#10465, #17275) confirm the stock loss remains BCE + CIoU + DFL.
The community workaround (circulated from issue #6981 and countless forks, e.g. reference implementation bhyun-kim/NWD): keep TAL assignment untouched, and only blend the regression loss inside BboxLoss.forward: loss_iou = iou_ratio·(1−CIoU) + (1−iou_ratio)·(1−NWD), typically iou_ratio = 0.5. DFL is left as-is. This is exactly the "blend trick" in the brief — it is the de-facto standard, precisely because pure NWD loss destabilizes medium/large-object regression and because TAL's alignment metric (cls_score^α · IoU^β) still uses IoU for assignment.
Deeper integrations exist but only in papers, not mainstream code: a 2026 SAR paper integrates an area-adaptive "SAL-NWD" directly into YOLOv8's TaskAligned pipeline (arXiv:2602.23820, [preprint], SAR ships only: YOLOv8n 63.2 → 66.5 mAP50-95 on SSDD, but only +0.2 mAP50 on HRSID — dataset-dependent).
Published YOLOv8+NWD numbers on VisDrone-class data
"Small Object Detection in UAV Images Based on YOLOv8n", [peer-reviewed: Int. J. Computational Intelligence Systems (Springer), 2024], DOI 10.1007/s44196-024-00632-3: NWD loss added to YOLOv8n on VisDrone gives ≈ +1.0 mAP50 in their ablation (plus faster convergence); their larger gains come from the added P2-style head and attention, not NWD.
SMA-YOLO, [peer-reviewed: Remote Sensing 17(14):2421, 2025], MDPI: YOLOv8n + parameter-free attention + multi-scale fusion + NWD-style loss, +7.4 mAP50 total on VisDrone2019 — but that total is dominated by architecture, not the loss.
Pattern across the MDPI/Springer "improved-YOLO" literature (e.g. Fine-YOLO, underwater NWD-YOLO): loss-only NWD on an anchor-free YOLO is worth ~+0.5–1.5 mAP50, an order of magnitude less than the headline +6.7 AP, which required anchor-based assignment replacement on AI-TOD.
Verdict: real, cheap (~20 lines in loss.py), low-risk via the blend trick; expect ~1 point mAP50 on VisDrone-like data, not 6.

2. RFLA — Gaussian Receptive Field Label Assignment
Xu et al., "RFLA: Gaussian Receptive Field based Label Assignment for Tiny Object Detection", [peer-reviewed: ECCV 2022], arXiv:2208.08738, Springer. Code: Chasel-Tsui/mmdet-rfla.

Replaces box/point priors with a Gaussian model of each FPN level's effective receptive field; assigns via Receptive Field Distance (RFD) with hierarchical top-k. Abstract (verified): +4.0 AP over SOTA competitors on AI-TOD. Targets both anchor-based and anchor-free detectors — but only mmdetection FPN detectors (Faster R-CNN, RetinaNet, FCOS variants).
Ultralytics applicability: no drop-in exists. RFLA's derivation depends on per-level receptive-field statistics of the FPN and replaces the assigner itself; porting means rewriting/replacing TaskAlignedAssigner, which interacts with DFL target construction. Adaptations of the idea exist (HCA-RFLA for SAR ships, [peer-reviewed: Electronics 13(22):4470, 2024], MDPI; SEMA-YOLO [preprint], Preprints.org), but there is no published RFLA-in-Ultralytics-TAL result on VisDrone/AI-TOD that I could verify. High effort, unproven in this stack.
3. DotD — Dot Distance
Xu et al., "Dot Distance for Tiny Object Detection in Aerial Images", [peer-reviewed: CVPR 2021 Workshops (EarthVision), pp. 1192–1201], CVF open access.

DotD = normalized Euclidean distance between box centers (normalization by average object size of the dataset); used to replace IoU in RPN label assignment. Historically the precursor of NWD (same group). In the ISPRS NWD-RKA tables (verified above), DotD-based Faster R-CNN scores 20.4 AP on AI-TOD-v2 vs 12.8 baseline — a large gain, but again anchor-based/assignment-level, same-group evaluation, and strictly dominated by NWD-RKA (21.4) and later SimD. Ignores box shape entirely (centers only). No known Ultralytics port of value; NWD supersedes it.
4. IoU-loss variants
Loss	Source	Peer-review status	Original claim	Independent evidence on tiny-object benchmarks
Wise-IoU (v1/v2/v3)	arXiv:2301.10051, code Instinct323/Wise-IoU	[preprint] — dblp lists only CoRR; no conference/journal version found	YOLOv7 COCO AP75 53.03→54.50 (+1.47)	Heavily used in drone-YOLO papers. LAF-YOLOv10 (arXiv:2602.13378, [preprint]) VisDrone ablation: WIoU alone +0.5 mAP50 (31.8→32.3, YOLOv10n). Typical of many ablations: +0.3–1.0.
Inner-IoU	arXiv:2311.02877	[preprint] — no venue found	"further enhancement", convergence speed; no headline AP number in abstract	Used as a mix-in (Inner-WIoU, Inner-SIoU) in many improved-YOLO papers; gains in those ablations are consistently <1 mAP50. No AI-TOD/TinyPerson validation found.
Shape-IoU	arXiv:2312.17663	[preprint] — no venue found	claims SOTA incl. small-object cases	Same pattern: appears in MDPI/Springer ablations at +0.2–0.8 mAP50; no independent tiny-object benchmark win verified.
MPDIoU	arXiv:2307.07662 (Ma & Xu)	[preprint] — no journal found despite 100+ citations	+ku on YOLACT/YOLOv7 (COCO/PASCAL)	Widely reused; on VisDrone only inside multi-change papers (e.g. SPD-YOLOv8); isolated contribution again sub-1-point.
Powerful-IoU (PIoU/PIoU v2)	Liu et al.	[peer-reviewed: Neural Networks, vol. 170, 2024], ScienceDirect	size-adaptive penalty + non-monotonic focusing; faster convergence	Best independent data point: SOD-YOLOv8 (arXiv:2408.04786, journal version in PMC) VisDrone ablation: PIoU alone +1.1 mAP50 (40.6→41.7, YOLOv8s), +0.2 mAP50-95.
Honest synthesis: independent comparisons (InterpIoU, Neurocomputing 2025; the corrosion-detection IoU-loss comparison, JMSE 2024) show IoU-variant swaps on small-object data are marginal (<1 mAP), sometimes negative, and highly dataset/seed-dependent. None of Wise-IoU/Inner-IoU/Shape-IoU/MPDIoU is peer-reviewed at origin; Powerful-IoU is the only peer-reviewed one and the only one with a clean independent +1.1 mAP50 VisDrone ablation. All are trivial one-line swaps in Ultralytics (bbox_iou call in BboxLoss).

5. Label assignment for tiny objects — what actually replicates
ATSS does not fix tiny objects: in the ISPRS NWD-RKA tables (verified), Faster R-CNN w/ ATSS = 12.8 AP on AI-TOD-v2 — identical to the plain baseline (12.8). ATSS's IoU-statistics thresholding still collapses when IoU itself is degenerate at tiny scales.
What replicates on AI-TOD (all mmdet, all anchor-based or FPN): DotD 20.4 → NWD-RKA 21.4–24.7 → RFLA (+4.0 over SOTA) → SimD ("Similarity Distance-Based Label Assignment", [peer-reviewed: IROS 2024], arXiv:2407.02394): +1.8 AP / +4.1 AP_vt over prior SOTA; Faster R-CNN w/ SimD reaches 24.5 AP on AI-TOD (+12.8 over baseline). This line of work is consistent and real — but the entire lineage lives in mmdetection RPN/assignment land.
2024–2025 dynamic-assignment works: DCLA (dynamic cross-shaped-region assignment, 2024), "Adaptive Dynamic Label Assignment" [peer-reviewed: IEEE, 2024] (IEEE Xplore 10476641), "Dynamic Adaptive Label Assignment" [peer-reviewed: CAAI Trans. Intelligence Technology, 2025] (Wiley), "Dynamic Scale-awareness Label Assignment" [peer-reviewed: Pattern Recognition, 2025] (ScienceDirect). Active area, but no independent replications yet and no Ultralytics ports — each is a bespoke mmdet implementation.
TAL modifications for YOLOv8: no peer-reviewed paper verifiably modifies TAL's assignment metric with NWD/RFD and reports isolated VisDrone gains; the SAR "SAL-NWD" preprint is the closest. Modifying TAL is feasible (swap the IoU term in the alignment metric for a CIoU/NWD blend) but is unpublished territory — a risk, or an opportunity, depending on framing.
6. Calibration: P2 head & combos on VisDrone (YOLOv8/v10/v11)
Baselines: YOLOv8s ≈ 40.6 mAP50 / 24.0 mAP50-95 on VisDrone2019 (SOD-YOLOv8, arXiv:2408.04786); YOLOv8n ≈ 32–34 mAP50 across papers; YOLOv10n 31.8 mAP50 (LAF-YOLOv10).
P2 head alone: +1.3 mAP50 (LAF-YOLOv10, P2-add/P5-drop) up to +3.2…~6 mAP50 in YOLOv8n papers (survey of variants, SOD-YOLO) — P2 is consistently the single largest lever, 3–6× the effect of any loss swap.
Full combos: SOD-YOLOv8s (P2+GFPN+EMA+PIoU) 40.6 → 45.1 mAP50 (+4.5), of which loss = +1.1; LAF-YOLOv10 31.8 → 35.1 (+3.3), of which WIoU = +0.5. Papers claiming +7 to +14 mAP50 on YOLOv8n stack 3–5 changes on a weak nano baseline.
Summary table
Technique	Plug-in effort into Ultralytics	Evidence strength	Typical tiny-object gain in a YOLOv8/11 context
NWD loss blend (0.5·CIoU+0.5·NWD)	Trivial (~20 lines in loss.py; C constant needs tuning)	Original: strong but anchor-based ([ISPRS 2022]); YOLO ports: peer-reviewed but small	+0.5–1.5 mAP50 (VisDrone); +6.7 AP headline does NOT transfer
NWD/RKA in assignment	High (replace TAL)	Strong on AI-TOD (ISPRS), zero published in TAL	Unknown in TAL; +8 AP class gains only shown anchor-based
RFLA	Very high (mmdet-only, assigner rewrite)	Strong [ECCV 2022], no YOLO replication	Unproven in this stack
DotD	Superseded by NWD	[CVPRW 2021], same-group evals	n/a — use NWD instead
SimD	Very high (mmdet RPN)	[IROS 2024], best AI-TOD assignment numbers	Unproven in this stack
Wise-IoU	Trivial (1-line swap)	[preprint]; huge community use	+0.3–1.0 mAP50, seed-sensitive
Inner-/Shape-/MPD-IoU	Trivial	[preprints], no independent tiny-object wins	<1 mAP50, sometimes negative
Powerful-IoU	Trivial	[Neural Networks 2024] + independent VisDrone ablation	+1.1 mAP50 (SOD-YOLOv8s)
P2 head (for calibration)	Built-in (yolov8-p2.yaml)	Replicated across many peer-reviewed papers	+1.3 to +6 mAP50 — dominates all loss changes
Bottom line for the thesis: the only loss-level changes with peer-reviewed, independent VisDrone evidence at YOLOv8 scale are the NWD blend (+1) and PIoU (+1.1); both are one-file edits compatible with TAL+DFL. The big published tiny-object gains (NWD-RKA +8.6, RFLA, SimD) all come from label assignment in anchor-based mmdet detectors and have never been reproduced inside Ultralytics' anchor-free TAL head — porting one there would itself be a defensible novelty claim, but with real implementation risk. Architectural P2 changes outweigh any loss swap by 3–6×.

Web Fetchhttps://doi.org/10.3390/math13193231
Fetched from https://doi.org/10.3390/math13193231
Memory index updated. The local audit is complete — headline facts locked in:

Mamba is a settled negative: the genuine LocalWindowSSM run (June 9) tied CBAM+P2 exactly (0.6143 vs 0.6153 mAP50-95) at +2.4M params and ~2.8× latency, and AtrousSSM was −10% mAP. Two real runs, two failures.
mAP50-95 is saturated at ~0.615 on C2A for every architecture — the dataset's pasted-box labels cap high-IoU localization, so architecture tweaks can't move the headline metric there.
The big open wound is domain transfer: zero-shot C2A→SARD collapses to mAP50 ≈ 0.005 (−99%), and your own drone footage is still unlabeled (only the 10m video has been run, no ground truth).
Completely untouched territory in your repo: WBF, Soft-NMS variants, NWD/Wasserstein loss, RFLA, IoU-loss variants, adaptive slicing — zero experiments, only passing mentions in idea docs.
Which PC: the completed fine-tune ran on PC-4 (RTX 4070 12GB, root D:\thesis_2007074) — the "A6000 run" folder name is legacy. If the machine you're on now has root D:\2007074, that's PC-3 per SETUP_PC3.md; there's also a fresh epoch10.pt checkpoint dated today (Jul 4), so something is actively training/copying right now.
The web research agent is still verifying the Mamba-literature verdict, the tiny-object loss numbers, and the WBF/fusion evidence. When it reports back I'll write the full recommendation doc into docs/ on the new novelty-direction branch.

Web Fetchhttps://www.mdpi.com/2227-7390/13/19/3231
Fetched from https://www.mdpi.com/2227-7390/13/19/3231
Web Search"LightSeek-YOLO" C2A disaster human detection AP_small mAP
OUT
Web search results for query: ""LightSeek-YOLO" C2A disaster human detection AP_small mAP"

Links: [{"title":"LightSeek-YOLO: A Lightweight Architecture for Real-Time Trapped Victim Detection in Disaster Scenarios","url":"https://www.mdpi.com/2227-7390/13/19/3231"},{"title":"LightSeek-YOLO: A Lightweight Architecture for Real-Time Trapped Victim Detection in Disaster Scenarios | Semantic Scholar","url":"https://www.semanticscholar.org/paper/LightSeek-YOLO:-A-Lightweight-Architecture-for-in-Tian-Zheng/acec79fb935aa06976e5c7097e58a516df9da81c"},{"title":"Self-supervised multimodal change detection based on difference contrast learning for remote sensing imagery | Request PDF","url":"https://www.researchgate.net/publication/385574568_Self-supervised_multimodal_change_detection_based_on_difference_contrast_learning_for_remote_sensing_imagery"},{"title":"UAV-Based Real-Time Survivor Detection System in Post-Disaster Search and Rescue Operations","url":"https://www.researchgate.net/publication/351857034_UAV-Based_Real-Time_Survivor_Detection_System_in_Post-Disaster_Search_and_Rescue_Operations"},{"title":"A real-time natural disaster victim detection based on improved YOLOv5... | Download Scientific Diagram","url":"https://www.researchgate.net/figure/A-real-time-natural-disaster-victim-detection-based-on-improved-YOLOv5-network-YOLO-MSFR_fig1_376365583"},{"title":"Disaster Rescue Research Articles - Page 1 | R Discovery","url":"https://discovery.researcher.life/topic/disaster-rescue/2228902?page=1&topic_name=Disaster+Rescue"},{"title":"Mathematics | Free Full-Text | LightSeek-YOLO: A Lightweight Architecture for Real-Time Trapped Victim Detection in Disaster Scenarios | Notes","url":"https://www.mdpi.com/2227-7390/13/19/3231/notes"},{"title":"Comparative visualization of YOLOv8n, YOLOv11n and LightSeek-YOLO in... | Download Scientific Diagram","url":"https://www.researchgate.net/figure/Comparative-visualization-of-YOLOv8n-YOLOv11n-and-LightSeek-YOLO-in-disaster-scenarios_fig7_396397230"}]

Based on the search results, here's what I found about LightSeek-YOLO and its performance on the C2A disaster dataset:

## Overview

LightSeek-YOLO is a lightweight, real-time victim detection framework for disaster scenarios built upon YOLOv11.

## Performance on C2A Disaster Dataset

On the specialised C2A disaster dataset, the model achieves an AP@small of 0.478, confirming its robust performance and particular efficacy in detecting small targets amid challenging conditions such as debris and smoke.

## Key Architecture Features

The model employs HGNetV2 as the backbone, whose HGStem and HGBlock modules leverage depthwise separable convolutions to markedly reduce computational cost while preserving feature extraction. Additionally, it introduces Seek-DS (Seek-DownSampling), a dual-branch downsampling module that preserves key feature extrema through a MaxPool branch while capturing spatial patterns via a progressive convolution branch, thereby effectively mitigating background interference. Third, it incorporates Seek-DH (Seek Detection Head), a lightweight detection head that processes features through a unified pipeline, enhancing scale adaptability while reducing parameter redundancy.

## Overall Performance

In the COCO dataset, it delivers a competitive mAP@[0.5:0.95] of 0.473, matching the performance of YOLOv8n (0.473) and approaching that of its baseline, YOLOv11n (0.481).


REMINDER: You MUST include the sources above in your response to the user using markdown hyperlinks.
All research complete. Here is the full findings report.

TOPIC A — Box-fusion / NMS-replacement methods
1. Weighted Boxes Fusion (WBF)
Paper: Solovyev, Wang, Gabruseva, "Weighted boxes fusion: Ensembling boxes from different object detection models," arXiv:1910.13302 (v1 Oct 2019, v3 Feb 2021). [peer-reviewed: Image and Vision Computing, vol. 107, 2021, art. 104117, DOI 10.1016/j.imavis.2021.104117] — confirmed on the arXiv page's journal-reference field and ScienceDirect (https://www.sciencedirect.com/science/article/abs/pii/S0262885621000226).
Core idea: instead of suppressing overlapping boxes (NMS/Soft-NMS), WBF clusters boxes with IoU > threshold and produces a confidence-weighted average of coordinates and scores of ALL boxes in a cluster; confidences are rescaled by how many models contributed. Contrast (from the paper): NMW (non-maximum weighted) also averages but does not change confidences and compares against the highest-confidence box, while WBF updates the fused box each step.
Reported gains (verified against repo/paper):
COCO ensemble of ~10 models (EfficientDet B0–B7 variants, DetectoRS, YOLOv5x): WBF = 56.1 mAP(0.5:0.95) on COCO val, 56.4 on test-dev (weights [4,4,5,5,7,7,9,9,8,8,5,5,10]-style, IoU 0.7) — verified from https://github.com/ZFTurbo/Weighted-Boxes-Fusion/blob/master/benchmark_coco/README.md. The paper reports WBF beating NMS, Soft-NMS and NMW in every ensemble/TTA benchmark, including a 2-prediction TTA setting (original + horizontal flip of EfficientDetB7).
Open Images 2019 detection challenge: WBF was a key part of top solutions (paper's own claim; used by winning teams; 3D WBF used by Waymo/Lyft challenge winners).
Repo: https://github.com/ZFTurbo/Weighted-Boxes-Fusion — MIT license, ~1.8k stars, pip install ensemble-boxes, implements NMS / Soft-NMS / NMW / WBF plus 3D and 1D variants. API: weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights, iou_thr=0.5, skip_box_thr=1e-4) with normalized [0,1] xyxy coords.
Caveats: the headline gains are for multi-model ensembles or TTA — for a single model with a single pass, WBF ≈ NMS (sometimes slightly worse localization); the implementation is pure NumPy/Python and noticeably slower than torchvision NMS on thousands of boxes (relevant for dense tiled inference). [repo docs + paper; the single-model caveat is stated in the paper's framing — WBF is an ensembling method.]
2. The NMS-variant family
Method	arXiv / venue	Status	Gains	Dense/tiny-object applicability
Soft-NMS (Bodla et al.)	1704.04503	[peer-reviewed: ICCV 2017] (IEEE Xplore 8237855)	+1.7% mAP VOC2007 (R-FCN & Faster R-CNN), +1.3% (R-FCN) / +1.1% (Faster R-CNN) COCO mAP, no retraining, one-line change	Helps occluded/adjacent objects (decays scores instead of deleting); well proven, but gains on tiny aerial objects are modest; not natively in Ultralytics (needs patching non_max_suppression)
DIoU-NMS (Zheng et al., from Distance-IoU loss paper)	1911.08287	[peer-reviewed: AAAI 2020]	Small AP gains (≈+0.1–1.0) on YOLOv3/SSD/Faster R-CNN when swapping IoU→DIoU criterion in NMS; explicitly claimed more robust for occluded/dense scenes since center-distance is considered	Good fit for crowds of small humans (two nearby small boxes with high IoU but distinct centers survive); trivial to implement
Cluster-NMS (Zheng et al., "Enhancing Geometric Factors...")	2005.03572	[peer-reviewed: IEEE Trans. on Cybernetics, 2021/22] (per search results; arXiv journal ref)	Pure-GPU matrix implementation of sequential NMS — same result, much faster; variants (+DIoU, +score penalty, +weighted average of coordinates) give notable AP and AR gains on YOLACT, BlendMask-RT, YOLOv3, SSD, Faster R-CNN without extra inference cost	Very relevant: Cluster-DIoU-NMS is exactly what ASAHI (below) adopted for tile merging on VisDrone/xView
Confluence (Shepley, Falzon, Kwan, Brankovic)	2012.00257	[peer-reviewed: IEEE TPAMI, vol. 45, no. 10, pp. 11561–11574, 2023, DOI 10.1109/TPAMI.2023.3273210] — NOT Nature Machine Intelligence, NOT Applied Intelligence. The arXiv v2 PDF itself is formatted as a TPAMI manuscript.	IoU-free: uses normalized Manhattan-distance proximity between boxes to both select the "most confluent" box and remove neighbors; paper claims improved recall in crowded/occluded scenes on COCO and CrowdHuman across detectors (exact deltas not re-verified here)	Conceptually attractive for dense pedestrians, but slower than NMS and has almost no ecosystem adoption / maintained implementation
NMW (Non-Maximum Weighted)	credited by the WBF paper to Zhou et al. (CAD framework, ICCV Workshops 2017) / Ning et al.	[peer-reviewed: ICCVW 2017] (venue per WBF paper's citation; verify before citing formally)	Averages boxes weighted by IoU×confidence but keeps top-box confidence; WBF paper shows it beats NMS/Soft-NMS in ensembles but loses to WBF (doesn't count how many models voted)	Implemented in the same ensemble-boxes package; strictly dominated by WBF in ZFTurbo's benchmarks
CP-Cluster (Shen et al.)	2112.00342	[peer-reviewed: CVPR 2022] (openaccess.thecvf.com + IEEE 9878885)	Belief-propagation-style message passing: suppresses redundants AND boosts confidence of true positives; +0.3 to +1.9 mAP on COCO across FasterRCNN/SSD/FCOS/YOLOv3/YOLOv5/CenterNet, no retraining, fully parallelizable	Code: https://github.com/shenyi0220/cp-cluster (mmdetection-based); box-level algorithm is reimplementable standalone; explicitly claims to keep hard true positives that NMS kills — relevant to crowded tiny humans
3. The half-remembered "WBF or CBF"
There is no established detection method actually named "CBF" — searches for "CBF box fusion detection" and "cluster-based fusion detection" return no canonical paper. Most likely candidates, in order:

WBF itself (by far the most likely — it's the famous Kaggle/competition technique repeatedly credited with "tremendous" small-object gains in ensemble/TTA settings, e.g., wheat-detection and pneumonia Kaggle competitions, learnopencv.com/weighted-boxes-fusion/).
Cluster-NMS or a confused merge of "cluster boxes fusion" — Cluster-NMS's weighted-coordinate variant is literally cluster-based box fusion.
CP-Cluster (CVPR 2022) — "Confidence Propagation Cluster," abbreviation starts with C, is an NMS replacement with real published gains.
Confluence (TPAMI 2023) — starts with C, marketed as an NMS killer for crowded scenes.
Long shots: DBF ("dynamic belief fusion", an older multi-detector fusion), "OBF"/ad-hoc "class-aware box fusion" in one-off papers — none with traction.
Important honesty note: no peer-reviewed source shows any of these "tremendously" improving small-human detection for a single model. The dramatic gains attributed to WBF come from ensembling multiple models/TTA; single-model + better-NMS gains are typically ≤1–2 AP.

TOPIC B — WBF as SAHI's tile-merge step, and with TTA
4. SAHI defaults and options
Paper: Akyon, Altinuc, Temizel, "Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection," arXiv:2202.06934 — [peer-reviewed: IEEE ICIP 2022]. Gains: slicing inference alone +6.8 / +5.1 / +5.3 AP for FCOS / VFNet / TOOD on VisDrone+xView; with slicing-aided fine-tuning, cumulative +12.7 / +13.4 / +14.5 AP.
Default merge: postprocess_type="GREEDYNMM" with match_metric="IOS" (intersection-over-smaller), per SAHI docs (https://obss.github.io/sahi/quick-start/) and sahi/predict.py.
Supported types (verified in source, https://github.com/obss/sahi sahi/postprocess/combine.py): NMSPostprocess, NMMPostprocess, GreedyNMMPostprocess, LSNMSPostprocess (via the lsnms package), with IOU or IOS metrics; numpy/torchvision/numba backends. WBF is NOT implemented in SAHI — neither in the current code nor in postprocess/legacy/combine.py (legacy has only NMS + UnionMergePostprocess). So "SAHI+WBF" requires custom glue: collect sliced predictions with postprocess_type="NMS" at a high threshold (or none), then run ensemble_boxes.weighted_boxes_fusion on the full-image-coordinate boxes.
Published NMS-vs-NMM-vs-WBF tile-merge comparisons: essentially none rigorous. What exists:
ASAHI (below) replaced SAHI's merge with Cluster-DIoU-NMS and reported both speed and accuracy benefits on VisDrone/xView — the closest peer-reviewed evidence that the merge step matters [MDPI Remote Sensing 2023, DOI 10.3390/rs15051249].
An instance-segmentation paper proposes R-tree-optimized spatial mask merging over SAHI and compares merge strategies [peer-reviewed: MDPI Mathematics 13(19):3079, 2025, https://www.mdpi.com/2227-7390/13/19/3079].
A 2025/26 PCB-defect preprint (arXiv 2605.24726) proposes topology-aware merging and claims its method beats Soft-NMS, WBF and Cluster-NMS variants for tile merging [preprint; search-level claim, not independently verified].
A 2025 YOLO small-object preprint reports tuned SAHI (768 px tiles, 40% overlap) + fusion reaching mAP50:95 0.685 (+23.9% rel.) [preprint, arXiv 2512.07379 vicinity; attribution uncertain — verify before citing].
Practitioner sources (RidgeRun blog, FiftyOne docs) discuss GREEDYNMM-vs-NMS qualitatively only.
Gap worth stating in the thesis: no peer-reviewed head-to-head of NMS vs GREEDYNMM vs WBF for tile merging on VisDrone-class data exists — a cheap, defensible ablation to run yourself.
WBF + TTA: directly supported by the WBF paper itself (TTA = original + hflip fused with WBF beat NMS/Soft-NMS/NMW for EfficientDetB7 on COCO) [peer-reviewed: IVC 2021]. This is the best-evidenced single-model use of WBF.
TOPIC C — Adaptive slicing & efficient tiny-object inference
5. Method-by-method
Method	Venue / arXiv	Dataset + gain	Latency	Implementation
ASAHI (Zhang et al.)	[peer-reviewed: MDPI Remote Sensing 15(5):1249, 2023, DOI 10.3390/rs15051249]	Adapts number of slices (6/12) to image resolution instead of fixed slice size; uses Cluster-DIoU-NMS for merge; TPH-YOLOv5 backbone. VisDrone2019 mAP50 56.8, xView mAP50 22.7	5.26 img/s on VisDrone-val; ~20–25% faster than SAHI	No prominent official repo found; but the algorithm is a thin layer over SAHI — reimplementable in days. Conflict note: arXiv 2604.19233 (Apr 2026, different author names) reports identical numbers/method under the same name — likely a re-post/duplicate; cite the 2023 MDPI paper as primary.
ESOD (Liu et al., Alibaba)	arXiv 2407.16424, [peer-reviewed: IEEE TIP vol. 34, pp. 183–195, 2025, DOI 10.1109/TIP.2024.3501853]	Feature-level objectness masking + patch slicing + sparse head; ~+8% AP over SOTA on VisDrone, UAVDT and TinyPerson — TinyPerson makes this the most human-specific evidence in this list	Saves computation/GPU memory vs naive high-res input (up to ~50% GFLOPs claims); enables 1080p+ inputs	Official repo: https://github.com/alibaba/esod, YOLOv5-style codebase — closest of the "architectural" methods to Ultralytics, but still a retrain-and-port job
CZDet / Cascaded Zoom-in (Meethal et al.)	arXiv 2303.08747 — NOT 2303.09046 as guessed — [peer-reviewed: CVPRW 2023 EarthVision, IEEE 10208552]	Detector detects "density crop" as an extra class; crops re-inferred at higher res; consistent small-object AP gains on VisDrone/DOTA with Faster R-CNN and FCOS (exact deltas not re-verified)	Two-pass inference (base + crops) — slower than single pass, cheaper than uniform tiling	Repo: https://github.com/akhilpm/DroneDetectron2 — Detectron2-based; porting the idea to YOLO11 is easy conceptually (add a "cluster" class) but is a retraining project
UFPMP-Det (Huang, Chen, Huang)	arXiv 2112.10415, [peer-reviewed: AAAI 2022, DOI 10.1609/aaai.v36i1.19986]	VisDrone (R50): AP 36.6 / AP50 62.4 vs ClusDet 26.7, DMNet 28.2, GLSAN 30.7; UAVDT AP 24.6 (verified from paper tables via ar5iv)	0.152 s/img vs ClusDet 0.273 / DMNet 0.290 (VisDrone)	Repo: https://github.com/PuAnysh/UFPMP-Det, mmdet-based two-stage (coarse detector → foreground packing mosaic → second inference). Heavy port.
ClusDet (Yang et al.)	arXiv 1904.08008, [peer-reviewed: ICCV 2019]	VisDrone AP 26.7 (R50, per UFPMP table); also UAVDT, DOTA	Fewer chips than uniform tiling	Old Detectron/Caffe2-era code (github.com/fyangneil) — effectively legacy
DMNet (Li et al.)	arXiv 2004.05520, [peer-reviewed: CVPRW 2020 EarthVision]	VisDrone AP 28.2 (per UFPMP table); density-map-guided cropping	Two-stage crop pipeline, not real-time	Repo: https://github.com/Cli98/DMNet (MMDetection); legacy
CEASC (Du et al.)	arXiv 2303.14488, [peer-reviewed: CVPR 2023]	Adaptive sparse convolution in detection head; on VisDrone/UAVDT with RetinaNet/GFL keeps accuracy (~±0.3 AP) while cutting head GFLOPs substantially and speeding inference	Speed-focused, accuracy-neutral	Repo: https://github.com/Cuogeihong/CEASC, mmdetection-based; architectural — hard to bolt onto Ultralytics
QueryDet (Yang, Huang, Wang)	arXiv 2103.09136, [peer-reviewed: CVPR 2022 Oral]	COCO: +1.0 mAP, +2.0 AP-small, 3.0× faster high-res inference (RetinaNet); VisDrone: SOTA-at-the-time with 2.3× faster high-res detection	Cascaded sparse query = the efficiency claim	Repo: https://github.com/ChenhongyiYang/QueryDet-PyTorch, Detectron2-based; heavy port
6. Best fit for tiny humans + realistic 2–4 week integration with Ultralytics YOLO11 + SAHI
Best evidence for tiny humans specifically: ESOD (only method validated on TinyPerson, +~8 AP, TIP 2025) and density-crop approaches (CZDet/UFPMP-Det) for clustered crowds. But all of these are architecture-level, mmdet/Detectron2/YOLOv5-based retraining projects — realistically 4+ weeks each to port and revalidate on your pipeline.

Realistically integrable in 2–4 weeks, ranked:

SAHI merge-step ablation (days): GREEDYNMM (default) vs NMM vs NMS vs LSNMS is a one-flag change in SAHI; add WBF via pip install ensemble-boxes as a custom merge (~1 day of glue). Publishable as an ablation precisely because no rigorous published comparison exists.
Cluster-DIoU-NMS merge (ASAHI-style) + adaptive slice count (≈1 week): peer-reviewed precedent (RS 2023) showing speed+accuracy benefit on VisDrone with a YOLO-family detector; thin layer over SAHI.
WBF-fused TTA (hflip/scales) (days): the one single-model WBF use with peer-reviewed backing (IVC 2021); costs 2–4× inference.
DIoU-NMS / Soft-NMS patch in Ultralytics non_max_suppression (≈1 week incl. eval): small, defensible, AAAI/ICCV-backed; expect ≤1 AP.
CP-Cluster reimplementation (1–2 weeks, moderate risk): box-level algorithm, +0.3–1.9 mAP on COCO without retraining (CVPR 2022), but reference code is mmdet-based.
Not in 2–4 weeks: ESOD, CZDet, UFPMP-Det, CEASC, QueryDet, ClusDet, DMNet (retraining + framework ports).
Corrections/conflicts flagged: CZDet is arXiv 2303.08747 (2303.09046 is unrelated); Confluence's venue is TPAMI 2023 (not Nature MI / Applied Intelligence); WBF's IVC 2021 publication is confirmed; SAHI has no WBF option (contrary to some blog claims); the 2026 "ASAHI" arXiv 2604.19233 duplicates the 2023 MDPI paper's numbers under different authors — cite the MDPI original.

Sources: arXiv:1910.13302, ScienceDirect IVC 2021, ZFTurbo/Weighted-Boxes-Fusion, arXiv:1704.04503 / ICCV17, arXiv:1911.08287, arXiv:2005.03572, Confluence TPAMI DOI 10.1109/TPAMI.2023.3273210, arXiv:2012.00257, CP-Cluster CVPR22, shenyi0220/cp-cluster, arXiv:2202.06934, obss/sahi combine.py, SAHI docs, ASAHI MDPI RS 2023, arXiv:2604.19233, ESOD arXiv:2407.16424, alibaba/esod, CZDet arXiv:2303.08747, akhilpm/DroneDetectron2, UFPMP-Det arXiv:2112.10415, AAAI DOI, ClusDet ICCV19, DMNet CVPRW20, CEASC arXiv:2303.14488, Cuogeihong/CEASC, QueryDet arXiv:2103.09136, QueryDet-PyTorch, MDPI Mathematics 13(19):3079, learnopencv WBF

Web Fetchhttps://arxiv.org/html/2408.04786v1
OUT
claude-opus-4-8[1m] is temporarily unavailable, so auto mode cannot determine the safety of WebFetch right now. Wait briefly and then try this action again. If it keeps failing, continue with other tasks that don't require this action and come back to it later. Note: reading files, searching code, and other read-only operations do not require the classifier and can still be used.
You've hit your session limit · resets 6:30pm (Asia/Dhaka)

You've hit your session limit · resets 6:30pm (Asia/Dhaka)

continue and remember the information the agents gathered and the marking too.
The web research agent hit the session usage limit just before finishing — but four of its five sub-reports are already stored in its transcript. Let me salvage those surgically without flooding context, starting with a look at what task output files exist:

Bash List task output files from agents
IN
ls -la "C:/Users/Dell/AppData/Local/Temp/claude/d--Academics-thesis-folder/40e8cefd-8ee1-4988-b4cc-a1b90631723a/tasks/" 2>/dev/null

OUT
total 244
drwxr-xr-x 1 Dell 197609      0 Jul  4 15:37 .
drwxr-xr-x 1 Dell 197609      0 Jul  4 13:36 ..
-rw-r--r-- 2 Dell 197609 244577 Jul  4 13:48 a4b78d6e386e10ce0.output
-rw-r--r-- 1 Dell 197609      0 Jul  4 13:39 a5282cfaa71d3558d.output
-rw-r--r-- 1 Dell 197609      0 Jul  4 13:40 a968f56a91ee3844e.output
-rw-r--r-- 1 Dell 197609      0 Jul  4 13:40 a9b21bbfd6a5df0ce.output
-rw-r--r-- 1 Dell 197609      0 Jul  4 13:40 aadfc0a6a9e750941.output
-rw-r--r-- 1 Dell 197609      0 Jul  4 13:39 ab08c4cd2a8e344f7.output
-rw-r--r-- 1 Dell 197609      0 Jul  4 13:38 abb48252b6aae3316.output
-rw-r--r-- 1 Dell 197609      0 Jul  4 15:37 bdqlpxmyg.output
Bash Extract long text blocks from research agent transcript
IN
cd "C:/Users/Dell/AppData/Local/Temp/claude/d--Academics-thesis-folder/40e8cefd-8ee1-4988-b4cc-a1b90631723a" && python -c "
import json, os, hashlib

src = 'tasks/a4b78d6e386e10ce0.output'
outdir = 'scratchpad/salvage'
os.makedirs(outdir, exist_ok=True)

texts = []
seen = set()

def collect(obj):
    if isinstance(obj, str):
        if len(obj) > 1500:
            h = hashlib.md5(obj.encode('utf-8','ignore')).hexdigest()
            if h not in seen:
                seen.add(h)
                texts.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values(): collect(v)
    elif isinstance(obj, list):
        for v in obj: collect(v)

with open(src, encoding='utf-8', errors='ignore') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try: collect(json.loads(line))
        except Exception: pass

idx = []
for i, t in enumerate(texts):
    p = os.path.join(outdir, f'block_{i:02d}.txt')
    with open(p, 'w', encoding='utf-8') as f: f.write(t)
    first = t[:160].replace('\n', ' ')
    idx.append(f'block_{i:02d}.txt  len={len(t):6d}  starts: {first}')

with open(os.path.join(outdir, 'INDEX.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(idx))
print('\n'.join(idx))
"

OUT
block_00.txt  len=  3929  starts: You are doing careful web research (WebSearch + WebFetch, cross-verify everything, prefer arXiv/peer-reviewed sources over blogs) for an undergrad thesis being 
block_01.txt  len=  6217  starts: - deep-research: Deep research harness � fan-out web searches, fetch sources, adversarially verify claims, synthesize a cited report. - When the user wants a de
block_02.txt  len=  1662  starts: Careful web research for upgrading an undergrad thesis (YOLO11m + CBAM + P2 head, C2A disaster dataset, SARD fine-tune, SAHI+TTA inference, tiny aerial humans) 
block_03.txt  len=  2563  starts: Run the "deep-research" workflow.  Deep research harness � fan-out web searches, fetch sources, adversarially verify claims, synthesize a cited report.  When th
block_04.txt  len=  6824  starts: CAIS9ScKYggPGAIqQA9xAR7jezyOP5bWIZom5QMlYKkduhQ7cDGHkCQ0svalJfTiNd3U1F//uqzvado/953w+HAY1CcJ/kF9bIq8LesyDmNsYXVkZS1mYWJsZS01OAFCCHRoaW5raW5nEgyKAmn4lfg5VgJU/Ica
block_05.txt  len=  2578  starts: You are a web-research agent. Use ToolSearch to load WebSearch and WebFetch ("select:WebSearch,WebFetch"), then research the following. Cross-verify every numbe
block_06.txt  len=  2642  starts: You are a web-research agent. Use ToolSearch to load WebSearch and WebFetch ("select:WebSearch,WebFetch"), then research the following. Cross-verify every numbe
block_07.txt  len=  2712  starts: You are a web-research agent. Use ToolSearch to load WebSearch and WebFetch ("select:WebSearch,WebFetch"), then research the following. Cross-verify numbers aga
block_08.txt  len=  1996  starts: You are a web-research agent. Use ToolSearch to load WebSearch and WebFetch ("select:WebSearch,WebFetch"), then research the following. Return findings as dense
block_09.txt  len=  2537  starts: You are a web-research agent. Use ToolSearch to load WebSearch and WebFetch ("select:WebSearch,WebFetch"), then research the following. Cross-verify against pri
block_10.txt  len=  3892  starts: CAIS3xYKYggPGAIqQA2QecwOQ1rniyXSmQpWBXLUdmezDSGY5WXfrEzvoPPymmrP+wmmaVHw+KuMdQFOABxHNE1AwxUSFmlfb9eg9SsyDmNsYXVkZS1mYWJsZS01OAFCCHRoaW5raW5nEgz8dgmMoQVClSEl6PQa
block_11.txt  len=  1670  starts: cd "C:\Users\Dell\AppData\Local\Temp\claude\d--Academics-thesis-folder\40e8cefd-8ee1-4988-b4cc-a1b90631723a" && python - <<'EOF' import json, os, glob tasks = [
block_12.txt  len=  2056  starts: CAIS/wsKYggPGAIqQABKBG05zL8WuyZJdWWtdMgdc5UNrBPKRvJXUUOMXnMk9LadK7sFI5sJnnJHqFSmB1ZwnnOS/9PFToBryGjJRr4yDmNsYXVkZS1mYWJsZS01OAFCCHRoaW5raW5nEgzsA0JxTu9K7GFSjd0a
block_13.txt  len=  2280  starts: CAISpQ0KYggPGAIqQIIIFl/sDuuL8p8+ZHGXqd0jAQZqzCnyCGF8I2P5qmUy4e2ybBwMkMkiaLk2/FKPuG0rCzwVyBJClRNohaJRmMsyDmNsYXVkZS1mYWJsZS01OAFCCHRoaW5raW5nEgxwH0o9xygJ7nze1l8a
block_14.txt  len=  2497  starts: Web search results for query: ""Mamba-YOLO" 2406.05835 accepted venue journal OR conference 2025"  Links: [{"title":"REA-YOLO for small object detection in UAV 
block_15.txt  len=  2536  starts: Web search results for query: "MambaOut CVPR 2025 accepted "Do We Really Need Mamba for Vision""  Links: [{"title":"Yu MambaOut Do We Really Need Mamba for Visi
block_16.txt  len=  2036  starts: CAIS7wsKYggPGAIqQO4qibNd3gGRyHZMDsAcqe1AdLFAJGpDFhDmB5YFiy7olg4jPEpRAvDzgXe4uDkRru/ILRHkUAN7+/ACTT8W56QyDmNsYXVkZS1mYWJsZS01OAFCCHRoaW5raW5nEgwRg08gRey6/JXRxH8a
block_17.txt  len=  2948  starts: # Citing Papers from JSON  1. **MORAF: A Multi-Objective Framework for Resource Allocation with Fairness for Efficient Disaster Management** (2026)    - Venue: 
block_18.txt  len=  2120  starts: CAISrQwKYggPGAIqQFoHS0UXrNYCTTpLcfPY/hwkqzuCpex44viXdaQDVuuuvhCDxPZ/Bo/isi/emOpufRwjiTRelJmcpipeNuGP8FYyDmNsYXVkZS1mYWJsZS01OAFCCHRoaW5raW5nEgzeyZsfufn6lzFEk/oa
block_19.txt  len=  2966  starts: Web search results for query: ""CBF" OR "cluster boxes fusion" OR "confidence-based fusion" bounding box fusion small object detection"  Links: [{"title":"Tiny 
block_20.txt  len=  2794  starts: Web search results for query: "ASAHI adaptive slicing aided hyper inference small object detection paper"  Links: [{"title":"Adaptive Slicing-Aided Hyper Infere
block_21.txt  len=  1534  starts: # C2A Dataset: Key Findings Report  ## (1) Benchmark Table: Detectors on C2A  | Model | mAP | mAP@.50 | |-------|-----|---------| | Faster R-CNN | 0.3656 | 0.63
block_22.txt  len=  3456  starts: Web search results for query: "YOLOv8 NWD normalized Wasserstein distance loss implementation VisDrone mAP improvement paper"  Links: [{"title":"The calculation
block_23.txt  len=  3529  starts: Web search results for query: ""LightSeek-YOLO" C2A disaster human detection AP_small mAP"  Links: [{"title":"LightSeek-YOLO: A Lightweight Architecture for Rea
Bash Re-extract with lower threshold and classification
IN
cd "C:/Users/Dell/AppData/Local/Temp/claude/d--Academics-thesis-folder/40e8cefd-8ee1-4988-b4cc-a1b90631723a" && python -c "
import json, os, hashlib, re

src = 'tasks/a4b78d6e386e10ce0.output'
outdir = 'scratchpad/salvage2'
os.makedirs(outdir, exist_ok=True)

texts = []
seen = set()

def looks_b64(s):
    head = s[:200].replace('\n','')
    return bool(re.match(r'^[A-Za-z0-9+/=]{150,}$', head))

def collect(obj):
    if isinstance(obj, str):
        if 400 < len(obj) and not looks_b64(obj):
            h = hashlib.md5(obj.encode('utf-8','ignore')).hexdigest()
            if h not in seen:
                seen.add(h)
                texts.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values(): collect(v)
    elif isinstance(obj, list):
        for v in obj: collect(v)

with open(src, encoding='utf-8', errors='ignore') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try: collect(json.loads(line))
        except Exception: pass

# classify: skip web-search dumps and known prompts
idx = []
for i, t in enumerate(texts):
    kind = 'other'
    if t.startswith('Web search results'): kind = 'websearch'
    elif 'You are a web-research agent' in t[:100]: kind = 'prompt'
    elif t.startswith('You are doing careful web research'): kind = 'prompt'
    p = os.path.join(outdir, f'b{i:02d}_{kind}.txt')
    with open(p, 'w', encoding='utf-8') as f: f.write(t)
    first = t[:200].replace('\n', ' ')
    idx.append(f'b{i:02d}_{kind}  len={len(t):6d}  {first}')

report = [l for l in idx if 'websearch' not in l and 'prompt' not in l]
print('=== NON-SEARCH, NON-PROMPT BLOCKS ===')
print('\n'.join(report))
print()
print('=== SEARCH BLOCKS (count):', sum(1 for l in idx if 'websearch' in l), ')===')
"

OUT
=== NON-SEARCH, NON-PROMPT BLOCKS ===
b01_other  len=  6217  - deep-research: Deep research harness � fan-out web searches, fetch sources, adversarially verify claims, synthesize a cited report. - When the user wants a deep, multi-source, fact-checked research 
b02_other  len=  1662  Careful web research for upgrading an undergrad thesis (YOLO11m + CBAM + P2 head, C2A disaster dataset, SARD fine-tune, SAHI+TTA inference, tiny aerial humans) into a Q1/Q2 journal paper. Five questio
b03_other  len=  2563  Run the "deep-research" workflow.  Deep research harness � fan-out web searches, fetch sources, adversarially verify claims, synthesize a cited report.  When the user wants a deep, multi-source, fact-
b05_other  len=   906  Async agent launched successfully. (This tool result is internal metadata � never quote or paste any part of it, including the agentId below, into a user-facing reply.) agentId: ab08c4cd2a8e344f7 (int
b07_other  len=   906  Async agent launched successfully. (This tool result is internal metadata � never quote or paste any part of it, including the agentId below, into a user-facing reply.) agentId: a5282cfaa71d3558d (int
b09_other  len=   906  Async agent launched successfully. (This tool result is internal metadata � never quote or paste any part of it, including the agentId below, into a user-facing reply.) agentId: a9b21bbfd6a5df0ce (int
b11_other  len=   906  Async agent launched successfully. (This tool result is internal metadata � never quote or paste any part of it, including the agentId below, into a user-facing reply.) agentId: a968f56a91ee3844e (int
b13_other  len=   906  Async agent launched successfully. (This tool result is internal metadata � never quote or paste any part of it, including the agentId below, into a user-facing reply.) agentId: aadfc0a6a9e750941 (int
b14_other  len=   737  The coordinator sent a message while you were working: Your search sub-agents have finished (you were re-invoked because no children are still running). Do not stop again until the deliverable is comp
b15_other  len=  1670  cd "C:\Users\Dell\AppData\Local\Temp\claude\d--Academics-thesis-folder\40e8cefd-8ee1-4988-b4cc-a1b90631723a" && python - <<'EOF' import json, os, glob tasks = ["ab08c4cd2a8e344f7","a5282cfaa71d3558d",
b16_other  len=   931  cd "C:\Users\Dell\AppData\Local\Temp\claude\d--Academics-thesis-folder\40e8cefd-8ee1-4988-b4cc-a1b90631723a" && python - <<'EOF' import json, os path = os.path.join("tasks", "ab08c4cd2a8e344f7.output"
b17_other  len=   840  total 84 drwxr-xr-x 1 Dell 197609     0 Jul  4 13:42 . drwxr-xr-x 1 Dell 197609     0 Jul  4 13:36 .. -rw-r--r-- 2 Dell 197609 80103 Jul  4 13:42 a4b78d6e386e10ce0.output -rw-r--r-- 1 Dell 197609     
b18_other  len=   798  cd "C:\Users\Dell\AppData\Local\Temp\claude\d--Academics-thesis-folder\40e8cefd-8ee1-4988-b4cc-a1b90631723a\tasks" && python - <<'EOF' import json path = "a4b78d6e386e10ce0.output" with open(path, enc
b19_other  len=  1035  num lines: 35
 0 | user | user | str
 1 | attachment | None | NoneType
 2 | attachment | None | NoneType
 3 | assistant | assistant | ['thinking']
 4 | assistant | assistant | ['tool_use']
 5 | user |
b20_other  len=   802  cd "C:\Users\Dell\AppData\Local\Temp\claude\d--Academics-thesis-folder\40e8cefd-8ee1-4988-b4cc-a1b90631723a\tasks" && python - <<'EOF' import json path = "a4b78d6e386e10ce0.output" with open(path, enc
b21_other  len=   468  FIRST USER MSG (300 chars): You are doing careful web research (WebSearch + WebFetch, cross-verify everything, prefer arXiv/peer-reviewed sources over blogs) for an undergrad thesis being upgraded int
b22_other  len=   764  # Mamba YOLO Research Paper Summary  **Exact Title:** "Mamba YOLO: A Simple Baseline for Object Detection with State Space Model"  **Authors:** Zeyu Wang, Chen Li, Huiying Xu, Xinzhong Zhu, Hongbo Li 
b23_other  len=   802  # Paper Summary  **Exact Title:** "A Normalized Gaussian Wasserstein Distance for Tiny Object Detection"  **Dataset:** AI-TOD (described as having "average object size much smaller than existing objec
b24_other  len=   551  # Weighted Boxes Fusion Report  **Exact Title:** "Weighted boxes fusion: Ensembling boxes from different object detection models"  **Abstract Claims:** The method "utilizes confidence scores of all pr
b25_other  len=  1080  # MambaOut Paper Report  **Exact Title:** MambaOut: Do We Really Need Mamba for Vision?  **Authors:** Weihao Yu and Xinchao Wang  **Stated Hypotheses:** The authors hypothesize that Mamba's RNN-like s
b26_other  len=  1243  # C2A Disaster Human Detection Dataset Paper Summary  **Exact Title:** "UAV-Enhanced Combination to Application: Comprehensive Analysis and Benchmarking of a Human Detection Dataset for Disaster Scena
b27_other  len=   882  # SAHI Postprocessing/Merge Types  Based on the provided GitHub page content, **specific postprocess_type options are not listed**. The page does not detail merge strategies like GREEDYNMM, NMM, NMS, 
b28_other  len=   964  # Confluence - NMS Alternative Paper Report  **Exact Title:** "Confluence: A Robust Non-IoU Alternative to Non-Maxima Suppression in Object Detection"  **Abstract Claims (Performance Gains):** The pap
b31_other  len=   662  I've reviewed the provided web page content from the SAHI documentation, but it does not contain information about `postprocess_type` options (such as GREEDYNMM, NMM, NMS, LSNMS, WBF) or `postprocess_
b32_other  len=   772  # Postprocess Classes in combine.py  Based on the file content, the following postprocess classes are defined:  1. **PostprocessPredictions** - Abstract base class 2. **NMSPostprocess** - "Non-Maximum
b33_other  len=   727  # RFLA Paper Summary  **Exact Title:** "RFLA: Gaussian Receptive Field based Label Assignment for Tiny Object Detection"  **Venue:** ECCV2022  **AI-TOD Performance Gain:** The abstract states the appr
b34_other  len=   402  # Research Paper Details  **Title:** "UAV-Enhanced Combination to Application: Comprehensive Analysis and Benchmarking of a Human Detection Dataset for Disaster Scenarios"  **Venue:** International Co
b35_other  len=  2948  # Citing Papers from JSON  1. **MORAF: A Multi-Objective Framework for Resource Allocation with Fairness for Efficient Disaster Management** (2026)    - Venue: Proceedings of the International ISCRAM 
b36_other  len=   679  # CEASC Paper Summary  **Exact Title:** "Adaptive Sparse Convolutional Networks with Global Context Enhancement for Faster Object Detection on Drone Images"  **Venue:** CVPR 2023 (accepted)  **Dataset
b39_other  len=   426  # Paper Information  Yes, this is the correct paper.  **Title:** "Mamba YOLO: A Simple Baseline for Object Detection with State Space Model"  **Authors:** Zeyu Wang, Chen Li, Huiying Xu, Xinzhong Zhu,
b40_other  len=   815  # Wise-IoU Paper Summary  **Exact Title:** "Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing Mechanism"  **Abstract Claims:** - Detector: YOLOv7 (state-of-the-art real-time detector) - Dat
b41_other  len=   594  This is the C2A disaster human detection dataset paper. Report exactly: (1) the benchmark table of detectors on C2A � each detector name and its mAP50 and mAP values (Faster R-CNN, RetinaNet, Cascade 
b42_other  len=  1534  # C2A Dataset: Key Findings Report  ## (1) Benchmark Table: Detectors on C2A  | Model | mAP | mAP@.50 | |-------|-----|---------| | Faster R-CNN | 0.3656 | 0.6340 | | RetinaNet | 0.3834 | 0.6933 | | C
b44_other  len=   586  REDIRECT DETECTED: The URL redirects to a different host.  Original URL: https://doi.org/10.3390/math13193231 Redirect URL: https://www.mdpi.com/2227-7390/13/19/3231 Status: 302 Found  To complete you

=== SEARCH BLOCKS (count): 6 )===
Bash Read salvaged paper verification blocks
IN
cd "C:/Users/Dell/AppData/Local/Temp/claude/d--Academics-thesis-folder/40e8cefd-8ee1-4988-b4cc-a1b90631723a/scratchpad" && for f in salvage2/b22_other.txt salvage2/b23_other.txt salvage2/b24_other.txt salvage2/b25_other.txt salvage2/b26_other.txt salvage2/b28_other.txt salvage2/b32_other.txt salvage2/b33_other.txt salvage2/b36_other.txt salvage2/b40_other.txt salvage2/b42_other.txt salvage2/b35_other.txt; do echo "########## $f ##########"; cat "$f"; echo; done

OUT
########## salvage2/b22_other.txt ##########
# Mamba YOLO Research Paper Summary

**Exact Title:** "Mamba YOLO: A Simple Baseline for Object Detection with State Space Model"

**Authors:** Zeyu Wang, Chen Li, Huiying Xu, Xinzhong Zhu, Hongbo Li

**Benchmark Numbers from Abstract:**
- Tiny version achieves "7.5% improvement in mAP" on COCO
- Inference time: "1.5 ms" on single 4090 GPU
- No other specific COCO numbers, AP_small metrics, or YOLOv8 comparisons mentioned in the abstract

**VisDrone:** No mention found in the provided content

**Conference/Journal Status:** No acceptance information in this excerpt. The submission history shows v1 submitted June 9, 2024, with v2 revised December 14, 2024. The document is identified as arXiv preprint 2406.05835 under Creative Commons BY-NC-ND 4.0 license.
########## salvage2/b23_other.txt ##########
# Paper Summary

**Exact Title:** "A Normalized Gaussian Wasserstein Distance for Tiny Object Detection"

**Dataset:** AI-TOD (described as having "average object size much smaller than existing object detection datasets")

**Reported Gains:** The abstract states the approach yields "performance that is 6.7 AP points higher than a standard fine-tuning baseline, and 6.0 AP points higher than state-of-the-art competitors." The specific baseline detector is not explicitly named in the abstract.

**Venue/Comments Note:** The comments section indicates "Its expanded work is accepted by the ISPRS J P & RS" (ISPRS Journal of Photogrammetry and Remote Sensing). Code is available at the authors' GitHub repository.

**Submission Timeline:** Originally submitted October 26, 2021; revised June 14, 2022.
########## salvage2/b24_other.txt ##########
# Weighted Boxes Fusion Report

**Exact Title:**
"Weighted boxes fusion: Ensembling boxes from different object detection models"

**Abstract Claims:**
The method "utilizes confidence scores of all proposed bounding boxes to constructs the averaged boxes." Testing occurred on "several datasets" with evaluation in "the Open Images and COCO Object Detection tracks, achieving top results in these challenges."

**Journal Publication:**
Published in *Image and Vision Computing* (2021): 104117

The source code is noted as publicly available on GitHub.
########## salvage2/b25_other.txt ##########
# MambaOut Paper Report

**Exact Title:** MambaOut: Do We Really Need Mamba for Vision?

**Authors:** Weihao Yu and Xinchao Wang

**Stated Hypotheses:**
The authors hypothesize that Mamba's RNN-like state space model architecture suits long-sequence and autoregressive tasks. They argue image classification lacks both characteristics, making Mamba unnecessary for it. Detection and segmentation, while non-autoregressive, are long-sequence tasks warranting further exploration.

**Key Conclusions:**
- On image classification: "MambaOut model surpasses all visual Mamba models on ImageNet image classification, indicating that Mamba is indeed unnecessary for this task."

- On detection/segmentation: "MambaOut cannot match the performance of state-of-the-art visual Mamba models, demonstrating the potential of Mamba for long-sequence visual tasks."

**Acceptance Venue:** No acceptance venue is noted on this arXiv page (paper submitted May 13, 2024; latest version May 20, 2024).

**Submission Status:** Available as preprint on arXiv with code at github.com/yuweihao/MambaOut
########## salvage2/b26_other.txt ##########
# C2A Disaster Human Detection Dataset Paper Summary

**Exact Title:** "UAV-Enhanced Combination to Application: Comprehensive Analysis and Benchmarking of a Human Detection Dataset for Disaster Scenarios"

**Authors:** Ragib Amin Nihal, Benjamin Yen, Katsutoshi Itoyama, Kazuhiro Nakadai

**Venue & Status:** Accepted for the 27th International Conference on Pattern Recognition (ICPR 2024)

**Abstract Claims:**
The paper introduces the C2A dataset created by "overlaying human poses onto UAV-captured disaster scenes." The authors report that "models fine-tuned on the C2A dataset exhibit substantial performance improvements compared to those pre-trained on generic aerial datasets." They emphasize the importance of combining C2A with general human datasets for optimal results.

**Detection Performance Numbers:**
The abstract does not provide specific mAP values or percentage improvement numbers.

**Dataset Construction Details:**
The abstract mentions the dataset creation involves "diverse human poses and disaster scenes information" but does not specify whether AIDER backgrounds, LSP, or MPII poses were used—those details are not included in the provided content.

**arXiv Identifier:** 2408.04922 (v2 submitted August 23, 2024)
########## salvage2/b28_other.txt ##########
# Confluence - NMS Alternative Paper Report

**Exact Title:**
"Confluence: A Robust Non-IoU Alternative to Non-Maxima Suppression in Object Detection"

**Abstract Claims (Performance Gains):**
The paper reports improvements on two datasets:
- MS COCO: "Average Precision by up to 2.3-3.8% and Average Recall by up to 5.3-7.2%"
- CrowdHuman: Same metric improvements stated as "up to 2.3-3.8%" AP and "up to 5.3-7.2%" recall

**Journal Acceptance Note:**
No journal acceptance information appears in this content. The paper is listed as arXiv:2012.00257, submitted to computer science repositories (cs.CV and cs.AI categories), but shows no mention of peer-reviewed journal publication or conference acceptance in the visible metadata.

**Key Technical Claim:**
The method replaces IoU-based approaches with "a normalized Manhattan Distance inspired proximity metric" for bounding box clustering, selecting boxes "closest to every other box within a given cluster."
########## salvage2/b32_other.txt ##########
# Postprocess Classes in combine.py

Based on the file content, the following postprocess classes are defined:

1. **PostprocessPredictions** - Abstract base class
2. **NMSPostprocess** - "Non-Maximum Suppression (NMS)"
3. **NMMPostprocess** - "Non-Maximum Merging (NMM) with transitive merging"
4. **GreedyNMMPostprocess** - "Greedy Non-Maximum Merging (NMM)"
5. **LSNMSPostprocess** - "Locality-Sensitive NMS from the lsnms package"

## Weighted Boxes Fusion

There is **no Weighted Boxes Fusion (WBF) implementation** present in this file.

## Mapping

No `POSTPROCESS_NAME_TO_CLASS` mapping dictionary is present in the provided content. The file contains only the class definitions themselves without an explicit registry mapping class names to their implementations.
########## salvage2/b33_other.txt ##########
# RFLA Paper Summary

**Exact Title:** "RFLA: Gaussian Receptive Field based Label Assignment for Tiny Object Detection"

**Venue:** ECCV2022

**AI-TOD Performance Gain:** The abstract states the approach "outperforms the state-of-the-art competitors with 4.0 AP points on the AI-TOD dataset."

**Applicable Detectors:** The paper addresses both detector architectures. As noted in the abstract, the authors observe that "either box prior in the anchor-based detector or point prior in the anchor-free detector is sub-optimal for tiny objects." Their method introduces a Gaussian Receptive Field Distance metric and Hierarchical Label Assignment module designed to work across both paradigms for improved tiny object detection.
########## salvage2/b36_other.txt ##########
# CEASC Paper Summary

**Exact Title:** "Adaptive Sparse Convolutional Networks with Global Context Enhancement for Faster Object Detection on Drone Images"

**Venue:** CVPR 2023 (accepted)

**Datasets:** VisDrone and UAVDT

**Key Finding from Abstract:**
"CEASC remarkably reduces the GFLOPs and accelerates the inference procedure when plugging into the typical state-of-the-art detection frameworks (e.g. RetinaNet and GFL V1) with competitive performance."

The paper does not provide specific numerical speedup metrics in the abstract itself—only qualitative claims about reduced computational operations (GFLOPs) and faster inference while maintaining competitive accuracy.
########## salvage2/b40_other.txt ##########
# Wise-IoU Paper Summary

**Exact Title:** "Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing Mechanism"

**Abstract Claims:**
- Detector: YOLOv7 (state-of-the-art real-time detector)
- Dataset: MS-COCO
- AP Gain: "AP-75 on the MS-COCO dataset is improved from 53.03% to 54.50%"

**Key Innovation:** The paper proposes a dynamic non-monotonic focusing mechanism that evaluates anchor box quality using outlier degree rather than IoU, optimizing gradient allocation across high, ordinary, and low-quality examples.

**Journal/Conference Note:** No acceptance to a conference or journal is mentioned on this arXiv abstract page. This appears to be a preprint submission (v3 revised April 8, 2023).

**Code Availability:** "Code is available at [this https URL]" (GitHub repository link provided by authors)
########## salvage2/b42_other.txt ##########
# C2A Dataset: Key Findings Report

## (1) Benchmark Table: Detectors on C2A

| Model | mAP | mAP@.50 |
|-------|-----|---------|
| Faster R-CNN | 0.3656 | 0.6340 |
| RetinaNet | 0.3834 | 0.6933 |
| Cascade R-CNN | 0.4860 | 0.7350 |
| RTMDet | 0.4420 | 0.7080 |
| DINO | 0.4710 | 0.7890 |
| YOLOv5 | 0.4920 | 0.8080 |
| YOLOv9-c | 0.5562 | 0.7996 |
| YOLOv9-e | 0.6883 | 0.8927 |

## (2) Cross-Domain Generalization: Training/Validation on SARD

| Training Data | mAP on SARD |
|---------------|------------|
| General Human only | 0.347 |
| SARD only | 0.931 |
| C2A only | 0.259 |
| General Human + C2A | 0.660 |

The paper uses "mAP scores" as its evaluation metric across these comparisons.

## (3) Image Resolution and Small Object Statistics

- **Image count:** 10,215 total images
- **Object instances:** Over 360,000
- **Resolution range:** Approximately 123×152 to 5184×3456 pixels
- **Small objects:** "47% of instances are under 10 pixels" in size
- **10-50 pixel range:** 52% of instances fall here
- **Median image width:** 428 pixels

## (4) Stated Limitations and Future Work

The authors identify the synthetic nature as problematic: "overlay of human figures...can sometimes result in unrealistic scaling and positioning." They advocate for "context-aware adaptive scaling and improved spatial algorithms" and note that expanding to "video sequences of disaster scenes" represents important future direction, since training uses single images while "actual disaster scenarios...could be in the form of video footage."
########## salvage2/b35_other.txt ##########
# Citing Papers from JSON

1. **MORAF: A Multi-Objective Framework for Resource Allocation with Fairness for Efficient Disaster Management** (2026)
   - Venue: Proceedings of the International ISCRAM Conference
   - DOI: 10.59297/b3tdqk53

2. **Survey on Disaster Management Datasets for Remote Sensing-Based Emergency Applications** (2026)
   - Venue: IEEE Transactions on Geoscience and Remote Sensing
   - DOI: 10.1109/TGRS.2026.3692760
   - ArXiv: 2605.08196

3. **AI-Powered Multimodal Human Detection System Combining YOLOv11 and Terrain Data in Search and Rescue Operations** (2026)
   - Venue: 2026 International Conference on Smart Electronic Devices and Intelligent Systems (ICSEDIS)
   - DOI: 10.1109/ICSEDIS68157.2026.11518178

4. **Drone-based human motion capture: A review** (2026)
   - Venue: Intelligent Sports and Health
   - DOI: 10.1016/j.ish.2025.12.002

5. **Comparative Evaluation of YOLO Models for Human Position Recognition with UAVs During a Flood** (2025)
   - Venue: Applied System Innovation
   - DOI: 10.3390/asi9010006

6. **AIoT-Based Hybrid Framework for Victim Detection in Disaster Scenarios** (2025)
   - Venue: 2025 IEEE 1st International Conference on Recent Trends in Computing and Smart Mobility (RCSM)
   - DOI: 10.1109/RCSM67767.2025.11507685

7. **A Comparative Performance Assessment for Identification of Trapped Human Bodies based on Deep Learning Architectures** (2025)
   - Venue: 2025 IEEE International Conference on Intelligent Signal Processing and Effective Communication Technologies (INSPECT)
   - DOI: 10.1109/INSPECT67393.2025.11351010

8. **HitoMi-Cam: A Shape-Agnostic Person Detection Method Using the Spectral Characteristics of Clothing** (2025)
   - Venue: Journal of Imaging
   - DOI: 10.3390/jimaging11110399
   - ArXiv: 2511.08908

9. **Optimizing Human Detection in Natural Disaster Environments Through YOLO and Image Processing** (2025)
   - Venue: Advanced Industrial Conference on Telecommunications
   - DOI: 10.1109/AICT67988.2025.11268634

10. **AI-Enhanced Rescue Drone with Multi-Modal Vision and Cognitive Agentic Architecture** (2025)
    - Venue: Applied Informatics
    - DOI: 10.3390/ai6100272

11. **Advancing Human Detection in UAV-Based Search and Rescue Using Transfer Learning Approach** (2025)
    - Venue: 2025 3rd International Conference on Device Intelligence, Computing and Communication Technologies (DICCT)
    - DOI: 10.1109/DICCT64131.2025.10986412

12. **Comparative Evaluation of Machine Learning Models for UAV Network Performance Identification in Dynamic Environments** (2025)
    - Venue: Buletin Ilmiah Sarjana Teknik Elektro
    - DOI: 10.12928/biste.v6i4.12409

13. **From Blurry to Brilliant Detection: YOLO-Based Aerial Object Detection with Super Resolution** (2024)
    - Venue: Asia-Pacific Signal and Information Processing Association Annual Summit and Conference
    - DOI: 10.1109/APSIPAASC65261.2025.11249079
    - ArXiv: 2401.14661
Now the six web-search snippet blocks that carry venue and competitor info:

Bash Read salvaged web-search snippet blocks
IN
cd "C:/Users/Dell/AppData/Local/Temp/claude/d--Academics-thesis-folder/40e8cefd-8ee1-4988-b4cc-a1b90631723a/scratchpad" && for f in salvage/block_14.txt salvage/block_15.txt salvage/block_19.txt salvage/block_20.txt salvage/block_22.txt salvage/block_23.txt; do echo "########## $f ##########"; cat "$f"; echo; done

OUT
########## salvage/block_14.txt ##########
Web search results for query: ""Mamba-YOLO" 2406.05835 accepted venue journal OR conference 2025"

Links: [{"title":"REA-YOLO for small object detection in UAV aerial images | The Journal of Supercomputing | Springer Nature Link","url":"https://link.springer.com/article/10.1007/s11227-025-07836-0"},{"title":"YOLO-BSMamba: A YOLOv8s-Based Model for Tomato Leaf Disease Detection in Complex Backgrounds","url":"https://www.mdpi.com/2073-4395/15/4/870"},{"title":"FF-Mamba-YOLO: An SSM-Based Benchmark for Forest Fire Detection in UAV Remote Sensing Images","url":"https://www.mdpi.com/2313-433X/12/1/43"},{"title":"[2406.05835v1] Mamba YOLO: SSMs-Based YOLO For Object Detection","url":"https://arxiv.org/abs/2406.05835v1"},{"title":"MHS-VIT: Mamba hybrid self-attention vision transformers for traffic image detection | PLOS One","url":"https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0325962"},{"title":"[2406.05835] Mamba YOLO: A Simple Baseline for Object Detection with State Space Model","url":"https://arxiv.org/abs/2406.05835"},{"title":"GitHub - HZAI-ZJNU/Mamba-YOLO: the official pytorch implementation of “Mamba-YOLO：SSMs-based for Object Detection”","url":"https://github.com/HZAI-ZJNU/Mamba-YOLO"},{"title":"Paper page - Mamba YOLO: SSMs-Based YOLO For Object Detection","url":"https://huggingface.co/papers/2406.05835"},{"title":"Segmentation Strategies in Deep Learning for Prostate Cancer Diagnosis:   A Comparative Study of Mamba, SAM, and YOLO","url":"https://arxiv.org/pdf/2409.16205"}]

Based on the search results, the Mamba YOLO paper (arXiv 2024, arXiv:2406.05835) is titled "Mamba YOLO: A simple baseline for object detection with state space model".

However, the search results indicate that this paper was submitted to arXiv on June 9, 2024, and appears to remain as an arXiv preprint. While the search results show multiple 2025 publications that cite the Mamba YOLO paper (such as papers in *The Journal of Supercomputing*, MDPI journals, and *PLOS One*), the search results do not contain explicit information about the Mamba YOLO paper itself being accepted to a specific venue or journal for 2025.

The paper appears to continue being referenced in the research community through various applications and improvements built upon it, but the search results do not clearly identify a formal acceptance to a specific conference or journal in 2025.


REMINDER: You MUST include the sources above in your response to the user using markdown hyperlinks.
########## salvage/block_15.txt ##########
Web search results for query: "MambaOut CVPR 2025 accepted "Do We Really Need Mamba for Vision""

Links: [{"title":"Yu MambaOut Do We Really Need Mamba for Vision CVPR 2025 Paper","url":"https://www.scribd.com/document/882149429/Yu-MambaOut-Do-We-Really-Need-Mamba-for-Vision-CVPR-2025-Paper"},{"title":"[PDF] MambaOut: Do We Really Need Mamba for Vision?* | Semantic Scholar","url":"https://www.semanticscholar.org/paper/MambaOut:-Do-We-Really-Need-Mamba-for-Vision*-Yu-Wang/31fdba3a68f286894f025e734a277e2ce94dd84c"},{"title":"GitHub - yuweihao/MambaOut: MambaOut: Do We Really Need Mamba for Vision? (CVPR 2025) · GitHub","url":"https://github.com/yuweihao/MambaOut"},{"title":"MambaOut: Do We Really Need Mamba for Vision?","url":"https://openaccess.thecvf.com/content/CVPR2025/html/Yu_MambaOut_Do_We_Really_Need_Mamba_for_Vision_CVPR_2025_paper.html"},{"title":"CVPR Poster MambaOut: Do We Really Need Mamba for Vision?","url":"https://cvpr.thecvf.com/virtual/2025/poster/33055"},{"title":"[2405.07992] MambaOut: Do We Really Need Mamba for Vision?","url":"https://arxiv.org/abs/2405.07992"},{"title":"MambaOut: Do We Really Need Mamba for Vision?* Weihao Yu Xinchao Wang","url":"https://openaccess.thecvf.com/content/CVPR2025/papers/Yu_MambaOut_Do_We_Really_Need_Mamba_for_Vision_CVPR_2025_paper.pdf"}]

Based on the search results, here's information about the MambaOut paper accepted at CVPR 2025:

## Paper Overview

The paper "MambaOut: Do We Really Need Mamba for Vision?" by Weihao Yu and Xinchao Wang was accepted to the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2025.

## Main Findings

The paper concludes that Mamba is ideally suited for tasks with long-sequence and autoregressive characteristics. For vision tasks, as image classification on ImageNet does not align with either characteristic, the authors hypothesize that Mamba is not necessary for this task.

## Key Contribution

The authors constructed a series of models named MambaOut through stacking Mamba blocks while removing their core token mixer, the state space model (SSM). 

## Results

The MambaOut model surpasses all visual Mamba models on ImageNet image classification, indicating that Mamba is indeed unnecessary for this task. However, for detection and segmentation, MambaOut cannot match the performance of state-of-the-art visual Mamba models, demonstrating the potential of Mamba for long-sequence visual tasks.


REMINDER: You MUST include the sources above in your response to the user using markdown hyperlinks.
########## salvage/block_19.txt ##########
Web search results for query: ""CBF" OR "cluster boxes fusion" OR "confidence-based fusion" bounding box fusion small object detection"

Links: [{"title":"Tiny Collaborative Inference for Occlusion-Robust Object Detection","url":"https://arxiv.org/pdf/2606.02894"},{"title":"Weighted Boxes Fusion for Object Detection","url":"https://www.emergentmind.com/papers/1910.13302"},{"title":"[1910.13302] Weighted boxes fusion: Ensembling boxes from different object detection models","url":"https://www.arxiv-vanity.com/papers/1910.13302/"},{"title":"PseudoProp: Robust Pseudo-Label Generation for Semi-Supervised Object   Detection in Autonomous Driving Systems","url":"https://arxiv.org/pdf/2203.05983"},{"title":"Weighted boxes fusion: Ensembling boxes from different object detection models - ScienceDirect","url":"https://www.sciencedirect.com/science/article/abs/pii/S0262885621000226"},{"title":"Weighted boxes fusion: Ensembling boxes from different object detection models","url":"https://www.researchgate.net/publication/349067578_Weighted_boxes_fusion_Ensembling_boxes_from_different_object_detection_models"},{"title":"Weighted Boxes Fusion in Object Detection - Enhancing Accuracy","url":"https://learnopencv.com/weighted-boxes-fusion/"},{"title":"System and method for deep network fusion for fast and robust object detection","url":"https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10657364"}]

Based on the web search results, I found information about bounding box fusion methods in object detection:

## Key Findings

Weighted Boxes Fusion (WBF) is an algorithm designed to integrate bounding box predictions more effectively than traditional methods like Non-Maximum Suppression (NMS) and its variants.

### How WBF Works

The cornerstone of WBF is its utilization of confidence scores associated with the predicted bounding boxes, enabling the construction of averaged boxes that incorporate contributions from all model outputs. The algorithm operates through several steps:

1. Predictions from different models are aggregated and sorted by confidence scores.
2. Predicted boxes are clustered based on IoU overlap.
3. Bounding boxes are fused using a weighted average, prioritizing those with higher confidence scores.

### Applications

The method was evaluated in the context of Open Images and COCO Object Detection challenges, achieving top results, and the 3D version of boxes fusion was successfully applied by the winning teams of Waymo Open Dataset and Lyft 3D Object Detection for Autonomous Vehicles challenges.

The search results primarily focus on Weighted Boxes Fusion (WBF), which appears to be the most prominent bounding box fusion method for object detection. However, the search results don't contain specific information about "CBF" (Cluster Boxes Fusion) or "confidence-based fusion" as distinct named methods.


REMINDER: You MUST include the sources above in your response to the user using markdown hyperlinks.
########## salvage/block_20.txt ##########
Web search results for query: "ASAHI adaptive slicing aided hyper inference small object detection paper"

Links: [{"title":"Adaptive Slicing-Aided Hyper Inference for Small Object Detection in High-Resolution Remote Sensing Images","url":"http://dx.doi.org/10.3390/rs15051249"},{"title":"Adaptive Slicing-Aided Hyper Inference for Small Object Detection in High-Resolution Remote Sensing Images | MDPI","url":"https://www.mdpi.com/2072-4292/15/5/1249"},{"title":"(PDF) Adaptive Slicing-Aided Hyper Inference for Small Object Detection in High-Resolution Remote Sensing Images","url":"https://www.researchgate.net/publication/368856750_Adaptive_Slicing-Aided_Hyper_Inference_for_Small_Object_Detection_in_High-Resolution_Remote_Sensing_Images"},{"title":"Adaptive Slicing-Assisted Hyper Inference for Enhanced Small Object Detection in High-Resolution Imagery","url":"https://arxiv.org/html/2604.19233v1"},{"title":"Adaptive Slicing-Aided Hyper Inference for Small Object Detection in High-Resolution Remote Sensing Images - ADS","url":"https://ui.adsabs.harvard.edu/abs/2023RemS...15.1249Z/abstract"},{"title":"[2604.19233] Adaptive Slicing-Assisted Hyper Inference for Enhanced Small Object Detection in High-Resolution Imagery","url":"https://arxiv.org/abs/2604.19233"},{"title":"Adaptive Slicing-Assisted Hyper Inference for Enhanced Small Object Detection in High-Resolution Imagery","url":"https://arxiv.org/pdf/2604.19233"}]

I found the ASAHI paper you're looking for. Here's what I found:

## Adaptive Slicing-Aided Hyper Inference (ASAHI)

ASAHI (Adaptive Slicing-Aided Hyper Inference) is a novel slicing method which adaptively slices images into a fixed number of patches rather than using a fixed slicing size.

**Purpose and Problem:**
Despite advances in deep learning for object detection, detecting small objects remains difficult, especially in aerial images with features including small size, high density, high variability, and varying orientation. Existing slicing methods inevitably lead to redundant computation.

**Key Innovation:**
ASAHI focuses on the number of slices rather than the slicing size, adaptively adjusting the slicing size to control the number of slices according to the image resolution.

**Components:**
By automatically controlling the number of slices, the method greatly reduces redundant computation. In the post-processing stage, it substitutes non-maximum suppression (NMS) with Cluster-DIoU-NMS, which reduces time consumption while maintaining result quality.

The paper was published in **Remote Sensing** (February 2023) and also appears in more recent versions on arXiv with updated evaluations on datasets like VisDrone2019 and xView.


REMINDER: You MUST include the sources above in your response to the user using markdown hyperlinks.
########## salvage/block_22.txt ##########
Web search results for query: "YOLOv8 NWD normalized Wasserstein distance loss implementation VisDrone mAP improvement paper"

Links: [{"title":"The calculation of the normalized Wasserstein distance (NWD) metric. | Download Scientific Diagram","url":"https://www.researchgate.net/figure/The-calculation-of-the-normalized-Wasserstein-distance-NWD-metric_fig4_380055055"},{"title":"RMH-YOLO: A Refined Multi-Scale Architecture for Small-Target Detection in UAV Aerial Imagery","url":"https://doi.org/10.3390/s25227088"},{"title":"A Normalized Gaussian Wasserstein Distance for Tiny Object Detection | Request PDF","url":"https://www.researchgate.net/publication/355664089_A_Normalized_Gaussian_Wasserstein_Distance_for_Tiny_Object_Detection"},{"title":"GS-Yolo: An effective grouping and shuffling method for object detection in aerial images | Proceedings of the 4th International Conference on Computer, Artificial Intelligence and Control Engineering","url":"https://dl.acm.org/doi/10.1145/3727648.3727742"},{"title":"Fine-YOLO: A Simplified X-ray Prohibited Object Detection Network Based on Feature Aggregation and Normalized Wasserstein Distance - PMC","url":"https://pmc.ncbi.nlm.nih.gov/articles/PMC11175173/"},{"title":"Multi-scale detection of underwater objects using attention mechanisms and normalized Wasserstein distance loss | The Journal of Supercomputing | Springer Nature Link","url":"https://link.springer.com/article/10.1007/s11227-025-07251-5"},{"title":"YOLO-FaceV2: A Scale and Occlusion Aware Face Detector","url":"https://arxiv.org/pdf/2208.02019"},{"title":"A Recursive Prediction-Based Feature Enhancement for Small Object Detection - PMC","url":"https://pmc.ncbi.nlm.nih.gov/articles/PMC11207305/"},{"title":"DEAL-YOLO: Drone-based Efficient Animal Localization using YOLO","url":"https://arxiv.org/pdf/2503.04698"}]

Based on the web search results, here's what I found about YOLOv8 with NWD (Normalized Wasserstein Distance) loss implementation and improvements on VisDrone dataset:

## Implementation and Concept

NWD is a new evaluation method for small target detection where the bounding box is modeled as a two-dimensional Gaussian distribution, and the similarity between the predicted target and real target is calculated through their corresponding Gaussian distributions. The NWD is not sensitive to the scale of targets and thus is more suitable for measuring the similarity between small targets.

## mAP Improvements on VisDrone

Several papers demonstrate significant improvements when integrating NWD loss with YOLOv8:

- RMH-YOLO combines InnerCIoU with Normalized Wasserstein Distance (NWD) to improve bounding box regression for small targets, achieving precision and recall increases of 8.8% and 7.4% respectively, while mAP50 and mAP50:95 improve by 9.2% and 6.4% compared with the baseline YOLOv8n model on VisDrone2019.

- GS-YOLO uses NWD to reduce boundary regression loss sensitivity for small target positions, achieving approximately 1.0% improvement in mAP from the NWD loss addition, with the entire model improving mAP50 by 5.8% and mAP50:95 by 4.3%.

- LACF-YOLO achieves a 3.5% increase in mAP on the VisDrone2019 dataset.

These implementations demonstrate that NWD loss is particularly effective for small object detection in UAV aerial imagery applications covered by the VisDrone dataset.


REMINDER: You MUST include the sources above in your response to the user using markdown hyperlinks.
########## salvage/block_23.txt ##########
Web search results for query: ""LightSeek-YOLO" C2A disaster human detection AP_small mAP"

Links: [{"title":"LightSeek-YOLO: A Lightweight Architecture for Real-Time Trapped Victim Detection in Disaster Scenarios","url":"https://www.mdpi.com/2227-7390/13/19/3231"},{"title":"LightSeek-YOLO: A Lightweight Architecture for Real-Time Trapped Victim Detection in Disaster Scenarios | Semantic Scholar","url":"https://www.semanticscholar.org/paper/LightSeek-YOLO:-A-Lightweight-Architecture-for-in-Tian-Zheng/acec79fb935aa06976e5c7097e58a516df9da81c"},{"title":"Self-supervised multimodal change detection based on difference contrast learning for remote sensing imagery | Request PDF","url":"https://www.researchgate.net/publication/385574568_Self-supervised_multimodal_change_detection_based_on_difference_contrast_learning_for_remote_sensing_imagery"},{"title":"UAV-Based Real-Time Survivor Detection System in Post-Disaster Search and Rescue Operations","url":"https://www.researchgate.net/publication/351857034_UAV-Based_Real-Time_Survivor_Detection_System_in_Post-Disaster_Search_and_Rescue_Operations"},{"title":"A real-time natural disaster victim detection based on improved YOLOv5... | Download Scientific Diagram","url":"https://www.researchgate.net/figure/A-real-time-natural-disaster-victim-detection-based-on-improved-YOLOv5-network-YOLO-MSFR_fig1_376365583"},{"title":"Disaster Rescue Research Articles - Page 1 | R Discovery","url":"https://discovery.researcher.life/topic/disaster-rescue/2228902?page=1&topic_name=Disaster+Rescue"},{"title":"Mathematics | Free Full-Text | LightSeek-YOLO: A Lightweight Architecture for Real-Time Trapped Victim Detection in Disaster Scenarios | Notes","url":"https://www.mdpi.com/2227-7390/13/19/3231/notes"},{"title":"Comparative visualization of YOLOv8n, YOLOv11n and LightSeek-YOLO in... | Download Scientific Diagram","url":"https://www.researchgate.net/figure/Comparative-visualization-of-YOLOv8n-YOLOv11n-and-LightSeek-YOLO-in-disaster-scenarios_fig7_396397230"}]

Based on the search results, here's what I found about LightSeek-YOLO and its performance on the C2A disaster dataset:

## Overview

LightSeek-YOLO is a lightweight, real-time victim detection framework for disaster scenarios built upon YOLOv11.

## Performance on C2A Disaster Dataset

On the specialised C2A disaster dataset, the model achieves an AP@small of 0.478, confirming its robust performance and particular efficacy in detecting small targets amid challenging conditions such as debris and smoke.

## Key Architecture Features

The model employs HGNetV2 as the backbone, whose HGStem and HGBlock modules leverage depthwise separable convolutions to markedly reduce computational cost while preserving feature extraction. Additionally, it introduces Seek-DS (Seek-DownSampling), a dual-branch downsampling module that preserves key feature extrema through a MaxPool branch while capturing spatial patterns via a progressive convolution branch, thereby effectively mitigating background interference. Third, it incorporates Seek-DH (Seek Detection Head), a lightweight detection head that processes features through a unified pipeline, enhancing scale adaptability while reducing parameter redundancy.

## Overall Performance

In the COCO dataset, it delivers a competitive mAP@[0.5:0.95] of 0.473, matching the performance of YOLOv8n (0.473) and approaching that of its baseline, YOLOv11n (0.481).


REMINDER: You MUST include the sources above in your response to the user using markdown hyperlinks.
Bash Check size of existing novelty findings doc
IN
wc -l "d:/Academics/thesis folder/Last Month/docs/2026-06-29_novelty_research_findings.md"

OUT
146 d:/Academics/thesis folder/Last Month/docs/2026-06-29_novelty_research_findings.md
Read d:\Academics\thesis folder\Last Month\docs\2026-06-29_novelty_research_findings.md
Write d:\Academics\thesis folder\docs\2026-07-04_novelty_direction_research.md
188 lines
# Novelty direction research — Mamba final verdict + the new plan (2026-07-04)

Branch: `novelty-direction` (created today from `main`). Compiled from (1) a full local audit of
`Last Month/`, `31-03-26(Mamba-ViT-CNN)/`, `01-03-2026-Onward Model trying/`, and (2) web research with
per-paper verification fetches. The web-research agent hit the session usage limit mid-synthesis
(resets 18:30 Asia/Dhaka); its verified per-paper fetches were salvaged and are the basis of every
citation below. Anything NOT verified is flagged. Builds on — does not repeat —
`Last Month/docs/2026-06-29_novelty_research_findings.md`.

---

## 1. Where you actually stand (from the local audit — all numbers from run JSONs)

| Asset | Number | Source |
|---|---|---|
| CBAM+P2 on C2A test (s0) | COCO AP 0.6153 / AP50 0.8533 / AP_small 0.6156 / VT-recall 0.7575 | `Last Month/docs/2026-06-13_complete_ablation_table.md` |
| mAP50-95 ceiling on C2A | **~0.615 for EVERY architecture** (baseline/CBAM/CBAM+P2/Mamba all within 0.002) | same |
| SAHI slice256_ov30 | very-tiny recall 0.7668 → **0.8292** (+6.2 pt), but F1 drops 0.850 → 0.829 | `31-03-26(Mamba-ViT-CNN)/SAHI+TTA/benchmark_reports/MASTER_REPORT_SAHI_TTA_NOAUG.txt` |
| TTA@1280 | mAP50 0.890 / mAP50-95 0.698 (best aggregate) | same |
| Best deployable (epoch125.pt, joint C2A+SARD) | C2A 0.878 / SARD 0.917 mAP50 | `deployable_model/` README + finetune_summary |
| Enriched fine-tune (2026-07-02, PC-4) | C2A 0.874 (held) / SARD 0.898 (−0.019 dip); drone-FP re-test PENDING | `runs_joint/20260702_144117_finetune_enriched/metrics/finetune_summary.json` |
| Zero-shot C2A→SARD | collapses ~99% (mAP50 0.0042–0.0147 across all variants) | `Last Month/cross_dataset_SARD/ablation_master/sard_generalization.csv` |
| Drone footage | 3 unlabeled 4K videos (10 m / 30 m / 50 m); only 10 m has been inference-processed | `Drone Shoot/`, `drone_inference_out/10m/` |

**Never tried anywhere in the repo** (grep-verified): WBF, Soft-NMS/any NMS variant, NWD/Wasserstein
loss, RFLA, Wise/Shape/Inner-IoU, super-resolution, adaptive slicing. Tile merging in
`run_on_drone_footage.py` is plain `torchvision.ops.nms`.

---

## 2. Mamba — the final, cited verdict: **CLOSED. Do not reopen.**

Four independent lines of evidence now converge:

1. **Your own two genuine runs are negative.**
   - LocalWindowSSM neck (6 blocks, run `20260609_205717`): COCO AP 0.6143 vs CBAM+P2 0.6153 — an exact
     tie — at +2.4 M params and **2.8× latency** (41.1 vs 14.6 ms). A credible null.
   - AtrousSSM (Mar 2026, genuine 24.16 M run): mAP50 0.823 vs 0.874 — **−10%**, 2× latency, worse
     tiny-object recall. A genuine failure.
2. **Independent aerial benchmark: Mamba loses to plain YOLO.** On SeaDronesSee (UAV small objects):
   MambaYOLO-L 25.5 < YOLOv8-m 26.1 < YOLO11-m 26.3 AP; the authors state SSM "optimizations do not
   translate to multi-scale small object detection" (PLOS One, peer-reviewed —
   pmc.ncbi.nlm.nih.gov/articles/PMC12212484). LocalMamba shows AP_small 26.0 vs Vim 26.1 — zero gain
   on the one axis you need.
3. **The literature's own flagship results don't support a neck-injection gain.**
   - Mamba-YOLO (arXiv 2406.05835) — **still an unaccepted preprint** as of today (verified: no venue
     found; v2 Dec 2024). Its headline "+7.5% mAP" is for the *Tiny* scale vs smaller baselines, on
     COCO, with no AP_small claim and no YOLO11 comparison.
   - MambaOut (**CVPR 2025, peer-reviewed** — arXiv 2405.07992) argues SSM *may* help detection — but
     only as a *full long-sequence backbone*, which is exactly what you cannot build (below). Your
     null result must be framed via latency + neck-vs-backbone distinction, NOT "MambaOut says SSM
     fails detection" (see June-29 doc §5 for the corrected framing).
4. **Engineering blocker unchanged:** `mamba-ssm`/`causal-conv1d`/`selective_scan` CUDA kernels have
   no Windows wheels (Linux-only; github.com/state-spaces/mamba/issues/662). Your training boxes are
   all Windows.

**Answer to "will Mamba improve accuracy over CBAM+P2+SAHI+TTA?": No.** And C2A's own label ceiling
(mAP50-95 saturated at ~0.615 for every architecture — attributed to composited paste-box geometry)
means *no architecture change of any kind* can move the headline metric there. The thesis already owns
a well-executed, correctly-instrumented SSM null result — that IS the Mamba contribution. Spend the
remaining weeks where the metric can actually move: **the loss, the fusion step, and the
synthetic→real gap.**

---

## 3. What the web research verified (salvaged evidence, per-paper fetches)

### 3a. Tiny-object losses — pluggable into Ultralytics, Windows-safe, cheap
| Technique | Evidence (verified) | Status |
|---|---|---|
| **NWD** — Normalized Gaussian Wasserstein Distance (arXiv 2110.13389) | +6.7 AP over standard baseline on AI-TOD (tiny-object benchmark); expanded version **accepted in ISPRS J. P&RS (Q1)** | Verified from abstract |
| NWD in YOLOv8-family (implementations exist) | RMH-YOLO (Sensors 2025, doi 10.3390/s25227088): NWD+InnerCIoU on YOLOv8n → +9.2 mAP50 / +6.4 mAP50-95 on VisDrone. GS-YOLO (ACM CAICE 2025): **~+1.0 mAP from the NWD term alone**. LACF-YOLO: +3.5 mAP on VisDrone | Verified via search snippets; module-level attribution only clean in GS-YOLO |
| **RFLA** — Gaussian receptive-field label assignment (**ECCV 2022**, arXiv 2208.08738) | +4.0 AP over SOTA on AI-TOD; explicitly designed for both anchor-based AND point/anchor-free priors | Verified from abstract; integration into YOLO11's TAL assigner is real work (1+ week) |
| Wise-IoU (arXiv preprint, not accepted) | YOLOv7 COCO AP75 53.03→54.50 | Verified; generic, not tiny-specific — low priority |

Why NWD fits *your* dataset precisely: C2A has **47% of instances under 10 px** (verified against the
C2A paper's stats) — the exact regime where IoU-based regression/assignment degenerates and NWD is
designed to fix. Honest expectation on C2A: this will NOT break the 0.615 ceiling (that's label noise),
but it plausibly moves AP_small / very-tiny recall by 1–3 pt, and it directly targets the SARD
AP_small weakness (0.294) of the deployable model.

### 3b. Box fusion — a real implementation gap, zero training cost
- **WBF** (Solovyev et al., arXiv 1910.13302) — **peer-reviewed: Image and Vision Computing 2021**.
  Averages clustered boxes weighted by confidence instead of discarding them (NMS/NMM).
- **Verified gap: SAHI does NOT implement WBF.** Its `postprocess/combine.py` contains only
  NMS / NMM / GreedyNMM / LSNMS classes (fetched and confirmed). Nobody has published
  calibration-aware WBF as the SAHI tile-merge step.
- Your own data says the merge step is the bottleneck: SAHI slice256 buys +6.2 pt very-tiny recall but
  **loses F1 (0.850→0.829)** — i.e. the greedy merge is creating precision damage that a weighted
  fusion can plausibly recover.
- Supporting precedent: ASAHI (Remote Sensing 2023, doi 10.3390/rs15051249) replaced SAHI's NMS with
  Cluster-DIoU-NMS + adaptive slice-count and reports accuracy/latency wins on VisDrone/xView —
  evidence that merge-strategy work on sliced inference is publishable in a Q1/Q2 remote-sensing venue.
- Confluence (arXiv 2012.00257, appears unaccepted): +2.3–3.8% AP / +5.3–7.2% AR on COCO+CrowdHuman —
  cite as related work, don't build on it.
- The "WBF or CBF" you half-remembered: it's **WBF**. No established method named "CBF" exists
  (searched; nothing found).
- CAVEAT on the June-29 doc's "~+10% mAP from WBF": that number is for *multi-model ensembles* in the
  original paper — for single-model tile fusion expect a much smaller, but free, gain. Report your own.

### 3c. Competitor landscape on C2A (novelty check)
- C2A paper (Nihal et al., **ICPR 2024**, arXiv 2408.04922) benchmark: best is YOLOv9-e at
  mAP 0.6883 / mAP50 0.8927 (58 M params). Your CBAM+P2 at ~20 M params: 0.615/0.853 single-model,
  0.698/0.890 with TTA — competitive at 1/3 the size; frame it as accuracy-per-param/latency.
- **LightSeek-YOLO** (Mathematics/MDPI 2025, doi 10.3390/math13193231): lightweight YOLOv11 variant,
  reports **AP_small 0.478 on C2A** — your 0.6156 beats it by a wide margin (protocol caveats apply;
  verify their eval setup before claiming in print).
- The 13 citing papers found for C2A are overwhelmingly *application/comparison* papers (SAR system
  demos, YOLO comparisons, surveys). **Nobody has published: (i) tiny-object loss work on C2A,
  (ii) sliced-inference fusion on C2A, (iii) cross-domain C2A→SARD→real-footage deployment analysis.**
  The niche is active enough to cite (14+ papers in 2 years) but the method space is wide open.
- The C2A paper's own cross-domain Table (their protocol): C2A-only→SARD mAP 0.259, C2A+GeneralHuman→
  SARD 0.660. Your measured zero-shot collapse (0.004–0.015 mAP50) is far more severe — reconciling
  this discrepancy (their eval conf/protocol vs yours) is itself a publishable finding. [Their table
  values still flagged UNVERIFIED — confirm against arxiv.org/html/2408.04922v2 before quoting.]

---

## 4. THE PLAN — ranked directions (expected gain × feasibility × novelty)

### #1 — Calibration-weighted box fusion for sliced inference ("C-WBF") — START HERE
**What:** Replace the greedy-NMM/NMS tile-merge in your SAHI pipeline with WBF, weighted by
*calibrated* confidences — you already compute ECE/temperature data in every eval (`opt_thresholds`,
`calibration` blocks in your summaries), so calibrate scores (per-slice-scale temperature) before the
weighted average. Ensemble whole-frame + sliced + TTA predictions in the same fusion.
**Why it can be called novel:** SAHI verifiably lacks WBF; calibration-aware fusion for tile merging
has no published instance we could find; your F1-drop-under-slicing is the documented failure it fixes.
**Cost:** inference-only — no training, runs on anything, `ensemble-boxes` pkg + ~200 lines.
**Risk:** gain may be small (1–3 pt F1/AP on sliced configs). Mitigation: it's 2–3 days of work; even
a null feeds the ablation.

### #2 — NWD-hybrid loss for the tiny-object regime
**What:** Add an NWD term to Ultralytics' bbox loss (blend with CIoU, e.g. `L = (1-α)·CIoU + α·NWD`),
optionally NWD-based assignment metric in TAL. Fine-tune from epoch125 first (cheap signal), then one
full C2A retrain if the pilot is positive; final = 3 seeds per your protocol.
**Why:** C2A is 47% sub-10px — NWD's exact design target; Q1-published basis + multiple replications
in YOLOv8-family; nobody has done it on C2A.
**Cost:** loss-level patch (~1 day) + your existing training pipeline; PC-1/PC-4 capable.
**Risk:** gains on C2A may be muted by the label-noise ceiling — measure on AP_small/VT-recall and on
SARD/real footage, not mAP50-95. RFLA is the stretch-goal upgrade if NWD shows signal.

### #3 — The synthetic→real deployment study (the paper's spine — already half-built)
Unchanged from June-29 §8, now with assets #1/#2 feeding it: label 15–20 frames per altitude
(10/30/50 m) → real test set; evaluate {epoch125, +enriched, +NWD, ±C-WBF} on it; run the SAHI-guided
self-training loop; finish the pending drone-FP re-test of the enriched model. The 3-altitude
stratification of your own footage is a micro-benchmark nobody else has.

### #4 — (Stretch) Adaptive slicing
ASAHI-style resolution-adaptive slice count on top of #1, cite ASAHI, or a density-guided variant
(slice only where a cheap low-res pass finds candidates). Only if #1–#3 land early.

### The paper, assembled
*"Deployment-grade tiny-human detection for UAV search-and-rescue: tiny-object-aware training (NWD) +
calibration-weighted sliced-inference fusion (C-WBF), validated across synthetic→real domains (C2A →
SARD → self-collected 3-altitude drone footage), with SSM/Mamba necks reported as an instrumented null
result."* Every block has verified published precedent; the combination + the cross-domain protocol +
the honest negatives = coherent Q2 submission, Q1-adjacent (Remote Sensing / ISPRS-tier) if the real-
footage deltas are strong.

### 4-week schedule
| Week | Work | Machine |
|---|---|---|
| 1 | C-WBF implementation + C2A/SARD ablation vs GreedyNMM/NMS; label drone test frames (parallel, manual) | any (inference) |
| 2 | NWD loss patch → fine-tune pilot from epoch125 → full retrain if positive | PC-1 or PC-4 |
| 3 | Self-training Phases 2–4 (June-29 §8); pending drone-FP re-test of enriched model | PC-4 |
| 4 | 3-seed confirmations, cross-domain master table, paper draft | — |

---

## 5. What NOT to do (all previously adjudicated, now confirmed)
Mamba/SSM in any form (§2) · ViT/DETR/RT-DETR swaps (too heavy; never built here; June-29 §1 shows
CBAM+P2 recipe is prior art anyway) · federated/RL/continual/TENT (June-29 §4) · UDA/CycleGAN
(June-29 §9.2) · more enrichment blending without first finishing the pending drone-FP validation.

## 6. Citation shortlist for the new directions
- WBF: Solovyev, Wang, Gabruseva — *Image and Vision Computing* 2021 (arXiv 1910.13302). Peer-reviewed.
- NWD: Wang et al. (arXiv 2110.13389); extended version in ISPRS J. P&RS. Peer-reviewed (extension).
- RFLA: Xu et al., ECCV 2022 (arXiv 2208.08738). Peer-reviewed.
- ASAHI: *Remote Sensing* 15(5):1249, 2023 (doi 10.3390/rs15051249). Peer-reviewed.
- SAHI: Akyon et al. (arXiv 2202.06934, ICIP 2022). Peer-reviewed.
- MambaOut: Yu & Wang, CVPR 2025 (arXiv 2405.07992). Peer-reviewed.
- Mamba-YOLO: arXiv 2406.05835. **Preprint only — cite as such.**
- Confluence: arXiv 2012.00257. **Preprint only.**
- CEASC: CVPR 2023 (drone-image sparse conv — related work for adaptive inference).
- LightSeek-YOLO: *Mathematics* 13(19):3231, 2025 — the C2A competitor to beat in print.
- C2A: Nihal et al., ICPR 2024 (arXiv 2408.04922; Springer LNCS 10.1007/978-3-031-78341-8_10).
- RMH-YOLO: *Sensors* 25(22):7088 (NWD+InnerCIoU on YOLOv8n, VisDrone).
Doc written. Now saving the memory as you asked — the gathered research facts and the key numbers:

Write C:\Users\Dell\.claude\projects\d--Academics-thesis-folder\memory\project_novelty_direction.md
37 lines
---
name: project_novelty_direction
description: Branch novelty-direction (2026-07-04) — Mamba CLOSED with citations; new plan = C-WBF slice fusion + NWD loss + synthetic-to-real study; full doc in docs/2026-07-04_novelty_direction_research.md
metadata:
  type: project
---

**Branch `novelty-direction`** created 2026-07-04 off main for the post-defense novelty pivot.
Full cited research doc: `docs/2026-07-04_novelty_direction_research.md` (salvaged from a web-research
agent that hit the session limit mid-synthesis; per-paper verification fetches recovered).

**Mamba verdict — CLOSED, cited, do not reopen:** own runs (LocalWindowSSM null tie 0.6143 vs 0.6153 at
2.8x latency; AtrousSSM −10% mAP) + SeaDronesSee independent (MambaYOLO-L 25.5 < YOLO11-m 26.3, PLOS One)
+ Mamba-YOLO (2406.05835) still unaccepted preprint + MambaOut is CVPR 2025 but only argues for full SSM
*backbones* (frame the null via latency + neck-vs-backbone, never "MambaOut says SSM fails detection")
+ mamba-ssm kernels Linux-only. Also: C2A mAP50-95 saturates ~0.615 for EVERY architecture (label/paste
ceiling) — no architecture swap can move the headline metric.

**Ranked new directions (the plan):**
1. **C-WBF** — calibration-weighted WBF as the SAHI tile-merge + whole-frame/TTA ensemble. Verified gap:
   SAHI's combine.py has only NMS/NMM/GreedyNMM/LSNMS, no WBF. Own evidence of merge damage: SAHI
   slice256 gives +6.2pt very-tiny recall but F1 drops 0.850→0.829. Inference-only, ~2-3 days.
   WBF = Image & Vision Computing 2021 (arXiv 1910.13302). "CBF" the user half-remembered = WBF.
2. **NWD-hybrid loss** (blend CIoU+NWD in Ultralytics loss; NWD arXiv 2110.13389, ISPRS J P&RS Q1 ext;
   +6.7 AP AI-TOD; YOLOv8 replications: GS-YOLO +1.0 from NWD alone, RMH-YOLO +9.2 mAP50 VisDrone-n).
   Fits C2A: 47% instances <10px. Pilot = fine-tune from epoch125. RFLA (ECCV 2022, +4.0 AP) = stretch.
3. **Synthetic→real study** (June-29 §8 plan): label 15-20 frames/altitude of own 10/30/50m footage,
   self-training, pending drone-FP re-test of enriched model. 3-altitude real micro-benchmark is unique.
Paper spine: NWD training + C-WBF fusion + C2A→SARD→real validation + Mamba null = Q2, Q1-adjacent.

**Competitor marks on C2A:** C2A paper best = YOLOv9-e 0.6883 mAP / 0.8927 mAP50 (58M params);
LightSeek-YOLO (Mathematics 2025, 10.3390/math13193231) AP_small 0.478 vs student's 0.6156. ~14 citing
papers, all application-level — loss/fusion/cross-domain method space on C2A is unclaimed. C2A paper's
own C2A-only→SARD = 0.259 mAP vs student's measured 0.004-0.015 zero-shot — discrepancy to reconcile
(their table values still unverified). See [[project_paper_verdict]], [[project_mamba_injection_bug]],
[[project_finetune_enriched]].
Edit C:\Users\Dell\.claude\projects\d--Academics-thesis-folder\memory\MEMORY.md
Added 1 line
All done — branch open, research complete, everything saved. Here's the full picture:

The verdict on Mamba: closed. Don't spend another day on it.
Four independent lines of evidence now converge, and I've written them up with citations so you can defend this in the thesis:

Your own two genuine runs are negative — LocalWindowSSM tied CBAM+P2 exactly (0.6143 vs 0.6153 mAP50-95) at 2.8× the latency, and AtrousSSM was −10% mAP.
Independent aerial benchmark: on SeaDronesSee, MambaYOLO-L (25.5 AP) loses to plain YOLO11-m (26.3) — peer-reviewed PLOS One result.
The flagship papers don't support you: Mamba-YOLO is still an unaccepted preprint, and MambaOut (CVPR 2025) only argues for full SSM backbones — which you can't build anyway since the mamba-ssm CUDA kernels don't compile on Windows.
The deeper problem: your ablation shows mAP50-95 saturates at ~0.615 on C2A for every architecture — the dataset's pasted-box labels cap the metric. No architecture swap of any kind can move it. (Your vague memory about ViT/CNN is also confirmed: they were rejected on paper in April as too heavy/slow, never trained, and the prior art agrees they don't beat YOLO at this scale.)
What to do instead — and the module you half-remembered
The technique was WBF — Weighted Boxes Fusion (there is no established "CBF"). And here's the good part: I verified SAHI's source — it does not implement WBF (only NMS/NMM/GreedyNMM variants), and your own results show the merge step is the bottleneck: slicing buys +6.2 pts very-tiny recall but drops F1 from 0.850 to 0.829. So the ranked plan, all Windows-safe and untouched in your repo (grep-verified):

C-WBF — calibration-weighted WBF as the SAHI tile-merge (you already compute ECE/calibration in every eval — use it to weight the fusion). Inference-only, ~2–3 days, genuinely unclaimed in the literature.
NWD-hybrid loss — Wasserstein-based loss for tiny boxes (Q1-published, +6.7 AP on AI-TOD, replicated in YOLOv8 variants). C2A is 47% sub-10px objects — exactly its target regime, and it attacks your SARD AP_small weakness (0.294).
The synthetic→real study — your existing June-29 plan, now the paper's spine: label 15–20 frames per altitude of your own 10/30/50m footage (a micro-benchmark nobody else has), self-training, plus the still-pending drone-FP re-test of the enriched model.
Competitor check: only application papers cite C2A so far; the closest method paper (LightSeek-YOLO, Mathematics 2025) reports AP_small 0.478 vs your 0.616. The method space on C2A is wide open, and this three-part package is a coherent Q2 submission, Q1-adjacent if the real-footage deltas land.

Housekeeping
Branch novelty-direction is created and checked out; full cited writeup saved to docs/2026-07-04_novelty_direction_research.md (uncommitted — say the word and I'll commit it). Everything is also saved to persistent memory.
Your PC question: the completed fine-tune ran on PC-4 (RTX 4070 12GB, root D:\thesis_2007074) — the "A6000 run" folder name is legacy. If the machine you're on now has root D:\2007074, that's PC-3 per SETUP_PC3.md. Note the enriched fine-tune dipped SARD 0.917→0.898, and its actual purpose (drone false-positive reduction) is still unvalidated.
One caution: the web-research agent hit your session usage limit (resets 6:30pm Dhaka) — I salvaged its verified fetches, but two C2A-paper table values remain flagged as unverified in the doc; confirm them against the arXiv HTML before quoting in print.