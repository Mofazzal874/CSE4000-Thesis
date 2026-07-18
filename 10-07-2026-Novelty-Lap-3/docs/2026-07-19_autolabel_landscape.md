# Auto-labeling landscape verification (2026-07-19)

Web-verified per lap-3 standing rule (no memory-only claims). Context: decide annotation strategy
for ~500-1000 new frames (own drone 13-55 px people, staged lying-down, tree-occluded campus,
real-disaster eval-only videos). Existing assets: YOLO11m+CBAM+P2 (~0.85 mAP50 C2A), Roboflow account.

## 1. Roboflow Label Assist + Auto Label (state 2026)

**Verdict: both exist and are mature; Label Assist CAN use your own previously-trained Roboflow
model (any prior version), and the train→assist loop is first-class ("Instant Models" retrain as
you approve annotations). Auto Label = text-prompted foundation-model batch labeling
(Grounding DINO default, SAM-family for masks).**

- Sources: https://docs.roboflow.com/annotate/ai-labeling/model-assisted-labeling (Label Assist),
  https://docs.roboflow.com/annotate/ai-labeling/automated-annotation-with-autodistill (Auto Label),
  https://blog.roboflow.com/launch-auto-label/ ,
  https://docs.roboflow.com/changelog/explore-by-month/june-2025/use-instant-models-in-label-assist
- Label Assist lets you pick "a previous version of a trained model or a public model on Roboflow
  Universe" as the annotation assistant — so our own CBAM+P2-lineage checkpoints (retrained inside
  Roboflow, or an uploaded-weights deploy) can pre-label; per-image suggestions are then corrected
  by hand in the annotate UI.
- Auto Label is batch/zero-touch: text prompt ("person") → Grounding DINO (default) or
  Grounded-SAM/other foundation models label whole batches; also supports using a Roboflow-trained
  model. Free-tier note found: Auto Label capped at 1,000 images per labeling job (beta terms);
  free workspaces have overall image-capacity limits (~250k images) — fine for our 500-1000 frames.
- June-2025 changelog: "Instant Models" (auto-trained as you approve annotations) usable directly
  in Label Assist — i.e., the annotate→retrain-assist→annotate-faster loop is built in without
  manual retrain runs.
- **Implication for us:** the pre-label-then-correct workflow needs zero new tooling — upload
  frames, point Label Assist at our own model for the 30 m/50 m easy tiers, correct by hand.
  BUT our-model-assisted labeling must NOT touch the disaster eval benchmark (see item 6).

## 2. SAM 3 (Meta) — real, released 2025-11-19

**Verdict: SAM 3 is REAL (arXiv 2511.16719, released Nov 19 2025), does Promptable Concept
Segmentation — a short noun phrase ("person") returns masks+boxes+IDs for ALL matching instances
in image or video. SAM 3.1 follow-up exists (Meta AI blog, faster real-time video). Open
checkpoints + inference/fine-tune code under SAM License.**

- Sources: https://arxiv.org/abs/2511.16719 (SAM 3 paper), https://ai.meta.com/blog/segment-anything-model-3/
  (SAM 3.1 blog), https://docs.ultralytics.com/models/sam-3 , https://blog.roboflow.com/what-is-sam3/
- Key upgrade over SAM 1/2: SAM 1/2 segment ONE object per geometric prompt (click/box); SAM 3
  takes an open-vocabulary concept (text or image exemplar) and returns every instance — it is a
  detector+segmenter, so "person" → instance masks whose tight bounding boxes are directly usable
  as detection labels.
- SAM 2/2.1 remain the click/box-promptable interactive models (video-capable); they cannot find
  objects from text alone — pairing with a text-grounded detector (Grounded SAM 2) was the 2024-25
  workaround that SAM 3 subsumes.
- Small-object/aerial performance evidence: pending (item 5).
- **Implication for us:** SAM 3 with prompt "person" is the strongest zero-shot candidate for
  frames our own model was never trained on (lying-down poses, tree occlusion) — but expect the
  <16 px tier to be its weak zone; verify against item-5 evidence before trusting it on 50 m footage.

## 3. Open-vocabulary pre-labelers 2025-26

**Verdict: the IDEA-Research top line (Grounding DINO 1.5/1.6 Pro, DINO-X, T-Rex2) is API-gated,
not open-weights — repo is literally "Grounding-DINO-1.5-API". Free/local options: original
Grounding DINO (open), YOLO-World (fast, notably lower zero-shot accuracy), YOLOE, and now SAM 3
(open checkpoints). Best free LOCAL pre-labeler for "person" today = SAM 3 (accuracy + open
weights + boxes from instance masks), with original Grounding DINO as the lighter fallback.**

- Sources: https://github.com/IDEA-Research/Grounding-DINO-1.5-API ,
  https://arxiv.org/abs/2405.10300 (GD 1.5), https://arxiv.org/abs/2411.14347 (DINO-X),
  https://docs.ultralytics.com/models/sam-3
- Grounding DINO 1.5 Pro: 54.3 AP COCO / 55.7 AP LVIS zero-shot (trained on 20M grounded images)
  — strongest text-prompt detector line, but cloud API (paid tokens), so unusable as an offline
  lab-PC batch labeler and a data-governance question for thesis data.
- YOLO-World: open, ~real-time, but zero-shot accuracy trails Grounding DINO badly (reported
  ~35.4 vs 52.5 AP in comparable settings) — poor fit for hard tiny/occluded instances; fine only
  as a cheap first pass. YOLOE (2025) is the Ultralytics-supported successor in the same
  speed-first class.
- DINO-X adds universal prompting (text+visual), T-Rex2 adds visual-exemplar prompting (good for
  hard-to-describe classes) — both via IDEA API.
- **Implication for us:** "person" is an easy prompt, so we don't need visual-prompt exotica; run
  SAM 3 locally (PC-1/PC-2 GPUs are ample) for zero-shot pre-labels, skip paid APIs.

## 4. Free/local annotation tools with auto-label

**Verdict: all three major open tools now embed the SAM family incl. SAM 3; X-AnyLabeling is the
most model-rich desktop option and added local (ONNX) SAM 3 text-grounded labeling in April 2026.**

- X-AnyLabeling (https://github.com/CVHub520/X-AnyLabeling): Python/Qt desktop tool, AI-first;
  ships SAM/SAM2/SAM3, YOLO, Grounding DINO, VLMs; Apr-2026 release = client-side ONNX SAM 3
  text-grounded segmentation (example: examples/grounding/sam3/README.md). Fully local — best
  Roboflow alternative if data must stay off-cloud.
- CVAT (https://www.cvat.ai/resources/blog/automated-data-labeling-guide): self-hostable; SAM3
  interactive segmentation + custom-model auto-annotation via Nuclio/Hugging Face — heavier to
  operate (server + serverless functions) but supports plugging in OUR OWN YOLO11 checkpoint.
- Label Studio ML backend (https://labelstud.io/guide/ml_tutorials/grounding_sam ,
  https://github.com/HumanSignal/label-studio-ml-backend): GroundingDINO+SAM zero-shot example
  backend; text prompt per task batch; custom backends possible; middle ground in setup effort.
- **Implication for us:** we already pay zero and know Roboflow — stay there for the main flow;
  keep X-AnyLabeling as the local fallback (e.g., disaster eval videos we may not want uploaded,
  or if Roboflow free-tier caps bite).

## 5. Evidence: assisted-annotation speedup + tiny-object failure modes

**Verdict (speedup): consistent vendor/industry/research numbers of 40-70% time reduction for
correct-a-prelabel vs draw-from-scratch (not the mythical "10x" except in marketing); correcting
a slightly-off box is faster AND more consistent than drawing one.**
**Verdict (tiny objects): foundation models measurably degrade on small aerial objects; the
standard mitigation is tiling/SAHI at inference. No evidence any zero-shot model reliably finds
<16 px people full-frame — assume our own domain-trained model beats zero-shot there.**

- Speedup sources: https://scematics.io/resource/blogs/ai-assisted-vs-manual-annotation-cost-speed-quality-comparison
  (2026 industry analysis: ~50% manual-effort cut, ~4x cost cut),
  https://imerit.ai/resources/blog/pre-labeling-automation-accelerating-ai-annotation-with-smarter-first-drafts/
  (pre-labeling "up to 70%" when reviewing/tweaking suggestions),
  https://labelbox.com/guides/how-to-pre-label-images-using-model-assisted-labeling-in-your-annotation-project/ .
  These are vendor-side numbers; treat as 2-3x practical, not 10x. Caveat from the same
  literature: bad pre-labels (hallucinated/missed boxes) can be SLOWER than scratch — speedup
  assumes the assist model is decent on that data slice.
- Tiny-object sources: https://isprs-archives.copernicus.org/articles/XLVIII-M-6-2025/23/2025/
  (ISPRS 2025: Grounded-SAM F1 0.83 on aerial buildings but small structures were the failure
  case; SAM+CLIP collapsed to 0.49 IoU and "struggled with delineating smaller ones");
  https://arxiv.org/abs/2605.24639 (DisDop 2026: open-vocab detectors need domain-prior
  distillation to work on aerial object detection — direct evidence zero-shot underperforms in
  this domain); https://github.com/idea-research/grounded-sam-2 (README explicitly recommends
  SAHI slicing for "high resolution images with dense small objects (e.g. 4K images)" — i.e.,
  even the authors treat full-frame small-object inference as a known weakness).
- **Implication for us:** expect big assist gains on 10 m/30 m frames (people 25-55 px) and on
  poses/occlusion where SAM 3 sees "person"-like evidence; expect zero-shot to MISS much of the
  50 m <16 px tier unless run tiled (SAHI-style 4K slicing) — and our own CBAM+P2 model, trained
  on 63%-sub-16px C2A, is likely the stronger pre-labeler for exactly that tier. Recall (missed
  boxes) is the dangerous error type: a missed tiny person that the human also misses becomes a
  silent label hole, so tiny-tier frames need a deliberate second pass, not casual review.

## 6. Eval-set purity: don't label the test set with the model under evaluation

**Verdict: circularity is a documented, citable failure class — evaluating a model on labels it
(or a sibling) produced yields "false perfection". Best-practice protocols state primary
annotators must not see model outputs when building evaluation labels.**

- Citations: https://arxiv.org/abs/2106.12417 ("False perfection in machine prediction:
  Detecting and assessing circularity problems in machine learning") — the direct citation for
  why model-generated eval labels inflate scores; https://arxiv.org/abs/2103.14749 (Northcutt
  et al., NeurIPS 2021, "Pervasive Label Errors in Test Sets Destabilize Machine Learning
  Benchmarks") — why test-label quality matters more than train-label quality; recent annotation
  protocols (e.g. arXiv 2605.26070) explicitly require that "primary annotators should not have
  access to model outputs" and final labels come from human adjudication.
- The failure mode for detection specifically: pre-labels anchor the annotator (acceptance bias)
  — boxes the model missed stay missing, boxes it hallucinated get accepted, and the eval then
  measures agreement-with-self, not detection quality. FP/FN structure of the eval set becomes
  correlated with the evaluated model's error structure.
- **Implication for us:** the earthquake/flood benchmark MUST be labeled without assistance from
  our YOLO lineage (and ideally without any detector we later compare against). Interactive
  class-agnostic tools (SAM 2/3 click-to-refine a HUMAN-found person) are acceptable — the human
  finds, the tool only tightens geometry. Two-pass independent review; document the protocol in
  the paper. Same rule as our existing "test frames are NEVER trained on" guard, extended to
  "never machine-proposed either".

## 7. Semi-automatic pipelines in recent aerial datasets (precedents)

**Verdict: peer-reviewed precedent exists for model-assisted labeling of TRAIN data with
human-in-the-loop verification — safe to cite as methodology, not something to hide.**

- RealDroneVision (WACV 2026): 173k images "constructed via a semi-automatic pipeline...
  self-annotated labeling from videos, enhanced with a human-in-the-loop to iteratively reduce
  false positives and false negatives" (138k train / 34k test).
  https://openaccess.thecvf.com/content/WACV2026/papers/Sivapuram_RealDroneVision_Dataset_and_Architecture_Advancements_for_Small-Object_Drone_Detection_WACV_2026_paper.pdf
- VTSaR (IEEE JSTARS 2025, RGB-T aerial person detection): note — ISCP = "instance segmentation
  for copy-paste", a mechanism in VTSaRNet for extracting person boundaries to BUILD SYNTHETIC
  samples (copy-paste augmentation), NOT an assisted-annotation/verification protocol. Do not
  cite it as a labeling workflow. https://ieeexplore.ieee.org/document/10833840/ ,
  https://github.com/zxq309/VTSaR
- Related: "Aerial Person Detection for Search and Rescue: Survey and Benchmarks"
  (https://spj.science.org/doi/10.34133/remotesensing.0474) — useful survey to mine for how APD
  datasets report annotation procedure.
- **Implication for us:** describe our train-set pipeline exactly like RealDroneVision does
  (pre-label → human FP/FN reduction loop), and keep the eval benchmark fully manual — that split
  of rigor matches current venue norms.

## 8. Active-learning / batch bootstrapping (annotate → retrain assist → annotate faster)

**Verdict: first-class in Roboflow as of 2025-26 — "Instant Models" auto-retrain on every
approved batch and immediately become the Label Assist model; no manual retrain step needed.**

- Sources: https://docs.roboflow.com/train/roboflow-instant , https://blog.roboflow.com/roboflow-instant/ ,
  https://blog.roboflow.com/active-learning-workflow/
- Roboflow Instant: few-shot model trained free from ~half-dozen labeled images; auto-retrains
  whenever a new annotation batch is approved; usable immediately in Label Assist and Workflows.
  The Dataset Upload inference block closes the loop from deployed inference back into batches.
- This exactly answers the "train a small labeler from 100-200 seed frames" option: the platform
  already does it automatically, and it can be combined with (not instead of) our full custom model.
- **Implication for us:** order the annotation queue so each approved batch makes the next batch
  cheaper: start with the tier where assist is strongest (10 m), approve, let Instant absorb our
  lying-down/occluded corrections, then move to harder tiers.

## Recommended stack for this project

- **Disaster eval benchmark (earthquake/flood videos): 100% MANUAL.** No pre-labels from our YOLO
  lineage or any comparison model (circularity — arXiv 2106.12417). Allowed: SAM 2/3 interactive
  click-to-box AFTER a human has found the person, to tighten geometry. Two independent passes +
  adjudication; write the protocol into the paper. Label locally (X-AnyLabeling) or in a separate
  Roboflow project with Label Assist OFF.
- **Easy train tiers (10 m / 30 m, people ~25-55 px): Roboflow Label Assist with OUR OWN model**
  (epoch125 / CBAM+P2 lineage uploaded or retrained in Roboflow), correct-and-approve; expect
  2-3x speedup (40-70% literature range). Let Roboflow Instant auto-retrain per approved batch so
  assist improves as we go (item 8).
- **Tiny tier (50 m, <16 px people): pre-label with OUR model at low confidence, run TILED
  (SAHI-style) on the 4K frames** — zero-shot foundation models are documented to degrade here
  (items 2/5). Mandatory dedicated second human pass hunting MISSES (silent label holes are the
  killer failure mode for this tier).
- **Novel-appearance tiers (lying-down staged, tree-occluded campus): dual pre-label** — our model
  (will undershoot; these poses are out of its training distribution) UNION SAM 3 text prompt
  "person" run locally (open checkpoints; via X-AnyLabeling ONNX or a short script), human
  adjudicates the merged proposals. These corrected frames are exactly the highest-value
  fine-tuning data, so budget the most careful review here.
- **Do NOT train a separate 100-200-frame seed labeler** — dominated on every axis: Roboflow
  Instant already implements the seed→assist loop automatically, and our full model + SAM 3 are
  both stronger than any small seed model would be.
- **Documentation:** log per-tier provenance (which model pre-labeled, conf threshold, tiling,
  correction rules, reviewer passes) in the dataset doc from day 1 — RealDroneVision (WACV 2026)
  is the citable precedent for a semi-automatic train-set pipeline with a clean manual test set.
