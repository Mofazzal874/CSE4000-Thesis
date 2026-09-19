# ICCIT title and MSA-YOLO name audit — 2026-09-13

## Correction after full ICCIT source read (same session, user follow-up)

The initial recommendation below was based on selected thesis sections and stale August positioning, NOT the complete ICCIT manuscript. It is superseded for the current paper. Read the complete `Defense/ICCIT/draft3/main.tex` (abstract through conclusion, equations, tables and captions), the scope history in ICCIT_PLAN.md and DRAFT3_PLAN.md, and the implementation in lap-3 script 20. This was a textual/content review, not rendered-PDF verification or a fresh numerical audit.

The actual paper combines attention/P2/SSM ablations, the assignment blend, a scene-split audit, and real-footage adaptation. The August 27 originality audit already recognized YOLOv7-UAV as close prior art. The August 31 log records submission; no submission/title changes were performed. Current source and submitted PDF have not been established byte-equivalent.

Revised preferred title: **Attention, Scale, and Supervision for Tiny Human Detection in Aerial Search-and-Rescue Imagery**.

Broader alternative: **Tiny Human Detection in Aerial Disaster Imagery: Architecture, Supervision, and Real-World Evaluation**.

Label assignment chooses which candidate predictions learn each annotated object. The implementation blends clamped CIoU with normalized Wasserstein similarity within the existing task-aligned assigner, at gamma 0.5; it does not invent assignment or a new distance. This is a useful incremental adaptation and measured contribution, not established fundamental algorithmic novelty. The source reports seed-0 +2.08 pp recall below 8 px, three-seed mean +1.62 pp (SD 0.49), larger-band costs and essentially neutral AP50. Architecture and assignment experiments use different splits, so their effect-size ordering is not a same-protocol causal ranking.

Primary sources reverified by a search agent: TOOD (ICCV 2021), https://openaccess.thecvf.com/content/ICCV2021/papers/Feng_TOOD_Task-Aligned_One-Stage_Object_Detection_ICCV_2021_paper.pdf ; NWD-RKA (2022), https://arxiv.org/abs/2206.13996 ; YOLOv7-UAV (2023), https://www.mdpi.com/2079-9292/12/14/3141 . The last explicitly uses NWD/IoU for sample allocation and a 0.5:0.5 weight on TinyPerson. Some direct publisher requests failed; indexed primary-source text supported the findings. Therefore assignment can appear descriptively in a title, but should not be elevated into a claim of an unprecedented assignment method. The revised broad title fits the actual scope better.

## Initial provisional recommendation (superseded above)

Preferred focused ICCIT title: **Tiny-Aware Label Assignment for Human Detection in Aerial Disaster Imagery**.

Alternative emphasizing the target size: **Improving Tiny Human Detection in Aerial Disaster Imagery Through Label Assignment**.

If the ICCIT manuscript includes both assignment and the full supervised adaptation experiments: **Tiny Human Detection in Aerial Disaster Imagery with Label Assignment and Supervised Adaptation**.

If retaining the model family is preferred: **Tiny-Aware Label Assignment for YOLO-Based Human Detection in Aerial Disaster Imagery**.

These are editorial recommendations, not selected/approved titles. No new model acronym is necessary. State YOLO11m, CBAM, P2, and the assignment modification accurately in the abstract/method. Removing YOLO from the title does not remove the obligation to identify and cite the underlying model.

## Local evidence and scope

Read START_HERE.md, lap-3 README and RESEARCH_PROTOCOL, active lap-4 README, lap-4 2026-08-17 ICCIT/JSTARS risk audit, and (read-only) Defense/draft6_07_09_26 frontmatter abstract/title and methodology assignment subsection.

The current thesis title is MSA-YOLO: A Multi-Scale Attention Enhancement of YOLO for Tiny-Human Detection in Aerial Search-and-Rescue Imagery. The final abstract attributes the useful architecture change to stride-4 prediction, and reports about +1.6 percentage points below-eight-pixel recall from assignment across seeds. Lap-4 closes FCCG as a null result. Thus attention-led branding misaligns with the strongest supported contribution. The August ICCIT strategy already recommends a focused assignment paper; the title should match the actual manuscript scope, not attempt to contain the entire thesis.

Use disaster imagery as the application domain, without claiming robust unseen-disaster generalization. Avoid novel, state-of-the-art, occlusion-aware, lightweight, or real-time in the title unless the submitted evidence specifically supports the corresponding claim. Adaptation here is supervised, not an established unsupervised pipeline.

## Verified name collisions (primary publisher pages)

1. Su et al., **MSA-YOLO: A Remote Sensing Object Detection Model Based on Multi-Scale Strip Attention**, Sensors, published 30 July 2023. https://www.mdpi.com/1424-8220/23/15/6811 ; DOI 10.3390/s23156811.
2. **Pedestrian detection in aerial image based on convolutional neural network with attention mechanism and multi-scale prediction**, Scientific Reports, 2025. Explicitly names its method **Multi-Scale Attention YOLO (MSA-YOLO)** and addresses aerial pedestrians, including small/occluded targets. This is the closest naming collision. https://www.nature.com/articles/s41598-025-27441-8 ; DOI 10.1038/s41598-025-27441-8.
3. Huang et al., **MSA-YOLO: An Optimized UAV Object Detection Algorithm for Low-Visibility Maritime**, Remote Sensing, published 23 June 2026. Names its model Multi-Scale Adaptive YOLO. https://www.mdpi.com/2072-4292/18/13/2065 ; DOI 10.3390/rs18132065.

Conclusion: abandon MSA-YOLO for this paper because of substantial discoverability/identity confusion. A name collision alone does not establish duplicated research or plagiarism. Renaming alone does not establish methodological novelty either.

## IEEE guidance

IEEE Author Center conference guidance says titles should be specific, concise, descriptive, include useful search keywords, and avoid unnecessary new/novel wording: https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/structure-your-paper/ .

An IEEE-hosted conference template advises avoiding abbreviations in titles unless unavoidable: https://edu.ieee.org/ng-futo/wp-content/uploads/sites/347/2018/02/Standard-IEEE-Format-for-Writing-Papers.pdf . This is template guidance, not proof that YOLO titles are prohibited. No requirement to invent a model name or include YOLO was identified in the checked guidance.

ICCIT Bangladesh submission page checked: https://iccit.org.bd/2026/online-paper-submission-system/ . Initial retrieval succeeded; subsequent retrieval failed. No special ICCIT title-length rule is asserted here.

Searches included exact MSA-YOLO, MSA-YOLO detection, IEEE title guidance, and an exact candidate search for Tiny-Aware Label Assignment for Human Detection in Aerial Search-and-Rescue Imagery. No exact match for that candidate surfaced, but this is not a guarantee of uniqueness; recommended alternatives are not certified globally unique.

## Review

Thesis files were read only. Refreshed START_HERE RIGHT NOW and active lap-4 checklist with this audit. No commit made. Suggested commit message: `docs: audit ICCIT title options and MSA-YOLO name collisions`.
