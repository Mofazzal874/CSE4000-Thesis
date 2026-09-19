# Artifact availability and attribution

This is a self-contained research companion: its documents, figures, result exports, and videos can be inspected without the detector implementation. The files have been prepared locally. External publication status is recorded separately in the release notes; the package itself does not assume a particular host.

| Artifact | Included here | What the reader receives |
|---|---|---|
| Research narrative and diagrams | Yes | Completed-study overview and conceptual method |
| Thesis | Yes | Original final submission PDF, 88 pages |
| Defense presentation | Yes | Original 43-page PDF, six previews, and reading notes |
| Selected results | Yes | Nine CSVs and two original bootstrap summary JSONs |
| Result checker | Yes | Independent standard-library analysis and integrity checks |
| Drone demonstrations | Yes | Two complete annotated MP4s and their provenance |
| Successes and failures | Yes | Five selected archived detection overlays with measured counts |
| Training and detector evaluation implementation | No | Maintained outside this companion |
| Weights and exported model binaries | No | Retained in the complete research archive |
| Raw datasets and annotation dumps | No | Retained in the complete research archive |
| Editable presentation | No | Full PPTX, including embedded media, remains in the professor handover |

The package supports inspecting reported results and checking arithmetic. Re-evaluation, bootstrap recomputation from predictions, and retraining require additional research artifacts.

## Attribution

Research text, original diagrams, analysis, and own-drone demonstration outputs: Md Mofazzal Hosen, Department of Computer Science and Engineering, KUET, 2026. Supervised by Prof. Dr. Sk. Md. Masudul Ahsan.

The comparison/failure images use the **C2A dataset** by R. A. Nihal, B. Yen, K. Itoyama, and K. Nakadai. The original paper and dataset references are retained in the thesis bibliography. Their underlying image rights are separate from the author's detection overlays. The source archive records no explicit C2A redistribution licence. These few annotated research examples do not confer rights to redistribute the dataset; obtain source data from its authors. The raw dataset and reference-paper collection are not included.

The detector builds on Ultralytics YOLO11 and established methods credited in the thesis. Publicly presenting their use does not transfer their software or data licences. Third-party illustrations and citations embedded in the original thesis and presentation retain their original attribution.

No blanket open-source licence is applied to this mixed-material companion. Sharing access is distinct from granting unrestricted reuse rights. Cite the thesis when discussing these findings, and obtain applicable permission for reuse of underlying third-party material.

## Repository boundary

This folder contains an explicit selection of research artifacts. It grants no technical access to files outside the folder. However, publishing a companion does not change the visibility of any existing research repository. The owner must manage the original repository's access separately.
