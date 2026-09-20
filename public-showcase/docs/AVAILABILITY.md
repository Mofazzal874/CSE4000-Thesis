# Artifact availability and attribution

This is a self-contained research companion within [CSE4000-Thesis](https://github.com/Mofazzal874/CSE4000-Thesis): its documents, figures, result exports, and videos can be inspected without running the detector implementation. A portable copy is also included in the local professor handover.

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

**CSE4000-Thesis is a public repository.** Anyone can inspect its tracked files and accessible Git history, including source code outside this folder. The exclusions in the table describe this companion's contents; they are not access restrictions on the enclosing repository.

GitHub visibility applies to a repository, not individual folders or branches. A README link, a selected download, or a `.gitignore` rule cannot make already tracked source private. [GitHub repository visibility](https://docs.github.com/en/repositories/creating-and-managing-repositories/about-repositories).

A ZIP containing only this companion is useful for offline reading without installing the research code. It does not revoke access to code already available in the public repository.
