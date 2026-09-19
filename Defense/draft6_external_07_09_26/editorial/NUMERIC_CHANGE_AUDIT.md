# Numeric change audit

This note records the quantitative changes made during the 2026-08-16 editorial
pass. Its purpose is to distinguish correction of provenance from alteration of
experimental results.

## Values retained

The primary four-model ablation, dataset counts, per-size recall table, threshold
and calibration table, scene-split comparison, assignment study, real-data study,
latency, parameter counts, and energy measurements retain their recorded values.
The abstract now selects a smaller set of those same results; it does not introduce
a new experiment.

## Values removed from claims

- The repeated claim that 1280-px TTA raised very-tiny recall from 0.758 to 0.850
  at 60 ms was removed from the abstract, Chapters I and VII, and the final-model
  discussion. The archived study used an older Mamba+CBAM+P2 checkpoint, not the
  recommended CBAM+P2 checkpoint.
- A 75-km car-journey emissions analogy was removed because the conversion was not
  part of the saved measurement record. The measured 10.6 kWh and estimated
  9.0 kg CO2 values remain.
- Duplicate or over-interpreted numbers in the old SAHI/TTA table were removed when
  that table was rebuilt from the archived source summary.

## Values added or corrected

- The exploratory SAHI table now uses the archived measurements from
  `report_partB_stale_draft/FINDINGS_AND_SOURCES.md`: SAHI-512 has
  P/R/F1/F2/very-tiny-recall 0.867/0.865/0.866/0.865/0.808 and 187 ms;
  SAHI-256 has 0.790/0.872/0.829/0.854/0.829 and 534 ms. The associated TTA
  paragraph reports the archived changes of +1.64 AP50 and +5.39 COCO AP and
  notes degradation at 1920 px.
- The abstract includes the replicated assignment result already documented in
  `27-07-2026-Novelty-Lap-4/D2_SIGNIFICANCE.md`: mean sub-8-pixel recall change
  +1.6 +/- 0.5 points, and +2.08 points with a 95% bootstrap interval [1.77, 2.38]
  for the matched seed-0 pair.
- The software environment remains the one recorded in the saved `env.json`; its
  unusual versions were verified rather than normalised to more familiar values.

No detector score was changed merely to improve the narrative. Where the source
artifact did not support attribution to the recommended model, the claim was
removed or labelled exploratory.
