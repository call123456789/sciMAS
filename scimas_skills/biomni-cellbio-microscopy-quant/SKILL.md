---
name: biomni-cellbio-microscopy-quant
description: Use to quantify cell-cycle phases from Calcofluor-white microscopy, cluster cell-motility tracks, or analyze mitochondrial morphology & membrane potential from fluorescence images.
x-scimas-role: cell-biologist
x-scimas-server: biomni-cell_biology
x-scimas-tools:
  - quantify_cell_cycle_phases_from_microscopy
  - quantify_and_cluster_cell_motility
  - analyze_mitochondrial_morphology_and_potential
---

# Microscopy quantification (cell cycle, motility, mitochondria)

## When to use
- User has Calcofluor-stained microscopy images and wants G1 / S / G2-M proportions.
- User has cell-track time-series and wants motility-cluster classification.
- User has fluorescence images of mitochondria and wants morphology / membrane-potential quantification.

## Limitations
- Requires ``cellpose`` / ``scikit-image`` (NOT installed by default).
- Classifier is a heuristic random-forest trained on synthetic features; treat as exploratory.
