---
name: biomni-immuno-proliferation-cytokine
description: Use to analyze CFSE-based proliferation, intracellular cytokine staining in CD4 T-cells, EBV antibody titers, CNS lesion histology, or immunohistochemistry images.
x-scimas-role: cell-biologist
x-scimas-server: biomni-immunology
x-scimas-tools:
  - analyze_cfse_cell_proliferation
  - analyze_cytokine_production_in_cd4_tcells
  - analyze_ebv_antibody_titers
  - analyze_cns_lesion_histology
  - analyze_immunohistochemistry_image
  - analyze_bacterial_growth_curve
---

# Proliferation, cytokine & histology

## When to use
- User has CFSE dye-dilution data and wants division-index estimates.
- User has intracellular-cytokine staining and wants per-cell-type cytokine frequency.
- User has EBV antibody titer panel and wants recent / past infection classification.
- User has CNS-lesion histology slides and wants lesion quantification.
- User has IHC images and wants marker-positive cell counts.
- User has a bacterial growth curve and wants OD-vs-time fit.

## Limitations
- Requires ``flowkit`` / ``scikit-image`` (NOT installed by default).
- Histology / IHC image analysis is heuristic — verify with a trained pathologist before publication.
