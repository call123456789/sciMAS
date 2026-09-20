---
name: cell-mitochondrial-assays
description: Use for mitochondrial ATP (luminescence → mM), ETC complex-activity (ΔA → U/mg), glucose-uptake assay appropriateness, and combined mitochondrial-health scoring.
x-scimas-role: cell-biologist
x-scimas-server: biology-cell
x-scimas-tools:
  - calculate_atp_concentration
  - calculate_complex_activity
  - assess_glucose_uptake_relevance
  - comprehensive_mitochondrial_assessment
---
# Cell Mitochondrial Assays

Use this skill for mitochondrial functional readouts: ATP concentration from a luminescence standard curve, ETC complex activity from Δabsorbance kinetics, glucose-uptake assay suitability, and a combined health-score.

State the standard-curve slope/intercept, extinction coefficient, time window, and protein loading. Reject negative or zero inputs.