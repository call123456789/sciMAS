---
name: cell-stem-cell-polycomb
description: Use for Polycomb-mediated contact enrichment scoring, enhancer-promoter distance distribution, and Polycomb (PRC1 / PRC2) knockout simulation.
x-scimas-role: cell-biologist
x-scimas-server: biology-cell
x-scimas-tools:
  - calculate_polycomb_enrichment_score
  - analyze_enhancer_promoter_distance_distribution
  - simulate_polycomb_knockout_effect
---
# Cell Stem Cell Polycomb

Use this skill for embryonic-stem-cell 3D-genome analysis: Polycomb enrichment, enhancer-promoter distance summary statistics, and PRC1 / PRC2 knockout effect.

State the knockout complex and the target-gene list. Report enrichment multiplier (default 2.5× for polycomb targets).