---
name: genetics-epistasis-and-redundancy
description: Use for epistasis coefficient (ε), single / double-mutant effect decomposition, transcription-factor inference, and gene-redundancy classification.
x-scimas-role: geneticist
x-scimas-server: biology-genetics
x-scimas-tools:
  - calculate_epistasis_coefficient
  - analyze_single_mutant_effects
  - identify_transcription_factor
  - analyze_gene_redundancy
  - determine_epistatic_relationship
---
# Genetics Epistasis And Redundancy

Use this skill for two-gene epistasis analysis (additive model): ε > 0.5 → synergistic, ε < −0.5 → antagonistic, |ε| ≤ 0.5 → multiplicative.

State the genetic model and the threshold for classification. Report pairwise triples explicitly.