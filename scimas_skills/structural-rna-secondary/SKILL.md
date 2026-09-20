---
name: structural-rna-secondary
description: Use for RNA sequence parsing / validation, Watson-Crick base-pair detection, structure-complexity scoring, RNA-type classification, and catalytic-activity prediction.
x-scimas-role: structural-biologist
x-scimas-server: biology-structural
x-scimas-tools:
  - parse_rna_sequence
  - detect_base_pairs
  - calculate_structure_complexity
  - classify_rna_type
  - predict_catalytic_activity
---
# Structural RNA Secondary

Use this skill for RNA secondary-structure analysis: validation, Watson-Crick pair detection (no pseudoknots), complexity scoring with an approximate ΔG, RNA-type classification by size heuristic, and catalytic-activity heuristic.

State the minimum stem length (default 3). Truncate U vs. T automatically.