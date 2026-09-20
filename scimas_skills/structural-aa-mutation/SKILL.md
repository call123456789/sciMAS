---
name: structural-aa-mutation
description: Use for amino-acid biochemical properties (hydropathy / volume / charge), single-point mutation effect score, candidate ranking, binding-site residue scoring, and active-site composition.
x-scimas-role: structural-biologist
x-scimas-server: biology-structural
x-scimas-tools:
  - get_amino_acid_properties
  - calculate_mutation_effect_score
  - compare_mutation_candidates
  - analyze_binding_site_residue
  - analyze_active_site_composition
---
# Structural AA Mutation

Use this skill for protein-level mutation analysis: per-residue properties, single mutation impact (Δhydropathy + Δvolume + Δcharge), multi-mutation ranking, and active-site composition.

Reject non-standard amino-acid letters. Mutations must follow "X###Y" format.