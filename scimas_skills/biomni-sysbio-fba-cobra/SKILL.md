---
name: biomni-sysbio-fba-cobra
description: Use to run a flux-balance analysis on a COBRA SBML model with custom constraints / objective; returns per-reaction flux + objective value.
x-scimas-role: structural-biologist
x-scimas-server: biomni-systems_biology
x-scimas-tools:
  - perform_flux_balance_analysis
  - compare_protein_structures
---

# Flux-balance analysis (FBA)

## When to use
- User has a COBRA-format SBML model and wants FBA.
- User has two PDB files and wants a per-residue RMSD / contact map comparison.

## Limitations
- FBA requires COBRApy; not installed by default — wrapper returns JSON error if missing.
- Protein comparison does a coarse alignment; not TM-align.
