---
name: analytical-molecular-descriptors
description: Use for drug-likeness and 2D molecular descriptors: Lipinski, H-bond counts, TPSA, rotatable bonds, aromatic rings, QED, logP, and logS.
x-scimas-role: analytical-chemist
x-scimas-server: chemistry-analytical
x-scimas-tools:
  - lipinski_properties
  - h_bond_counts
  - topological_polar_surface_area
  - rotatable_bond_count
  - aromatic_ring_count
  - qed_druglikeness
  - logp_rdkit
  - logS_rdkit
---
# Analytical Molecular Descriptors

Use this skill for medicinal-chemistry style descriptor calculations from SMILES, including Lipinski compliance, hydrogen bonding, TPSA, rotatable bonds, aromaticity, QED, logP, and solubility estimates.

Be explicit about descriptor definitions, especially when the prompt defines a custom score. Avoid implying these heuristics are experimental measurements.
