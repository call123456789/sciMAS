---
name: analytical-formula-identity
description: Use for molecular identity calculations: formula, molecular weight, element composition, InChI, InChIKey, and heavy atom counts.
x-scimas-role: analytical-chemist
x-scimas-server: chemistry-analytical
x-scimas-tools:
  - molecular_formula_from_smiles
  - molecular_weight
  - molecular_formula_rdkit
  - molecular_weight_rdkit
  - element_composition_rdkit
  - inchi_from_smiles
  - inchikey_from_smiles
  - heavy_atom_count
---
# Analytical Formula And Identity

Use this skill when the task needs molecular formula, molecular mass, atom counts, elemental composition, or structure identifiers from SMILES.

Prefer RDKit-backed tools when valid SMILES are available and exact chemistry parsing matters. Use the stdlib formula or weight tools for simple formulas or fallback estimates.
