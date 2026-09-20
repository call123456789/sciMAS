---
name: computational-pubchem-identity
description: Use for PubChem name, CID, canonical SMILES, and molecular formula lookup.
x-scimas-role: computational-chemist
x-scimas-server: chemistry-computational
x-scimas-tools:
  - pubchem_cid_by_name
  - pubchem_smiles_by_cid
  - chemical_name_to_smiles
---
# Computational PubChem Identity

Use this skill for resolving compound names to PubChem CIDs, SMILES, and formulas.

Mention network dependency and verify that returned structures match the intended compound or isomer when the prompt is stereochemical or regioisomer-specific.
