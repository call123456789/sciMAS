---
name: biomni-pharmacology-rdkit-properties
description: Use to compute drug-likeness / physicochemical properties from a SMILES string using RDKit only.
x-scimas-role: drug-discovery-scientist
x-scimas-server: biomni-pharmacology
x-scimas-tools:
  - calculate_physicochemical_properties
---

# RDKit physicochemical property calculation

## When to use
- User has a SMILES string and wants MW, logP, HBA/HBD, TPSA, rotatable bonds, ring counts, etc.

## Limitations
- Requires RDKit; returns JSON error if RDKit is unavailable.
- For ADMET / binding-affinity predictions use ``DeepPurpose``-backed tools instead (Phase 2).
