---
name: computational-3d-conformers
description: Use for 3D conformer generation, force-field optimization, multiple conformers, and 3D molecular properties.
x-scimas-role: computational-chemist
x-scimas-server: chemistry-computational
x-scimas-tools:
  - generate_3d_conformer
  - mol_analyzer_generate_3d
  - optimize_geometry
  - get_3d_properties
  - generate_multiple_conformers_with_optimization
  - rdkit_generate_3d
---
# Computational 3D Conformers

Use this skill for ETKDG/RDKit 3D embedding, MMFF or UFF geometry optimization, conformer ensemble generation, and 3D shape descriptors.

State force field, convergence, conformer count, and whether a single conformer or ensemble was used. Use consistent protonation and stereochemistry across comparisons.
