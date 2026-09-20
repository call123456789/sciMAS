---
name: structural-pdb-composition
description: Use for PDB text parsing, atom / residue / chain counting, ligand / water / protein classification, and per-chain composition summary.
x-scimas-role: structural-biologist
x-scimas-server: biology-structural
x-scimas-tools:
  - parse_pdb_structure
  - classify_residue_type
  - count_ligand_chains
  - analyze_structure_composition
---
# Structural PDB Composition

Use this skill for PDB text-only analysis (no network): atom counts, residue classification, ligand vs. water vs. protein breakdown, and per-chain summaries.

Pass a PDB text block, not a file path. Optionally strip hydrogens before counting.