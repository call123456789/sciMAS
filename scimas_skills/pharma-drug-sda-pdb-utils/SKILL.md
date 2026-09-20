---
name: pharma-drug-sda-pdb-utils
description: Use to inspect a PDB structure or validate a protein sequence: basic statistics, atom composition, quality metrics, geometric properties, and sequence physicochemical properties.
x-scimas-role: structural-biologist
x-scimas-server: pharma-drug-sda
x-scimas-tools:
  - calculate_pdb_basic_info
  - calculate_pdb_composition_info
  - calculate_pdb_quality_metrics
  - calculate_pdb_structural_geometry
  - calculate_protein_sequence_properties
  - is_valid_protein_sequence
---
Characterise a structure or sequence before doing anything heavier with it.

Run `is_valid_protein_sequence` before submitting a sequence to any
prediction tool, and `calculate_pdb_quality_metrics` before trusting a
downloaded structure. These are cheap checks that catch most bad inputs.

Cα-based geometry and composition statistics describe the deposited model.
They do not by themselves establish experimental quality.

The remote server keeps its own filesystem. Tools that take a `*_file_path`
argument expect a path **on the server**, not a local one. To use a local
structure: call `base64_to_server_file` with the file's base64 content and a
name, then pass the returned server path to the analysis tool, and
`server_file_to_base64` to bring a result back (files must be under 10MB).
