---
name: pharma-drug-sda-structure-prep
description: Use to clean up and prepare protein structures: repair missing atoms and residues, rebuild incomplete models, repack sidechains, extract chains, and convert CIF to PDB.
x-scimas-role: structural-biologist
x-scimas-server: pharma-drug-sda
x-scimas-tools:
  - fix_pdb
  - pulchura_rebuild
  - pack_sidechains
  - extract_and_save_chains
  - extract_pdb_chains
  - convert_complex_cif_to_pdb
---
Prepare a structure for downstream simulation, docking, or design.

A typical order: `convert_complex_cif_to_pdb` (if needed) →
`fix_pdb` for missing atoms, hydrogens and heterogens →
`pulchura_rebuild` for grossly incomplete backbones →
`pack_sidechains` to restore full-atom sidechains.

`extract_pdb_chains` returns sequences; `extract_and_save_chains` writes chain
files. Report which repairs were applied and what was removed — deleting
heterogens or repairing residues changes what the structure represents.

The remote server keeps its own filesystem. Tools that take a `*_file_path`
argument expect a path **on the server**, not a local one. To use a local
structure: call `base64_to_server_file` with the file's base64 content and a
name, then pass the returned server path to the analysis tool, and
`server_file_to_base64` to bring a result back (files must be under 10MB).
