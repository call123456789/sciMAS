---
name: pharma-drug-sda-docking
description: Use to dock molecules and peptides into protein structures and to detect and score binding pockets: HDOCK, KarmaDock, QuickVina2-GPU, P2Rank, fpocket, EquiScore, and ProLIF pose summaries.
x-scimas-role: computational-chemist
x-scimas-server: pharma-drug-sda
x-scimas-tools:
  - hdock_tool
  - karmadock_tool
  - molecule_docking_quickvina_fullprocess
  - convert_pdb_to_pdbqt_dock
  - prolif_docking
  - equiscore_pipeline
  - equiscore_pocket
  - equiscore_screen
  - pred_pocket_prank
  - fpocket_toolkit
---
Dock ligands or peptides and score the poses.

If the binding site is unknown, detect pockets first with
`pred_pocket_prank` or `fpocket_toolkit`; otherwise docking tools either take
a box or select a site themselves. `convert_pdb_to_pdbqt_dock` prepares a
receptor when a tool needs PDBQT. `prolif_docking` summarises poses as
interaction fingerprints, and `equiscore_*` re-scores them (pocket →
pipeline → screen).

Docking scores rank poses within one run. They are not binding affinities and
do not compare across targets or protocols — say so when reporting a "best"
pose.

The remote server keeps its own filesystem. Tools that take a `*_file_path`
argument expect a path **on the server**, not a local one. To use a local
structure: call `base64_to_server_file` with the file's base64 content and a
name, then pass the returned server path to the analysis tool, and
`server_file_to_base64` to bring a result back (files must be under 10MB).
