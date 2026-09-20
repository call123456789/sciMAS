---
name: pharma-drug-sda-md-mmpbsa
description: Use to set up and run molecular dynamics and end-point free energy calculations: OpenMM and GROMACS MM/PBSA protein–ligand and protein–protein workflows, coarse-grained OpenAWSM and GoCa simulations, BioEmu conformational sampling, and trajectory extraction.
x-scimas-role: physical-chemist
x-scimas-server: pharma-drug-sda
x-scimas-tools:
  - protein_openmm_md
  - prepare_protein_md
  - prepare_complex
  - run_mmpbsa
  - analyze_mmpbsa
  - gmx_mmpbsa_propro
  - openmm_extract_frames
  - prolif_md
  - openawsem_sim
  - openawsem_traj_extract
  - goca_pipeline
  - run_bioemu
  - extract_bioemu_structures
---
Run simulations and free-energy calculations. These are the slowest tools in
the set — expect minutes, and treat a timeout as "not finished", not
"failed": do not silently retry with a shorter budget and report the
truncated result as converged.

Protein–ligand path: `prepare_complex` → `run_mmpbsa` → `analyze_mmpbsa`.
Protein–protein path: `prepare_protein_md` → `gmx_mmpbsa_propro`.
`protein_openmm_md` runs a plain OpenMM simulation and returns its run
directory. For conformational ensembles use `run_bioemu`, then
`extract_bioemu_structures`; for coarse-grained work `openawsem_sim` /
`openawsem_traj_extract` or `goca_pipeline`.

MM/PBSA and MM/GBSA end-point estimates are sensitive to the setup and are
not rigorous free energies. State the protocol, the sampling, and the
approximations alongside any number.

The remote server keeps its own filesystem. Tools that take a `*_file_path`
argument expect a path **on the server**, not a local one. To use a local
structure: call `base64_to_server_file` with the file's base64 content and a
name, then pass the returned server path to the analysis tool, and
`server_file_to_base64` to bring a result back (files must be under 10MB).
