---
name: pharma-drug-sda-binding-affinity
description: Use to estimate binding affinity and analyse interactions: Boltz-2 affinity prediction, FoldX stability and interface analysis, ProLIF interaction fingerprints, and residue numbering between UniProt, PDB, and tool conventions.
x-scimas-role: computational-chemist
x-scimas-server: pharma-drug-sda
x-scimas-tools:
  - pred_binding_affinity_boltz2
  - foldx_tool
  - analyze_protein_ligand_interactions
  - interaction_visualizer
  - prolif_pdb
  - prolif_protein_protein
  - residue_mapper
---
Estimate affinity and characterise interfaces.

Use `residue_mapper` whenever a residue number must line up between a
UniProt entry, a PDB file and a tool's internal numbering — mismatched
numbering silently produces wrong mutations and wrong interfaces.

`pred_binding_affinity_boltz2` gives a predicted affinity for a
protein–small-molecule pair; `foldx_tool` covers stability, mutation and
interface analysis; the ProLIF tools return interaction fingerprints for a
complex, a trajectory, or a protein–protein pair.

Predicted affinities are estimates with no experimental error bars. Report
them as such and prefer relative comparisons within one protocol.

The remote server keeps its own filesystem. Tools that take a `*_file_path`
argument expect a path **on the server**, not a local one. To use a local
structure: call `base64_to_server_file` with the file's base64 content and a
name, then pass the returned server path to the analysis tool, and
`server_file_to_base64` to bring a result back (files must be under 10MB).
