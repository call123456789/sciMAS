---
name: pharma-drug-sda-molecule-generation
description: Use for generative molecule design: de novo sampling, molecule-to-molecule optimisation, scaffold R-group and linker sampling, and peptide sampling.
x-scimas-role: drug-discovery-scientist
x-scimas-server: pharma-drug-sda
x-scimas-tools:
  - reinvent_denovo_sampling
  - reinvent_mol2mol_sampling
  - libinvent_rgroup_sampling_by_scaffold
  - libinvent_rgroup_sampling_by_scaffold_name
  - linkinvent_linker_sampling_by_warhead_pair_name
  - linkinvent_linker_sampling_by_warheads
  - pepinvent_peptide_sampling_by_peptide
  - pepinvent_peptide_sampling_by_template
  - get_pepinvent_info
---
Generate candidate molecules with the REINVENT family of models.

Call `get_pepinvent_info` before the pepinvent tools — it lists the preset
templates and amino-acid alphabets that the `*_by_template` variants accept.

Generated molecules are proposals. Validate them (`is_valid_smiles`) and
score them with the descriptor or ADMET skills before reporting, and never
present a generated structure as a synthesised or tested compound.
