---
name: pharma-drug-sda-mol-descriptors
description: Use for RDKit molecular descriptors over SMILES lists: basic properties, partial charges, complexity, drug-likeness, hydrogen bonding, hydrophobicity, and topology.
x-scimas-role: drug-discovery-scientist
x-scimas-server: pharma-drug-sda
x-scimas-tools:
  - calculate_mol_basic_info
  - calculate_mol_charge
  - calculate_mol_complexity
  - calculate_mol_drug_chemistry
  - calculate_mol_hbond
  - calculate_mol_hydrophobicity
  - calculate_mol_structure_complexity
  - calculate_mol_topology
  - is_valid_smiles
---
Compute descriptor sets for candidate molecules. Validate SMILES with
`is_valid_smiles` first and report invalid entries explicitly rather than
dropping them silently.

These are triage metrics. QED, charge, complexity and hydrophobicity
descriptors support comparison and prioritisation; they do not establish
safety, efficacy, or clinical viability.
