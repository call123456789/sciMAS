---
name: pharma-drug-sda-admet-similarity
description: Use for ADMET property prediction, disease-associated DLEPS scoring, Tanimoto similarity to a reference molecule, shared fragment counts, format conversion, and name-to-SMILES lookup.
x-scimas-role: drug-discovery-scientist
x-scimas-server: pharma-drug-sda
x-scimas-tools:
  - pred_mol_admet
  - calculate_dleps_score
  - calculate_morgan_fingerprint_similarity
  - calculate_common_fragments
  - convert_smiles_to_format
  - retrieve_smiles_by_compoundname
---
Predict ADMET profiles and rank candidates against a reference molecule.

`pred_mol_admet` returns over 90 key-value properties per molecule; summarise
the absorption/distribution/metabolism/excretion/toxicity headline values
rather than dumping the full dictionary. `calculate_dleps_score` needs a
disease name and identifies upregulated targets, so state the disease context
in your report.

ADMET predictions are model estimates. Present them as predicted
liabilities to investigate, not measured outcomes.
