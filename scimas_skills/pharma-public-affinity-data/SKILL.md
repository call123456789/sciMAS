---
name: pharma-public-affinity-data
description: Use for fetching BindingDB/ChEMBL affinity data and inspecting or filtering modeling datasets.
x-scimas-role: pharma-data-specialist
x-scimas-server: pharma-data
x-scimas-tools:
  - fetch_bindingdb_affinity
  - fetch_chembl_activities
  - inspect_dataset_columns
  - filter_dataset_columns
---
# Pharma Public Affinity Data

Use this skill to obtain target-ligand affinity data from public databases or
prepare tabular datasets for pharmaceutical modeling.

For BindingDB, use a UniProt accession when the prompt provides one; otherwise
search UniProt from the protein name and report the selected accession. For
ChEMBL, do not invent target IDs; if a name search is used, report the chosen
target_chembl_id.

When modifying a dataset, save a new file and report dropped, missing, and
remaining columns.
