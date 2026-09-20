---
name: pharma-druglikeness-screening
description: Use for local RDKit drug-likeness descriptors, structural alerts, and molecule PNG rendering from SMILES.
x-scimas-role: drug-discovery-scientist
x-scimas-server: pharma-drug-discovery
x-scimas-tools:
  - evaluate_druglikeness
  - draw_molecules
---
# Pharma Drug-Likeness Screening

Use this skill to triage one or more SMILES strings with local RDKit descriptors
and medicinal-chemistry structural alerts.

Report validity first. For valid molecules, summarize QED, synthetic
accessibility, LogP, TPSA, H-bond donors/acceptors, rotatable bonds, aromatic
rings, and PAINS/SureChEMBL/Glaxo/Brenk flags. Present these as screening
metrics only; they do not prove clinical safety or efficacy.

Use `draw_molecules` when a structural image helps interpret or communicate the
candidate set.
