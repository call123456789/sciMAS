---
name: ms-fragmentation-prediction
description: Use for SMILES → molecular-property prediction (formula, MW, ring count) and fragmentation prediction by neutral-loss subtraction.
x-scimas-role: mass-spectrometrist
x-scimas-server: biology-mass-spec
x-scimas-tools:
  - calculate_molecular_properties
  - predict_fragmentation_pattern
---
# MS Fragmentation Prediction

Use this skill for SMILES-level MS prediction: molecular formula, MW, approximate ring count, and a small set of predicted fragment m/z values by neutral loss.

Default neutral losses: H₂O (18), HCl (36.46), NH₃ (17.03), CO (28.01). Sort fragments by descending m/z.