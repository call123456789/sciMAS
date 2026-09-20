---
name: organic-rdkit-reactions
description: Use for RDKit or SMARTS-based organic reaction parsing and product prediction.
x-scimas-role: organic-chemist
x-scimas-server: chemistry-organic
x-scimas-tools:
  - parse_smiles_rdkit
  - predict_smarts_product
  - generate_product_via_smarts
  - predict_hydrolysis_product
  - predict_organic_reaction_products
---
# Organic RDKit Reactions

Use this skill for SMARTS reaction templates, RDKit SMILES parsing, named organic reaction product prediction, and broader product generation.

Validate input SMILES first when product prediction fails. State template limitations and avoid claiming comprehensive retrosynthesis coverage.
