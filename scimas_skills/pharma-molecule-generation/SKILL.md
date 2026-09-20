---
name: pharma-molecule-generation
description: Use for MADD disease-case molecule generation, MADD property prediction, and follow-up local screening of generated SMILES.
x-scimas-role: drug-discovery-scientist
x-scimas-server: pharma-drug-discovery
x-scimas-tools:
  - list_madd_local_checkpoints
  - generate_molecules_with_local_madd
  - generate_molecules_by_case
  - predict_with_local_madd_checkpoint
  - predict_properties_by_smiles
  - evaluate_druglikeness
  - draw_molecules
---
# Pharma Molecule Generation

Use this skill when the task asks to generate or score candidate molecules with
MADD drug-discovery services or local MADD checkpoints.

MADD generation can run locally only when the required checkpoint directories
are present and the Python environment matches MADD's requirements. Use
`list_madd_local_checkpoints` before local generation or local checkpoint
prediction. If local generation is unavailable, MADD API generation requires
URL_GEN; MADD API prediction requires URL_PRED.

Predefined MADD generation cases include Alzheimer, Parkinson, multiple
sclerosis, dyslipidemia, acquired drug resistance, lung cancer, and random
generation. Use exact user-trained case names for custom models.

After molecules are returned, use local drug-likeness screening when useful and
make the service dependency explicit in the answer.
