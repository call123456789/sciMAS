---
name: pharma-drug-sda-structure-prediction-design
description: Use to predict protein 3D structure from sequence and to design protein sequences or binders: ESMFold, Chai-1, Chroma, ProteinMPNN, and EvoBind2.
x-scimas-role: structural-biologist
x-scimas-server: pharma-drug-sda
x-scimas-tools:
  - pred_protein_structure_esmfold
  - chai1_predict
  - chroma_monomer
  - chroma_complex
  - chroma_symmetry
  - proteinmpnn_tool
  - evobind_tool
---
Predict structures and design sequences.

`pred_protein_structure_esmfold` and `chai1_predict` take a sequence and
return a PDB path on the remote server. Chroma tools generate de novo
backbones (monomer, complex, symmetric); `proteinmpnn_tool` designs sequences
onto a given backbone and can constrain the design interface;
`evobind_tool` designs linear or cyclic peptide binders from a receptor
sequence.

These are predictions and designs. Report the model used and treat the output
as a hypothesis — a predicted fold is not an experimental structure.

The remote server keeps its own filesystem. Tools that take a `*_file_path`
argument expect a path **on the server**, not a local one. To use a local
structure: call `base64_to_server_file` with the file's base64 content and a
name, then pass the returned server path to the analysis tool, and
`server_file_to_base64` to bring a result back (files must be under 10MB).
