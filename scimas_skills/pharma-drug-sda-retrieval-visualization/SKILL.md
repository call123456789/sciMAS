---
name: pharma-drug-sda-retrieval-visualization
description: Use to fetch protein sequences and structures by identifier and to render protein or molecule images, plus moving files to and from the remote server.
x-scimas-role: structural-biologist
x-scimas-server: pharma-drug-sda
x-scimas-tools:
  - retrieve_protein_sequence
  - retrieve_protein_structure_by_gene_name
  - retrieve_protein_structure_by_pdb_id
  - retrieve_protein_structure_by_uniprot_id
  - visualize_protein
  - visualize_molecule
  - base64_to_server_file
  - server_file_to_base64
---
Retrieve inputs and render figures.

Structure lookups accept a PDB ID, UniProt ID, or gene name and fall back
from `.pdb` to `.cif` automatically — check which format came back before
passing the path on. `retrieve_protein_sequence` takes a gene name or UniProt
ID plus an organism.

This skill also owns the file bridge to the remote server, which the other
pharma-drug-sda skills depend on.

The remote server keeps its own filesystem. Tools that take a `*_file_path`
argument expect a path **on the server**, not a local one. To use a local
structure: call `base64_to_server_file` with the file's base64 content and a
name, then pass the returned server path to the analysis tool, and
`server_file_to_base64` to bring a result back (files must be under 10MB).
