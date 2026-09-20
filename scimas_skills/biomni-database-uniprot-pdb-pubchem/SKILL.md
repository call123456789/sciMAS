---
name: biomni-database-uniprot-pdb-pubchem
description: Use to fetch protein, structure, or compound metadata by ID from UniProt, RCSB PDB, or PubChem.
x-scimas-role: pharma-data-specialist
x-scimas-server: biomni-database
x-scimas-tools:
  - query_uniprot
  - query_pdb
  - query_pubchem
---

# UniProt / PDB / PubChem lookups

## When to use
- UniProt accession → canonical sequence, organism, function annotations.
- PDB ID → entry metadata, polymer entity list, experimental method.
- Compound name / SMILES / CID → PubChem properties.

## Limitations
- Results are research-log strings, not JSON records.
- Network availability reflects BiOMNI's REST wrappers, not sciMAS.
