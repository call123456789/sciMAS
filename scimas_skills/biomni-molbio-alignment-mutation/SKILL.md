---
name: biomni-molbio-alignment-mutation
description: Use to align a long sequence against short probes, retrieve a gene coding sequence from NCBI Entrez, find sequence mutations between a query and a reference, or get a plasmid sequence from AddGene.
x-scimas-role: molecular-biologist
x-scimas-server: biomni-molecular_biology
x-scimas-tools:
  - align_sequences
  - find_sequence_mutations
  - get_gene_coding_sequence
  - get_plasmid_sequence
---

# Sequence alignment & mutation analysis

## When to use
- User has a long sequence + a list of short probes and wants the locations of each probe.
- User has two sequences (query vs reference) and wants a mutation table (SNPs / indels).
- User wants a NCBI gene coding sequence (NCBI Entrez).
- User wants an AddGene plasmid full sequence by ID.

## Limitations
- ``get_gene_coding_sequence`` / ``get_plasmid_sequence`` hit live HTTP endpoints; require network access and may rate-limit.
