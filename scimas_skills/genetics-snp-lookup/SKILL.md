---
name: genetics-snp-lookup
description: Use for dbSNP rsID lookup, flanking-sequence retrieval (mock), and SNP-region composition.
x-scimas-role: geneticist
x-scimas-server: biology-genetics
x-scimas-tools:
  - fetch_snp_from_ncbi
  - fetch_flanking_sequence_ensembl
---
# Genetics SNP Lookup

Use this skill for SNP metadata retrieval (gene, position, ref/alt, consequence, AA change) and mock flanking-sequence retrieval.

Report the chromosome and GRCh coordinates explicitly. Reject unknown rsIDs.