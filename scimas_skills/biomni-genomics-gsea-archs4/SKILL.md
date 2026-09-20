---
name: biomni-genomics-gsea-archs4
description: Use to run a gene-set enrichment analysis on a ranked gene list (Enrichr / GSEA), or fetch ARCHS4 co-expression / expression-by-gene signatures.
x-scimas-role: geneticist
x-scimas-server: biomni-genomics
x-scimas-tools:
  - gene_set_enrichment_analysis
  - get_gene_set_enrichment_analysis_supported_database_list
  - get_rna_seq_archs4
---

# Gene-set enrichment & ARCHS4

## When to use
- User has a ranked gene list and wants Enrichr / GSEA pathway enrichment.
- User wants the supported Enrichr database list to pick the right library.
- User wants ARCHS4 co-expression / sample-by-gene matrix.

## Limitations
- Requires ``gseapy`` (NOT installed by default).
- Returns a research-log string; pathway p-values must be parsed.
