---
name: biomni-genomics-scrna-celltype
description: Use to run cell-type annotation on a single-cell RNA-seq AnnData (CellTypist / scANVI / UCE / IMAP + interpretation), batch-corrected embeddings (scVI / Harmony / STATE), or run gene-set enrichment on the marker list.
x-scimas-role: geneticist
x-scimas-server: biomni-genomics
x-scimas-tools:
  - annotate_celltype_scRNA
  - annotate_celltype_with_panhumanpy
  - create_scvi_embeddings_scRNA
  - create_harmony_embeddings_scRNA
  - get_uce_embeddings_scRNA
  - map_to_ima_interpret_scRNA
  - unsupervised_celltype_transfer_between_scRNA_datasets
  - generate_embeddings_with_state
---

# scRNA-seq annotation & embedding

## When to use
- User has a single-cell AnnData and wants cell-type labels (CellTypist / scANVI / UCE / IMAP) or batch-corrected embeddings (scVI / Harmony / STATE).
- User wants to transfer labels from a reference atlas to a query dataset.

## Limitations
- Requires ``scanpy`` + a CellTypist / scANVI / UCE checkpoint (NOT installed by default — wrapper returns JSON error if missing).
- Heavy GPU recommended for scVI / UCE embeddings; CPU runs are very slow.
- Returns a research-log string; embedding vectors must be parsed out if needed downstream.
