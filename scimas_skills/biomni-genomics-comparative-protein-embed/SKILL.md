---
name: biomni-genomics-comparative-protein-embed
description: Use to run comparative-genomics / haplotype analysis on a multi-species alignment, run interspecies gene-name conversion, or generate protein / transcript embeddings with ESM / TranscriptFormer.
x-scimas-role: geneticist
x-scimas-server: biomni-genomics
x-scimas-tools:
  - analyze_comparative_genomics_and_haplotypes
  - interspecies_gene_conversion
  - generate_gene_embeddings_with_ESM_models
  - generate_transcriptformer_embeddings
---

# Comparative genomics & protein-language embeddings

## When to use
- User has a multi-species alignment and wants a haplotype / selection scan.
- User has mouse / rat / zebrafish gene symbols and wants the human ortholog.
- User has a protein sequence and wants ESM / TranscriptFormer embeddings.

## Limitations
- Protein / transcript embeddings require ``esm`` / ``transcriptformer`` + a downloaded checkpoint (GB-scale).
- Variant callers (somatic / SV) are not part of this skill — they live in the
  ``biomni-cancer-genomics-network`` skill bound to ``biomni-cancer_biology``.
