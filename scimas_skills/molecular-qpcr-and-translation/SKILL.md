---
name: molecular-qpcr-and-translation
description: Use for qPCR Ct / ΔΔCt relative-expression analysis and DNA → protein translation (single-frame).
x-scimas-role: molecular-biologist
x-scimas-server: biology-molecular
x-scimas-tools:
  - analyze_qrt_pcr
  - calculate_rna_concentration
  - translate_dna_to_protein
---
# Molecular qPCR And Translation

Use this skill for qPCR fold-change calculations (ΔΔCt, optional efficiency correction), RNA concentration by Beer-Lambert, and first-frame DNA → protein translation.

State the efficiency assumption (default 2.0 = 100%). Report fold-change and log2 fold-change together. For translation, truncate to a multiple of 3.