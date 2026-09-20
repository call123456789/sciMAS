---
name: biomni-synbio-codon-design
description: Use to optimize codons of a coding sequence for a target host, design a therapeutic-delivery bacterial genome, or analyze bacterial growth rate from an OD600 time-course.
x-scimas-role: molecular-biologist
x-scimas-server: biomni-synthetic_biology
x-scimas-tools:
  - optimize_codons_for_heterologous_expression
  - engineer_bacterial_genome_for_therapeutic_delivery
  - analyze_bacterial_growth_rate
---

# Codon & genome design

## When to use
- User has a coding sequence and a target host (CAI optimization).
- User has a bacterial FASTA + a list of genetic parts and wants an engineered construct.
- User has OD600 vs time and wants growth-rate fit (μ, lag, carrying capacity).

## Limitations
- Codon optimization uses simple CAI; no mRNA-structure / ribosomal queueing constraints.
- Growth-rate fit uses a simple logistic; no diauxic shift.
