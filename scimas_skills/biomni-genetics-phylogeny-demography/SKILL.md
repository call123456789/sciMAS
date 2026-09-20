---
name: biomni-genetics-phylogeny-demography
description: Use to build a protein phylogeny from aligned sequences, or to simulate a demographic history under a specified effective population size / time schedule.
x-scimas-role: geneticist
x-scimas-server: biomni-genetics
x-scimas-tools:
  - analyze_protein_phylogeny
  - simulate_demographic_history
---

# Phylogeny & demographic inference

## When to use
- User has multiple aligned protein sequences and wants a tree.
- User has population-size / generation schedule and wants a simulated allele-frequency trajectory.

## Limitations
- Phylogeny uses Biopython NJ; no bootstrap support.
- Demographic simulator uses a custom coalescent approximation; not msprime.
