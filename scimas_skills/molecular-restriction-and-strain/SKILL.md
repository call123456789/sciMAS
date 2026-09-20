---
name: molecular-restriction-and-strain
description: Use for restriction-enzyme lookup / digest simulation, methylation-sensitivity screening, and DNA-purity (OD 260/280, 260/230) metrics.
x-scimas-role: molecular-biologist
x-scimas-server: biology-molecular
x-scimas-tools:
  - query_restriction_enzyme
  - simulate_restriction_digest
  - analyze_digest_pattern
  - check_methylation_sensitivity
  - calculate_dna_quality_metrics
---
# Molecular Restriction And Strain

Use this skill for restriction-enzyme selection, in-silico digest fragment-size prediction, methylation-aware strain choice (dam/dcm), and DNA-purity assessment.

State the host strain and its dam/dcm methylation. Report expected fragment sizes in bp. For purity, distinguish protein vs. RNA vs. carbohydrate contamination.