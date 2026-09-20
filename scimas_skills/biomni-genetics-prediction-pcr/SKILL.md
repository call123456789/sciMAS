---
name: biomni-genetics-prediction-pcr
description: Use to fit a genomic prediction model (GBLUP-style) on a phenotype + genotype matrix, or to simulate a PCR + gel electrophoresis.
x-scimas-role: geneticist
x-scimas-server: biomni-genetics
x-scimas-tools:
  - fit_genomic_prediction_model
  - perform_pcr_and_gel_electrophoresis
  - liftover_coordinates
---

# Genomic prediction & in-silico PCR

## When to use
- User has a SNP genotype matrix + phenotype column and wants a GBLUP prediction.
- User has primer pair + template and wants a simulated PCR + gel image.
- User has hg19 coordinates and wants hg38 (liftover).

## Limitations
- GBLUP is a single-kernel ridge regression; no multi-kernel / Bayes models.
- Liftover needs an external chain file at the path BiOMNI expects.
