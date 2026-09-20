---
name: biomni-cancer-genomics-network
description: Use to model the DNA-damage-response network in a tumor sample, detect / annotate somatic mutations and structural variations from a VCF, run NMF on expression data, or estimate copy-number purity / ploidy.
x-scimas-role: cell-biologist
x-scimas-server: biomni-cancer_biology
x-scimas-tools:
  - analyze_ddr_network_in_cancer
  - detect_and_annotate_somatic_mutations
  - detect_and_characterize_structural_variations
  - perform_gene_expression_nmf_analysis
  - analyze_copy_number_purity_ploidy_and_focal_events
---

# Cancer DDR network & mutation / SV analysis

## When to use
- User has a tumor mutation list and wants DDR-pathway enrichment.
- User has a VCF and wants somatic-mutation annotation + SV characterization.
- User has an expression matrix and wants NMF factor decomposition.
- User has a copy-number profile and wants purity / ploidy / focal-event estimation.

## Limitations
- Requires ``gseapy`` / ``scanpy`` + variant-caller dependencies (NOT installed by default).
- Results are research-log strings; numeric estimates must be parsed out.
