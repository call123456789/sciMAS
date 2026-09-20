---
name: biomni-genetics-crispr-editing
description: Use to analyze the outcomes of a Cas9 edit (original vs edited sequence + guide RNA), classify indel type, or simulate a CRISPR HDR repair template.
x-scimas-role: geneticist
x-scimas-server: biomni-genetics
x-scimas-tools:
  - analyze_cas9_mutation_outcomes
  - analyze_crispr_genome_editing
  - identify_transcription_factor_binding_sites
---

# CRISPR / Cas9 outcome analysis

## When to use
- User has Cas9-edited sequence vs WT and wants indel classification.
- User has a guide RNA + optional HDR repair template and wants a predicted edited outcome.

## Limitations
- ``analyze_cas9_mutation_outcomes`` does not call NHEJ vs MMEJ — only the net indel spectrum.
- ``identify_transcription_factor_binding_sites`` uses a PWM approximation; not a ChIP-seq replacement.
