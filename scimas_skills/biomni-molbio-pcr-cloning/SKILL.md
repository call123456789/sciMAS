---
name: biomni-molbio-pcr-cloning
description: Use to design primers / oligos for a PCR, Golden Gate assembly, or knockout sgRNA; simulate a PCR on a linear / circular template; simulate a restriction digest.
x-scimas-role: molecular-biologist
x-scimas-server: biomni-molecular_biology
x-scimas-tools:
  - pcr_simple
  - digest_sequence
  - design_primer
  - design_verification_primers
  - design_golden_gate_oligos
  - golden_gate_assembly
---

# PCR / cloning workflow helpers

## When to use
- User has template + primer pair and wants the simulated PCR product.
- User has a multi-fragment Golden Gate assembly and wants junction oligo design.
- User wants primer Tm / hairpin / dimer-aware design for a target region.

## Limitations
- ``pcr_simple`` is a strict 3' / 5' identity match — does not model mismatch / wobble bases.
- ``golden_gate_assembly`` assumes the standard Type IIS overhang set (BsaI / BsmBI).
