---
name: genetics-sequence-composition
description: Use for nucleotide sequence GC/AT counts, FASTA-style formatting with group spacing, and chromosome-level metrics (length, GC, AT-skew).
x-scimas-role: geneticist
x-scimas-server: biology-genetics
x-scimas-tools:
  - analyze_sequence_composition
  - format_sequence_with_spacing
  - chromosome_metrics
---
# Genetics Sequence Composition

Use this skill for sequence composition statistics: per-base counts, GC content, and chromosome-level GC/AT-skew.

Strip ambiguous bases (N, R, Y, ...) before computing ratios. For AT-skew, reject any gene with G+C == 0.