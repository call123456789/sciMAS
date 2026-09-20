---
name: biomni-genomics-chipseq-motifs
description: Use to call peaks on ChIP-seq aligned reads (MACS2), find enriched motifs (HOMER), or analyze a genomic region's overlap with a known annotation set.
x-scimas-role: geneticist
x-scimas-server: biomni-genomics
x-scimas-tools:
  - perform_chipseq_peak_calling_with_macs2
  - find_enriched_motifs_with_homer
  - analyze_genomic_region_overlap
  - analyze_chromatin_interactions
---

# ChIP-seq peak calling & motif analysis

## When to use
- User has aligned ChIP-seq reads (BAM) and wants MACS2 peaks.
- User has a peak set and wants HOMER motif enrichment.
- User has a genomic region BED and wants overlap with a reference annotation.
- User has Hi-C / ChIA-PET data and wants loop / TAD annotation.

## Limitations
- MACS2 + HOMER must be installed system-wide (NOT in environment.yml).
- Chromatin-interaction tools may require ``cooltools`` / ``HiCExplorer``.
