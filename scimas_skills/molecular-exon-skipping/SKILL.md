---
name: molecular-exon-skipping
description: Use for DMD exon-skipping therapy: frame-shift analysis, morpholino binding prediction, retained-transcript fraction, and skip-exon recommendation.
x-scimas-role: molecular-biologist
x-scimas-server: biology-molecular
x-scimas-tools:
  - analyze_exon_frame_shift
  - simulate_morpholino_binding
  - analyze_dmd_mutation_therapy
  - query_rna_structure_involvement
---
# Molecular Exon Skipping

Use this skill for DMD exon-skipping analysis: deleted-exon list → reading-frame preservation, morpholino design (length 18–30 nt, GC 40–60%), and retained-transcript fraction.

Report bp removed, frame preserved (Δ % 3 == 0), and the recommended skip exon near the deleted cluster.