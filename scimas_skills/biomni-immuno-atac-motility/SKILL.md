---
name: biomni-immuno-atac-motility
description: Use to run differential accessibility analysis on ATAC-seq peak counts, isolate / purify immune-cell populations in silico, track immune cells under flow, or estimate cell-cycle phase durations from a time-course.
x-scimas-role: cell-biologist
x-scimas-server: biomni-immunology
x-scimas-tools:
  - analyze_atac_seq_differential_accessibility
  - isolate_purify_immune_cells
  - track_immune_cells_under_flow
  - estimate_cell_cycle_phase_durations
---

# ATAC-seq + immune-cell motility

## When to use
- User has ATAC-seq peak counts for two conditions and wants DAR analysis.
- User wants an immune-cell subset isolation plan from a marker panel.
- User has immune-cell tracking data under flow and wants speed / direction stats.
- User has a time-course BrdU / EdU staining and wants cell-cycle phase durations.

## Limitations
- Requires ``scanpy`` / ``diffbind`` / cell-tracking libs (NOT installed by default).
- ``isolate_purify_immune_cells`` is a planning helper; not connected to a FACS sorter.
