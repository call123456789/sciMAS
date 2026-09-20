---
name: biomni-cellbio-flow-facs
description: Use to perform simulated FACS cell sorting, or analyze flow-cytometry immunophenotyping panels.
x-scimas-role: cell-biologist
x-scimas-server: biomni-cell_biology
x-scimas-tools:
  - perform_facs_cell_sorting
  - analyze_flow_cytometry_immunophenotyping
---

# Flow cytometry & FACS

## When to use
- User has marker-expression data and wants a simulated FACS sort.
- User has an immunophenotyping panel and wants per-cluster abundance.

## Limitations
- Requires ``flowkit`` / ``anndata`` (NOT installed by default).
- FACS sort is a simulation; not connected to a physical sorter.
