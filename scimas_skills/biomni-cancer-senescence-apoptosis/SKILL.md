---
name: biomni-cancer-senescence-apoptosis
description: Use to score senescence and apoptosis signatures from bulk or single-cell expression data.
x-scimas-role: cell-biologist
x-scimas-server: biomni-cancer_biology
x-scimas-tools:
  - analyze_cell_senescence_and_apoptosis
---

# Cell senescence & apoptosis scoring

## When to use
- User has an expression matrix and wants senescence / apoptosis signature scores.

## Limitations
- Requires ``scanpy`` + the underlying signature sets (NOT installed by default).
