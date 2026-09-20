---
name: biomni-biochemistry-protein-conservation
description: Use for multi-sequence-alignment + phylogenetic tree + position-wise conservation across a small protein set (BiOMNI uses Biopython / MUSCLE).
x-scimas-role: molecular-biologist
x-scimas-server: biomni-biochemistry
x-scimas-tools:
  - analyze_protein_conservation
---

# Protein conservation analysis

## When to use
- User has 2+ protein sequences (plain or FASTA) and wants alignment + tree + per-position conservation scores.

## Limitations
- Requires Biopython; falls back to a padding-based alignment if MUSCLE is not on PATH, but the fallback is not biologically meaningful.
- Returns a research log string, not a JSON alignment record.
