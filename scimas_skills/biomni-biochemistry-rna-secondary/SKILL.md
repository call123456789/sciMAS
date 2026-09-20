---
name: biomni-biochemistry-rna-secondary
description: Use for dot-bracket → stems / loops / paired-fraction analysis; optional nearest-neighbor free-energy if the RNA sequence is supplied.
x-scimas-role: molecular-biologist
x-scimas-server: biomni-biochemistry
x-scimas-tools:
  - analyze_rna_secondary_structure_features
---

# RNA secondary-structure features

## When to use
- User has a dot-bracket RNA structure (and optionally the matching sequence) and wants stems, loops, paired-fraction, or nearest-neighbor free energy.

## Limitations
- Only ``()``, ``[]``, ``{}``, and ``.`` are accepted brackets; mismatched or unbalanced input returns an error string, not a JSON envelope.
- Energy parameters are simplified (AU=-0.9, GC=-2.1, GU=-0.5 kcal/mol); treat as approximate.
