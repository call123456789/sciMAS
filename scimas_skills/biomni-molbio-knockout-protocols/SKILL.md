---
name: biomni-molbio-knockout-protocols
description: Use to design a knockout sgRNA for a target locus or fetch a wet-lab protocol (oligo annealing, Golden Gate assembly, bacterial transformation).
x-scimas-role: molecular-biologist
x-scimas-server: biomni-molecular_biology
x-scimas-tools:
  - design_knockout_sgrna
  - get_oligo_annealing_protocol
  - get_golden_gate_assembly_protocol
  - get_bacterial_transformation_protocol
---

# CRISPR knockout design & wet-lab protocols

## When to use
- User wants a sgRNA designed against a target gene (PAM = NGG).
- User wants a step-by-step protocol for oligo annealing, Golden Gate assembly, or bacterial transformation.

## Limitations
- Protocol helpers return static protocol text; not the latest vendor-specific procedure.
- ``design_knockout_sgrna`` scores off-targets heuristically; for production CRISPR use a CRISPOR-style tool.
