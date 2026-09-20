---
name: biomni-synbio-network-bifurcation
description: Use to simulate a gene circuit with growth feedback, scan a bifurcation diagram over a parameter sweep, or analyze barcode-sequencing data for lineage tracing.
x-scimas-role: molecular-biologist
x-scimas-server: biomni-synthetic_biology
x-scimas-tools:
  - simulate_gene_circuit_with_growth_feedback
  - analyze_bifurcation_diagram
  - analyze_barcode_sequencing_data
  - create_biochemical_network_sbml_model
  - identify_fas_functional_domains
---

# Gene-circuit & bifurcation simulation

## When to use
- User has a gene-circuit ODE and wants a growth-coupled simulation.
- User has a 1D parameter sweep and wants the bifurcation diagram.
- User has barcode count table and wants lineage / clonal-frequency analysis.
- User wants an SBML model for a biochemical network.
- User has a FAS protein sequence and wants functional-domain annotation.

## Limitations
- SBML emission requires ``libsbml`` (not installed by default — wrapper returns JSON error if missing).
- Bifurcation diagram uses a coarse parameter grid; no continuation method (pseudo-arclength).
