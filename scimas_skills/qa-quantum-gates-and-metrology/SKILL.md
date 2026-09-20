---
name: qa-quantum-gates-and-metrology
description: Use for CNOT and anti-CNOT gate matrix construction, and quantum Fisher information for pure-state metrology.
x-scimas-role: quantum-atomic-physicist
x-scimas-server: physics-quantum-atomic
x-scimas-tools:
  - cnot_gate
  - anti_cnot_gate
  - quantum_fisher_information_pure
---
# QA Quantum Gates And Metrology

Use this skill for two-qubit controlled gate matrices and pure-state quantum Fisher information (single-parameter sensitivity).

Specify the control / target qubit convention for CNOT. For QFI, the parameter is a phase (or other generator) and the bound gives 1/Δθ²_min.