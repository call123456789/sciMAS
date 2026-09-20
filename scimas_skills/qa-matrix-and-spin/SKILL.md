---
name: qa-matrix-and-spin
description: Use for matrix parsing, Hermitian / unitary checking, trace, matrix exponential, Pauli-matrix generation, spin density matrix construction, and expectation value of an observable.
x-scimas-role: quantum-atomic-physicist
x-scimas-server: physics-quantum-atomic
x-scimas-tools:
  - parse_matrix_string
  - is_hermitian
  - is_unitary
  - matrix_trace
  - matrix_exponential
  - pauli_matrices
  - spin_density_matrix
  - expectation_value
---
# QA Matrix And Spin

Use this skill for matrix property verification, matrix exponential, Pauli algebra, single-spin density matrices, and expectation values of observables.

Report tolerances for Hermitian / unitary checks. Reject empty matrices. Use the convention ⟨ψ|A|ψ⟩ for expectation value.