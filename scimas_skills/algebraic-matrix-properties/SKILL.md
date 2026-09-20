---
name: algebraic-matrix-properties
description: Use for matrix parsing, Hermitian / unitary checks, matrix trace / determinant / Frobenius norm / rank, matrix exponential, and dense linear-system solves.
x-scimas-role: algebraic-mathematician
x-scimas-server: math-algebraic
x-scimas-tools:
  - parse_matrix_string
  - is_hermitian
  - is_unitary
  - matrix_properties
  - matrix_exponential
  - solve_linear_system
---
# Algebraic Matrix Properties

Use this skill for matrix property checks (Hermitian, unitary, trace,
determinant, Frobenius norm, rank), closed-form matrix exponential,
and small dense linear-system solves.

State the numerical tolerance explicitly for any near-equality check.
Reject non-square inputs for Hermitian / unitary / matrix exponential.