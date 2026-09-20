---
name: algebraic-eigendecomposition
description: Use for Hermitian matrix eigendecomposition (eigenvalues + eigenvectors) and sandwich-style covariance construction.
x-scimas-role: algebraic-mathematician
x-scimas-server: math-algebraic
x-scimas-tools:
  - hermitian_eigendecomposition
  - compute_sandwich_covariance
---
# Algebraic Eigendecomposition

Use this skill for symmetric eigendecomposition (real symmetric /
Hermitian matrix) and for computing a sandwich covariance
H = K1 @ Γ1 @ K1ᵀ.

For symmetric inputs, the function symmetrises the matrix before
solving. For sandwich, ensure K1 (m×p) and Γ1 (p×p) shapes are
consistent.