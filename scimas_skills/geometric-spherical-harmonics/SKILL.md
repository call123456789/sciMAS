---
name: geometric-spherical-harmonics
description: Use for analytic Laplace-Beltrami eigenvalues on S^(d−1), evaluation of axis-aligned spherical harmonics, sampling-vector construction, and alignment-sign correction.
x-scimas-role: geometric-mathematician
x-scimas-server: math-geometric
x-scimas-tools:
  - sphere_laplacian_eigenvalue
  - evaluate_sphere_eigenfunction
  - compute_sampling_vector
  - compute_alignment_sign
  - compute_euclidean_error
---
# Geometric Spherical Harmonics

Use this skill for analytic λ_k = k(k+d−2) eigenvalues, low-k
harmonic evaluations on points, weighted sampling vectors
|w|^p, alignment-sign flips, and Euclidean error ‖α v − φ‖₂.

Reject k < 0. Pad the embedding to 3 dimensions for the harmonic
evaluator.