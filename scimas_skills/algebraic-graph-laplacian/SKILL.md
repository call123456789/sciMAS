---
name: algebraic-graph-laplacian
description: Use for graph-Laplacian machinery: degree matrix, unnormalised / random-walk Laplacian, smallest-K eigenpairs, and Gaussian RBF affinity matrices.
x-scimas-role: algebraic-mathematician
x-scimas-server: math-algebraic
x-scimas-tools:
  - compute_degree_matrix
  - compute_graph_laplacian
  - compute_random_walk_laplacian
  - compute_laplacian_eigenpairs
  - compute_gaussian_affinity
---
# Algebraic Graph Laplacian

Use this skill for graph-spectrum analysis: degree matrix,
(unnormalised) Laplacian L = D − W, random-walk Laplacian
L_rw = I − D⁻¹ W, smallest-K Laplacian eigenpairs for spectral
clustering, and Gaussian RBF affinity matrices from feature rows.

Reject isolated nodes (degree 0) for the random-walk variant.
Default K = 4 eigenpairs.