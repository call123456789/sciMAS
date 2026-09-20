---
name: geometric-manifold-sampling
description: Use for uniform sampling on the unit sphere S^(d−1) and the 3-D torus, and graph-Laplacian construction from a sample of points.
x-scimas-role: geometric-mathematician
x-scimas-server: math-geometric
x-scimas-tools:
  - sample_sphere
  - sample_torus
  - build_graph_laplacian_from_samples
---
# Geometric Manifold Sampling

Use this skill to draw uniform samples on a sphere or a torus, and
to build a Gaussian-RBF graph Laplacian from those samples.

Specify n, ambient dimension d, and the bandwidth epsilon. For
torus sampling, set major radius R and minor radius r.