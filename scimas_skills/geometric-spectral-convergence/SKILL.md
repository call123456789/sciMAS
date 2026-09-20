---
name: geometric-spectral-convergence
description: Use for theoretical error bounds C·ε^α, empirical spectral convergence error, and bandwidth scaling sweeps.
x-scimas-role: geometric-mathematician
x-scimas-server: math-geometric
x-scimas-tools:
  - theoretical_error_bound
  - compute_spectral_convergence_error
  - analyze_bandwidth_scaling
---
# Geometric Spectral Convergence

Use this skill for spectral-graph convergence analysis: theoretical
bounds, empirical error from graph Laplacian eigenvalues vs.
analytic λ_k, and a sweep over sample sizes n.

Reject k values larger than the available non-zero eigenvalues.