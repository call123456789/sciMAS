---
name: numerical-stability-and-smoothing
description: Use for matrix condition number, central finite-difference derivative, running-mean smoothing, Richardson extrapolation, and spectral radius stability.
x-scimas-role: numerical-mathematician
x-scimas-server: math-numerical
x-scimas-tools:
  - matrix_condition_number
  - finite_difference_derivative
  - running_mean_smooth
  - richardson_extrapolation
  - spectral_radius
---
# Numerical Stability And Smoothing

Use this skill for numerical-stability analysis: matrix condition
number (2-norm), central finite-difference derivative at an interior
point, running-mean smoothing with an odd window, Richardson
extrapolation of two estimates at step h and h/2, and spectral
radius for stability checks.

Reject non-square matrices for condition number and spectral radius.