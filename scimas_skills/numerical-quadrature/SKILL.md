---
name: numerical-quadrature
description: Use for numerical quadrature: composite midpoint / trapezoid / Simpson's 1/3 rule, 2-point Gauss-Legendre on a polynomial family, and adaptive Simpson with Richardson extrapolation.
x-scimas-role: numerical-mathematician
x-scimas-server: math-numerical
x-scimas-tools:
  - numerical_midpoint_quadrature
  - numerical_trapezoid_quadrature
  - numerical_simpson_quadrature
  - gauss_legendre_2pt
  - adaptive_simpson
---
# Numerical Quadrature

Use this skill for definite integrals: composite midpoint (n ≥ 1),
composite trapezoid (n ≥ 2), composite Simpson's 1/3 (n odd, ≥ 3),
2-point Gauss-Legendre for linear / quadratic / cubic families, and
adaptive Simpson with Richardson extrapolation.

For Simpson's 1/3 rule, n must be odd.