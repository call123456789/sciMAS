---
name: optimization-quadratic-and-lp
description: Use for quadratic-form gradient / Hessian, convexity check, and a 2-variable toy linear program (vertex enumeration).
x-scimas-role: optimization-mathematician
x-scimas-server: math-optimization
x-scimas-tools:
  - quadratic_gradient
  - quadratic_hessian
  - convexity_check
  - linear_program_2d
---
# Optimization Quadratic And LP

Use this skill for quadratic-form analysis (gradient ∇f = A x + b,
Hessian = A), convexity check on the symmetric part of A (positive
eigenvalues → convex), and a toy 2-variable linear program solved
by vertex enumeration.

The LP solver handles up to a 2-D problem with arbitrary linear
constraints. For larger problems, hand off to a numerical solver.