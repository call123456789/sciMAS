---
name: numerical-ode-integrator
description: Use for ODE integration: RK4 and Euler methods on exponential_decay, logistic, and linear_oscillator families.
x-scimas-role: numerical-mathematician
x-scimas-server: math-numerical
x-scimas-tools:
  - rk4_integrate
  - euler_integrate
---
# Numerical ODE Integrator

Use this skill to integrate simple 1-D ODE families. Both RK4 and
Euler methods are supported. State the step dt and n_steps; the
trajectory output is down-sampled to ≤ 10 points.

Reject dt ≤ 0 or n_steps ≤ 0.