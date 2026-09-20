---
name: statistical-wls-and-sandwich
description: Use for weighted least squares (WLS) parameter estimation, sandwich-covariance estimator, and stationary AR(1)-style time-series simulation.
x-scimas-role: statistical-mathematician
x-scimas-server: math-statistical
x-scimas-tools:
  - estimate_wls_parameters
  - compute_sandwich_covariance_estimator
  - simulate_stationary_time_series
---
# Statistical WLS And Sandwich

Use this skill for WLS regression with optional weights, sandwich
heteroskedasticity-robust covariance estimation, and stationary
AR(1) time-series simulation.

For WLS, supply an n×p design matrix X, target y, and optional
weights. For simulation, ensure |phi| < 1.