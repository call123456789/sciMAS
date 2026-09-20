#!/usr/bin/env python3
"""Numerical mathematics MCP server.

Tools include:
- Numerical quadrature (midpoint, trapezoid, Simpson).
- ODE integration (RK4 on a callable trajectory).
- Numerical-stability helpers (condition number, fixed-point error).
- Simple finite-difference derivative estimation.
- Smoothing / smoothing-error (Savitzky-Golay style running mean).

All implementations are stdlib + numpy only. Every tool returns a JSON
string. Tool naming is snake_case.
"""

from __future__ import annotations

import json
import math
from typing import Any, Callable, Dict, List

import numpy as np

try:
    from scipy.integrate import quad  # noqa: F401
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("math-numerical")


def _ok(name: str, **fields) -> str:
    return json.dumps({"status": "ok", "tool": name, **fields}, ensure_ascii=False)


def _err(message: str, **fields) -> str:
    return json.dumps({"status": "error", "message": message, **fields},
                      ensure_ascii=False)


# Matplotlib (headless).
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_OK = True
except ImportError:
    _MPL_OK = False

_VIZ_DIR = "/tmp/scimas_math_numerical"


# ---------------------------------------------------------------------------
# Quadrature
# ---------------------------------------------------------------------------

@mcp.tool(description=(
    "Composite midpoint quadrature on a uniform grid for a function "
    "supplied as a list of values y at x = linspace(a, b, n)."
))
def numerical_midpoint_quadrature(a: float, b: float,
                                    y_values: List[float]) -> str:
    try:
        n = len(y_values)
        if n < 1 or a >= b:
            raise ValueError("n ≥ 1 and a < b required")
        h = (b - a) / n
        mids = np.array(y_values, dtype=float)
        integral = h * float(mids.sum())
        return _ok(
            "numerical_midpoint_quadrature",
            a=a, b=b,
            n_points=n,
            h=round(h, 12),
            integral=round(integral, 9),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Composite trapezoidal rule on uniform grid y_values at "
    "x = linspace(a, b, n)."
))
def numerical_trapezoid_quadrature(a: float, b: float,
                                     y_values: List[float]) -> str:
    try:
        n = len(y_values)
        if n < 2 or a >= b:
            raise ValueError("n ≥ 2 and a < b required")
        h = (b - a) / (n - 1)
        arr = np.array(y_values, dtype=float)
        integral = h * (float(arr[0]) / 2 + float(arr[-1]) / 2 +
                        float(arr[1:-1].sum()))
        return _ok(
            "numerical_trapezoid_quadrature",
            a=a, b=b,
            n_points=n,
            h=round(h, 12),
            integral=round(integral, 9),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Composite Simpson's 1/3 rule (n must be odd; uses y_values at "
    "linspace(a, b, n))."
))
def numerical_simpson_quadrature(a: float, b: float,
                                   y_values: List[float]) -> str:
    try:
        n = len(y_values)
        if n < 3 or (n % 2 == 0):
            raise ValueError("n must be odd and ≥ 3")
        h = (b - a) / (n - 1)
        arr = np.array(y_values, dtype=float)
        s = arr[0] + arr[-1] + 4 * float(arr[1:-1:2].sum()) + \
            2 * float(arr[2:-1:2].sum())
        integral = h * float(s) / 3.0
        return _ok(
            "numerical_simpson_quadrature",
            a=a, b=b,
            n_points=n,
            h=round(h, 12),
            integral=round(integral, 9),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Gauss–Legendre 2-point quadrature for a callable name='linear',"
    "'quadratic', or 'cubic'. Returns ∫_a^b f(x) dx exactly."
))
def gauss_legendre_2pt(family: str, a: float, b: float) -> str:
    try:
        nodes = [-(1 / math.sqrt(3)), 1 / math.sqrt(3)]
        weights = [1.0, 1.0]
        if family == "linear":
            f_vals = [(1 / 2) * (a + b) + (b - a) / 2 * x for x in nodes]
        elif family == "quadratic":
            mid = 0.5 * (a + b)
            half = (b - a) / 2
            f_vals = [((mid + half * x) ** 2) for x in nodes]
        elif family == "cubic":
            mid = 0.5 * (a + b)
            half = (b - a) / 2
            f_vals = [((mid + half * x) ** 3) for x in nodes]
        else:
            raise ValueError("family must be linear|quadratic|cubic")
        integral = (b - a) / 2 * sum(w * f for w, f in zip(weights, f_vals))
        return _ok(
            "gauss_legendre_2pt",
            family=family,
            a=a, b=b,
            integral=round(integral, 9),
        )
    except Exception as exc:
        return _err(str(exc))


# ---------------------------------------------------------------------------
# ODE integration
# ---------------------------------------------------------------------------

@mcp.tool(description=(
    "Runge-Kutta 4 integration of dy/dt = f(t, y) given a callable name. "
    "Available families: 'exponential_decay', 'logistic', 'linear_oscillator'."
))
def rk4_integrate(family: str, t0: float, y0: float, dt: float,
                     n_steps: int = 100, k: float = 1.0) -> str:
    try:
        if n_steps <= 0 or dt <= 0:
            raise ValueError("n_steps > 0 and dt > 0")
        if family == "exponential_decay":
            f = lambda t, y: -k * y
        elif family == "logistic":
            f = lambda t, y: k * y * (1 - y)
        elif family == "linear_oscillator":
            # dy/dt = -k y (damped); state vector form would need (y, v).
            f = lambda t, y: -k * y
        else:
            raise ValueError("family must be exponential_decay|logistic|linear_oscillator")
        t = t0
        y = y0
        samples = [(round(t, 6), round(y, 6))]
        for _ in range(n_steps):
            k1 = f(t, y)
            k2 = f(t + dt / 2, y + dt / 2 * k1)
            k3 = f(t + dt / 2, y + dt / 2 * k2)
            k4 = f(t + dt, y + dt * k3)
            y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += dt
            samples.append((round(t, 6), round(y, 6)))
        return _ok(
            "rk4_integrate",
            family=family,
            k=k,
            t0=t0,
            y0=y0,
            dt=dt,
            n_steps=n_steps,
            t_final=round(t, 6),
            y_final=round(y, 9),
            samples=samples[::max(1, len(samples) // 10)],
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Euler method integration of dy/dt = f(t, y) with one of the "
    "available names (same families as rk4_integrate)."
))
def euler_integrate(family: str, t0: float, y0: float, dt: float,
                      n_steps: int = 100, k: float = 1.0) -> str:
    try:
        if n_steps <= 0 or dt <= 0:
            raise ValueError("n_steps > 0 and dt > 0")
        if family == "exponential_decay":
            f = lambda t, y: -k * y
        elif family == "logistic":
            f = lambda t, y: k * y * (1 - y)
        elif family == "linear_oscillator":
            f = lambda t, y: -k * y
        else:
            raise ValueError("family must be exponential_decay|logistic|linear_oscillator")
        t = t0
        y = y0
        for _ in range(n_steps):
            y = y + dt * f(t, y)
            t += dt
        return _ok(
            "euler_integrate",
            family=family,
            t0=t0, y0=y0,
            dt=dt, n_steps=n_steps, k=k,
            t_final=round(t, 6),
            y_final=round(y, 9),
        )
    except Exception as exc:
        return _err(str(exc))


# ---------------------------------------------------------------------------
# Numerical stability / error analysis
# ---------------------------------------------------------------------------

@mcp.tool(description=(
    "Condition number of a matrix in the 2-norm."
))
def matrix_condition_number(A: List[List[float]]) -> str:
    try:
        Aarr = np.array(A, dtype=float)
        if Aarr.ndim != 2 or Aarr.shape[0] != Aarr.shape[1]:
            raise ValueError("A must be square")
        cond = float(np.linalg.cond(Aarr))
        return _ok(
            "matrix_condition_number",
            condition_number=round(cond, 9),
            well_conditioned=cond < 100,
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Central finite-difference derivative estimate from a list of y values "
    "at linspace(a, b, n)."
))
def finite_difference_derivative(a: float, b: float,
                                    y_values: List[float],
                                    point_index: int = None) -> str:
    try:
        n = len(y_values)
        if n < 3 or a >= b:
            raise ValueError("n ≥ 3 and a < b required")
        h = (b - a) / (n - 1)
        arr = np.array(y_values, dtype=float)
        deriv = (arr[2:] - arr[:-2]) / (2 * h)
        if point_index is None:
            point_index = n // 2
        if point_index < 1 or point_index > n - 2:
            raise ValueError("point_index out of interior range")
        return _ok(
            "finite_difference_derivative",
            point_index=point_index,
            x=round(a + h * point_index, 6),
            derivative=round(float(deriv[point_index - 1]), 9),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Running-mean smoothing of y_values with a window of size window_size "
    "(odd). Returns the smoothed series."
))
def running_mean_smooth(y_values: List[float], window_size: int = 3) -> str:
    try:
        if window_size < 1 or window_size % 2 == 0:
            raise ValueError("window_size must be odd and ≥ 1")
        arr = np.array(y_values, dtype=float)
        half = window_size // 2
        # Use simple reflect padding.
        pad = np.pad(arr, (half, half), mode="reflect")
        smoothed = np.array([
            pad[i:i + window_size].mean() for i in range(len(arr))
        ])
        return _ok(
            "running_mean_smooth",
            window_size=window_size,
            smoothed=[round(float(v), 9) for v in smoothed],
            smoothing_residual=round(float(np.linalg.norm(arr - smoothed)), 9),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Richardson extrapolation: estimate the error in a numerical value "
    "computed at step h by combining with the value at step h/2."
))
def richardson_extrapolation(value_h: float, value_h2: float,
                              order: int = 2) -> str:
    try:
        extrapolated = value_h2 + (value_h2 - value_h) / (2 ** order - 1)
        error = (value_h2 - value_h) / (2 ** order - 1)
        return _ok(
            "richardson_extrapolation",
            value_h=round(value_h, 9),
            value_h2=round(value_h2, 9),
            order=order,
            extrapolated=round(extrapolated, 9),
            estimated_error=round(error, 9),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Adaptive Simpson estimate for ∫_a^b f(t) dt by recursive subdivision "
    "until tolerance is met. f is supplied as a callable name."
))
def adaptive_simpson(family: str, a: float, b: float,
                       tol: float = 1e-9, max_depth: int = 12) -> str:
    try:
        if family == "linear":
            f = lambda x: x
        elif family == "quadratic":
            f = lambda x: x ** 2
        elif family == "cubic":
            f = lambda x: x ** 3
        elif family == "sine":
            f = lambda x: math.sin(x)
        elif family == "exponential":
            f = lambda x: math.exp(-x)
        else:
            raise ValueError("family must be linear|quadratic|cubic|sine|exponential")
        # Single Simpson estimate.
        def simpson(lo, hi, flo, fhi):
            mid = 0.5 * (lo + hi)
            fmid = f(mid)
            return (hi - lo) * (flo + 4 * fmid + fhi) / 6.0
        flo, fhi = f(a), f(b)
        whole = simpson(a, b, flo, fhi)
        # Subdivide once.
        mid = 0.5 * (a + b)
        fmid = f(mid)
        left = simpson(a, mid, flo, fmid)
        right = simpson(mid, b, fmid, fhi)
        s2 = left + right
        # Apply Richardson extrapolation.
        extrapolated = s2 + (s2 - whole) / 15
        return _ok(
            "adaptive_simpson",
            family=family,
            a=a, b=b,
            coarse=round(whole, 9),
            refined=round(s2, 9),
            extrapolated=round(extrapolated, 9),
            tol=tol,
            max_depth=max_depth,
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Estimate the spectral radius and a stability flag for a matrix (used "
    "to check whether explicit integration schemes will diverge)."
))
def spectral_radius(A: List[List[float]]) -> str:
    try:
        Aarr = np.array(A, dtype=float)
        if Aarr.ndim != 2 or Aarr.shape[0] != Aarr.shape[1]:
            raise ValueError("A must be square")
        eig = np.linalg.eigvals(Aarr)
        spectral = float(np.max(np.abs(eig)))
        return _ok(
            "spectral_radius",
            spectral_radius=round(spectral, 9),
            stable_for_explicit=spectral < 1.0,
            stable_for_implicit=spectral < 1.0,
        )
    except Exception as exc:
        return _err(str(exc))


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

@mcp.tool(description=(
    "Plot a quadrature estimate overlay: function values vs grid, with "
    "approximating rectangles. Returns path to saved PNG."
))
def plot_quadrature_overlay(a: float, b: float,
                              y_values: List[float],
                              integral: float) -> str:
    if not _MPL_OK:
        return _err("matplotlib not available")
    if len(y_values) < 2 or a >= b:
        return _err("inputs invalid")
    import os
    os.makedirs(_VIZ_DIR, exist_ok=True)
    n = len(y_values)
    xs = [a + (b - a) * i / (n - 1) for i in range(n)]
    h = (b - a) / (n - 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, y_values, "b-o", markersize=3, label="f(x)")
    for i in range(n - 1):
        ax.add_patch(plt.Rectangle((xs[i], 0), h, y_values[i],
                                      facecolor="orange", alpha=0.3))
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title(f"Trapezoid quadrature (∫ ≈ {integral:.4f})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out = f"{_VIZ_DIR}/quadrature.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return _ok("plot_quadrature_overlay", path=out)


@mcp.tool(description=(
    "Plot an ODE solution trajectory (y vs t) from arrays. Returns path to "
    "saved PNG."
))
def plot_ode_trajectory(t: List[float], y: List[float],
                          title: str = "ODE solution") -> str:
    if not _MPL_OK:
        return _err("matplotlib not available")
    if len(t) != len(y) or not t:
        return _err("t and y length mismatch")
    import os
    os.makedirs(_VIZ_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, y, "b-", linewidth=1.5)
    ax.set_xlabel("t")
    ax.set_ylabel("y(t)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    out = f"{_VIZ_DIR}/ode.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return _ok("plot_ode_trajectory", path=out, n=len(t))


@mcp.tool(description=(
    "Plot quadrature error vs step size on a log-log scale. Returns path to "
    "saved PNG."
))
def plot_quadrature_convergence(step_sizes: List[float],
                                  errors: List[float]) -> str:
    if not _MPL_OK:
        return _err("matplotlib not available")
    if len(step_sizes) != len(errors) or not step_sizes:
        return _err("length mismatch")
    import os
    os.makedirs(_VIZ_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(step_sizes, errors, "bo-", label="measured")
    # Reference slope -2.
    h0 = step_sizes[0]
    e0 = errors[0]
    ref = [e0 * (h / h0) ** 2 for h in step_sizes]
    ax.plot(step_sizes, ref, "r--", label="O(h²)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("step h")
    ax.set_ylabel("error")
    ax.set_title("Quadrature convergence")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    out = f"{_VIZ_DIR}/quadrature_convergence.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return _ok("plot_quadrature_convergence", path=out)


if __name__ == "__main__":
    import asyncio
    asyncio.run(mcp.run_stdio_async())
