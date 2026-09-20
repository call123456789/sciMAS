#!/usr/bin/env python3
"""Geometric mathematics MCP server.

Tools ported from
SciAgentGYM-main/toolkits/statistics/statistical_analysis/
  - manifold_spectral_convergence_toolkit_claude_7.py
    (sphere / torus sampling, sphere eigenfunctions, alignment sign, "
    "theoretical error bound, spectral convergence error, bandwidth "
    "scaling)

All implementations are stdlib + numpy only. Every tool returns a JSON
string. Tool naming is snake_case.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List

import numpy as np

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("math-geometric")


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

_VIZ_DIR = "/tmp/scimas_math_geometric"


# ---------------------------------------------------------------------------
# Manifold sampling
# ---------------------------------------------------------------------------

@mcp.tool(description=(
    "Sample n points uniformly on the unit sphere in d dimensions."
))
def sample_sphere(n: int, d: int, seed: int = 42) -> str:
    try:
        if n <= 0 or d < 1:
            raise ValueError("n > 0 and d ≥ 1 required")
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, d))
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.maximum(norms, 1e-12)
        return _ok(
            "sample_sphere",
            n=n,
            d=d,
            samples=json.dumps([[round(float(x), 6) for x in row] for row in X[:10]]),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Sample n points uniformly on a torus embedded in 3D with major radius "
    "R and minor radius r."
))
def sample_torus(n: int, R: float = 2.0, r: float = 1.0, seed: int = 42) -> str:
    try:
        if n <= 0 or R <= 0 or r <= 0:
            raise ValueError("n, R, r must be > 0")
        rng = np.random.default_rng(seed)
        u = rng.uniform(0, 2 * math.pi, n)
        v = rng.uniform(0, 2 * math.pi, n)
        x = (R + r * np.cos(v)) * np.cos(u)
        y = (R + r * np.cos(v)) * np.sin(u)
        z = r * np.sin(v)
        return _ok(
            "sample_torus",
            n=n,
            R=R,
            r=r,
            samples=json.dumps(
                [[round(float(x), 4), round(float(y), 4), round(float(z), 4)]
                 for x, y, z in zip(x[:10], y[:10], z[:10])]),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Build a graph Laplacian from samples (Gaussian RBF affinity) and "
    "return the smallest K eigenpairs."
))
def build_graph_laplacian_from_samples(X: List[List[float]],
                                          epsilon: float = 0.5,
                                          K: int = 4) -> str:
    try:
        Xarr = np.array(X, dtype=float)
        if Xarr.ndim != 2:
            raise ValueError("X must be 2-D")
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        sq = np.sum(Xarr ** 2, axis=1, keepdims=True)
        d2 = sq + sq.T - 2 * Xarr @ Xarr.T
        W = np.exp(-d2 / (epsilon ** 2))
        np.fill_diagonal(W, 0.0)
        d = W.sum(axis=1)
        L = np.diag(d) - W
        if K < 1 or K > L.shape[0]:
            raise ValueError("K must be in [1, n]")
        vals, vecs = np.linalg.eigh(L)
        vals = vals[:K]
        vecs = vecs[:, :K]
        return _ok(
            "build_graph_laplacian_from_samples",
            K=K,
            epsilon=epsilon,
            eigenvalues=[round(float(v), 6) for v in vals],
            eigenvectors=json.dumps(
                [[round(float(x), 6) for x in col] for col in vecs.T]),
        )
    except Exception as exc:
        return _err(str(exc))


# ---------------------------------------------------------------------------
# Eigenfunctions on the sphere
# ---------------------------------------------------------------------------

@mcp.tool(description=(
    "Compute the analytic eigenvalue of the k-th spherical harmonic on "
    "S^(d-1)."
))
def sphere_laplacian_eigenvalue(k: int, d: int) -> str:
    try:
        if k < 0 or d < 2:
            raise ValueError("k ≥ 0 and d ≥ 2 required")
        # Eigenvalues of the Laplace-Beltrami operator on S^(d-1) for
        # degree-k spherical harmonics: λ_k = k(k + d − 2).
        eig = k * (k + d - 2)
        return _ok(
            "sphere_laplacian_eigenvalue",
            k=k,
            d=d,
            eigenvalue=eig,
            multiplicity=math.comb(k + d - 2, k) - math.comb(k + d - 4, k - 2) if k >= 2 else 1,
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Evaluate a Y_1,0 spherical harmonic (z-direction) on points on the "
    "sphere. Returns the function values."
))
def evaluate_sphere_eigenfunction(X: List[List[float]], k: int = 1) -> str:
    try:
        Xarr = np.array(X, dtype=float)
        if Xarr.ndim != 2:
            raise ValueError("X must be 2-D")
        if Xarr.shape[1] < 3:
            # Embedding dimension must allow for an axis.
            Xarr = np.hstack([Xarr, np.zeros((Xarr.shape[0], 3 - Xarr.shape[1]))])
        if k == 1:
            f = Xarr[:, 2]  # Y_1,0 ∝ z
        elif k == 2:
            f = Xarr[:, 0] ** 2 - Xarr[:, 1] ** 2  # Y_2,2-like
        else:
            f = Xarr[:, 2] ** k
        return _ok(
            "evaluate_sphere_eigenfunction",
            k=k,
            n=Xarr.shape[0],
            values=[round(float(v), 6) for v in f],
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Compute a sampling vector from per-sample eigenfunction values using "
    "a power p."
))
def compute_sampling_vector(f_values: List[float],
                              n: int = None,
                              p: float = 1.0) -> str:
    try:
        if not f_values:
            raise ValueError("f_values required")
        arr = np.array(f_values, dtype=float)
        weights = np.power(np.abs(arr) + 1e-12, p)
        weights = weights / weights.sum()
        return _ok(
            "compute_sampling_vector",
            n=len(arr),
            p=p,
            weights=[round(float(w), 6) for w in weights],
            min_weight=round(float(weights.min()), 6),
            max_weight=round(float(weights.max()), 6),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Compute the alignment sign that flips a vector v to maximally align "
    "with reference φ."
))
def compute_alignment_sign(v: List[float], phi: List[float]) -> str:
    try:
        if len(v) != len(phi):
            raise ValueError("v and phi must be same length")
        va = np.array(v)
        pa = np.array(phi)
        s = +1.0 if float(np.dot(va, pa)) >= 0 else -1.0
        aligned = s * va
        return _ok(
            "compute_alignment_sign",
            sign=float(s),
            aligned=[round(float(x), 6) for x in aligned],
            cosine=round(float(np.dot(va, pa) /
                                (np.linalg.norm(va) * np.linalg.norm(pa) + 1e-12)), 6),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Euclidean error ‖α v − φ‖₂ between a (possibly scaled) vector and "
    "reference φ."
))
def compute_euclidean_error(v: List[float], phi: List[float],
                              alpha: float = 1.0) -> str:
    try:
        if len(v) != len(phi):
            raise ValueError("v and phi must be same length")
        va = np.array(v)
        pa = np.array(phi)
        err = float(np.linalg.norm(alpha * va - pa))
        return _ok(
            "compute_euclidean_error",
            alpha=alpha,
            error=round(err, 9),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Theoretical error bound C * ε^α for spectral-convergence style "
    "analysis."
))
def theoretical_error_bound(epsilon: float, n: int, d: int,
                              alpha: float = 1.0,
                              C: float = 1.0) -> str:
    try:
        if epsilon <= 0 or n <= 0 or d <= 0:
            raise ValueError("epsilon, n, d must be > 0")
        bound = C * (epsilon ** alpha) * math.log(max(2.0, n)) ** (1.0 / max(1.0, d))
        return _ok(
            "theoretical_error_bound",
            epsilon=epsilon,
            n=n,
            d=d,
            alpha=alpha,
            C=C,
            bound=round(bound, 9),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Spectral convergence error: build a graph Laplacian from samples and "
    "compare to the analytic eigenvalue for the requested harmonic k."
))
def compute_spectral_convergence_error(X: List[List[float]],
                                          epsilon: float,
                                          k: int,
                                          d: int = 3) -> str:
    try:
        Xarr = np.array(X, dtype=float)
        if Xarr.ndim != 2:
            raise ValueError("X must be 2-D")
        sq = np.sum(Xarr ** 2, axis=1, keepdims=True)
        d2 = sq + sq.T - 2 * Xarr @ Xarr.T
        W = np.exp(-d2 / (epsilon ** 2))
        np.fill_diagonal(W, 0.0)
        d_w = W.sum(axis=1)
        L = np.diag(d_w) - W
        vals = np.linalg.eigvalsh(L)
        # The first nonzero eigenvalue approximates k(k+d-2).
        if len(vals) < 2:
            raise ValueError("not enough eigenvalues to compute error")
        nonzero = vals[vals > 1e-9]
        if k - 1 >= len(nonzero):
            raise ValueError(f"only {len(nonzero)} non-zero eigenvalues; "
                             f"cannot compare to harmonic k={k}")
        observed = float(nonzero[k - 1])
        analytic = float(k * (k + d - 2))
        err = abs(observed - analytic) / max(1.0, analytic)
        return _ok(
            "compute_spectral_convergence_error",
            epsilon=epsilon,
            k=k,
            d=d,
            observed_eigenvalue=round(observed, 6),
            analytic_eigenvalue=analytic,
            relative_error=round(err, 6),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Bandwidth scaling sweep: vary n and report empirical error + theoretical "
    "bound for a fixed k."
))
def analyze_bandwidth_scaling(n_values: List[int],
                                d: int = 3,
                                k: int = 1,
                                epsilon: float = 0.5) -> str:
    try:
        if not n_values:
            raise ValueError("n_values required")
        rows = []
        rng = np.random.default_rng(0)
        for n in n_values:
            if n <= 0:
                continue
            X = rng.normal(size=(n, d))
            X = X / np.linalg.norm(X, axis=1, keepdims=True)
            sq = np.sum(X ** 2, axis=1, keepdims=True)
            d2 = sq + sq.T - 2 * X @ X.T
            W = np.exp(-d2 / (epsilon ** 2))
            np.fill_diagonal(W, 0.0)
            d_w = W.sum(axis=1)
            L = np.diag(d_w) - W
            vals = np.linalg.eigvalsh(L)
            nonzero = vals[vals > 1e-9]
            if k - 1 < len(nonzero):
                observed = float(nonzero[k - 1])
                analytic = float(k * (k + d - 2))
                emp_err = abs(observed - analytic) / max(1.0, analytic)
                theo = math.sqrt(2) * math.log(max(2.0, n)) / n ** 0.5
                rows.append({
                    "n": n,
                    "observed": round(observed, 6),
                    "empirical_error": round(emp_err, 6),
                    "theoretical_bound": round(theo, 6),
                })
        return _ok(
            "analyze_bandwidth_scaling",
            k=k,
            d=d,
            epsilon=epsilon,
            results=rows,
        )
    except Exception as exc:
        return _err(str(exc))


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

@mcp.tool(description=(
    "Scatter-plot sphere samples (3-D) projected to 2-D. Returns path to PNG."
))
def plot_sphere_samples(samples: List[List[float]],
                          title: str = "Sphere samples") -> str:
    if not _MPL_OK:
        return _err("matplotlib not available")
    if not samples:
        return _err("samples empty")
    import os
    os.makedirs(_VIZ_DIR, exist_ok=True)
    arr = np.array(samples, dtype=float)
    if arr.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(arr[:, 0], arr[:, 1], s=8, alpha=0.7)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    else:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.scatter(arr[:, 0], np.zeros(arr.shape[0]), s=8)
        ax.set_xlabel("x")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    out = f"{_VIZ_DIR}/sphere.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return _ok("plot_sphere_samples", path=out, n=arr.shape[0])


@mcp.tool(description=(
    "Plot a bandwidth-scaling curve: empirical error vs n. Returns path to PNG."
))
def plot_bandwidth_scaling(n_values: List[float],
                              empirical_errors: List[float],
                              theoretical_bounds: List[float] = None) -> str:
    if not _MPL_OK:
        return _err("matplotlib not available")
    if len(n_values) != len(empirical_errors):
        return _err("length mismatch")
    import os
    os.makedirs(_VIZ_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(n_values, empirical_errors, "bo-", label="empirical error")
    if theoretical_bounds and len(theoretical_bounds) == len(n_values):
        ax.plot(n_values, theoretical_bounds, "r--", label="theoretical bound")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("error")
    ax.set_title("Bandwidth scaling")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    out = f"{_VIZ_DIR}/bandwidth.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return _ok("plot_bandwidth_scaling", path=out)


if __name__ == "__main__":
    import asyncio
    asyncio.run(mcp.run_stdio_async())
