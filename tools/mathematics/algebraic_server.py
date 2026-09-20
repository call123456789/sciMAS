#!/usr/bin/env python3
"""Algebraic mathematics MCP server.

Tools ported from
SciAgentGYM-main/toolkits/statistics/statistical_analysis/
  - asymptotic_statistics_toolkit_claude_12.py  (matrix K1, Gamma1, sandwich covariance)
  - manifold_spectral_convergence_toolkit_claude_7.py  (degree matrix, laplacian)
plus stdlib implementations of Hermitian / unitary checks, matrix trace,
matrix exponential, and graph Laplacian eigenpairs.

All implementations are stdlib + numpy only. Every tool returns a JSON
string. Tool naming is snake_case.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Tuple

import numpy as np

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("math-algebraic")


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

_VIZ_DIR = "/tmp/scimas_math_algebraic"


# ---------------------------------------------------------------------------
# Linear-algebra primitives
# ---------------------------------------------------------------------------

@mcp.tool(description=(
    "Parse a flat numeric list into a square (n,n) numpy matrix."
))
def parse_matrix_string(rows: List[List[float]]) -> str:
    try:
        if not rows:
            raise ValueError("rows is empty")
        n = len(rows)
        for i, r in enumerate(rows):
            if len(r) != n:
                raise ValueError(f"row {i} has length {len(r)} ≠ {n}")
        A = np.array(rows, dtype=float)
        return _ok("parse_matrix_string", shape=list(A.shape),
                   matrix=json.dumps(A.tolist()))
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Check whether a matrix is Hermitian (or symmetric for real entries) "
    "within tolerance."
))
def is_hermitian(matrix: List[List[float]], tol: float = 1e-9) -> str:
    try:
        A = np.array(matrix, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("matrix must be square")
        diff = np.max(np.abs(A - A.T))
        return _ok(
            "is_hermitian",
            max_asymmetry=round(float(diff), 12),
            tolerance=tol,
            is_hermitian=bool(diff < tol),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Check whether a matrix is unitary (A^T U = I) within tolerance."
))
def is_unitary(matrix: List[List[float]], tol: float = 1e-9) -> str:
    try:
        A = np.array(matrix, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("matrix must be square")
        n = A.shape[0]
        delta = np.max(np.abs(A.T @ A - np.eye(n)))
        return _ok(
            "is_unitary",
            max_orthogonality_error=round(float(delta), 12),
            tolerance=tol,
            is_unitary=bool(delta < tol),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Compute matrix trace, determinant, Frobenius norm, and rank."
))
def matrix_properties(matrix: List[List[float]]) -> str:
    try:
        A = np.array(matrix, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("matrix must be square")
        return _ok(
            "matrix_properties",
            shape=list(A.shape),
            trace=round(float(np.trace(A)), 9),
            determinant=round(float(np.linalg.det(A)), 9),
            frobenius_norm=round(float(np.linalg.norm(A, "fro")), 9),
            rank=int(np.linalg.matrix_rank(A)),
            eigenvalues_real=[round(float(x), 6) for x in np.linalg.eigvals(A).real],
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Compute matrix exponential via the closed-form eigendecomposition."
))
def matrix_exponential(matrix: List[List[float]]) -> str:
    try:
        A = np.array(matrix, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("matrix must be square")
        # For general A use scipy.linalg.expm; otherwise closed-form.
        try:
            from scipy.linalg import expm  # noqa: F401
            result = expm(A)
        except ImportError:
            vals, vecs = np.linalg.eig(A)
            result = (vecs * np.exp(vals)) @ np.linalg.inv(vecs)
        return _ok(
            "matrix_exponential",
            shape=list(A.shape),
            result=json.dumps([[round(float(x), 9) for x in row]
                                for row in result.real]),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Compute eigenvalues and (optionally) eigenvectors of a Hermitian "
    "matrix, sorted ascending."
))
def hermitian_eigendecomposition(matrix: List[List[float]],
                                  compute_vectors: bool = True) -> str:
    try:
        A = np.array(matrix, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("matrix must be square")
        sym = (A + A.T) / 2.0
        if compute_vectors:
            vals, vecs = np.linalg.eigh(sym)
            return _ok(
                "hermitian_eigendecomposition",
                eigenvalues=[round(float(v), 9) for v in vals],
                eigenvectors=json.dumps(
                    [[round(float(x), 6) for x in col] for col in vecs.T]),
            )
        vals = np.linalg.eigvalsh(sym)
        return _ok(
            "hermitian_eigendecomposition",
            eigenvalues=[round(float(v), 9) for v in vals],
        )
    except Exception as exc:
        return _err(str(exc))


# ---------------------------------------------------------------------------
# Graph-Laplacian primitives (from manifold_spectral_convergence_toolkit_claude_7)
# ---------------------------------------------------------------------------

@mcp.tool(description=(
    "Compute the degree matrix from a list of edge weights indexed by "
    "row-sum of an adjacency matrix."
))
def compute_degree_matrix(adjacency: List[List[float]]) -> str:
    try:
        W = np.array(adjacency, dtype=float)
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError("adjacency must be square")
        d = W.sum(axis=1)
        D = np.diag(d)
        return _ok(
            "compute_degree_matrix",
            shape=list(W.shape),
            degrees=[round(float(x), 4) for x in d],
            diagonal=json.dumps([[round(float(x), 4) for x in row]
                                 for row in D]),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Compute the unnormalized graph Laplacian L = D − W from an adjacency."
))
def compute_graph_laplacian(adjacency: List[List[float]]) -> str:
    try:
        W = np.array(adjacency, dtype=float)
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError("adjacency must be square")
        d = W.sum(axis=1)
        L = np.diag(d) - W
        return _ok(
            "compute_graph_laplacian",
            shape=list(W.shape),
            laplacian=json.dumps([[round(float(x), 6) for x in row]
                                    for row in L]),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Compute the random-walk normalised Laplacian L_rw = D^-1 (D − W). "
    "Returns identity-safe result when D has zero entries."
))
def compute_random_walk_laplacian(adjacency: List[List[float]]) -> str:
    try:
        W = np.array(adjacency, dtype=float)
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError("adjacency must be square")
        d = W.sum(axis=1)
        if np.any(d <= 0):
            raise ValueError("random-walk Laplacian requires positive degree")
        D_inv = np.diag(1.0 / d)
        L_rw = np.eye(W.shape[0]) - D_inv @ W
        return _ok(
            "compute_random_walk_laplacian",
            laplacian=json.dumps([[round(float(x), 6) for x in row]
                                    for row in L_rw]),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Compute the smallest K eigenvalues/eigenvectors of a graph Laplacian "
    "for spectral-clustering style analysis."
))
def compute_laplacian_eigenpairs(adjacency: List[List[float]],
                                  K: int = 4) -> str:
    try:
        W = np.array(adjacency, dtype=float)
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError("adjacency must be square")
        d = W.sum(axis=1)
        L = np.diag(d) - W
        if K < 1 or K > W.shape[0]:
            raise ValueError("K must be in [1, n]")
        vals, vecs = np.linalg.eigh(L)
        vals = vals[:K]
        vecs = vecs[:, :K]
        return _ok(
            "compute_laplacian_eigenpairs",
            K=K,
            eigenvalues=[round(float(v), 9) for v in vals],
            eigenvectors=json.dumps(
                [[round(float(x), 6) for x in col] for col in vecs.T]),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Compute a Gaussian affinity (RBF) matrix between rows of a feature "
    "matrix."
))
def compute_gaussian_affinity(X: List[List[float]],
                               epsilon: float = 1.0) -> str:
    try:
        Xarr = np.array(X, dtype=float)
        if Xarr.ndim != 2:
            raise ValueError("X must be 2-D")
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        # Squared pairwise distances.
        sq = np.sum(Xarr ** 2, axis=1, keepdims=True)
        d2 = sq + sq.T - 2 * Xarr @ Xarr.T
        W = np.exp(-d2 / (epsilon ** 2))
        np.fill_diagonal(W, 0.0)
        return _ok(
            "compute_gaussian_affinity",
            shape=list(W.shape),
            epsilon=epsilon,
            affinity=json.dumps([[round(float(x), 6) for x in row]
                                  for row in W]),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Compute sandwich-style covariance H = K1 @ Gamma1 @ K1.T for "
    "weighted-least-squares style estimator."
))
def compute_sandwich_covariance(K1: List[List[float]],
                                  Gamma1: List[List[float]]) -> str:
    try:
        A = np.array(K1, dtype=float)
        G = np.array(Gamma1, dtype=float)
        if A.ndim != 2 or G.ndim != 2:
            raise ValueError("inputs must be 2-D")
        if A.shape[1] != G.shape[0] or G.shape[0] != G.shape[1]:
            raise ValueError("shape mismatch: K1 (m×p), Gamma1 (p×p)")
        H = A @ G @ A.T
        return _ok(
            "compute_sandwich_covariance",
            shape=list(H.shape),
            covariance=json.dumps([[round(float(x), 9) for x in row]
                                    for row in H]),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Solve a small dense linear system Ax = b using numpy's LU factorisation."
))
def solve_linear_system(A: List[List[float]], b: List[float]) -> str:
    try:
        Aarr = np.array(A, dtype=float)
        barr = np.array(b, dtype=float)
        if Aarr.ndim != 2 or Aarr.shape[0] != Aarr.shape[1]:
            raise ValueError("A must be square")
        if Aarr.shape[0] != len(barr):
            raise ValueError("A and b dimension mismatch")
        x = np.linalg.solve(Aarr, barr)
        residual = float(np.linalg.norm(Aarr @ x - barr))
        return _ok(
            "solve_linear_system",
            solution=[round(float(v), 9) for v in x],
            residual=round(residual, 12),
        )
    except Exception as exc:
        return _err(str(exc))


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

@mcp.tool(description=(
    "Plot a matrix as a heatmap. Returns path to saved PNG."
))
def plot_matrix_heatmap(matrix: List[List[float]],
                          title: str = "matrix heatmap") -> str:
    if not _MPL_OK:
        return _err("matplotlib not available")
    if not matrix or not matrix[0]:
        return _err("matrix is empty")
    import os
    os.makedirs(_VIZ_DIR, exist_ok=True)
    M = np.array(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(M, cmap="viridis", aspect="auto")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="value")
    out = f"{_VIZ_DIR}/matrix_heatmap.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return _ok("plot_matrix_heatmap", path=out, shape=list(M.shape))


@mcp.tool(description=(
    "Plot the sorted eigenvalues of a matrix as a 'scree' plot. Returns "
    "path to saved PNG."
))
def plot_eigenvalue_spectrum(eigenvalues: List[float],
                                title: str = "eigenvalue spectrum") -> str:
    if not _MPL_OK:
        return _err("matplotlib not available")
    if not eigenvalues:
        return _err("eigenvalues empty")
    import os
    os.makedirs(_VIZ_DIR, exist_ok=True)
    vals = sorted(eigenvalues)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(vals) + 1), vals, "bo-")
    ax.set_xlabel("index")
    ax.set_ylabel("eigenvalue")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    out = f"{_VIZ_DIR}/eigenvalues.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return _ok("plot_eigenvalue_spectrum", path=out, n=len(vals))


if __name__ == "__main__":
    import asyncio
    asyncio.run(mcp.run_stdio_async())
