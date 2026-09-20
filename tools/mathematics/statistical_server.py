#!/usr/bin/env python3
"""Statistical mathematics MCP server.

Tools ported from
SciAgentGYM-main/toolkits/statistics/statistical_analysis/
  - asymptotic_statistics_toolkit_claude_12.py  (WLS, sandwich covariance, hypothesis test, CI)
  - proximal_mediation_eif_toolkit_claude_5.py  (mediation parameter ψ, EIF component)
plus stdlib implementations of time-series simulation and basic
hypothesis-test helpers.

All implementations are stdlib + numpy + scipy.stats only. Every tool
returns a JSON string. Tool naming is snake_case.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    from scipy import stats as _stats  # noqa: F401
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("math-statistical")


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

_VIZ_DIR = "/tmp/scimas_math_statistical"


# ---------------------------------------------------------------------------
# Weighted least squares / sandwich covariance
# ---------------------------------------------------------------------------

@mcp.tool(description=(
    "Estimate WLS regression parameters: β = (Xᵀ W X)⁻¹ Xᵀ W y, with "
    "residual standard error and R²."
))
def estimate_wls_parameters(X: List[List[float]],
                              y: List[float],
                              weights: List[float] = None) -> str:
    try:
        Xarr = np.array(X, dtype=float)
        yarr = np.array(y, dtype=float)
        if Xarr.ndim != 2:
            raise ValueError("X must be 2-D")
        n_obs, p = Xarr.shape
        if yarr.shape != (n_obs,):
            raise ValueError("y length must match X rows")
        if weights is None:
            W = np.eye(n_obs)
        else:
            W = np.diag(np.array(weights, dtype=float))
        XtW = Xarr.T @ W
        XtWX = XtW @ Xarr
        XtWy = XtW @ yarr
        try:
            beta = np.linalg.solve(XtWX, XtWy)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(XtWX, XtWy, rcond=None)[0]
        resid = yarr - Xarr @ beta
        s2 = float(resid @ W @ resid) / max(1, n_obs - p)
        # R² for unweighted case (informative).
        y_mean = float(np.mean(yarr))
        ss_tot = float(np.sum((yarr - y_mean) ** 2))
        ss_res = float(np.sum(resid ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return _ok(
            "estimate_wls_parameters",
            beta=[round(float(b), 9) for b in beta],
            residual_variance=round(s2, 9),
            r_squared=round(r2, 9),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Estimate asymptotic sandwich covariance H = K1 @ Γ1 @ K1ᵀ where "
    "K1 = (Xᵀ W X)⁻¹ Xᵀ W and Γ1 = diag of squared weighted residuals."
))
def compute_sandwich_covariance_estimator(X: List[List[float]],
                                            y: List[float],
                                            weights: List[float] = None) -> str:
    try:
        Xarr = np.array(X, dtype=float)
        yarr = np.array(y, dtype=float)
        if Xarr.ndim != 2:
            raise ValueError("X must be 2-D")
        n_obs, p = Xarr.shape
        if weights is None:
            W = np.eye(n_obs)
        else:
            W = np.diag(np.array(weights, dtype=float))
        K1 = np.linalg.solve(Xarr.T @ W @ Xarr, Xarr.T @ W)
        resid = yarr - Xarr @ (K1 @ yarr)
        Gamma1 = np.diag((W @ resid) ** 2)
        H = K1 @ Gamma1 @ K1.T
        return _ok(
            "compute_sandwich_covariance_estimator",
            shape=list(H.shape),
            covariance=json.dumps([[round(float(x), 9) for x in row]
                                    for row in H]),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Simulate a stationary AR(1)-like time-series with Gaussian noise. "
    "Returns n samples plus summary stats."
))
def simulate_stationary_time_series(n: int = 200,
                                      phi: float = 0.5,
                                      sigma: float = 1.0,
                                      seed: int = 42) -> str:
    try:
        if n <= 0:
            raise ValueError("n must be > 0")
        if abs(phi) >= 1.0:
            raise ValueError("|phi| must be < 1 for stationarity")
        rng = np.random.default_rng(seed)
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = phi * x[t - 1] + rng.normal(0.0, sigma)
        return _ok(
            "simulate_stationary_time_series",
            n=n,
            phi=phi,
            sigma=sigma,
            mean=round(float(x.mean()), 6),
            std=round(float(x.std()), 6),
            samples=[round(float(v), 4) for v in x[:20]],
        )
    except Exception as exc:
        return _err(str(exc))


# ---------------------------------------------------------------------------
# Hypothesis testing / confidence intervals
# ---------------------------------------------------------------------------

@mcp.tool(description=(
    "Two-sample z-test for equal means (assumes known sigma). Returns z, "
    "p-value (two-tailed), and 95% CI for the mean difference."
))
def two_sample_z_test(x: List[float], y: List[float],
                       sigma_x: float = None,
                       sigma_y: float = None) -> str:
    try:
        if not x or not y:
            raise ValueError("both samples required")
        xa, ya = np.array(x), np.array(y)
        nx, ny = len(xa), len(ya)
        sx = float(xa.std(ddof=1)) if sigma_x is None else float(sigma_x)
        sy = float(ya.std(ddof=1)) if sigma_y is None else float(sigma_y)
        se = math.sqrt(sx ** 2 / nx + sy ** 2 / ny)
        if se == 0:
            raise ValueError("standard error is zero")
        z = (xa.mean() - ya.mean()) / se
        # Two-tailed p-value (normal).
        if _SCIPY_OK:
            p = 2 * (1 - _stats.norm.cdf(abs(z)))
        else:
            # erfc-based fallback.
            p = math.erfc(abs(z) / math.sqrt(2))
        ci_low = (xa.mean() - ya.mean()) - 1.96 * se
        ci_high = (xa.mean() - ya.mean()) + 1.96 * se
        return _ok(
            "two_sample_z_test",
            n_x=nx,
            n_y=ny,
            mean_diff=round(float(xa.mean() - ya.mean()), 6),
            z=round(float(z), 6),
            p_value=round(float(p), 9),
            ci_95=[round(ci_low, 6), round(ci_high, 6)],
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Two-sided t-test (Welch's variant). Returns t, df (Welch–Satterthwaite), "
    "p-value, and 95% CI."
))
def welch_t_test(x: List[float], y: List[float]) -> str:
    try:
        if not x or not y:
            raise ValueError("both samples required")
        xa, ya = np.array(x), np.array(y)
        nx, ny = len(xa), len(ya)
        mx, my = float(xa.mean()), float(ya.mean())
        vx, vy = float(xa.var(ddof=1)), float(ya.var(ddof=1))
        se = math.sqrt(vx / nx + vy / ny)
        if se == 0:
            raise ValueError("standard error is zero")
        t = (mx - my) / se
        df = (vx / nx + vy / ny) ** 2 / (
            (vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1)
        )
        if _SCIPY_OK:
            p = 2 * (1 - _stats.t.cdf(abs(t), df))
        else:
            # Crude normal fallback.
            p = math.erfc(abs(t) / math.sqrt(2))
        return _ok(
            "welch_t_test",
            n_x=nx,
            n_y=ny,
            mean_diff=round(mx - my, 6),
            t=round(t, 6),
            df=round(df, 4),
            p_value=round(p, 9),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "95% confidence interval for the mean using sample mean, sample std, "
    "and a t/z multiplier (default 1.96 for z, or scipy t for small n)."
))
def confidence_interval_for_mean(sample: List[float],
                                    confidence: float = 0.95,
                                    use_t: bool = False) -> str:
    try:
        if not sample:
            raise ValueError("sample required")
        arr = np.array(sample)
        n = len(arr)
        m = float(arr.mean())
        s = float(arr.std(ddof=1))
        if use_t and _SCIPY_OK:
            mult = float(_stats.t.ppf((1 + confidence) / 2, n - 1))
        else:
            # z-multiplier via inverse erfc.
            alpha = 1 - confidence
            mult = math.sqrt(2) * 1.4821  # placeholder
            try:
                from scipy.special import erfinv
                mult = math.sqrt(2) * erfinv(1 - alpha)
            except ImportError:
                mult = 1.96
        se = s / math.sqrt(n)
        return _ok(
            "confidence_interval_for_mean",
            n=n,
            mean=round(m, 6),
            std=round(s, 6),
            confidence=confidence,
            ci=[round(m - mult * se, 6), round(m + mult * se, 6)],
        )
    except Exception as exc:
        return _err(str(exc))


# ---------------------------------------------------------------------------
# Mediation / EIF helpers
# ---------------------------------------------------------------------------

@mcp.tool(description=(
    "Compute the EIF-style estimator for a treatment effect from per-row "
    "pseudo-outcomes. Returns ψ̂ and its variance."
))
def estimate_eif_psi(pseudo_outcomes: List[float],
                      weights: List[float] = None) -> str:
    try:
        if not pseudo_outcomes:
            raise ValueError("pseudo_outcomes required")
        psi_arr = np.array(pseudo_outcomes, dtype=float)
        if weights is None:
            psi_hat = float(psi_arr.mean())
            var = float(psi_arr.var(ddof=1) / len(psi_arr))
        else:
            w = np.array(weights, dtype=float)
            if w.shape != psi_arr.shape:
                raise ValueError("weights shape mismatch")
            psi_hat = float((w * psi_arr).sum() / w.sum())
            # Approximate variance.
            var = float((w ** 2 * (psi_arr - psi_hat) ** 2).sum() / w.sum() ** 2)
        return _ok(
            "estimate_eif_psi",
            psi=round(psi_hat, 6),
            variance=round(var, 9),
            standard_error=round(math.sqrt(var), 6),
            n=len(psi_arr),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Compute mediation parameter psi and 95% CI using a Wald-style "
    "approximation from observed treatment effect and bootstrap SE."
))
def estimate_mediation_parameter_psi(treatment_effect: float,
                                       indirect_effect: float,
                                       direct_effect: float,
                                       bootstrap_se: float) -> str:
    try:
        # Natural direct effect.
        psi = direct_effect + indirect_effect / 2.0
        z = psi / bootstrap_se if bootstrap_se > 0 else 0.0
        ci_low = psi - 1.96 * bootstrap_se
        ci_high = psi + 1.96 * bootstrap_se
        return _ok(
            "estimate_mediation_parameter_psi",
            psi=round(psi, 6),
            bootstrap_se=round(bootstrap_se, 6),
            z=round(z, 6),
            ci_95=[round(ci_low, 6), round(ci_high, 6)],
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Bootstrap standard error for a statistic over a sample (default 200 "
    "resamples). Returns SE and a 95% CI."
))
def bootstrap_standard_error(data: List[float],
                               statistic: str = "mean",
                               n_resamples: int = 200,
                               seed: int = 42) -> str:
    try:
        if not data:
            raise ValueError("data required")
        if statistic not in ("mean", "median", "std"):
            raise ValueError("statistic must be mean|median|std")
        rng = np.random.default_rng(seed)
        arr = np.array(data)
        n = len(arr)
        if statistic == "mean":
            stat_fn = np.mean
        elif statistic == "median":
            stat_fn = np.median
        else:
            stat_fn = lambda a: a.std(ddof=1)
        boot_stats = []
        for _ in range(n_resamples):
            idx = rng.integers(0, n, n)
            boot_stats.append(float(stat_fn(arr[idx])))
        boot_stats_arr = np.array(boot_stats)
        point = float(stat_fn(arr))
        se = float(boot_stats_arr.std(ddof=1))
        return _ok(
            "bootstrap_standard_error",
            statistic=statistic,
            point_estimate=round(point, 6),
            bootstrap_se=round(se, 6),
            ci_95=[round(float(np.percentile(boot_stats_arr, 2.5)), 6),
                   round(float(np.percentile(boot_stats_arr, 97.5)), 6)],
            n_resamples=n_resamples,
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Compute the Pearson correlation coefficient and 95% CI (Fisher-z "
    "approximation) for paired samples."
))
def correlation_with_ci(x: List[float], y: List[float]) -> str:
    try:
        if len(x) != len(y) or len(x) < 3:
            raise ValueError("x, y same length ≥ 3")
        xa, ya = np.array(x), np.array(y)
        r = float(np.corrcoef(xa, ya)[0, 1])
        n = len(xa)
        # Fisher-z CI.
        z = 0.5 * math.log((1 + r) / (1 - r)) if abs(r) < 1.0 else math.inf
        se = 1.0 / math.sqrt(n - 3)
        z_lo = z - 1.96 * se
        z_hi = z + 1.96 * se
        r_lo = math.tanh(z_lo)
        r_hi = math.tanh(z_hi)
        if _SCIPY_OK:
            t = r * math.sqrt((n - 2) / max(1e-12, 1 - r ** 2))
            p = 2 * (1 - _stats.t.cdf(abs(t), n - 2))
        else:
            p = math.erfc(abs(z))
        return _ok(
            "correlation_with_ci",
            n=n,
            r=round(r, 6),
            ci_95=[round(r_lo, 6), round(r_hi, 6)],
            p_value=round(float(p), 9),
        )
    except Exception as exc:
        return _err(str(exc))


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

@mcp.tool(description=(
    "Plot a histogram of the sample with a normal overlay. Returns path to "
    "saved PNG."
))
def plot_sample_distribution(sample: List[float],
                                bins: int = 20) -> str:
    if not _MPL_OK:
        return _err("matplotlib not available")
    if not sample:
        return _err("sample is empty")
    if bins < 2:
        return _err("bins must be >= 2")
    import os
    os.makedirs(_VIZ_DIR, exist_ok=True)
    arr = np.array(sample, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(arr, bins=bins, density="True", alpha=0.7, color="tab:blue")
    m, s = float(arr.mean()), float(arr.std(ddof=1))
    xs = np.linspace(arr.min(), arr.max(), 200)
    pdf = (1 / (s * math.sqrt(2 * math.pi))) * np.exp(-(xs - m) ** 2 / (2 * s ** 2))
    ax.plot(xs, pdf, "r-", label=f"N({m:.2f}, {s:.2f}²)")
    ax.set_xlabel("value")
    ax.set_ylabel("density")
    ax.set_title(f"Sample distribution (n = {len(arr)})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out = f"{_VIZ_DIR}/sample_distribution.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return _ok("plot_sample_distribution", path=out)


@mcp.tool(description=(
    "Plot bootstrap-distribution histogram with mean and 95% CI markers. "
    "Returns path to saved PNG."
))
def plot_bootstrap_distribution(bootstrap_stats: List[float],
                                  ci_low: float = None,
                                  ci_high: float = None,
                                  bins: int = 30) -> str:
    if not _MPL_OK:
        return _err("matplotlib not available")
    if not bootstrap_stats:
        return _err("bootstrap_stats empty")
    import os
    os.makedirs(_VIZ_DIR, exist_ok=True)
    arr = np.array(bootstrap_stats, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(arr, bins=bins, color="tab:purple", alpha=0.7)
    ax.axvline(float(arr.mean()), color="r", linestyle="--",
                 label=f"mean = {arr.mean():.4f}")
    if ci_low is not None:
        ax.axvline(ci_low, color="k", linestyle=":",
                     label=f"CI low = {ci_low:.4f}")
    if ci_high is not None:
        ax.axvline(ci_high, color="k", linestyle="--",
                     label=f"CI high = {ci_high:.4f}")
    ax.set_xlabel("bootstrap statistic")
    ax.set_ylabel("count")
    ax.set_title(f"Bootstrap distribution ({len(arr)} resamples)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out = f"{_VIZ_DIR}/bootstrap.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return _ok("plot_bootstrap_distribution", path=out)


if __name__ == "__main__":
    import asyncio
    asyncio.run(mcp.run_stdio_async())
