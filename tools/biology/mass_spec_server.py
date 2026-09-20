#!/usr/bin/env python3
"""Mass spectrometry MCP server.

Tools ported from
SciAgentGYM-main/toolkits/life_science/mass_spectrometry/
  - mass_spectrometry_toolkit_0050000.py  (chlorine isotope pattern)
  - mass_spectrometry_toolkit_0070000.py  (spectrum-to-SMILES matching)

All implementations are stdlib + numpy only. Every tool returns a JSON
string. Tool naming is snake_case.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from typing import Any, Dict, List, Tuple

import numpy as np

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("biology-mass-spec")


# Isotope natural abundances.
_ISOTOPES: Dict[str, Dict[int, float]] = {
    "Cl": {35: 0.7578, 37: 0.2422},
    "C":  {12: 0.9893, 13: 0.0107},
    "Br": {79: 0.5069, 81: 0.4931},
    "H":  {1: 0.9999, 2: 0.0001},
}


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

_VIZ_DIR = "/tmp/scimas_biology_mass_spec"


# ---------------------------------------------------------------------------
# Chlorine-isotope / isotope-cluster helpers
# ---------------------------------------------------------------------------

@mcp.tool(description=(
    "Compute the theoretical isotope pattern for n chlorines (and optional "
    "n carbons). Returns relative intensities at the +0, +2, +4 ... "
    "isotope peaks."
))
def calculate_theoretical_isotope_pattern(num_chlorine: int,
                                          num_carbon: int = 0,
                                          max_peaks: int = 8) -> str:
    try:
        if num_chlorine < 0 or num_carbon < 0:
            raise ValueError("counts must be ≥ 0")
        if max_peaks <= 0:
            raise ValueError("max_peaks must be > 0")
        # Build a discrete convolution over Cl + C isotope distributions.
        cl = _ISOTOPES["Cl"]
        c = _ISOTOPES["C"]
        cl_pmf = [cl[35], cl[37]]  # at mass offsets 0, 2
        c_pmf = [c[12], c[13]]     # at mass offsets 0, 1
        # Start with single Cl.
        cl_dist = np.array(cl_pmf, dtype=float)
        for _ in range(max(0, num_chlorine - 1)):
            cl_dist = np.convolve(cl_dist, np.array(cl_pmf, dtype=float))
        if num_chlorine == 0:
            cl_dist = np.array([1.0])
        c_dist = np.array(c_pmf, dtype=float)
        for _ in range(max(0, num_carbon - 1)):
            c_dist = np.convolve(c_dist, np.array(c_pmf, dtype=float))
        if num_carbon == 0:
            c_dist = np.array([1.0])
        # Convolve across both elements (with offset).
        # Resulting 2D array of size len(cl_dist) x len(c_dist); flatten with
        # offset = 2*i + j.
        flat = np.zeros(len(cl_dist) + len(c_dist) - 1 + 10)
        for i, p_cl in enumerate(cl_dist):
            for j, p_c in enumerate(c_dist):
                flat[2 * i + j] += p_cl * p_c
        # Normalise.
        flat = flat / flat.sum()
        peaks = [(round(2 * i + j, 0) if False else None, 0.0) for i, j in []]
        # Easier: rebuild a clean list of (m+offset, intensity) up to max_peaks.
        masses = []
        for offset in range(min(max_peaks, len(cl_dist) + len(c_dist))):
            # Sum over all (i, j) with 2*i + j == offset.
            val = 0.0
            for i in range(len(cl_dist)):
                for j in range(len(c_dist)):
                    if 2 * i + j == offset:
                        val += float(cl_dist[i]) * float(c_dist[j])
            masses.append({"offset": offset, "intensity": round(val, 6)})
        # Renormalise again.
        total = sum(p["intensity"] for p in masses) or 1.0
        for p in masses:
            p["intensity"] = round(p["intensity"] / total, 6)
        return _ok(
            "calculate_theoretical_isotope_pattern",
            num_chlorine=num_chlorine,
            num_carbon=num_carbon,
            peaks=masses,
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Identify the chlorine count from an observed M / M+2 intensity ratio."
))
def determine_chlorine_number_from_ratio(observed_ratio: float) -> str:
    try:
        if observed_ratio <= 0:
            raise ValueError("observed_ratio must be > 0")
        # Theoretical ratios for n Cl: C(n,k) * 0.7578^(n-k) * 0.2422^k at k=0..n
        # r = (M+2)/(M) = n * 0.2422 / 0.7578
        # Solve r = 0.3196 * n for low n.
        approx_n = observed_ratio / 0.3196
        nearest = int(round(approx_n))
        candidates = []
        for n in range(max(0, nearest - 1), nearest + 3):
            if n == 0:
                ratio = 0.0
            else:
                ratio = n * 0.2422 / 0.7578
            candidates.append({"n_chlorine": n, "expected_ratio": round(ratio, 4)})
        return _ok(
            "determine_chlorine_number_from_ratio",
            observed_ratio=observed_ratio,
            approx_n=round(approx_n, 2),
            nearest_n=nearest,
            candidates=candidates,
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Extract peak m/z / intensity pairs from a raw spectrum of "
    "(m/z, intensity) tuples. Filters by min_intensity."
))
def extract_peaks_from_spectrum(mz_values: List[float],
                                 intensities: List[float],
                                 min_intensity: float = 0.05) -> str:
    try:
        if len(mz_values) != len(intensities):
            raise ValueError("m/z and intensity arrays must match in length")
        peaks = [{"mz": round(m, 4), "intensity": round(i, 4)}
                 for m, i in zip(mz_values, intensities) if i >= min_intensity]
        peaks.sort(key=lambda p: p["mz"])
        return _ok(
            "extract_peaks_from_spectrum",
            n_input_peaks=len(mz_values),
            n_extracted=len(peaks),
            peaks=peaks,
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Find an isotope cluster (M, M+2, M+4 ...) starting at peak_mz[0]. "
    "Returns the matched cluster and inferred chlorine count."
))
def find_isotope_cluster(peak_mz: List[float], peak_intensity: List[float],
                          spacing_da: float = 2.0) -> str:
    try:
        if not peak_mz:
            raise ValueError("peak_mz is empty")
        cluster = [{"mz": round(peak_mz[0], 4),
                     "intensity": round(peak_intensity[0], 4)}]
        for m, i in zip(peak_mz[1:], peak_intensity[1:]):
            if abs(m - cluster[-1]["mz"] - spacing_da) < 0.05:
                cluster.append({"mz": round(m, 4), "intensity": round(i, 4)})
            else:
                break
        ratio = (cluster[1]["intensity"] / cluster[0]["intensity"]
                  if len(cluster) > 1 and cluster[0]["intensity"] > 0 else 0.0)
        n_cl = int(round(ratio / 0.3196))
        return _ok(
            "find_isotope_cluster",
            cluster_size=len(cluster),
            cluster=cluster,
            m_to_m2_ratio=round(ratio, 4),
            inferred_n_chlorine=n_cl,
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Analyse a spectrum for chlorine: combines extract + find_isotope "
    "into one shot."
))
def analyze_spectrum_for_chlorine(mz_values: List[float],
                                   intensities: List[float]) -> str:
    try:
        extract = json.loads(extract_peaks_from_spectrum(mz_values, intensities))
        if extract["status"] != "ok":
            return json.dumps(extract)
        peaks = extract["peaks"]
        if not peaks:
            return _err("no peaks above threshold")
        cluster = json.loads(find_isotope_cluster(
            [p["mz"] for p in peaks],
            [p["intensity"] for p in peaks],
        ))
        return _ok(
            "analyze_spectrum_for_chlorine",
            n_peaks=len(peaks),
            cluster=cluster["cluster"],
            inferred_n_chlorine=cluster["inferred_n_chlorine"],
        )
    except Exception as exc:
        return _err(str(exc))


# ---------------------------------------------------------------------------
# SMILES / structure matching helpers
# ---------------------------------------------------------------------------

# Minimal atomic-mass table for property prediction.
_ATOMIC_MASS: Dict[str, float] = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999,
    "F": 18.998, "P": 30.974, "S": 32.06, "Cl": 35.45,
    "Br": 79.904, "I": 126.904,
}


@mcp.tool(description=(
    "Compute molecular properties (formula, MW, #atoms) from a SMILES "
    "string (organic subset)."
))
def calculate_molecular_properties(smiles: str) -> str:
    try:
        s = smiles.strip()
        if not s:
            raise ValueError("SMILES is empty")
        # Atom counts via simple regex over [A-Z][a-z]? and bracket atoms.
        atom_pattern = re.compile(r"\[([A-Z][a-z]?)(?::\d+)?\]|([A-Z][a-z]?)")
        counts: Dict[str, int] = {}
        for m in atom_pattern.finditer(s):
            sym = m.group(1) or m.group(2)
            if sym and sym in _ATOMIC_MASS:
                counts[sym] = counts.get(sym, 0) + 1
        mw = sum(_ATOMIC_MASS[a] * n for a, n in counts.items())
        # Hydrogen count: implicit-H heuristic for organic subset.
        implicit_h = sum(
            max(0, 4 - counts.get("C", 0) * 0) +  # crude
            n for n in [counts.get("C", 0)]
        )
        # Approximate ring count from digit counts.
        n_rings = sum(1 for c in s if c.isdigit()) // 2
        return _ok(
            "calculate_molecular_properties",
            smiles=smiles,
            formula="".join(f"{a}{n if n > 1 else ''}" for a, n in counts.items()),
            molecular_weight=round(mw, 3),
            n_atoms=sum(counts.values()),
            n_rings=n_rings,
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Predict a simple fragmentation pattern: SMILES → top fragment ions by "
    "subtracting common neutral losses (H2O=18, HCl=36, NH3=17, CO=28)."
))
def predict_fragmentation_pattern(smiles: str,
                                   neutral_losses: List[float] = None) -> str:
    try:
        props = json.loads(calculate_molecular_properties(smiles))
        if props["status"] != "ok":
            return json.dumps(props)
        mw = props["molecular_weight"]
        losses = neutral_losses if neutral_losses else [18.0, 36.46, 17.03, 28.01]
        fragments = []
        for loss in losses:
            m = mw - loss
            if m > 0:
                fragments.append({"neutral_loss_da": loss,
                                   "fragment_mz": round(m, 4)})
        fragments.sort(key=lambda f: f["fragment_mz"], reverse=True)
        return _ok(
            "predict_fragmentation_pattern",
            parent_mw=round(mw, 4),
            fragments=fragments,
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Match a peak list against a list of candidate SMILES by computing "
    "each candidate's MW and counting peaks within a tolerance."
))
def match_spectrum_to_structure(mz_values: List[float],
                                 intensities: List[float],
                                 candidate_smiles: List[str],
                                 tolerance_da: float = 0.5) -> str:
    try:
        scores = []
        for smi in candidate_smiles:
            props = json.loads(calculate_molecular_properties(smi))
            if props["status"] != "ok":
                continue
            mw = props["molecular_weight"]
            # Score = count of peaks within tolerance of MW.
            n_match = sum(1 for mz in mz_values
                           if abs(mz - mw) < tolerance_da)
            scores.append({"smiles": smi, "mw": round(mw, 4),
                            "n_peaks_matched": n_match})
        scores.sort(key=lambda s: s["n_peaks_matched"], reverse=True)
        return _ok(
            "match_spectrum_to_structure",
            tolerance_da=tolerance_da,
            candidates=scores,
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Compute spectrum-level summary statistics: number of peaks, mean / "
    "median m/z, base-peak intensity, dynamic range."
))
def analyze_spectrum_characteristics(mz_values: List[float],
                                      intensities: List[float]) -> str:
    try:
        if not mz_values or not intensities:
            raise ValueError("inputs must be non-empty")
        if len(mz_values) != len(intensities):
            raise ValueError("length mismatch")
        max_i = max(intensities)
        return _ok(
            "analyze_spectrum_characteristics",
            n_peaks=len(mz_values),
            mean_mz=round(statistics.mean(mz_values), 4),
            median_mz=round(statistics.median(mz_values), 4),
            base_peak_mz=round(mz_values[intensities.index(max_i)], 4),
            base_peak_intensity=round(max_i, 4),
            dynamic_range_db=round(20 * math.log10(max_i / max(1e-9, min(intensities))), 2),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Batch-structure screening: rank each candidate SMILES by how many "
    "peaks it matches within tolerance."
))
def batch_structure_screening(mz_values: List[float],
                               candidate_smiles: List[str],
                               top_k: int = 5) -> str:
    try:
        scored = json.loads(match_spectrum_to_structure(
            mz_values, [1.0] * len(mz_values), candidate_smiles))
        if scored["status"] != "ok":
            return json.dumps(scored)
        ranked = sorted(scored["candidates"],
                        key=lambda c: c.get("n_peaks_matched", 0),
                        reverse=True)[:top_k]
        return _ok(
            "batch_structure_screening",
            top_k=len(ranked),
            ranked=ranked,
        )
    except Exception as exc:
        return _err(str(exc))


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

@mcp.tool(description=(
    "Plot a mass spectrum (stick plot of m/z vs intensity). Returns path to "
    "saved PNG."
))
def plot_mass_spectrum(mz_values: List[float],
                          intensities: List[float],
                          title: str = "Mass spectrum") -> str:
    if not _MPL_OK:
        return _err("matplotlib not available")
    if len(mz_values) != len(intensities) or not mz_values:
        return _err("m/z and intensity length mismatch")
    import os
    os.makedirs(_VIZ_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.vlines(mz_values, 0, intensities, color="k", linewidth=1)
    ax.set_xlabel("m/z")
    ax.set_ylabel("intensity")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    out = f"{_VIZ_DIR}/mass_spectrum.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return _ok("plot_mass_spectrum", path=out, n_peaks=len(mz_values))


@mcp.tool(description=(
    "Plot a theoretical isotope pattern (sticks) for n chlorine and "
    "n carbon. Returns path to saved PNG."
))
def plot_isotope_pattern(intensities: List[float],
                            mz_offsets: List[float],
                            title: str = "Isotope pattern") -> str:
    if not _MPL_OK:
        return _err("matplotlib not available")
    if len(intensities) != len(mz_offsets):
        return _err("length mismatch")
    import os
    os.makedirs(_VIZ_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.vlines(mz_offsets, 0, intensities, color="tab:blue", linewidth=2)
    ax.set_xlabel("m/z offset (Da)")
    ax.set_ylabel("relative intensity")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    out = f"{_VIZ_DIR}/isotope.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return _ok("plot_isotope_pattern", path=out, n_peaks=len(intensities))


if __name__ == "__main__":
    import asyncio
    asyncio.run(mcp.run_stdio_async())
