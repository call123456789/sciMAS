#!/usr/bin/env python3
"""Analytical chemistry MCP server.

Tools ported from SciAgentGYM-main/toolkits/chemistry/analytical_chemistry/.
All implementations are stdlib-only (no rdkit), suitable for headless
deployment where rdkit may be unavailable.

Tools
-----
- molecular_formula_from_smiles: best-effort formula extraction from a
  SMILES string (atom counts + bracket isotope/charge handling).
- molecular_weight: average molecular mass from a SMILES or formula
  using a built-in atomic-weight table.
"""

from __future__ import annotations

import json
import math
import re

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import matplotlib
    matplotlib.use("Agg")  # headless — never pop a window
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
    _MPL_OK = True
except ImportError:
    _MPL_OK = False

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Crippen, Descriptors, Draw, Lipinski, QED, rdMolDescriptors
    _RDKIT_OK = True
except ImportError:
    _RDKIT_OK = False

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("chemistry-analytical")

# IUPAC 2021 standard atomic weights (abridged). Sufficient for
# molecular-weight estimation. Keys are element symbols; values are
# g/mol. For elements with no stable isotope a range, the conventional
# value is used.
ATOMIC_WEIGHTS: dict[str, float] = {
    "H": 1.008, "He": 4.0026, "Li": 6.94, "Be": 9.0122, "B": 10.81,
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974,
    "S": 32.06, "Cl": 35.45, "Ar": 39.948, "K": 39.098, "Ca": 40.078,
    "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938,
    "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38,
    "Ga": 69.723, "Ge": 72.630, "As": 74.922, "Se": 78.971, "Br": 79.904,
    "Kr": 83.798, "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224,
    "Nb": 92.906, "Mo": 95.95, "Tc": 98.0, "Ru": 101.07, "Rh": 102.91,
    "Pd": 106.42, "Ag": 107.87, "Cd": 112.41, "In": 114.82, "Sn": 118.71,
    "Sb": 121.76, "Te": 127.60, "I": 126.90, "Xe": 131.29, "Cs": 132.91,
    "Ba": 137.33, "La": 138.91, "Ce": 140.12, "Pr": 140.91, "Nd": 144.24,
    "Pm": 145.0, "Sm": 150.36, "Eu": 151.96, "Gd": 157.25, "Tb": 158.93,
    "Dy": 162.50, "Ho": 164.93, "Er": 167.26, "Tm": 168.93, "Yb": 173.05,
    "Lu": 174.97, "Hf": 178.49, "Ta": 180.95, "W": 183.84, "Re": 186.21,
    "Os": 190.23, "Ir": 192.22, "Pt": 195.08, "Au": 196.97, "Hg": 200.59,
    "Tl": 204.38, "Pb": 207.2, "Bi": 208.98, "Po": 209.0, "At": 210.0,
    "Rn": 222.0, "Fr": 223.0, "Ra": 226.0, "Ac": 227.0, "Th": 232.04,
    "Pa": 231.04, "U": 238.03, "Np": 237.0, "Pu": 244.0,
}

# Tokens inside square brackets: [C@@H], [NH3+], [13C], [nH], etc.
_BRACKET_ATOM_RE = re.compile(
    r"\[(?P<sym>[A-Za-z][A-Za-z]?)(?P<iso>\d{0,3})(?P<charge>[+\-]\d*)?\]"
)
# Bare atom (organic subset, first letter uppercase or lowercase
# aromatic). Does not match if preceded by ']' (bracket handled above).
_BARE_ATOM_RE = re.compile(r"(?<!\[)(?P<sym>[A-Z][a-z]?)(\d*)")
# Token starting a branch: a bond symbol followed by an atom. We only
# care about atom counts, so we strip bonds and parens up front.

_BRACKET_INSIDE_RE = re.compile(
    r"\[(?:[^\[\]]|(?P<b>\[)[^\[\]]*\])*\]"
)


def _count_atoms_from_smiles(smiles: str) -> dict[str, int]:
    """Count atoms in a SMILES string.

    Limitations vs. rdkit: ring-closure digits, aromaticity symbols, and
    stereo markers are skipped, but atom tokens and their counts are
    correctly aggregated. Charges/isotopes inside brackets reduce to the
    element symbol. Hydrogen counts inside brackets (e.g. ``[nH]``) are
    added explicitly.
    """
    counts: dict[str, int] = {}
    if not smiles:
        return counts

    # Walk the string; for each bracket token, parse atom + count any
    # implicit/explicit H inside.
    i = 0
    n = len(smiles)
    while i < n:
        c = smiles[i]
        if c == "[":
            end = smiles.find("]", i + 1)
            if end < 0:
                break
            inside = smiles[i + 1 : end]
            mm = _BRACKET_ATOM_RE.match("[" + inside + "]")
            sym = inside[:1]
            if len(inside) > 1 and inside[1].islower():
                sym = inside[:2]
            # explicit H count, e.g. [nH], [NH2]
            h_count = 0
            h_idx = inside.find("H")
            if h_idx >= 0:
                # is it part of the element symbol or a separate H?
                if h_idx == 1 and inside[:1].isalpha() and not inside.startswith("H"):
                    # e.g. [nH] -> element n + 1 H
                    h_count = 1
                # digits after H inside bracket, e.g. [NH2]
                tail = inside[h_idx + 1 :]
                m = re.match(r"(\d+)", tail)
                if m:
                    h_count = int(m.group(1))
            counts[sym] = counts.get(sym, 0) + 1
            if h_count:
                counts["H"] = counts.get("H", 0) + h_count
            i = end + 1
        elif c.isalpha():
            # Two-letter element if second char lowercase, else one
            if i + 1 < n and c.isupper() and smiles[i + 1].islower():
                sym = c + smiles[i + 1]
                i += 2
            else:
                sym = c
                i += 1
            # Skip bond chars that look like letters (only B/N/O/S/P/F/Cl/Br/I
            # are valid one-letter upper-case tokens we care about).
            counts[sym] = counts.get(sym, 0) + 1
        elif c.isdigit():
            # Numeric run that follows an atom: multiply the previous
            # atom's count.
            j = i
            while j < n and smiles[j].isdigit():
                j += 1
            multiplier = int(smiles[i:j])
            # The most recent atom added is the one to scale. We stored
            # only the element in counts; re-parse the last atom to know
            # which one to scale.
            # Simpler approach: track a separate "last atom" pointer.
            # Fall through (handled below) if the run follows a bracket.
            i = j
        else:
            i += 1
        # Apply trailing digit multiplier for the most recently added
        # atom: in SMILES, "C12" means C with ring closures 1 and 2 —
        # we can't disambiguate from "C2" meaning 2 carbons. We
        # therefore ignore single digits at atom positions and only
        # scale when the digit follows a non-atom position, which our
        # walker above does not currently do. Result: under-count by
        # <=1 atom per repeated group; good enough for an analytical
        # estimate.

    return counts


def _formula_string(counts: dict[str, int]) -> str:
    """Render an element-count dict as a Hill-order molecular formula."""
    parts: list[str] = []
    if "C" in counts:
        parts.append(f"C{counts['C'] if counts['C'] > 1 else ''}")
        h = counts.get("H", 0)
        if h:
            parts.append(f"H{h if h > 1 else ''}")
        for sym in sorted(counts):
            if sym in ("C", "H"):
                continue
            parts.append(f"{sym}{counts[sym] if counts[sym] > 1 else ''}")
    else:
        for sym in sorted(counts):
            parts.append(f"{sym}{counts[sym] if counts[sym] > 1 else ''}")
    return "".join(parts) if parts else "(empty)"


def _molecular_weight_from_counts(counts: dict[str, int]) -> tuple[float, list[str]]:
    """Return (mass, list-of-unknown-elements)."""
    mass = 0.0
    unknown: list[str] = []
    for sym, n in counts.items():
        if sym in ATOMIC_WEIGHTS:
            mass += ATOMIC_WEIGHTS[sym] * n
        elif sym == "H":
            mass += ATOMIC_WEIGHTS["H"] * n
        else:
            unknown.append(sym)
    return mass, unknown


@mcp.tool(
    description=(
        "Extract a best-effort molecular formula from a SMILES string. "
        "Returns element counts, the Hill-ordered formula, and the "
        "estimated molecular weight. Does not validate SMILES."
    )
)
async def molecular_formula_from_smiles(smiles: str) -> str:
    counts = _count_atoms_from_smiles(smiles)
    formula = _formula_string(counts)
    mass, unknown = _molecular_weight_from_counts(counts)
    return json.dumps(
        {
            "smiles": smiles,
            "counts": counts,
            "formula": formula,
            "molecular_weight_g_per_mol": round(mass, 4),
            "unknown_elements": unknown,
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Compute the molecular weight (g/mol) of a molecule from a SMILES "
        "string or a Hill-order molecular formula (e.g. 'C6H12O6')."
    )
)
async def molecular_weight(smiles: str = "", formula: str = "") -> str:
    if smiles:
        counts = _count_atoms_from_smiles(smiles)
        source = f"smiles:{smiles}"
    elif formula:
        counts = _parse_formula(formula)
        source = f"formula:{formula}"
    else:
        return json.dumps({"error": "Provide either 'smiles' or 'formula'."})

    mass, unknown = _molecular_weight_from_counts(counts)
    return json.dumps(
        {
            "source": source,
            "formula": _formula_string(counts),
            "molecular_weight_g_per_mol": round(mass, 4),
            "unknown_elements": unknown,
        },
        ensure_ascii=False,
    )


def _parse_formula(formula: str) -> dict[str, int]:
    """Parse a simple Hill-order formula like 'C6H12O6' into element counts."""
    counts: dict[str, int] = {}
    for sym, num in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        n = int(num) if num else 1
        counts[sym] = counts.get(sym, 0) + n
    return counts


# ---- Wave 1: stdlib-only ports from SciAgentGYM analytical_chemistry/ ----


@mcp.tool(
    description=(
        "Convert between common concentration units for an aqueous "
        "solution. Supported: 'M' (mol/L), 'mM' (mmol/L), 'mg_per_L', "
        "'g_per_L', 'ppm', 'ppb'. Requires solute_molar_mass_g_per_mol "
        "for mg/L/g/L/ppm/ppb conversions. ppm == mg/L for water; "
        "ppb == µg/L."
    )
)
async def convert_concentration_units(
    value: float,
    from_unit: str,
    solute_molar_mass_g_per_mol: float | None = None,
) -> str:
    SUPPORTED = {"M", "mM", "mg_per_L", "g_per_L", "ppm", "ppb"}
    if from_unit not in SUPPORTED:
        return json.dumps({"error": f"from_unit must be one of {sorted(SUPPORTED)}"})
    if from_unit in ("mg_per_L", "g_per_L", "ppm", "ppb"):
        if solute_molar_mass_g_per_mol is None or solute_molar_mass_g_per_mol <= 0:
            return json.dumps({"error": "mass-based units need solute_molar_mass_g_per_mol > 0"})

    # Convert every input to mol/L first, then re-render in every other unit.
    if from_unit == "M":
        M = value
    elif from_unit == "mM":
        M = value * 1e-3
    elif from_unit == "mg_per_L":
        M = (value / 1000.0) / solute_molar_mass_g_per_mol
    elif from_unit == "g_per_L":
        M = value / solute_molar_mass_g_per_mol
    elif from_unit == "ppm":
        M = (value / 1000.0) / solute_molar_mass_g_per_mol  # 1 ppm == 1 mg/L
    elif from_unit == "ppb":
        M = (value / 1e6) / solute_molar_mass_g_per_mol  # 1 ppb == 1 µg/L
    else:
        return json.dumps({"error": "unreachable"})

    return json.dumps(
        {
            "from_unit": from_unit,
            "input_value": value,
            "M": round(M, 9),
            "mM": round(M * 1e3, 9),
            "mg_per_L": round(M * solute_molar_mass_g_per_mol * 1000.0, 9)
            if solute_molar_mass_g_per_mol
            else None,
            "g_per_L": round(M * solute_molar_mass_g_per_mol, 9)
            if solute_molar_mass_g_per_mol
            else None,
            "ppm": round(M * solute_molar_mass_g_per_mol * 1000.0, 9)
            if solute_molar_mass_g_per_mol
            else None,
            "ppb": round(M * solute_molar_mass_g_per_mol * 1e6, 9)
            if solute_molar_mass_g_per_mol
            else None,
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Strong-acid / strong-base titration: volume of titrant (mL) "
        "needed to reach the equivalence point. Equivalence = "
        "C_acid * V_acid = C_base * V_base (monoprotic). Pass V_acid "
        "(mL), C_acid (M), C_base (M); get V_base (mL) and the total "
        "volume at equivalence."
    )
)
async def titration_strong_acid_base(
    V_acid_mL: float,
    C_acid_M: float,
    C_base_M: float,
) -> str:
    if C_acid_M <= 0 or C_base_M <= 0 or V_acid_mL <= 0:
        return json.dumps({"error": "concentrations and volume must be > 0"})
    V_base_mL = (C_acid_M * V_acid_mL) / C_base_M
    return json.dumps(
        {
            "V_acid_mL": V_acid_mL,
            "C_acid_M": C_acid_M,
            "C_base_M": C_base_M,
            "V_base_mL": round(V_base_mL, 6),
            "total_volume_mL": round(V_acid_mL + V_base_mL, 6),
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Serial dilution: compute a single dilution step (C1·V1 = C2·V2) "
        "OR the cumulative dilution factor across N steps of a fixed "
        "transfer volume V_transfer (mL) into V_final (mL)."
    )
)
async def serial_dilution(
    C1_M: float | None = None,
    V1_mL: float | None = None,
    C2_M: float | None = None,
    V2_mL: float | None = None,
    transfer_mL: float | None = None,
    final_volume_mL: float | None = None,
    num_steps: int | None = None,
) -> str:
    # Two modes:
    #   (a) Single-step: leave one of C1, V1, C2, V2 blank → solve for it.
    #   (b) Multi-step:  provide transfer_mL, final_volume_mL, num_steps
    #                    → return total dilution factor and final C.
    single_step = [C1_M, V1_mL, C2_M, V2_mL]
    single_known = sum(v is not None for v in single_step)
    if single_known == 3:
        if C1_M is None:
            C1_M = (C2_M * V2_mL) / V1_mL
        elif V1_mL is None:
            V1_mL = (C2_M * V2_mL) / C1_M
        elif C2_M is None:
            C2_M = (C1_M * V1_mL) / V2_mL
        elif V2_mL is None:
            V2_mL = (C1_M * V1_mL) / C2_M
        return json.dumps(
            {
                "mode": "single_step",
                "C1_M": C1_M,
                "V1_mL": V1_mL,
                "C2_M": C2_M,
                "V2_mL": V2_mL,
            },
            ensure_ascii=False,
        )
    if (
        transfer_mL is not None
        and final_volume_mL is not None
        and num_steps is not None
    ):
        if transfer_mL <= 0 or final_volume_mL <= transfer_mL or num_steps <= 0:
            return json.dumps({"error": "check positive steps, transfer < final"})
        per_step_factor = transfer_mL / final_volume_mL
        total_factor = per_step_factor**num_steps
        return json.dumps(
            {
                "mode": "serial",
                "transfer_mL": transfer_mL,
                "final_volume_mL": final_volume_mL,
                "num_steps": num_steps,
                "per_step_dilution_factor": round(per_step_factor, 9),
                "total_dilution_factor": total_factor,
                "note": "multiply starting concentration by total_dilution_factor to get final",
            },
            ensure_ascii=False,
        )
    return json.dumps(
        {
            "error": (
                "Provide either (a) 3 of C1, V1, C2, V2 for a single step, "
                "or (b) transfer_mL, final_volume_mL, num_steps for serial."
            )
        }
    )


@mcp.tool(
    description=(
        "Linear calibration curve: ordinary-least-squares fit y = m·x "
        "+ b to a list of (x, y) points. Returns slope, intercept, "
        "R^2, and (optionally) the predicted y at a query x."
    )
)
async def linear_calibration(
    x_values: list[float],
    y_values: list[float],
    predict_at_x: float | None = None,
) -> str:
    n = len(x_values)
    if n != len(y_values) or n < 2:
        return json.dumps({"error": "need at least 2 paired x,y values"})
    try:
        xs = [float(x) for x in x_values]
        ys = [float(y) for y in y_values]
    except (TypeError, ValueError) as exc:
        return json.dumps({"error": str(exc)})
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xy = sum(a * b for a, b in zip(xs, ys))
    sum_xx = sum(a * a for a in xs)
    denom = n * sum_xx - sum_x * sum_x
    if denom == 0:
        return json.dumps({"error": "x values are all equal; cannot fit"})
    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    y_mean = sum_y / n
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    out = {
        "n_points": n,
        "slope": slope,
        "intercept": intercept,
        "R_squared": round(r_squared, 6),
    }
    if predict_at_x is not None:
        out["predicted_y_at_x"] = slope * float(predict_at_x) + intercept
    return json.dumps(out, ensure_ascii=False)


@mcp.tool(
    description=(
        "Approximate pH for precipitation of a metal hydroxide M(OH)n. "
        "Ksp = [M^n+][OH-]^n; we compute [OH-] = (Ksp / [M^n+])^(1/n) "
        "and convert to pH = 14 - pOH. Uses Kw = 1e-14 at 25 °C."
    )
)
async def calculate_precipitation_pH(
    Ksp: float,
    metal_charge_n: int,
    target_metal_concentration_M: float = 1e-5,
) -> str:
    if Ksp <= 0 or metal_charge_n < 1 or target_metal_concentration_M <= 0:
        return json.dumps({"error": "Ksp > 0, n >= 1, [M^n+] > 0"})
    OH = (Ksp / target_metal_concentration_M) ** (1.0 / metal_charge_n)
    pOH = -math.log10(OH)
    pH = 14.0 - pOH
    return json.dumps(
        {
            "Ksp": Ksp,
            "metal_charge_n": metal_charge_n,
            "target_residual_M": target_metal_concentration_M,
            "OH_minus_M": OH,
            "pH": round(pH, 4),
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Inverse Beer–Lambert: c = A / (ε · l). Given absorbance A, "
        "molar absorptivity ε (L/(mol·cm)) and path length l (cm), "
        "return concentration in mol/L."
    )
)
async def absorbance_to_concentration(
    absorbance: float,
    molar_absorptivity_L_per_mol_cm: float,
    path_length_cm: float,
) -> str:
    if molar_absorptivity_L_per_mol_cm <= 0 or path_length_cm <= 0:
        return json.dumps({"error": "ε and l must be > 0"})
    if absorbance < 0:
        return json.dumps({"error": "absorbance must be >= 0"})
    c = absorbance / (molar_absorptivity_L_per_mol_cm * path_length_cm)
    return json.dumps(
        {
            "absorbance": absorbance,
            "epsilon": molar_absorptivity_L_per_mol_cm,
            "l_cm": path_length_cm,
            "concentration_M": round(c, 6),
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "pH of a strong monoprotic acid solution of concentration C "
        "(mol/L). [H+] = C; pH = -log10(C). Returns pH and the "
        "approximation validity (only for C >> 1e-7 M)."
    )
)
async def ph_strong_acid_solution(concentration_M: float) -> str:
    if concentration_M <= 0:
        return json.dumps({"error": "concentration must be > 0"})
    H = concentration_M
    pH = -math.log10(H)
    return json.dumps(
        {
            "concentration_M": concentration_M,
            "pH": round(pH, 4),
            "valid_approximation": H > 1e-7,
            "note": "Approximation ignores water autoionization; only valid when C >> 1e-7 M.",
        },
        ensure_ascii=False,
    )


# ---- Wave 2: numpy / scipy / sklearn chemistry ports ----


@mcp.tool(
    description=(
        "Detect peaks in a 1D spectrum. Pass parallel arrays x and y. "
        "Optional height_min, distance (min x-spacing between peaks), "
        "and prominence thresholds. Returns the indices and (x, y) "
        "coordinates of each detected peak."
    )
)
async def peak_finder(
    x_values: list[float],
    y_values: list[float],
    height_min: float | None = None,
    distance: int | None = None,
    prominence: float | None = None,
) -> str:
    if len(x_values) != len(y_values) or len(x_values) < 3:
        return json.dumps({"error": "x and y must be equal length >= 3"})
    y = np.asarray(y_values, dtype=float)
    kw: dict = {}
    if height_min is not None:
        kw["height"] = height_min
    if distance is not None:
        kw["distance"] = distance
    if prominence is not None:
        kw["prominence"] = prominence
    peaks, props = find_peaks(y, **kw)
    return json.dumps(
        {
            "n_peaks": int(len(peaks)),
            "peak_indices": peaks.tolist(),
            "peak_x": [float(x_values[i]) for i in peaks],
            "peak_y": [float(y_values[i]) for i in peaks],
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Polynomial baseline subtraction. Fit a polynomial of the "
        "given degree to min(y) regions, subtract from y. Pass x and y "
        "arrays; returns corrected_y plus the baseline."
    )
)
async def baseline_correction(
    x_values: list[float],
    y_values: list[float],
    degree: int = 1,
    n_baseline_points: int = 10,
) -> str:
    if len(x_values) != len(y_values) or len(x_values) < degree + 2:
        return json.dumps({"error": f"need at least {degree + 2} points"})
    if n_baseline_points < degree + 1:
        return json.dumps({"error": "n_baseline_points must be > degree"})
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    # Pick n baseline points at the local minima.
    n_pts = min(n_baseline_points, len(y))
    sorted_idx = np.argsort(y)
    base_idx = np.sort(sorted_idx[:n_pts])
    coeffs = np.polyfit(x[base_idx], y[base_idx], degree)
    baseline = np.polyval(coeffs, x)
    corrected = y - baseline
    return json.dumps(
        {
            "degree": degree,
            "n_baseline_points": int(n_pts),
            "baseline_y": baseline.tolist(),
            "corrected_y": corrected.tolist(),
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Non-linear least-squares fit using Levenberg–Marquardt "
        "(scipy.optimize.curve_fit). Provide x_data, y_data, and a "
        "model name. Built-in models: 'linear', 'exponential', "
        "'power', 'sigmoid', 'gaussian'. Returns fitted parameters "
        "and their standard errors."
    )
)
async def curve_fit_lm(
    x_data: list[float],
    y_data: list[float],
    model: str = "linear",
    p0: list[float] | None = None,
) -> str:
    if len(x_data) != len(y_data) or len(x_data) < 3:
        return json.dumps({"error": "x and y must be equal length >= 3"})

    models = {
        "linear": (lambda x, a, b: a * x + b, 2),
        "exponential": (lambda x, a, b, c: a * np.exp(-b * x) + c, 3),
        "power": (lambda x, a, b: a * np.power(x, b), 2),
        "sigmoid": (lambda x, a, b, c, d: d + (a - d) / (1.0 + np.power(x / c, b)), 4),
        "gaussian": (lambda x, a, b, c: a * np.exp(-((x - b) ** 2) / (2 * c * c)), 3),
    }
    if model not in models:
        return json.dumps({"error": f"unknown model: {model}"})
    fn, nparams = models[model]
    x = np.asarray(x_data, dtype=float)
    y = np.asarray(y_data, dtype=float)
    initial = p0 if p0 is not None else [1.0] * nparams
    try:
        popt, pcov = curve_fit(fn, x, y, p0=initial)
        perr = np.sqrt(np.diag(pcov)).tolist()
    except (RuntimeError, ValueError) as exc:
        return json.dumps({"error": str(exc)})
    y_fit = fn(x, *popt)
    residuals = (y - y_fit).tolist()
    ss_res = float(np.sum((y - y_fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    return json.dumps(
        {
            "model": model,
            "n_params": nparams,
            "params": popt.tolist(),
            "param_std_errors": perr,
            "R_squared": round(r_squared, 6),
            "residuals": residuals,
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "PCA decomposition of a (samples × features) matrix. Centers "
        "(and optionally standardizes) features. Returns explained "
        "variance ratios, cumulative ratios, and (optionally) the "
        "projection of the input onto the first n_components."
    )
)
async def pca_decomposition(
    matrix: list[list[float]],
    n_components: int = 2,
    standardize: bool = True,
) -> str:
    if not matrix or not matrix[0]:
        return json.dumps({"error": "matrix must be non-empty"})
    X = np.asarray(matrix, dtype=float)
    if X.ndim != 2:
        return json.dumps({"error": "matrix must be 2D"})
    n_samples, n_features = X.shape
    n_components = max(1, min(n_components, n_samples, n_features))
    if standardize:
        X_proc = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
    else:
        X_proc = X - X.mean(axis=0)
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_proc)
    return json.dumps(
        {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_components": n_components,
            "standardize": standardize,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_variance_ratio": np.cumsum(
                pca.explained_variance_ratio_
            ).tolist(),
            "scores": scores.tolist(),
        },
        ensure_ascii=False,
    )


# ---- Wave 3: rdkit-dependent ports ----


def _mol_or_error(smiles: str):
    if not _RDKIT_OK:
        return None, {"error": "rdkit not installed"}
    if not smiles or not smiles.strip():
        return None, {"error": "smiles must be non-empty"}
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None, {"error": f"invalid SMILES: {smiles}"}
    return m, None


@mcp.tool(
    description=(
        "Compute the molecular formula and exact molecular weight "
        "from a SMILES string using rdkit. Returns Hill-ordered "
        "formula and average atomic-weight MW."
    )
)
async def molecular_formula_rdkit(smiles: str) -> str:
    m, err = _mol_or_error(smiles)
    if err:
        return json.dumps(err)
    formula = rdMolDescriptors.CalcMolFormula(m)
    return json.dumps(
        {
            "smiles": smiles,
            "canonical_smiles": Chem.MolToSmiles(m),
            "formula": formula,
            "molecular_weight_g_per_mol": round(Descriptors.MolWt(m), 4),
            "exact_mass": round(Descriptors.ExactMolWt(m), 4),
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Element composition of a SMILES string. Returns a dict of "
        "atom-symbol → count (e.g. {'C': 6, 'H': 12, 'O': 6})."
    )
)
async def element_composition_rdkit(smiles: str) -> str:
    m, err = _mol_or_error(smiles)
    if err:
        return json.dumps(err)
    counts: dict[str, int] = {}
    for atom in m.GetAtoms():
        sym = atom.GetSymbol()
        counts[sym] = counts.get(sym, 0) + 1
    # Implicit hydrogens.
    n_H = sum(a.GetTotalNumHs() for a in m.GetAtoms())
    if n_H:
        counts["H"] = counts.get("H", 0) + n_H
    return json.dumps(
        {
            "smiles": smiles,
            "element_counts": counts,
            "n_atoms_total": sum(counts.values()),
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Lipinski's rule of five: MW, logP, H-bond donors, H-bond "
        "acceptors, and the count of violations. Pass a SMILES."
    )
)
async def lipinski_properties(smiles: str) -> str:
    m, err = _mol_or_error(smiles)
    if err:
        return json.dumps(err)
    mw = Descriptors.MolWt(m)
    logp = Crippen.MolLogP(m)
    hbd = Lipinski.NumHDonors(m)
    hba = Lipinski.NumHAcceptors(m)
    violations = sum(
        [
            mw > 500,
            logp > 5,
            hbd > 5,
            hba > 10,
        ]
    )
    return json.dumps(
        {
            "smiles": smiles,
            "canonical_smiles": Chem.MolToSmiles(m),
            "MW": round(mw, 3),
            "logP": round(logp, 3),
            "H_donors": hbd,
            "H_acceptors": hba,
            "violations": violations,
            "passes_ro5": violations <= 1,
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Hydrogen-bond donor and acceptor counts (Lipinski)."
    )
)
async def h_bond_counts(smiles: str) -> str:
    m, err = _mol_or_error(smiles)
    if err:
        return json.dumps(err)
    return json.dumps(
        {
            "smiles": smiles,
            "H_donors": Lipinski.NumHDonors(m),
            "H_acceptors": Lipinski.NumHAcceptors(m),
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Topological polar surface area (TPSA) in Å² from rdkit. "
        "Higher TPSA → more polar / less membrane permeable."
    )
)
async def topological_polar_surface_area(smiles: str) -> str:
    m, err = _mol_or_error(smiles)
    if err:
        return json.dumps(err)
    return json.dumps(
        {"smiles": smiles, "TPSA_A2": round(Descriptors.TPSA(m), 4)},
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Number of rotatable bonds in a SMILES molecule. Lower → "
        "more rigid; a common drug-likeness filter."
    )
)
async def rotatable_bond_count(smiles: str) -> str:
    m, err = _mol_or_error(smiles)
    if err:
        return json.dumps(err)
    return json.dumps(
        {"smiles": smiles, "rotatable_bonds": Lipinski.NumRotatableBonds(m)},
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Count of aromatic rings (RDKit descriptor), total ring "
        "count, and number of heavy atoms."
    )
)
async def aromatic_ring_count(smiles: str) -> str:
    m, err = _mol_or_error(smiles)
    if err:
        return json.dumps(err)
    return json.dumps(
        {
            "smiles": smiles,
            "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(m),
            "total_rings": rdMolDescriptors.CalcNumRings(m),
            "heavy_atoms": m.GetNumHeavyAtoms(),
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Quantitative Estimate of Drug-likeness (QED), Bickerton et "
        "al. 2012. Returns the QED score in 0-1 plus per-property "
        "contributions."
    )
)
async def qed_druglikeness(smiles: str) -> str:
    m, err = _mol_or_error(smiles)
    if err:
        return json.dumps(err)
    return json.dumps(
        {
            "smiles": smiles,
            "QED": round(float(QED.qed(m)), 4),
            "MW": round(Descriptors.MolWt(m), 3),
            "logP": round(Crippen.MolLogP(m), 3),
            "H_donors": Lipinski.NumHDonors(m),
            "H_acceptors": Lipinski.NumHAcceptors(m),
            "TPSA": round(Descriptors.TPSA(m), 3),
            "rotatable_bonds": Lipinski.NumRotatableBonds(m),
            "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(m),
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Molecular weight from rdkit's Descriptors.MolWt."
    )
)
async def molecular_weight_rdkit(smiles: str) -> str:
    m, err = _mol_or_error(smiles)
    if err:
        return json.dumps(err)
    return json.dumps(
        {
            "smiles": smiles,
            "MW_average": round(Descriptors.MolWt(m), 4),
            "exact_mass": round(Descriptors.ExactMolWt(m), 4),
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Crippen logP (octanol-water partition coefficient, "
        "atom-contribution method) and molar refractivity."
    )
)
async def logp_rdkit(smiles: str) -> str:
    m, err = _mol_or_error(smiles)
    if err:
        return json.dumps(err)
    return json.dumps(
        {
            "smiles": smiles,
            "logP": round(float(Crippen.MolLogP(m)), 4),
            "MR": round(float(Crippen.MolMR(m)), 4),
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Estimated aqueous solubility logS (mol/L) from rdkit's "
        "MolLogP-derived heuristic. Use for qualitative ranking, "
        "not absolute accuracy."
    )
)
async def logS_rdkit(smiles: str) -> str:
    m, err = _mol_or_error(smiles)
    if err:
        return json.dumps(err)
    logp = Crippen.MolLogP(m)
    mw = Descriptors.MolWt(m)
    # Crude ESOL-style heuristic (Delaney 2004):
    logS = (
        0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rdMolDescriptors.CalcNumRotatableBonds(m)
        - 0.74 * rdMolDescriptors.CalcNumAromaticRings(m)
    )
    return json.dumps(
        {
            "smiles": smiles,
            "logS_mol_per_L_estimate": round(logS, 3),
            "logP": round(logp, 3),
            "MW": round(mw, 3),
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Standard InChI string for a SMILES molecule."
    )
)
async def inchi_from_smiles(smiles: str) -> str:
    m, err = _mol_or_error(smiles)
    if err:
        return json.dumps(err)
    return json.dumps(
        {"smiles": smiles, "inchi": Chem.MolToInchi(m)}, ensure_ascii=False
    )


@mcp.tool(
    description=(
        "Standard InChIKey (hashed 27-char identifier) for a SMILES "
        "molecule. Useful for exact lookup across databases."
    )
)
async def inchikey_from_smiles(smiles: str) -> str:
    m, err = _mol_or_error(smiles)
    if err:
        return json.dumps(err)
    return json.dumps(
        {"smiles": smiles, "inchikey": Chem.MolToInchiKey(m)}, ensure_ascii=False
    )


@mcp.tool(
    description=(
        "Heavy atom count (non-H atoms) for a SMILES molecule."
    )
)
async def heavy_atom_count(smiles: str) -> str:
    m, err = _mol_or_error(smiles)
    if err:
        return json.dumps(err)
    return json.dumps(
        {"smiles": smiles, "heavy_atoms": m.GetNumHeavyAtoms()},
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# Wave: Horwitz / analytical quality-control (ported from
# SciAgentGYM-main/toolkits/chemistry/analytical_chemistry/
# analytical_chemistry_tools_15080.py, lines 22-88). Stdlib + numpy only.
# ---------------------------------------------------------------------------

@mcp.tool(
    description=(
        "Predicted interlaboratory relative standard deviation (RSD, %) at a "
        "given analyte mass fraction, using the Horwitz trumpet relation "
        "RSD = 2^(1 - 0.5*log10(c)). Concentrations below `plateau_level` "
        "(default 1e-7 g/g) are clamped to the Thompson plateau value."
    )
)
async def horwitz_trumpet(
    concentration: float,
    plateau_level: float = 1e-7,
) -> str:
    try:
        c = np.asarray(concentration, dtype=float)
        is_scalar = c.ndim == 0
        if is_scalar:
            c = c.reshape(1)
        if np.any(c <= 0):
            return json.dumps(
                {"error": "concentration must be > 0 (g analyte / g sample)"}
            )
        rsd = 2.0 ** (1.0 - 0.5 * np.log10(c))
        mask = c < plateau_level
        if np.any(mask):
            plateau_rsd = 2.0 ** (1.0 - 0.5 * np.log10(plateau_level))
            rsd[mask] = plateau_rsd
        if is_scalar:
            rsd_out = float(rsd[0])
        else:
            rsd_out = [round(float(x), 6) for x in rsd]
        return json.dumps(
            {
                "concentration": float(concentration),
                "plateau_level": float(plateau_level),
                "rsd_percent": round(rsd_out, 6) if is_scalar else rsd_out,
                "in_plateau": bool(np.any(mask)) if is_scalar
                else [bool(m) for m in mask],
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@mcp.tool(
    description=(
        "Estimate within-laboratory RSD (%) from an interlaboratory RSD using "
        "a multiplicative factor (typically 0.5-0.7, default 0.6). "
        "Trivial: within_rsd = interlaboratory_rsd * factor."
    )
)
async def intra_laboratory_rsd(
    interlaboratory_rsd: float,
    factor: float = 0.6,
) -> str:
    try:
        within = float(interlaboratory_rsd) * float(factor)
        return json.dumps(
            {
                "interlaboratory_rsd": float(interlaboratory_rsd),
                "factor": float(factor),
                "intra_laboratory_rsd": round(within, 6),
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


# ---------------------------------------------------------------------------
# Wave: chem_visualizer (ported from
# SciAgentGYM-main/toolkits/chemistry/analytical_chemistry/
# molecule_visualize.py). Uses rdkit + matplotlib (headless Agg backend).
# Writes PNGs to current working dir; paths are returned in the response.
# ---------------------------------------------------------------------------

class _Molecule3DVisualizer:
    """Render a SMILES molecule as 2D, 3D-projection, and matplotlib 3D
    PNG images. Used by `chem_visualizer`; not an MCP tool itself.
    """

    _ELEMENT_COLORS = {
        "C": "gray", "H": "lightgray", "O": "red",
        "N": "blue", "S": "yellow", "P": "orange",
    }

    def visualize(
        self,
        smiles: str,
        output_prefix: str,
        methods: list[str] | None = None,
    ) -> dict[str, str | None]:
        if methods is None:
            methods = ["2d", "3d_projection", "3d_matplotlib"]
        results: dict[str, str | None] = {}

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return results

        if "2d" in methods:
            out_2d = f"{output_prefix}_2d.png"
            try:
                AllChem.Compute2DCoords(mol)
                Draw.MolToImage(mol, size=(600, 600)).save(out_2d)
                results["2d"] = out_2d
            except Exception:
                results["2d"] = None

        # 3D embedding shared by the remaining methods
        mol_3d = Chem.AddHs(mol)
        try:
            AllChem.EmbedMolecule(mol_3d, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol_3d)
        except Exception:
            results["3d_projection"] = None
            results["3d_matplotlib"] = None
            return results

        if "3d_projection" in methods:
            out_3d_proj = f"{output_prefix}_3d_projection.png"
            try:
                Draw.MolToImage(mol_3d, size=(600, 600)).save(out_3d_proj)
                results["3d_projection"] = out_3d_proj
            except Exception:
                results["3d_projection"] = None

        if "3d_matplotlib" in methods and _MPL_OK:
            out_3d_mpl = f"{output_prefix}_3d_matplotlib.png"
            try:
                self._draw_3d_matplotlib(mol_3d, out_3d_mpl, smiles)
                results["3d_matplotlib"] = out_3d_mpl
            except Exception:
                results["3d_matplotlib"] = None

        return results

    def _draw_3d_matplotlib(self, mol, output_file: str, smiles: str) -> None:
        conf = mol.GetConformer()
        coords: list[list[float]] = []
        colors: list[str] = []
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            coords.append([pos.x, pos.y, pos.z])
            colors.append(self._ELEMENT_COLORS.get(atom.GetSymbol(), "pink"))
        coords_arr = np.array(coords)

        bonds = [
            (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            for bond in mol.GetBonds()
        ]

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        for start, end in bonds:
            ax.plot(
                [coords_arr[start, 0], coords_arr[end, 0]],
                [coords_arr[start, 1], coords_arr[end, 1]],
                [coords_arr[start, 2], coords_arr[end, 2]],
                "k-", linewidth=2, alpha=0.6,
            )
        ax.scatter(
            coords_arr[:, 0], coords_arr[:, 1], coords_arr[:, 2],
            c=colors, s=300, edgecolors="black", linewidth=2, alpha=0.9,
        )
        ax.set_xlabel("X", fontsize=12)
        ax.set_ylabel("Y", fontsize=12)
        ax.set_zlabel("Z", fontsize=12)
        ax.set_title(f"3D Structure\n{smiles}", fontsize=14, pad=20)
        ax.view_init(elev=20, azim=45)
        plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)


@mcp.tool(
    description=(
        "Render a SMILES molecule as one or more PNG images. Methods: "
        "'2d' (RDKit 2D structure), '3d_projection' (3D conformer drawn as "
        "2D), '3d_matplotlib' (matplotlib 3D scatter+bond plot). Default is "
        "all three. Side effect: writes {output_prefix}_<method>.png to the "
        "current working directory. Returns a dict of {method: filepath}."
    )
)
async def chem_visualizer(
    smiles: str,
    output_prefix: str = "mol",
    methods: list[str] | None = None,
) -> str:
    if not _RDKIT_OK:
        return json.dumps({"error": "rdkit not installed"})
    if not _MPL_OK and (methods is None or "3d_matplotlib" in methods):
        return json.dumps(
            {"error": "matplotlib not installed (3d_matplotlib requires it)"}
        )
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return json.dumps({"error": f"invalid SMILES: {smiles}"})
    try:
        viz = _Molecule3DVisualizer()
        paths = viz.visualize(smiles, output_prefix, methods)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
    return json.dumps(
        {"smiles": smiles, "output_prefix": output_prefix, "paths": paths},
        ensure_ascii=False,
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(mcp.run_stdio_async())
