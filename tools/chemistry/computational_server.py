#!/usr/bin/env python3
"""Computational chemistry MCP server.

Tools ported from SciAgentGYM-main/toolkits/chemistry/computational_chemistry/.
Uses the PubChem PUG-REST API directly (no pubchempy SDK required).

Tools
-----
- pubchem_cid_by_name: compound name → PubChem CID list.
- pubchem_smiles_by_cid: PubChem CID → canonical SMILES.
- point_group_lookup: small curated name→point-group table (full
  automatic detection requires rdkit and is not in this port).
"""

from __future__ import annotations

import json
import math
import urllib.parse

import numpy as np
import requests
from scipy.integrate import quad

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors
    _RDKIT_OK = True
except ImportError:
    _RDKIT_OK = False

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("chemistry-computational")

PUG_REST = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
TIMEOUT = 15  # seconds


@mcp.tool(
    description=(
        "Look up PubChem CIDs by compound name. Returns the top 10 CIDs "
        "as reported by PubChem PUG-REST. Requires network access."
    )
)
async def pubchem_cid_by_name(name: str) -> str:
    if not name.strip():
        return json.dumps({"error": "name must be non-empty"})
    url = f"{PUG_REST}/compound/name/{urllib.parse.quote(name.strip())}/cids/JSON"
    try:
        r = requests.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as exc:
        return json.dumps({"error": f"PubChem request failed: {exc}"})
    except ValueError as exc:
        return json.dumps({"error": f"PubChem returned non-JSON: {exc}"})

    cids = data.get("IdentifierList", {}).get("CID", [])
    return json.dumps({"name": name, "cids": cids[:10]}, ensure_ascii=False)


@mcp.tool(
    description=(
        "Fetch the canonical isomeric SMILES for a PubChem CID. Returns "
        "the SMILES string and the molecular formula. Requires network "
        "access."
    )
)
async def pubchem_smiles_by_cid(cid: int) -> str:
    if cid <= 0:
        return json.dumps({"error": "cid must be a positive integer"})
    props = "CanonicalSMILES,MolecularFormula"
    url = f"{PUG_REST}/compound/cid/{cid}/property/{props}/JSON"
    try:
        r = requests.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as exc:
        return json.dumps({"error": f"PubChem request failed: {exc}"})
    except ValueError as exc:
        return json.dumps({"error": f"PubChem returned non-JSON: {exc}"})

    table = data.get("PropertyTable", {}).get("Properties", [])
    if not table:
        return json.dumps({"error": f"No properties for CID {cid}"})
    row = table[0]
    return json.dumps(
        {
            "cid": cid,
            "canonical_smiles": row.get("CanonicalSMILES"),
            "molecular_formula": row.get("MolecularFormula"),
        },
        ensure_ascii=False,
    )


# Curated name→point-group table. Used by point_group_lookup. This is a
# deliberately small subset; the full SciAgentGYM detector uses rdkit +
# a heuristic symmetry analyzer.
POINT_GROUP_TABLE: dict[str, str] = {
    "water": "C2v", "H2O": "C2v",
    "ammonia": "C3v", "NH3": "C3v",
    "methane": "Td", "CH4": "Td",
    "benzene": "D6h", "C6H6": "D6h",
    "boron trifluoride": "D3h", "BF3": "D3h",
    "carbon dioxide": "D∞h", "CO2": "D∞h",
    "ethylene": "D2h", "C2H4": "D2h", "ethene": "D2h",
    "acetylene": "D∞h", "C2H2": "D∞h", "ethyne": "D∞h",
    "hydrogen peroxide": "C2", "H2O2": "C2",
    "phosphorus pentachloride": "D3h", "PCl5": "D3h",
    "sulfur hexafluoride": "Oh", "SF6": "Oh",
    "xenon tetrafluoride": "D4h", "XeF4": "D4h",
    "ferrocene": "D5d",
    "allene": "D2d", "C3H4": "D2d",
    "formaldehyde": "C2v", "CH2O": "C2v",
    "glyoxal": "C2h", "C2H2O2": "C2h",
}


@mcp.tool(
    description=(
        "Look up the molecular point group for a common small molecule. "
        "Returns 'unknown' if the name is not in the curated table; for "
        "automatic symmetry detection use an rdkit-based tool."
    )
)
async def point_group_lookup(name: str) -> str:
    if not name.strip():
        return json.dumps({"error": "name must be non-empty"})
    key = name.strip()
    group = POINT_GROUP_TABLE.get(key) or POINT_GROUP_TABLE.get(key.lower())
    return json.dumps(
        {"name": key, "point_group": group or "unknown"},
        ensure_ascii=False,
    )


# ---- Wave 1: stdlib-only ports from SciAgentGYM computational_chemistry/ ----


@mcp.tool(
    description=(
        "Look up canonical SMILES by compound name via PubChem. Combines "
        "pubchem_cid_by_name and pubchem_smiles_by_cid. Requires network "
        "access; returns the first hit only."
    )
)
async def chemical_name_to_smiles(name: str) -> str:
    if not name.strip():
        return json.dumps({"error": "name must be non-empty"})
    cid_url = f"{PUG_REST}/compound/name/{urllib.parse.quote(name.strip())}/cids/JSON"
    try:
        r = requests.get(cid_url, timeout=TIMEOUT)
        r.raise_for_status()
        cids = r.json().get("IdentifierList", {}).get("CID", [])
    except (requests.RequestException, ValueError) as exc:
        return json.dumps({"error": f"PubChem CID lookup failed: {exc}"})
    if not cids:
        return json.dumps({"name": name, "canonical_smiles": None, "cid": None})
    cid = cids[0]
    smiles_url = f"{PUG_REST}/compound/cid/{cid}/property/CanonicalSMILES,MolecularFormula/JSON"
    try:
        r = requests.get(smiles_url, timeout=TIMEOUT)
        r.raise_for_status()
        props = r.json().get("PropertyTable", {}).get("Properties", [])
    except (requests.RequestException, ValueError) as exc:
        return json.dumps({"error": f"PubChem SMILES lookup failed: {exc}"})
    if not props:
        return json.dumps({"name": name, "canonical_smiles": None, "cid": cid})
    return json.dumps(
        {
            "name": name,
            "cid": cid,
            "canonical_smiles": props[0].get("CanonicalSMILES"),
            "molecular_formula": props[0].get("MolecularFormula"),
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Initialize a local SQLite symmetry-cache database. Creates a "
        "table (name, point_group) if absent. Returns the resolved "
        "absolute path. Default location: ~/.scimas/symmetry_cache.sqlite."
    )
)
async def init_local_symmetry_db(db_path: str = "") -> str:
    import os
    import sqlite3
    if not db_path:
        db_path = os.path.expanduser("~/.scimas/symmetry_cache.sqlite")
    db_path = os.path.abspath(db_path)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS symmetry ("
                "name TEXT PRIMARY KEY, point_group TEXT NOT NULL)"
            )
            conn.commit()
    except sqlite3.Error as exc:
        return json.dumps({"error": str(exc)})
    return json.dumps({"db_path": db_path, "initialized": True}, ensure_ascii=False)


@mcp.tool(
    description=(
        "Query the local symmetry-cache database for a name → "
        "point-group mapping. Returns null if not cached. Initialize "
        "the DB first with init_local_symmetry_db."
    )
)
async def query_local_symmetry(name: str, db_path: str = "") -> str:
    import os
    import sqlite3
    if not name.strip():
        return json.dumps({"error": "name must be non-empty"})
    if not db_path:
        db_path = os.path.expanduser("~/.scimas/symmetry_cache.sqlite")
    db_path = os.path.abspath(db_path)
    if not os.path.exists(db_path):
        return json.dumps({"name": name, "point_group": None, "cached": False})
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT point_group FROM symmetry WHERE name = ?", (name.strip(),)
            ).fetchone()
    except sqlite3.Error as exc:
        return json.dumps({"error": str(exc)})
    if row is None:
        return json.dumps({"name": name, "point_group": None, "cached": False})
    return json.dumps({"name": name, "point_group": row[0], "cached": True}, ensure_ascii=False)


# ---- Wave 2: numpy / scipy chemistry ports ----

# Physical constants (SI).
H_PLANCK = 6.62607015e-34  # J·s
K_BOLTZMANN = 1.380649e-23  # J/K
N_AVOGADRO = 6.02214076e23  # 1/mol
R_GAS = 8.314462618  # J/(mol·K)


@mcp.tool(
    description=(
        "Maxwell–Boltzmann most-probable speed v_p = sqrt(2·R·T/M) "
        "for a gas of molar mass M (g/mol) at temperature T (K). Also "
        "returns mean speed <v> = sqrt(8·R·T/(π·M)) and RMS speed "
        "c_rms = sqrt(3·R·T/M). All in m/s."
    )
)
async def maxwell_boltzmann_speed(
    temperature_K: float, molar_mass_g_per_mol: float
) -> str:
    if temperature_K <= 0 or molar_mass_g_per_mol <= 0:
        return json.dumps({"error": "T > 0 and M > 0"})
    M = molar_mass_g_per_mol / 1000.0  # kg/mol
    v_p = math.sqrt(2 * R_GAS * temperature_K / M)
    v_mean = math.sqrt(8 * R_GAS * temperature_K / (math.pi * M))
    v_rms = math.sqrt(3 * R_GAS * temperature_K / M)
    return json.dumps(
        {
            "temperature_K": temperature_K,
            "M_kg_per_mol": M,
            "v_most_probable_m_per_s": v_p,
            "v_mean_m_per_s": v_mean,
            "v_rms_m_per_s": v_rms,
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Morse potential V(R) = D_e · (1 - exp(-a · (R - R_e)))². "
        "R in Å; D_e in eV; a in Å^-1. Returns V(R) at the requested R, "
        "plus the second-derivative curvature k = 2·D_e·a² (eV/Å²) and "
        "the harmonic frequency ν in cm^-1 via k = μ·ω² (μ reduced "
        "mass in kg)."
    )
)
async def potential_energy_morse(
    R_angstrom: float,
    R_eq_angstrom: float,
    D_e_eV: float,
    a_per_angstrom: float,
    reduced_mass_kg: float,
) -> str:
    if D_e_eV <= 0 or a_per_angstrom <= 0 or reduced_mass_kg <= 0:
        return json.dumps({"error": "D_e, a, and reduced_mass must be > 0"})
    V = D_e_eV * (1.0 - math.exp(-a_per_angstrom * (R_angstrom - R_eq_angstrom))) ** 2
    # Second derivative at equilibrium: V''(R_e) = 2·D_e·a²  (in eV/Å²).
    k_eV_per_ang2 = 2.0 * D_e_eV * a_per_angstrom**2
    # Convert to SI: 1 eV/Å² = 1.602176634e-19 J / 1e-20 m² = 16.02176634 J/m².
    k_SI = k_eV_per_ang2 * 16.02176634
    omega_rad_per_s = math.sqrt(k_SI / reduced_mass_kg)
    nu_Hz = omega_rad_per_s / (2 * math.pi)
    nu_cm_inv = nu_Hz / 2.99792458e10
    return json.dumps(
        {
            "R_angstrom": R_angstrom,
            "R_eq_angstrom": R_eq_angstrom,
            "D_e_eV": D_e_eV,
            "V_eV": V,
            "curvature_eV_per_ang2": k_eV_per_ang2,
            "harmonic_freq_cm_inv": nu_cm_inv,
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Translational partition function per molecule in an ideal "
        "gas: q_trans = (2π·m·k_B·T / h²)^(3/2) · V. Pass molecular "
        "mass m (kg), temperature T (K), and volume V (m³). Returns "
        "the dimensionless q_trans."
    )
)
async def translational_partition_function(
    molecular_mass_kg: float, temperature_K: float, volume_m3: float
) -> str:
    if molecular_mass_kg <= 0 or temperature_K <= 0 or volume_m3 <= 0:
        return json.dumps({"error": "m, T, V must be > 0"})
    q = (
        (2.0 * math.pi * molecular_mass_kg * K_BOLTZMANN * temperature_K)
        / (H_PLANCK**2)
    ) ** 1.5 * volume_m3
    return json.dumps(
        {
            "m_kg": molecular_mass_kg,
            "T_K": temperature_K,
            "V_m3": volume_m3,
            "q_trans": q,
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Rotational partition function for a rigid linear rotor: "
        "q_rot = T / (σ · Θ_rot), where Θ_rot = h²/(8π²·I·k_B). "
        "Pass moment of inertia I (kg·m²) and symmetry number σ. "
        "Returns q_rot, Θ_rot (K), and the rotational temperature."
    )
)
async def rotational_partition_function(
    moment_of_inertia_kg_m2: float, symmetry_number: int = 1, temperature_K: float = 298.15
) -> str:
    if moment_of_inertia_kg_m2 <= 0 or symmetry_number < 1 or temperature_K <= 0:
        return json.dumps({"error": "I > 0, σ >= 1, T > 0"})
    theta_rot = H_PLANCK**2 / (8 * math.pi**2 * moment_of_inertia_kg_m2 * K_BOLTZMANN)
    q_rot = temperature_K / (symmetry_number * theta_rot)
    return json.dumps(
        {
            "I_kg_m2": moment_of_inertia_kg_m2,
            "sigma": symmetry_number,
            "theta_rot_K": theta_rot,
            "T_K": temperature_K,
            "q_rot": q_rot,
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Vibrational partition function for a single harmonic mode: "
        "q_vib = 1 / (1 - exp(-h·ν / (k_B·T))). Pass vibrational "
        "frequency ν in cm^-1 and temperature T (K). Returns q_vib "
        "and the dimensionless excitation energy hν/(k_B·T)."
    )
)
async def vibrational_partition_function(
    wavenumber_cm_inv: float, temperature_K: float = 298.15
) -> str:
    if wavenumber_cm_inv < 0 or temperature_K <= 0:
        return json.dumps({"error": "ν >= 0, T > 0"})
    # Convert cm^-1 to Hz: ν_Hz = ν_cm * c (c in cm/s).
    nu_Hz = wavenumber_cm_inv * 2.99792458e10
    x = H_PLANCK * nu_Hz / (K_BOLTZMANN * temperature_K)
    q_vib = 1.0 / (1.0 - math.exp(-x))
    return json.dumps(
        {
            "wavenumber_cm_inv": wavenumber_cm_inv,
            "T_K": temperature_K,
            "hnu_over_kT": x,
            "q_vib": q_vib,
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Numerical quadrature of f(x) over [a, b] using adaptive "
        "Gauss-Kronrod (scipy.integrate.quad). Pass a JSON-encodable "
        "expression understood by numpy: e.g. 'sin(x)/x', "
        "'exp(-x**2)', 'x**2 + 1'. Use 'x' as the variable. Returns "
        "the integral value and the estimated error."
    )
)
async def numerical_quadrature(
    expression: str,
    a: float,
    b: float,
) -> str:
    if a >= b:
        return json.dumps({"error": "a must be < b"})
    try:
        # Restricted namespace: only numpy + math are exposed.
        ns = {"np": np, "sin": np.sin, "cos": np.cos, "tan": np.tan,
              "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
              "abs": np.abs, "pow": np.power, "pi": math.pi, "e": math.e}
        fn = lambda x: eval(expression, {"__builtins__": {}}, {**ns, "x": x})
        val, err = quad(fn, a, b)
    except (ValueError, SyntaxError, NameError) as exc:
        return json.dumps({"error": f"expression failed: {exc}"})
    return json.dumps(
        {
            "expression": expression,
            "a": a,
            "b": b,
            "integral": float(val),
            "estimated_error": float(err),
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Normalize a 3×3 Cartesian moment-of-inertia tensor: returns "
        "principal moments I_A, I_B, I_C (in amu·Å²) and the rotational "
        "constants A, B, C (in MHz). Pass I_xx, I_xy, I_xz, I_yy, "
        "I_yz, I_zz in amu·Å². The tensor is diagonalized; result is "
        "returned sorted descending so A >= B >= C."
    )
)
async def principal_moment_of_inertia(
    I_xx_amu_A2: float,
    I_xy_amu_A2: float,
    I_xz_amu_A2: float,
    I_yy_amu_A2: float,
    I_yz_amu_A2: float,
    I_zz_amu_A2: float,
) -> str:
    I_mat = np.asarray(
        [
            [I_xx_amu_A2, I_xy_amu_A2, I_xz_amu_A2],
            [I_xy_amu_A2, I_yy_amu_A2, I_yz_amu_A2],
            [I_xz_amu_A2, I_yz_amu_A2, I_zz_amu_A2],
        ],
        dtype=float,
    )
    eigs = np.sort(np.linalg.eigvalsh(I_mat))[::-1]
    # Convert amu·Å² → kg·m²: 1 amu = 1.66053906660e-27 kg,
    # 1 Å² = 1e-20 m².
    conv = 1.66053906660e-27 * 1e-20
    eigs_SI = eigs * conv
    # Rotational constant B (MHz): B = h / (8π² · I · c), with
    # c in m/s and I in kg·m².
    c_ms = 2.99792458e8
    Bs_MHz = []
    for I_SI in eigs_SI:
        if I_SI > 0:
            B_Hz = H_PLANCK / (8 * math.pi**2 * I_SI)
            Bs_MHz.append(B_Hz / 1e6)
        else:
            Bs_MHz.append(float("inf"))
    return json.dumps(
        {
            "I_A_amu_A2": float(eigs[0]),
            "I_B_amu_A2": float(eigs[1]),
            "I_C_amu_A2": float(eigs[2]),
            "A_MHz": float(Bs_MHz[0]),
            "B_MHz": float(Bs_MHz[1]),
            "C_MHz": float(Bs_MHz[2]),
        },
        ensure_ascii=False,
    )


# ---- Wave 3: rdkit-dependent ports ----


@mcp.tool(
    description=(
        "Generate a 3D conformer for a SMILES molecule using rdkit's "
        "ETKDG algorithm, optionally with MMFF94 force-field "
        "minimization. Returns the 3D coordinates as a list of "
        "[x, y, z] in Å. Requires rdkit."
    )
)
async def generate_3d_conformer(
    smiles: str,
    minimize: bool = True,
    random_seed: int = 42,
) -> str:
    if not _RDKIT_OK:
        return json.dumps({"error": "rdkit not installed"})
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return json.dumps({"error": f"invalid SMILES: {smiles}"})
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        return json.dumps({"error": "rdkit failed to embed 3D conformer"})
    energy = None
    if minimize:
        try:
            res = AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
            ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol))
            energy = float(ff.CalcEnergy()) if ff is not None else None
        except Exception:
            energy = None
    conf = mol.GetConformer()
    coords = []
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        coords.append([round(pos.x, 4), round(pos.y, 4), round(pos.z, 4)])
    return json.dumps(
        {
            "smiles": smiles,
            "canonical_smiles": Chem.MolToSmiles(Chem.RemoveHs(mol)),
            "n_atoms": mol.GetNumAtoms(),
            "minimized": minimize,
            "energy_kcal_per_mol": energy,
            "coords_angstrom": coords,
        },
        ensure_ascii=False,
    )


# ---- Wave: 3D / optimization ports from SciAgentGYM molecule_analyzer ----
# All four tools use direct rdkit calls (no SMILES3DGenerator wrapper, no
# py_smiles / pubchempy dependency).

def _mol_or_error(smiles: str) -> tuple[Any, dict[str, str] | None]:
    """Parse SMILES → AddHs → return (mol, None) or (None, error_dict)."""
    if not _RDKIT_OK:
        return None, {"error": "rdkit not installed"}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, {"error": f"invalid SMILES: {smiles}"}
    return Chem.AddHs(mol), None


def _energy_or_none(mol, force_field: str = "MMFF", conf_id: int | None = None) -> float | None:
    """Compute single-point energy via MMFF (or UFF on failure). Returns
    kcal/mol or None if neither force field parameterizes the molecule.

    ``conf_id`` selects which conformer to evaluate. MUST be passed for
    multi-conformer molecules — without it, every call reads conformer
    0's energy and all conformer energies collapse to the same value.
    """
    try:
        props = AllChem.MMFFGetMoleculeProperties(mol)
        ff = AllChem.MMFFGetMoleculeForceField(
            mol, props, confId=conf_id if conf_id is not None else -1
        )
        if ff is None:
            raise ValueError("MMFF unavailable")
        return float(ff.CalcEnergy())
    except Exception:
        try:
            ff = AllChem.UFFGetMoleculeForceField(
                mol, confId=conf_id if conf_id is not None else -1
            )
            return float(ff.CalcEnergy()) if ff is not None else None
        except Exception:
            return None


@mcp.tool(
    description=(
        "Generate a single 3D conformer for a SMILES molecule and run a "
        "MMFF94 (or UFF fallback) minimization. Returns the conformer "
        "coordinates in Å, the final energy in kcal/mol, and the "
        "conformer id. Requires rdkit."
    )
)
async def mol_analyzer_generate_3d(
    smiles: str,
    method: str = "ETKDGv3",
) -> str:
    mol, err = _mol_or_error(smiles)
    if err is not None:
        return json.dumps(err)
    params = AllChem.ETKDGv3() if method == "ETKDGv3" else AllChem.ETKDG()
    params.randomSeed = 42
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        return json.dumps({"error": "rdkit failed to embed 3D conformer"})
    try:
        ff_code = AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        ff_used = "MMFF" if ff_code != -1 else "UFF"
        if ff_code == -1:
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        ff_used = "UFF"
    energy = _energy_or_none(mol)
    conf = mol.GetConformer()
    coords = [
        [round(conf.GetAtomPosition(i).x, 4),
         round(conf.GetAtomPosition(i).y, 4),
         round(conf.GetAtomPosition(i).z, 4)]
        for i in range(mol.GetNumAtoms())
    ]
    return json.dumps(
        {
            "smiles": smiles,
            "method": method,
            "conformer_id": 0,
            "force_field": ff_used,
            "energy_kcal_mol": round(energy, 6) if energy is not None else None,
            "coordinates": coords,
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Embed a SMILES molecule with the chosen ETKDG method and run an "
        "iterative MMFF (or UFF fallback) geometry optimization, "
        "reporting convergence and the final energy in kcal/mol. "
        "Requires rdkit."
    )
)
async def optimize_geometry(
    smiles: str,
    method: str = "ETKDGv3",
    force_field: str = "MMFF",
    max_iters: int = 200,
) -> str:
    mol, err = _mol_or_error(smiles)
    if err is not None:
        return json.dumps(err)
    params = AllChem.ETKDGv3() if method == "ETKDGv3" else AllChem.ETKDG()
    params.randomSeed = 42
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        return json.dumps({"error": "rdkit failed to embed 3D conformer"})
    ff = force_field.upper()
    converged = True
    try:
        if ff == "MMFF":
            code = AllChem.MMFFOptimizeMolecule(mol, maxIters=int(max_iters))
            converged = code == 0
            if code == -1:
                code = AllChem.UFFOptimizeMolecule(mol, maxIters=int(max_iters))
                ff = "UFF"
                converged = code == 0
        else:
            code = AllChem.UFFOptimizeMolecule(mol, maxIters=int(max_iters))
            converged = code == 0
    except Exception:
        converged = False
    energy = _energy_or_none(mol, ff)
    conf = mol.GetConformer()
    coords = [
        [round(conf.GetAtomPosition(i).x, 4),
         round(conf.GetAtomPosition(i).y, 4),
         round(conf.GetAtomPosition(i).z, 4)]
        for i in range(mol.GetNumAtoms())
    ]
    return json.dumps(
        {
            "smiles": smiles,
            "method": method,
            "force_field": ff,
            "iterations": int(max_iters),
            "converged": converged,
            "final_energy_kcal_mol": round(energy, 6) if energy is not None else None,
            "coordinates": coords,
        },
        ensure_ascii=False,
    )


def _classify_shape(npr1: float, npr2: float) -> str:
    """Rod / disk / sphere label per rdkit NPR ranges used in
    SciAgentGYM's molecule_analyzer.
    """
    if npr1 >= 0.8:
        return "rod"
    if npr2 <= 0.4:
        return "sphere"
    return "disk"


@mcp.tool(
    description=(
        "Compute 3D shape descriptors for an embedded SMILES molecule: "
        "principal moments of inertia (PMI), normalized PMI ratios "
        "(NPR1/NPR2), shape classification (rod / disk / sphere), "
        "radius of gyration (Å), asphericity, and eccentricity. "
        "Requires rdkit."
    )
)
async def get_3d_properties(
    smiles: str,
    method: str = "ETKDGv3",
    conf_id: int = 0,
) -> str:
    mol, err = _mol_or_error(smiles)
    if err is not None:
        return json.dumps(err)
    params = AllChem.ETKDGv3() if method == "ETKDGv3" else AllChem.ETKDG()
    params.randomSeed = 42
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        return json.dumps({"error": "rdkit failed to embed 3D conformer"})
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass
    if conf_id >= mol.GetNumConformers():
        return json.dumps({"error": f"conf_id {conf_id} out of range"})
    pmi1, pmi2, pmi3 = rdMolDescriptors.CalcPMI1(mol, conf_id), \
                      rdMolDescriptors.CalcPMI2(mol, conf_id), \
                      rdMolDescriptors.CalcPMI3(mol, conf_id)
    npr1 = pmi1 / pmi3 if pmi3 != 0 else 0.0
    npr2 = pmi2 / pmi3 if pmi3 != 0 else 0.0
    radius = rdMolDescriptors.CalcRadiusOfGyration(mol, conf_id)
    asphericity = rdMolDescriptors.CalcAsphericity(mol, conf_id)
    eccentricity = rdMolDescriptors.CalcEccentricity(mol, conf_id)
    return json.dumps(
        {
            "smiles": smiles,
            "pmi": [round(pmi1, 6), round(pmi2, 6), round(pmi3, 6)],
            "npr": [round(npr1, 6), round(npr2, 6)],
            "shape": _classify_shape(npr1, npr2),
            "radius": round(radius, 6),
            "asphericity": round(asphericity, 6),
            "eccentricity": round(eccentricity, 6),
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Embed N conformers of a SMILES molecule (ETKDG), minimize each "
        "with the chosen force field (MMFF preferred, UFF fallback), and "
        "return per-conformer energies plus E_min, E_max, the id of the "
        "minimum-energy conformer, and the energy span in kcal/mol. "
        "Requires rdkit."
    )
)
async def generate_multiple_conformers_with_optimization(
    smiles: str,
    num_confs: int = 10,
    method: str = "ETKDGv3",
    force_field: str = "MMFF",
    max_iters: int = 200,
) -> str:
    mol, err = _mol_or_error(smiles)
    if err is not None:
        return json.dumps(err)
    if num_confs < 1:
        return json.dumps({"error": "num_confs must be >= 1"})
    params = AllChem.ETKDGv3() if method == "ETKDGv3" else AllChem.ETKDG()
    params.randomSeed = 42
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=int(num_confs), params=params)
    if not cids:
        return json.dumps({"error": "rdkit failed to embed any conformers"})
    use_uff = False
    energies: list[dict[str, Any]] = []
    ff_label = force_field.upper()
    for cid in cids:
        try:
            if ff_label == "MMFF" and not use_uff:
                code = AllChem.MMFFOptimizeMolecule(mol, confId=cid, maxIters=int(max_iters))
                if code == -1:
                    use_uff = True
                    AllChem.UFFOptimizeMolecule(mol, confId=cid, maxIters=int(max_iters))
            else:
                AllChem.UFFOptimizeMolecule(mol, confId=cid, maxIters=int(max_iters))
        except Exception:
            continue
        e = _energy_or_none(mol, "UFF" if use_uff else ff_label, conf_id=cid)
        energies.append({"conformer_id": int(cid), "energy_kcal_mol": round(e, 6) if e is not None else None})
    valid = [e["energy_kcal_mol"] for e in energies if e["energy_kcal_mol"] is not None]
    if not valid:
        return json.dumps({"error": "no conformer energies computed"})
    min_e = min(valid)
    max_e = max(valid)
    min_cid = next(e["conformer_id"] for e in energies if e["energy_kcal_mol"] == min_e)
    return json.dumps(
        {
            "smiles": smiles,
            "num_confs": len(energies),
            "force_field": "UFF" if use_uff else ff_label,
            "energies": energies,
            "E_min": round(min_e, 6),
            "E_max": round(max_e, 6),
            "min_conf_id": int(min_cid),
            "delta_E_span": round(max_e - min_e, 6),
        },
        ensure_ascii=False,
    )


# ---- Wave 2: symmetry / planarity tools ported from
# SciAgentGYM-main/toolkits/chemistry/computational_chemistry/
# gpqa_physics_chemsirty_16.py. Pure numpy; thresholds are tuned for
# small drug-like molecules and may need adjustment for larger systems.
# ---------------------------------------------------------------------------

_PLANARITY_RMS_THRESHOLD = 0.05  # Å RMS deviation from best-fit plane
_ANGLE_CLUSTER_TOL = 30.0          # degrees


@mcp.tool(
    description=(
        "Compute planarity of a set of 3D points via best-fit plane RMS "
        "deviation. coords is a list of [x, y, z] in Å. Returns the RMS "
        "deviation, centroid, and the plane normal. A molecule with "
        "RMS < 0.05 Å is considered planar."
    )
)
async def compute_planarity(coords: list[list[float]]) -> str:
    try:
        if len(coords) < 3:
            return json.dumps({"error": "need at least 3 points"})
        P = np.asarray(coords, dtype=float)
        centroid = P.mean(axis=0)
        Q = P - centroid
        _, _, Vt = np.linalg.svd(Q, full_matrices=False)
        normal = Vt[-1, :]
        distances = np.abs(Q.dot(normal)) / np.linalg.norm(normal)
        rms = float(np.sqrt((distances**2).mean()))
        return json.dumps(
            {
                "n_points": len(coords),
                "rms_angstrom": round(rms, 6),
                "centroid": [round(float(c), 4) for c in centroid],
                "normal": [round(float(n), 6) for n in normal],
                "is_planar": rms <= _PLANARITY_RMS_THRESHOLD,
                "threshold_angstrom": _PLANARITY_RMS_THRESHOLD,
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@mcp.tool(
    description=(
        "Detect a σh mirror plane by planarity threshold. Pass the "
        "planarity_rms (Å) from compute_planarity. Returns whether a "
        "horizontal mirror plane is consistent with that RMS."
    )
)
async def detect_sigma_h(planarity_rms: float) -> str:
    try:
        rms = float(planarity_rms)
        return json.dumps(
            {
                "planarity_rms_angstrom": rms,
                "threshold_angstrom": _PLANARITY_RMS_THRESHOLD,
                "sigma_h_detected": rms <= _PLANARITY_RMS_THRESHOLD,
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@mcp.tool(
    description=(
        "Heuristically detect a C3 rotational axis by clustering projected "
        "angles around the molecular centroid into 3 sectors of ~120°. "
        "Pass coords (list of [x,y,z]) and the centroid + plane normal "
        "(from compute_planarity). Returns detected flag and metadata."
    )
)
async def detect_c3_axis(
    coords: list[list[float]],
    centroid: list[float],
    normal: list[float],
) -> str:
    try:
        P = np.asarray(coords, dtype=float)
        c = np.asarray(centroid, dtype=float)
        n = np.asarray(normal, dtype=float)
        n = n / np.linalg.norm(n)
        a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = a - n * np.dot(a, n)
        u = u / np.linalg.norm(u)
        v = np.cross(n, u)
        v = v / np.linalg.norm(v)
        Q = P - c
        x = Q.dot(u)
        y = Q.dot(v)
        angles = np.degrees(np.arctan2(y, x))
        angles = (angles + 360.0) % 360.0
        r = np.sqrt(x**2 + y**2)
        mask = r > (np.mean(r) * 0.3)
        angles_filt = angles[mask]
        if angles_filt.size < 6:
            return json.dumps({"c3_detected": False, "info": "insufficient angular features"})
        std_mod = float(np.std(angles_filt % 120.0))
        hist, _ = np.histogram(angles_filt, bins=12, range=(0.0, 360.0))
        seg = hist.reshape(3, 4).sum(axis=1)
        seg_var = float(np.var(seg))
        c3_detected = (std_mod < _ANGLE_CLUSTER_TOL) and (seg_var < float(np.mean(seg)) * 0.5)
        return json.dumps(
            {
                "c3_detected": bool(c3_detected),
                "std_mod_deg": round(std_mod, 4),
                "segment_counts": seg.tolist(),
                "segment_variance": round(seg_var, 4),
                "angle_cluster_tol_deg": _ANGLE_CLUSTER_TOL,
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@mcp.tool(
    description=(
        "Heuristically detect C2 axes perpendicular to the main C3 axis "
        "(used to discriminate D3h from C3h point groups). Pass coords, "
        "centroid and plane normal. Returns a coarse boolean plus "
        "per-angle symmetry score."
    )
)
async def detect_c2_perp_axes(
    coords: list[list[float]],
    centroid: list[float],
    normal: list[float],
) -> str:
    try:
        P = np.asarray(coords, dtype=float)
        c = np.asarray(centroid, dtype=float)
        n = np.asarray(normal, dtype=float)
        n = n / np.linalg.norm(n)
        a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = a - n * np.dot(a, n)
        u = u / np.linalg.norm(u)
        v = np.cross(n, u)
        v = v / np.linalg.norm(v)
        Q = P - c
        x = Q.dot(u)
        y = Q.dot(v)
        # Reflect across candidate axis at angle θ in the plane and compare.
        scores: list[float] = []
        for theta_deg in range(0, 180, 30):
            theta = math.radians(theta_deg)
            ax = np.array([math.cos(theta), math.sin(theta)])
            # Reflection matrix across axis ax in 2D.
            R = np.array([
                [math.cos(2 * theta), math.sin(2 * theta)],
                [math.sin(2 * theta), -math.cos(2 * theta)],
            ])
            xy = np.column_stack([x, y])
            xy_ref = xy @ R.T
            # Distance from each reflected point to nearest original point.
            from scipy.spatial import cKDTree
            tree = cKDTree(xy)
            d, _ = tree.query(xy_ref, k=1)
            score = float(np.mean(d))
            scores.append(score)
        best = float(min(scores))
        c2_detected = best < 0.5  # Å in-plane
        return json.dumps(
            {
                "c2_detected": bool(c2_detected),
                "best_inplane_symmetry_angstrom": round(best, 4),
                "per_axis_scores": [round(s, 4) for s in scores],
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


# ---- Wave 3: rdkit_generate_3d (ported from
# SciAgentGYM-main/toolkits/chemistry/computational_chemistry/
# gpqa_physics_chemsirty_16.py:299). Returns XYZ-style output from
# a SMILES, with element list and per-atom coords in Å.
# ---------------------------------------------------------------------------


@mcp.tool(
    description=(
        "Generate a 3D conformer from a SMILES string using rdkit's "
        "ETKDGv3 embedder and a UFF force-field minimization. Returns "
        "the per-atom coords (list of [x, y, z] in Å) and the parallel "
        "element symbol list. Falls back to random-coord embedding if "
        "ETKDGv3 fails. Requires rdkit."
    )
)
async def rdkit_generate_3d(smiles: str) -> str:
    if not _RDKIT_OK:
        return json.dumps({"error": "rdkit not installed"})
    if not isinstance(smiles, str) or not smiles.strip():
        return json.dumps({"error": "smiles must be a non-empty string"})
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return json.dumps({"error": f"invalid SMILES: {smiles}"})
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        ok = AllChem.EmbedMolecule(mol, params)
        if ok != 0:
            AllChem.EmbedMolecule(mol, useRandomCoords=True)
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        conf = mol.GetConformer()
        coords: list[list[float]] = []
        elements: list[str] = []
        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            coords.append([round(float(pos.x), 6),
                           round(float(pos.y), 6),
                           round(float(pos.z), 6)])
            elements.append(atom.GetSymbol())
        return json.dumps(
            {
                "smiles": smiles,
                "n_atoms": mol.GetNumAtoms(),
                "elements": elements,
                "coords_angstrom": coords,
                "force_field": "UFF",
                "source": "RDKit",
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    import asyncio

    asyncio.run(mcp.run_stdio_async())
