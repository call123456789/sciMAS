#!/usr/bin/env python3
"""Physical chemistry MCP server.

Tools ported from SciAgentGYM-main/toolkits/chemistry/physical_chemistry/.
Pure-math, stdlib-only implementations of the most common formulas.

Tools
-----
- ideal_gas: solve PV = nRT for any of P, V, T, n given the other three.
- henderson_hasselbalch: buffer pH = pKa + log10([A-]/[HA]).
- nernst_equation: equilibrium electrode potential E = E° - (RT/nF) ln Q.
"""

from __future__ import annotations

import json
import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors
    _RDKIT_OK = True
except ImportError:
    _RDKIT_OK = False

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("chemistry-physical")

# SI unit conversion factors. Internal state: Pa, m^3, K, mol.
_UNIT_TO_PA = {"Pa": 1.0, "kPa": 1e3, "atm": 101325.0, "bar": 1e5, "mmHg": 133.322}
_UNIT_TO_M3 = {"m^3": 1.0, "L": 1e-3, "mL": 1e-6}
_UNIT_FROM_K = {"K": 1.0, "C": lambda c: c + 273.15}
_KELVIN_FROM_C = lambda c: c + 273.15


def _solve_for(
    pressure: float | None,
    volume: float | None,
    temperature: float | None,
    quantity: float | None,
    gas_constant: float,
) -> dict:
    """Solve PV = nRT for the missing variable."""
    given = {
        "pressure": pressure,
        "volume": volume,
        "temperature": temperature,
        "quantity": quantity,
    }
    missing = [k for k, v in given.items() if v is None]
    if len(missing) != 1:
        return {"error": f"Provide exactly 3 of P, V, T, n. Missing: {missing}"}
    target = missing[0]
    if target == "pressure":
        return {"solved": "pressure", "value": quantity * gas_constant * temperature / volume}
    if target == "volume":
        return {"solved": "volume", "value": quantity * gas_constant * temperature / pressure}
    if target == "temperature":
        return {"solved": "temperature", "value": pressure * volume / (quantity * gas_constant)}
    if target == "quantity":
        return {"solved": "quantity", "value": pressure * volume / (gas_constant * temperature)}
    return {"error": "unreachable"}


@mcp.tool(
    description=(
        "Ideal-gas equation PV = nRT. Leave exactly one of pressure, "
        "volume, temperature, quantity (mol) blank to solve for it. "
        "Default gas constant is 8.314 J/(mol·K). Units: pressure in Pa "
        "(accepts kPa/atm/bar/mmHg via the units field), volume in m^3 "
        "(accepts L/mL), temperature in K (accepts °C). Use 'units' = "
        "{'P': 'atm', 'V': 'L', 'T': 'C'} to mix."
    )
)
async def ideal_gas(
    pressure: float | None = None,
    volume: float | None = None,
    temperature: float | None = None,
    quantity: float | None = None,
    gas_constant: float = 8.314,
    units: dict | None = None,
) -> str:
    units = units or {}

    # Convert to SI.
    if pressure is not None:
        u = units.get("P", "Pa")
        if u not in _UNIT_TO_PA:
            return json.dumps({"error": f"unknown P unit: {u}"})
        pressure = pressure * _UNIT_TO_PA[u]
    if volume is not None:
        u = units.get("V", "m^3")
        if u not in _UNIT_TO_M3:
            return json.dumps({"error": f"unknown V unit: {u}"})
        volume = volume * _UNIT_TO_M3[u]
    if temperature is not None:
        u = units.get("T", "K")
        if u == "C":
            temperature = _KELVIN_FROM_C(temperature)
        elif u != "K":
            return json.dumps({"error": f"unknown T unit: {u}"})

    result = _solve_for(pressure, volume, temperature, quantity, gas_constant)
    if "error" in result:
        return json.dumps(result)
    return json.dumps(
        {
            "solved": result["solved"],
            "value": result["value"],
            "value_unit": {
                "pressure": "Pa",
                "volume": "m^3",
                "temperature": "K",
                "quantity": "mol",
            }[result["solved"]],
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Henderson-Hasselbalch: pH = pKa + log10([A-]/[HA]). Provide the "
        "pKa and the molar concentrations of the conjugate base (A-) "
        "and the conjugate acid (HA). Returns pH and a flag indicating "
        "whether the buffer is in the recommended 0.1 < [A-]/[HA] < 10 "
        "range."
    )
)
async def henderson_hasselbalch(
    pKa: float,
    base_conc_M: float,
    acid_conc_M: float,
) -> str:
    if acid_conc_M <= 0:
        return json.dumps({"error": "acid_conc_M must be > 0"})
    if base_conc_M < 0:
        return json.dumps({"error": "base_conc_M must be >= 0"})
    ratio = base_conc_M / acid_conc_M
    pH = pKa + math.log10(ratio) if ratio > 0 else float("-inf")
    return json.dumps(
        {
            "pKa": pKa,
            "base_conc_M": base_conc_M,
            "acid_conc_M": acid_conc_M,
            "ratio_A_minus_over_HA": round(ratio, 6),
            "pH": round(pH, 4) if math.isfinite(pH) else None,
            "in_effective_buffer_range": 0.1 <= ratio <= 10.0,
        },
        ensure_ascii=False,
    )


# Faraday constant, C/mol.
FARADAY = 96485.33212


@mcp.tool(
    description=(
        "Nernst equation: E = E° - (R·T / (n·F)) · ln(Q), where R = "
        "8.314 J/(mol·K), F = 96485 C/mol. Provide standard potential E° "
        "(V), number of electrons transferred n, temperature T (K; "
        "defaults to 298.15 K), and the reaction quotient Q."
    )
)
async def nernst_equation(
    E_standard_V: float,
    n_electrons: int,
    Q: float,
    temperature_K: float = 298.15,
    gas_constant: float = 8.314,
) -> str:
    if n_electrons <= 0:
        return json.dumps({"error": "n_electrons must be > 0"})
    if Q <= 0:
        return json.dumps({"error": "Q must be > 0"})
    E = E_standard_V - (gas_constant * temperature_K / (n_electrons * FARADAY)) * math.log(Q)
    return json.dumps(
        {
            "E_standard_V": E_standard_V,
            "n_electrons": n_electrons,
            "Q": Q,
            "temperature_K": temperature_K,
            "E_V": round(E, 6),
        },
        ensure_ascii=False,
    )


# ---- Wave 1: stdlib-only ports from SciAgentGYM physical_chemistry/ ----

GAS_CONSTANT_R = 8.314  # J/(mol·K), default


@mcp.tool(
    description=(
        "Arrhenius equation: k = A · exp(-Ea / (R·T)). Returns the "
        "rate constant k given the pre-exponential factor A, "
        "activation energy Ea (J/mol), and temperature T (K)."
    )
)
async def arrhenius(
    pre_exponential_A: float,
    activation_energy_J_per_mol: float,
    temperature_K: float,
    gas_constant: float = GAS_CONSTANT_R,
) -> str:
    if temperature_K <= 0:
        return json.dumps({"error": "temperature_K must be > 0"})
    k = pre_exponential_A * math.exp(
        -activation_energy_J_per_mol / (gas_constant * temperature_K)
    )
    return json.dumps(
        {
            "pre_exponential_A": pre_exponential_A,
            "activation_energy_J_per_mol": activation_energy_J_per_mol,
            "temperature_K": temperature_K,
            "k": k,
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Gibbs free-energy change: ΔG = ΔH - T·ΔS. ΔH in J/mol, ΔS in "
        "J/(mol·K), T in K. ΔG < 0 means spontaneous. Returns ΔG and "
        "the spontaneity flag."
    )
)
async def gibbs_free_energy(
    enthalpy_J_per_mol: float,
    entropy_J_per_mol_K: float,
    temperature_K: float,
) -> str:
    dG = enthalpy_J_per_mol - temperature_K * entropy_J_per_mol_K
    return json.dumps(
        {
            "enthalpy_J_per_mol": enthalpy_J_per_mol,
            "entropy_J_per_mol_K": entropy_J_per_mol_K,
            "temperature_K": temperature_K,
            "delta_G_J_per_mol": round(dG, 6),
            "spontaneous": dG < 0,
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Equilibrium constant from Gibbs free energy: K = exp(-ΔG / "
        "(R·T)). ΔG in J/mol, T in K. If ΔG is positive, K < 1; if "
        "negative, K > 1."
    )
)
async def equilibrium_constant_from_gibbs(
    delta_G_J_per_mol: float,
    temperature_K: float,
    gas_constant: float = GAS_CONSTANT_R,
) -> str:
    if temperature_K <= 0:
        return json.dumps({"error": "temperature_K must be > 0"})
    K = math.exp(-delta_G_J_per_mol / (gas_constant * temperature_K))
    return json.dumps(
        {
            "delta_G_J_per_mol": delta_G_J_per_mol,
            "temperature_K": temperature_K,
            "K": K,
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Van 't Hoff equation — temperature-dependence of K. Given K "
        "at T1 and ΔH, return K at T2: ln(K2/K1) = -ΔH/R · (1/T2 - "
        "1/T1). ΔH in J/mol."
    )
)
async def vant_hoff(
    K1: float,
    T1_K: float,
    T2_K: float,
    delta_H_J_per_mol: float,
    gas_constant: float = GAS_CONSTANT_R,
) -> str:
    if K1 <= 0 or T1_K <= 0 or T2_K <= 0:
        return json.dumps({"error": "K1, T1, T2 must be > 0"})
    ln_ratio = -delta_H_J_per_mol / gas_constant * (1.0 / T2_K - 1.0 / T1_K)
    K2 = K1 * math.exp(ln_ratio)
    return json.dumps(
        {
            "K1": K1,
            "T1_K": T1_K,
            "T2_K": T2_K,
            "delta_H_J_per_mol": delta_H_J_per_mol,
            "K2": K2,
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Beer–Lambert law: A = ε · l · c. ε in L/(mol·cm), l in cm, "
        "c in mol/L. Returns A and transmittance T = 10^(-A)."
    )
)
async def beer_lambert(
    molar_absorptivity_L_per_mol_cm: float,
    path_length_cm: float,
    concentration_M: float,
) -> str:
    if molar_absorptivity_L_per_mol_cm < 0 or path_length_cm < 0 or concentration_M < 0:
        return json.dumps({"error": "ε, l, c must be non-negative"})
    A = molar_absorptivity_L_per_mol_cm * path_length_cm * concentration_M
    T = 10.0 ** (-A)
    return json.dumps(
        {
            "epsilon": molar_absorptivity_L_per_mol_cm,
            "l_cm": path_length_cm,
            "c_M": concentration_M,
            "absorbance": round(A, 6),
            "transmittance": round(T, 6),
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Clausius–Clapeyron equation: ln(P2/P1) = -ΔHvap/R · (1/T2 - "
        "1/T1). ΔHvap in J/mol. Returns the vapor pressure P2 at T2."
    )
)
async def clausius_clapeyron(
    P1_Pa: float,
    T1_K: float,
    T2_K: float,
    delta_H_vap_J_per_mol: float,
    gas_constant: float = GAS_CONSTANT_R,
) -> str:
    if P1_Pa <= 0 or T1_K <= 0 or T2_K <= 0:
        return json.dumps({"error": "P1, T1, T2 must be > 0"})
    ln_ratio = -delta_H_vap_J_per_mol / gas_constant * (1.0 / T2_K - 1.0 / T1_K)
    P2 = P1_Pa * math.exp(ln_ratio)
    return json.dumps(
        {
            "P1_Pa": P1_Pa,
            "T1_K": T1_K,
            "T2_K": T2_K,
            "delta_H_vap_J_per_mol": delta_H_vap_J_per_mol,
            "P2_Pa": P2,
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Langmuir isotherm: θ = (K·P) / (1 + K·P). Returns fractional "
        "surface coverage θ at pressure P (Pa) for adsorption constant "
        "K (Pa^-1)."
    )
)
async def langmuir_isotherm(
    pressure_Pa: float,
    equilibrium_constant_Pa_inv: float,
) -> str:
    if pressure_Pa < 0 or equilibrium_constant_Pa_inv < 0:
        return json.dumps({"error": "P and K must be non-negative"})
    KP = equilibrium_constant_Pa_inv * pressure_Pa
    theta = KP / (1.0 + KP) if (1.0 + KP) > 0 else 0.0
    return json.dumps(
        {
            "pressure_Pa": pressure_Pa,
            "K_Pa_inv": equilibrium_constant_Pa_inv,
            "theta": round(theta, 6),
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "First-order half-life: t½ = ln(2) / k. k in s^-1, t½ in "
        "seconds. Also returns the time-constants τ = 1/k."
    )
)
async def first_order_half_life(rate_constant_s_inv: float) -> str:
    if rate_constant_s_inv <= 0:
        return json.dumps({"error": "k must be > 0"})
    t_half = math.log(2) / rate_constant_s_inv
    tau = 1.0 / rate_constant_s_inv
    return json.dumps(
        {
            "k_s_inv": rate_constant_s_inv,
            "t_half_s": round(t_half, 6),
            "tau_s": round(tau, 6),
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Reaction quotient Q from product/reactant concentrations "
        "(or partial pressures). Pass concentrations as lists aligned "
        "with stoich_products / stoich_reactants (the integer "
        "coefficients). Returns Q = Π products^c_i / Π reactants^c_i."
    )
)
async def reaction_quotient(
    product_concentrations: list[float],
    reactant_concentrations: list[float],
    product_stoich: list[int],
    reactant_stoich: list[int],
) -> str:
    if (
        len(product_concentrations) != len(product_stoich)
        or len(reactant_concentrations) != len(reactant_stoich)
    ):
        return json.dumps({"error": "concentration and stoich lists must align"})
    num = 1.0
    for c, s in zip(product_concentrations, product_stoich):
        if c <= 0:
            return json.dumps({"error": "all concentrations must be > 0"})
        num *= c**s
    den = 1.0
    for c, s in zip(reactant_concentrations, reactant_stoich):
        if c <= 0:
            return json.dumps({"error": "all concentrations must be > 0"})
        den *= c**s
    Q = num / den
    return json.dumps(
        {
            "Q": Q,
            "num": num,
            "den": den,
            "note": "Q > K means reverse direction; Q < K means forward.",
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Dalton's law partial pressure: p_i = y_i · P_total. y_i is "
        "the mole fraction (0-1), P_total in Pa. Returns p_i."
    )
)
async def partial_pressure(mole_fraction: float, total_pressure_Pa: float) -> str:
    if not (0.0 <= mole_fraction <= 1.0):
        return json.dumps({"error": "mole_fraction must be in 0-1"})
    if total_pressure_Pa < 0:
        return json.dumps({"error": "total_pressure_Pa must be >= 0"})
    p_i = mole_fraction * total_pressure_Pa
    return json.dumps(
        {"mole_fraction": mole_fraction, "P_total_Pa": total_pressure_Pa, "p_i_Pa": p_i},
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Convert molality (mol solute / kg solvent) to molarity "
        "(mol solute / L solution), given the solute molar mass "
        "(g/mol) and the solution density (kg/L)."
    )
)
async def molality_to_molarity(
    molality_mol_per_kg: float,
    solute_molar_mass_g_per_mol: float,
    solution_density_kg_per_L: float,
) -> str:
    if molality_mol_per_kg < 0 or solute_molar_mass_g_per_mol <= 0 or solution_density_kg_per_L <= 0:
        return json.dumps({"error": "check non-negativity and positive molar mass/density"})
    mass_solute_per_L = solution_density_kg_per_L * molality_mol_per_kg * solute_molar_mass_g_per_mol / 1000.0
    M = mass_solute_per_L / solute_molar_mass_g_per_mol
    return json.dumps(
        {
            "molality": molality_mol_per_kg,
            "density_kg_per_L": solution_density_kg_per_L,
            "molarity_M": round(M, 6),
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Boiling-point elevation: ΔTb = Kb · m · i. m is molality "
        "(mol/kg), Kb the ebullioscopic constant of the solvent "
        "(K·kg/mol, water = 0.512), i the van 't Hoff factor."
    )
)
async def boiling_point_elevation(
    molality_mol_per_kg: float,
    Kb_K_kg_per_mol: float,
    vant_hoff_i: float = 1.0,
    normal_boiling_point_K: float | None = None,
) -> str:
    if molality_mol_per_kg < 0:
        return json.dumps({"error": "molality must be >= 0"})
    delta_Tb = Kb_K_kg_per_mol * molality_mol_per_kg * vant_hoff_i
    out = {"delta_Tb_K": round(delta_Tb, 6)}
    if normal_boiling_point_K is not None:
        out["new_boiling_point_K"] = round(normal_boiling_point_K + delta_Tb, 6)
    return json.dumps(out, ensure_ascii=False)


# ---- Wave 2: numpy / scipy chemistry ports ----


@mcp.tool(
    description=(
        "Solve a first- or second-order kinetics ODE numerically. "
        "Pass reaction_order = 1 or 2, initial_concentration_M, "
        "rate_constant (units depend on order: s^-1 for 1st, "
        "M^-1 s^-1 for 2nd), and the time span (t_start_s, t_end_s). "
        "Returns concentration vs. time at ~50 sample points plus "
        "the half-life extracted from the integration."
    )
)
async def kinetics_solver(
    reaction_order: int,
    initial_concentration_M: float,
    rate_constant: float,
    t_start_s: float = 0.0,
    t_end_s: float = 1000.0,
) -> str:
    if reaction_order not in (1, 2):
        return json.dumps({"error": "reaction_order must be 1 or 2"})
    if initial_concentration_M <= 0 or rate_constant <= 0:
        return json.dumps({"error": "initial_concentration and rate_constant must be > 0"})
    if t_end_s <= t_start_s:
        return json.dumps({"error": "t_end_s must be > t_start_s"})

    def ode(t, y):
        c = y[0]
        if reaction_order == 1:
            return [-rate_constant * c]
        return [-rate_constant * c * c]

    sol = solve_ivp(
        ode,
        (t_start_s, t_end_s),
        [initial_concentration_M],
        t_eval=np.linspace(t_start_s, t_end_s, 51),
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )
    if not sol.success:
        return json.dumps({"error": f"integration failed: {sol.message}"})
    c_half = initial_concentration_M / 2.0
    times = sol.t.tolist()
    concs = sol.y[0].tolist()
    # Find half-life by interpolation on the descending trajectory.
    t_half = None
    for i in range(1, len(concs)):
        if concs[i] <= c_half:
            t0, t1 = times[i - 1], times[i]
            c0, c1 = concs[i - 1], concs[i]
            if c0 != c1:
                t_half = t0 + (c0 - c_half) / (c0 - c1) * (t1 - t0)
            else:
                t_half = t0
            break
    return json.dumps(
        {
            "reaction_order": reaction_order,
            "initial_concentration_M": initial_concentration_M,
            "rate_constant": rate_constant,
            "t_start_s": t_start_s,
            "t_end_s": t_end_s,
            "t_half_s": t_half,
            "n_points": len(times),
            "times_s": times,
            "concentrations_M": concs,
        },
        ensure_ascii=False,
    )


# ---- Wave 3: rdkit-dependent ports ----


@mcp.tool(
    description=(
        "Compute a comprehensive set of physical-chemistry "
        "descriptors from a SMILES string using rdkit: MW, logP, "
        "TPSA, HBD, HBA, rotatable bonds, aromatic rings, fraction "
        "Csp3, molar refractivity. Returns a flat dict."
    )
)
async def rdkit_all_descriptors(smiles: str) -> str:
    if not _RDKIT_OK:
        return json.dumps({"error": "rdkit not installed"})
    if not smiles.strip():
        return json.dumps({"error": "smiles must be non-empty"})
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return json.dumps({"error": f"invalid SMILES: {smiles}"})
    return json.dumps(
        {
            "smiles": smiles,
            "canonical_smiles": Chem.MolToSmiles(mol),
            "MW": round(Descriptors.MolWt(mol), 4),
            "exact_mass": round(Descriptors.ExactMolWt(mol), 4),
            "logP": round(float(Crippen.MolLogP(mol)), 4),
            "MR": round(float(Crippen.MolMR(mol)), 4),
            "TPSA": round(Descriptors.TPSA(mol), 4),
            "H_donors": Lipinski.NumHDonors(mol),
            "H_acceptors": Lipinski.NumHAcceptors(mol),
            "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
            "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "heavy_atoms": mol.GetNumHeavyAtoms(),
            "fraction_csp3": round(
                rdMolDescriptors.CalcFractionCSP3(mol), 4
            ),
        },
        ensure_ascii=False,
    )


# ---- Wave: ideal_gas_calculation (ported from
# SciAgentGYM-main/toolkits/chemistry/physical_chemistry/
# physical_chemistry_toolkit_4720.py:18). Solves PV = nRT for the one
# missing variable; the other three must be provided.
# ---------------------------------------------------------------------------


@mcp.tool(
    description=(
        "Solve the ideal gas law PV = n·R·T for any one missing variable. "
        "Pass three of (pressure Pa, volume m^3, temperature K, quantity "
        "mol) and leave the fourth as null; the gas constant defaults to "
        "8.314 J/(mol·K). Units can be customized via the units dict. "
        "Returns the resolved values."
    )
)
async def ideal_gas_calculation(
    pressure: float | None = None,
    volume: float | None = None,
    temperature: float | None = None,
    gas_constant: float = 8.314,
    quantity: float | None = None,
    units: dict[str, str] | None = None,
) -> str:
    if gas_constant <= 0:
        return json.dumps({"error": "gas_constant must be > 0"})

    # Default unit labels.
    unit_labels = {
        "pressure": "Pa",
        "volume": "m^3",
        "temperature": "K",
        "quantity": "mol",
    }
    if units:
        unit_labels.update({k: v for k, v in units.items() if k in unit_labels})

    provided = {
        "pressure": pressure,
        "volume": volume,
        "temperature": temperature,
        "quantity": quantity,
    }
    unknowns = [k for k, v in provided.items() if v is None]
    if len(unknowns) != 1:
        return json.dumps(
            {"error": f"exactly one variable must be None; got {len(unknowns)}"}
        )
    target = unknowns[0]
    p = pressure
    V = volume
    T = temperature
    n = quantity

    try:
        if target == "pressure":
            if None in (V, T, n):
                return json.dumps({"error": "V, T, n must all be provided"})
            value = n * gas_constant * T / V
        elif target == "volume":
            if None in (p, T, n):
                return json.dumps({"error": "p, T, n must all be provided"})
            value = n * gas_constant * T / p
        elif target == "temperature":
            if None in (p, V, n):
                return json.dumps({"error": "p, V, n must all be provided"})
            value = p * V / (n * gas_constant)
        elif target == "quantity":
            if None in (p, V, T):
                return json.dumps({"error": "p, V, T must all be provided"})
            value = p * V / (gas_constant * T)
        else:
            return json.dumps({"error": f"unknown target: {target}"})
    except ZeroDivisionError:
        return json.dumps({"error": "division by zero — check inputs"})

    return json.dumps(
        {
            "pressure": p,
            "volume": V,
            "temperature": T,
            "quantity": n,
            "gas_constant": gas_constant,
            "solved": target,
            "value": value,
            "units": unit_labels,
        },
        ensure_ascii=False,
    )


# ---- Wave 2: physical-chemistry tools ported from
# SciAgentGYM-main/toolkits/chemistry/physical_chemistry/.
# All six are pure stdlib + scipy (no rdkit dependency).

_HBAR = 1.054571817e-34   # J·s
_E_CHARGE = 1.602176634e-19  # C
_EPSILON_0 = 8.8541878128e-12  # F/m
_EV_PER_J = 6.241509074e18    # 1 J = 6.24e18 eV


def _round_energy(x: float) -> float:
    """Round a tiny or huge energy to 6 significant figures without
    dropping to 0 or inf. Returns float(x) when |x| is in a sane range.
    """
    if x == 0:
        return 0.0
    import math as _math
    mag = _math.floor(_math.log10(abs(x)))
    if -3 <= mag <= 6:
        return float(round(x, 6))
    # For tiny or huge values, keep ~6 significant digits.
    return float(f"{x:.6g}")


@mcp.tool(
    description=(
        "Sum up bond energies for a molecule given per-bond-type counts "
        "and per-bond-type energies. Typical use: estimate enthalpy of "
        "formation ΔHf ≈ Σ(atomization energies) - Σ(bond energies). "
        "Returns the total bond energy sum in kJ/mol."
    )
)
async def compute_bond_energy_sum(
    bond_counts: dict[str, int],
    bond_energy_data: dict[str, float],
) -> str:
    try:
        total = 0.0
        breakdown: dict[str, float] = {}
        for bond_type, count in bond_counts.items():
            if bond_type not in bond_energy_data:
                return json.dumps(
                    {"error": f"missing bond_energy_data for '{bond_type}'"}
                )
            count_i = int(count)
            energy_f = float(bond_energy_data[bond_type])
            contribution = count_i * energy_f
            breakdown[bond_type] = round(contribution, 4)
            total += contribution
        return json.dumps(
            {
                "bond_counts": bond_counts,
                "breakdown_kJ_per_mol": breakdown,
                "total_kJ_per_mol": round(total, 4),
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@mcp.tool(
    description=(
        "Compute the product mole amounts from a 3-gas mixture (O2, H2, "
        "X2 where X2 is a halogen) with proportional volumes (treated as "
        "moles at fixed T, P). Implements H2 + 0.5 O2 -> H2O and H2 + "
        "X2 -> 2 HX. Returns per-product moles and leftover H2."
    )
)
async def compute_products_from_mixture(
    volumes: dict[str, float],
    gases: dict[str, str],
) -> str:
    for k in ("A", "B", "C"):
        if k not in volumes or k not in gases:
            return json.dumps({"error": f"missing key {k} in volumes/gases"})
        if not isinstance(volumes[k], (int, float)) or volumes[k] < 0:
            return json.dumps({"error": f"invalid volume for {k}"})
        if not isinstance(gases[k], str):
            return json.dumps({"error": f"invalid gas formula for {k}"})
    mapping = {"O2": None, "X2": None, "H2": None}
    for tag in ("A", "B", "C"):
        g = gases[tag]
        if g == "O2":
            mapping["O2"] = tag
        elif g == "H2":
            mapping["H2"] = tag
        else:
            mapping["X2"] = tag
    if None in mapping.values():
        return json.dumps({"error": "gases must include O2, H2 and one halogen X2"})
    n_O2 = float(volumes[mapping["O2"]])
    n_X2 = float(volumes[mapping["X2"]])
    n_H2 = float(volumes[mapping["H2"]])
    halogen = gases[mapping["X2"]]
    if halogen == "O2" or halogen == "H2":
        return json.dumps({"error": f"X2 must be a halogen, got {halogen}"})
    # Stoichiometry: H2 + 0.5 O2 -> H2O  → 1 H2 per 1 O2, 1 H2O per 1 O2.
    #               H2 + X2  -> 2 HX      → 1 H2 per 1 X2, 2 HX per 1 X2.
    H2_needed = n_O2 + n_X2
    if n_H2 >= H2_needed:
        n_H2O = n_O2
        n_HX = 2.0 * n_X2
        leftover_H2 = n_H2 - H2_needed
        limiting = "none"
    else:
        # Distribute H2 to O2 first (water is preferred), then to halogen.
        if n_H2 <= n_O2:
            n_H2O = n_H2
            n_HX = 0.0
            leftover_H2 = 0.0
            limiting = "H2 (vs O2)"
        else:
            n_H2O = n_O2
            n_HX = 2.0 * (n_H2 - n_O2)
            leftover_H2 = 0.0
            limiting = "H2 (vs O2+X2)"
    return json.dumps(
        {
            "inputs": {"volumes": volumes, "gases": gases},
            "n_H2O": round(n_H2O, 6),
            "n_HX": round(n_HX, 6),
            "halogen": halogen,
            "leftover_H2": round(leftover_H2, 6),
            "limiting_reagent": limiting,
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Compute the first n energy levels (J) of a quantum system. "
        "system_type is one of 'particle_in_box' (parameters: mass kg, "
        "box_length m), 'harmonic_oscillator' (parameters: mass kg, "
        "force_constant N/m), 'hydrogen_atom' (parameters: reduced_mass "
        "kg, defaults to electron mass). Returns levels as a JSON list."
    )
)
async def quantum_energy_levels(
    system_type: str,
    parameters: dict[str, float],
    n_levels: int = 5,
) -> str:
    if n_levels < 1:
        return json.dumps({"error": "n_levels must be >= 1"})
    st = system_type.strip().lower()
    levels: list[float] = []
    try:
        if st == "particle_in_box":
            m = float(parameters["mass"])
            L = float(parameters["box_length"])
            for n in range(1, n_levels + 1):
                levels.append((n**2 * math.pi**2 * _HBAR**2) / (2 * m * L**2))
        elif st == "harmonic_oscillator":
            m = float(parameters["mass"])
            k = float(parameters["force_constant"])
            omega = math.sqrt(k / m)
            for n in range(n_levels):
                levels.append((n + 0.5) * _HBAR * omega)
        elif st == "hydrogen_atom":
            mu = float(parameters.get("reduced_mass", 9.1093837015e-31))
            # Rydberg in joules: Ry = μ·e^4 / (8 ε0^2 h^2). Use ħ for clarity:
            # Ry = μ e^4 / (2 (4π ε0)^2 ħ^2)
            factor = (mu * _E_CHARGE**4) / (
                2.0 * (4.0 * math.pi * _EPSILON_0) ** 2 * _HBAR**2
            )
            for n in range(1, n_levels + 1):
                levels.append(-factor / (n**2))
        else:
            return json.dumps(
                {"error": f"unsupported system_type: {system_type}"}
            )
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
    return json.dumps(
        {
            "system_type": st,
            "n_levels": n_levels,
            "energies_J": [_round_energy(e) for e in levels],
            "energies_eV": [_round_energy(e * _EV_PER_J) for e in levels],
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Compute basic thermodynamic properties of a gas: volume "
        "(auto-solved from ideal-gas or van der Waals eq if not given), "
        "internal energy U, enthalpy H, entropy S, Gibbs G, Helmholtz A "
        "— all SI (J or J/K). Pass moles, gas_constant, and optionally "
        "van_der_waals_params={'a': a, 'b': b} for a non-ideal gas."
    )
)
async def thermodynamic_properties(
    temperature: float,
    pressure: float,
    volume: float | None = None,
    moles: float = 1.0,
    gas_constant: float = 8.314,
    is_ideal_gas: bool = True,
    van_der_waals_params: dict[str, float] | None = None,
) -> str:
    try:
        if volume is None:
            if is_ideal_gas:
                volume = (moles * gas_constant * temperature) / pressure
            elif van_der_waals_params is not None:
                a = float(van_der_waals_params["a"])
                b = float(van_der_waals_params["b"])
                # Solve for V: (R T / (V - nb) - a n² / V²) - p = 0.
                def _vdw(v):
                    return (
                        gas_constant * temperature / (v - b * moles)
                        - a * moles**2 / v**2
                    ) - pressure
                guess = b * moles * 1.5 if b * moles > 0 else moles * gas_constant * temperature / pressure
                volume = float(fsolve(_vdw, guess)[0])
            else:
                return json.dumps(
                    {"error": "non-ideal gas requires van_der_waals_params"}
                )
        if is_ideal_gas:
            internal_energy = 1.5 * moles * gas_constant * temperature
        else:
            a = float(van_der_waals_params["a"])
            internal_energy = (
                1.5 * moles * gas_constant * temperature
                - a * moles**2 / volume
            )
        enthalpy = internal_energy + pressure * volume
        # Sackur–Tetrode-like residual entropy (no h, no m — uses mole-scale).
        entropy = moles * gas_constant * (
            math.log(volume / moles) + 2.5 * math.log(temperature)
        )
        gibbs = enthalpy - temperature * entropy
        helmholtz = internal_energy - temperature * entropy
        return json.dumps(
            {
                "temperature_K": temperature,
                "pressure_Pa": pressure,
                "volume_m3": round(volume, 8),
                "moles": moles,
                "internal_energy_J": round(internal_energy, 4),
                "enthalpy_J": round(enthalpy, 4),
                "entropy_J_per_K": round(entropy, 4),
                "gibbs_energy_J": round(gibbs, 4),
                "helmholtz_energy_J": round(helmholtz, 4),
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@mcp.tool(
    description=(
        "Compute vapor-liquid equilibrium for a multi-component mixture "
        "given Antoine coefficients [A, B, C] (log10 P_sat = A - B/(T+C), "
        "P_sat in mmHg) and liquid-phase mole fractions. method='raoult' "
        "(default) or 'modified_raoult' (requires interaction_parameters). "
        "Returns vapor-phase composition, K-values, and a bubble-point "
        "estimate."
    )
)
async def phase_equilibrium_calculator(
    components: list[dict],
    temperature: float,
    pressure: float = 101325.0,
    interaction_parameters: dict[str, float] | None = None,
    method: str = "raoult",
) -> str:
    try:
        k_values: dict[str, float] = {}
        vapor: dict[str, float] = {}
        liquid: dict[str, float] = {}
        for comp in components:
            name = comp["name"]
            x_i = float(comp["mole_fraction"])
            A, B, C = comp["antoine_coefficients"]
            log_p_sat = A - B / (temperature + C)
            p_sat = 10**log_p_sat * 133.322
            k = p_sat / pressure
            k_values[name] = k
            liquid[name] = x_i
            if method == "raoult":
                y_i = x_i * k
            elif method == "modified_raoult":
                if interaction_parameters is None:
                    return json.dumps(
                        {"error": "modified_raoult requires interaction_parameters"}
                    )
                gamma_i = float(interaction_parameters.get(name, 1.0))
                y_i = x_i * gamma_i * k
            else:
                return json.dumps({"error": f"unsupported method: {method}"})
            vapor[name] = y_i
        total_y = sum(vapor.values()) or 1.0
        for n in vapor:
            vapor[n] = round(vapor[n] / total_y, 6)
            liquid[n] = round(liquid[n], 6)
            k_values[n] = round(k_values[n], 6)
        # Bubble-point estimate: T such that Σ x_i P_sat_i(T) = P.
        def _bubble_eq(T_guess):
            total = 0.0
            for comp in components:
                x_i = float(comp["mole_fraction"])
                A, B, C = comp["antoine_coefficients"]
                log_p = A - B / (T_guess + C)
                p_sat = 10**log_p * 133.322
                total += x_i * p_sat
            return total - pressure
        try:
            T_bubble = float(fsolve(_bubble_eq, temperature)[0])
            bubble_point_K = round(T_bubble, 4)
        except Exception:
            bubble_point_K = None
        return json.dumps(
            {
                "temperature_K": temperature,
                "pressure_Pa": pressure,
                "method": method,
                "vapor_phase": vapor,
                "liquid_phase": liquid,
                "k_values": k_values,
                "bubble_point_K": bubble_point_K,
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@mcp.tool(
    description=(
        "Integrate a chemical kinetics ODE system for the time interval "
        "time_span = (t0, t1) (s) given rate_constants (1/s or 1/(M·s) "
        "depending on order), initial_concentrations dict, and an "
        "optional custom reaction_model(t, y, k)->dy/dt. Built-in "
        "defaults: 2 species → first-order A->B, 3 species → second-order "
        "A+B->C. Returns dense time grid (1000 points) and concentration "
        "matrix."
    )
)
async def reaction_kinetics_solver(
    rate_constants: list[float],
    initial_concentrations: dict[str, float] | list[float],
    time_span: list[float],
    reaction_model: str | None = None,
    temperature_dependent: bool = False,
    activation_energies: list[float] | None = None,
    pre_exponential_factors: list[float] | None = None,
    temperature: float = 298.15,
) -> str:
    try:
        if isinstance(initial_concentrations, dict):
            species = list(initial_concentrations.keys())
            y0 = [float(initial_concentrations[s]) for s in species]
        else:
            y0 = [float(c) for c in initial_concentrations]
            species = [f"Species_{i}" for i in range(len(y0))]
        ks = list(rate_constants)
        if temperature_dependent:
            if activation_energies is None or pre_exponential_factors is None:
                return json.dumps(
                    {"error": "temperature_dependent requires Ea and A"}
                )
            R = 8.314
            ks = [
                A * math.exp(-Ea / (R * temperature))
                for A, Ea in zip(pre_exponential_factors, activation_energies)
            ]
        # Default reaction model by species count.
        if reaction_model is None or reaction_model == "first_order":
            if len(y0) == 2:
                def _model(t, y, k):
                    return [-k[0] * y[0], k[0] * y[0]]
            elif len(y0) == 3:
                def _model(t, y, k):
                    return [
                        -k[0] * y[0] * y[1],
                        -k[0] * y[0] * y[1],
                        k[0] * y[0] * y[1],
                    ]
            else:
                return json.dumps(
                    {"error": "default model supports 2 (A->B) or 3 (A+B->C) species; pass reaction_model"}
                )
        else:
            return json.dumps(
                {"error": "custom reaction_model not yet supported in this port"}
            )
        sol = solve_ivp(
            lambda t, y: _model(t, y, ks),
            (float(time_span[0]), float(time_span[1])),
            y0,
            method="RK45",
            dense_output=True,
            rtol=1e-6,
            atol=1e-9,
        )
        if not sol.success:
            return json.dumps({"error": f"solve_ivp failed: {sol.message}"})
        t_dense = np.linspace(float(time_span[0]), float(time_span[1]), 200)
        y_dense = sol.sol(t_dense)
        concentrations: dict[str, list[float]] = {
            species[i]: [round(float(v), 6) for v in y_dense[i]]
            for i in range(len(species))
        }
        return json.dumps(
            {
                "species": species,
                "rate_constants_per_s": ks,
                "time_grid": [round(float(t), 6) for t in t_dense],
                "concentrations": concentrations,
                "t_final_concentrations": {
                    species[i]: round(float(y_dense[i, -1]), 6)
                    for i in range(len(species))
                },
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


# ---- Wave 3: electrochemistry / partition tools ported from
# SciAgentGYM-main/toolkits/chemistry/physical_chemistry/
# nernst_potential.py and partition_coefficient_func_expose.py.
# All inline (no SciAgentGYM helper deps).
# ---------------------------------------------------------------------------

_F_CONST = 96485.33212   # C/mol — Faraday constant
_R_GAS = 8.314462618       # J/(mol·K) — gas constant


@mcp.tool(
    description=(
        "Compute the remaining (unreacted) ion concentration after "
        "electrodeposition. Uses the Nernst equation in the limit where "
        "the applied potential E_applied plus overpotential drives the "
        "reaction to completion. Returns mol/kg, floored at 1e-20 to "
        "avoid numerical underflow. Inputs: applied potential (V), "
        "standard potential E0 (V), electron count n, temperature (K), "
        "overpotential (V)."
    )
)
async def solve_remaining_concentration(
    E_applied: float,
    E0: float,
    n: int,
    T: float = 298.15,
    overpotential: float = 0.0,
) -> str:
    try:
        E_eff = float(E_applied) + float(overpotential)
        exponent = (float(E0) - E_eff) * (float(n) * _F_CONST) / (_R_GAS * float(T))
        c_remaining = math.exp(exponent)
        c_remaining = max(c_remaining, 1e-20)
        return json.dumps(
            {
                "E_applied_V": float(E_applied),
                "E0_V": float(E0),
                "n_electrons": int(n),
                "T_K": float(T),
                "overpotential_V": float(overpotential),
                "E_effective_V": round(E_eff, 6),
                "remaining_concentration": _round_energy(c_remaining),
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@mcp.tool(
    description=(
        "Solve the apparent partition coefficient problem for a single "
        "acidic or basic compound. Uses logD = logP_neutral + "
        "log10(fraction_neutral) for acids, or logD = logP_neutral + "
        "log10(1 - fraction_neutral) for bases. fraction_neutral comes "
        "from the Henderson–Hasselbalch equation. Returns logD and K_ow."
    )
)
async def solve_partition_problem(
    compound: str,
    smiles: str,
    logP_neutral: float,
    pKa: float,
    pH: float,
    acidic: bool = True,
    temperature: float = 298.15,
) -> str:
    issues: list[str] = []
    for name, val in (("logP_neutral", logP_neutral), ("pKa", pKa), ("pH", pH)):
        if not isinstance(val, (int, float)):
            issues.append(f"{name}_missing_or_invalid")
    if issues:
        return json.dumps(
            {
                "ok": False,
                "compound": compound,
                "smiles": smiles,
                "logP_neutral": logP_neutral,
                "pKa": pKa,
                "pH": pH,
                "issues": issues,
                "message": "solve_partition_problem requires numeric logP_neutral, pKa, pH.",
            },
            ensure_ascii=False,
        )
    try:
        L = float(logP_neutral)
        pKa_f = float(pKa)
        pH_f = float(pH)
        # Henderson–Hasselbalch: ratio [A-]/[HA] = 10^(pH - pKa) (acid)
        #                         ratio [BH+]/[B] = 10^(pKa - pH) (base)
        if acidic:
            ratio = 10 ** (pH_f - pKa_f)
            frac_neutral = 1.0 / (1.0 + ratio)
        else:
            ratio = 10 ** (pKa_f - pH_f)
            frac_neutral = 1.0 / (1.0 + ratio)
        # logD = logP + log10(frac_neutral) — only the neutral species partitions.
        logD = L + math.log10(max(frac_neutral, 1e-30))
        K_ow = 10**logD
        return json.dumps(
            {
                "ok": True,
                "compound": compound,
                "smiles": smiles,
                "logP_neutral": round(L, 4),
                "pKa": round(pKa_f, 4),
                "pH": round(pH_f, 4),
                "acidic": acidic,
                "frac_neutral": _round_energy(frac_neutral),
                "logD_apparent": round(logD, 4),
                "K_ow": _round_energy(K_ow),
                "analysis": {
                    "dominant_species": "neutral" if frac_neutral > 0.5 else (
                        "anion" if acidic else "cation"
                    ),
                    "ionization_pct": round((1.0 - frac_neutral) * 100, 2),
                },
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@mcp.tool(
    description=(
        "Analyze cathode-reaction priorities for a multi-ion aqueous "
        "electrolyte. For each ion, compute its Nernst reduction "
        "potential at its bulk concentration, subtract overpotential to "
        "get the effective deposition potential, and compute the driving "
        "force (E_cathode - E_effective). The most-positive effective "
        "potential with negative driving force is the dominant reaction; "
        "otherwise HER (hydrogen evolution) wins. Returns per-reaction "
        "data, dominant reaction, and (for metal deposition) the "
        "remaining ion concentration."
    )
)
async def analyze_cathode_reaction(
    ions: list[dict],
    pH: float,
    T: float,
    E_cathode: float,
    overpotentials: dict[str, float],
    electrode_type: str,
) -> str:
    try:
        T_K = float(T) + 273.15 if float(T) < 100.0 else float(T)
        F = _F_CONST
        reactions: list[dict] = []
        for ion in ions:
            name = ion["name"]
            E0 = float(ion["E0"])
            n = int(ion["n"])
            c0 = float(ion["c0"])
            product = ion["product"]
            eta_ion = float(ion.get("overpotential", 0.0))
            # Nernst: E = E0 + (R T / nF) ln(c0); for reduction from Mn+ to M,
            # activity of M is 1, so use ln(c_oxidized).
            E_ion = E0 + (_R_GAS * T_K) / (n * F) * math.log(c0)
            E_ion_eff = E_ion - eta_ion
            driving_force = float(E_cathode) - E_ion_eff
            reactions.append(
                {
                    "reaction": f"{name} -> {product}",
                    "ion": name,
                    "type": "metal",
                    "E_thermo_V": round(E_ion, 6),
                    "E_effective_V": round(E_ion_eff, 6),
                    "driving_force_V": round(driving_force, 6),
                    "c0_mol_per_kg": c0,
                    "n_electrons": n,
                }
            )
        # HER.
        eta_H = float(overpotentials.get(electrode_type, 0.0))
        # E_H = 0 - (R T / F) ln(10) * pH  (Nernst for H+/H2 at unit activity).
        E_H_thermo = -(_R_GAS * T_K) / F * math.log(10) * float(pH)
        E_H_eff = E_H_thermo - eta_H
        driving_force_H = float(E_cathode) - E_H_eff
        reactions.append(
            {
                "reaction": "2H+ + 2e- -> H2",
                "ion": "H+",
                "type": "HER",
                "E_thermo_V": round(E_H_thermo, 6),
                "E_effective_V": round(E_H_eff, 6),
                "driving_force_V": round(driving_force_H, 6),
                "c0_mol_per_kg": None,
                "n_electrons": 2,
            }
        )
        feasible = [r for r in reactions if r["driving_force_V"] <= 0]
        if not feasible:
            dominant = None
            reaction_type = "No reaction"
        else:
            # Most positive E_effective wins.
            dominant = max(feasible, key=lambda r: r["E_effective_V"])
            reaction_type = dominant["type"]
        remaining = None
        if dominant and dominant["type"] == "metal":
            ion_match = next(
                (i for i in ions if i["name"] == dominant["ion"]), None
            )
            if ion_match is not None:
                # Inline solve_remaining_concentration.
                E_eff = float(E_cathode) + float(ion_match.get("overpotential", 0.0))
                exponent = (float(ion_match["E0"]) - E_eff) * (
                    int(ion_match["n"]) * F
                ) / (_R_GAS * T_K)
                remaining = max(math.exp(exponent), 1e-20)
        return json.dumps(
            {
                "all_reactions": reactions,
                "dominant_reaction": dominant,
                "reaction_type": reaction_type,
                "remaining_concentration": _round_energy(remaining) if remaining is not None else None,
                "temperature_K": round(T_K, 4),
                "pH": float(pH),
                "E_cathode_V": float(E_cathode),
                "electrode_type": electrode_type,
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    import asyncio

    asyncio.run(mcp.run_stdio_async())
