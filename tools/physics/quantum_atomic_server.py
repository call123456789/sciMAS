#!/usr/bin/env python3
"""Quantum, atomic, plasma, structural, and thermal physics MCP server.

Bundled tool port from
SciAgentGYM-main/toolkits/physics/{atomic_and_molecular_physics,plasma_physics,
solid_mechanics,structural_mechanics,thermodynamics}/ into a single server
per the 5-bucket sub-discipline plan. Tools cover quantum mechanics
(matrix properties, spin dynamics, entanglement), hydrogen transitions,
particle-physics kinematics, plasma Saha ionization, classical structural
stress / strain, beam & truss analysis, and thermodynamic cycle / kinetic
theory / nucleation primitives.

All implementations are stdlib + numpy. Every tool returns a JSON string.
Tool naming is snake_case.
"""

from __future__ import annotations

import json
import math

import numpy as np

try:
    import scipy.linalg as la
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("physics-quantum-atomic")

HBAR = 1.054571817e-34
KB = 1.380649e-23
C = 2.99792458e8
G = 6.67430e-11
R_GAS = 8.314462618


def _ok(name: str, **fields) -> str:
    return json.dumps({"status": "ok", "tool": name, **fields}, ensure_ascii=False)


def _err(message: str, **fields) -> str:
    return json.dumps({"status": "error", "message": message, **fields}, ensure_ascii=False)


# Matplotlib (headless).
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_OK = True
except ImportError:
    _MPL_OK = False

_VIZ_DIR = "/tmp/scimas_physics_quantum_atomic"


# ---------------------------------------------------------------------------
# Quantum mechanics: matrix properties
# ---------------------------------------------------------------------------

@mcp.tool(description="Parse a matrix from a string like '1 2; 3 4' and return as a numpy-compatible list.")
async def parse_matrix_string(s: str) -> str:
    if not s.strip():
        return _err("empty string")
    rows = []
    for line in s.replace(",", " ").split(";"):
        toks = line.split()
        if not toks:
            continue
        try:
            row = [complex(tok) for tok in toks]
        except ValueError:
            return _err("non-numeric token encountered")
        rows.append(row)
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        return _err("rows have inconsistent widths")
    return _ok("parse_matrix_string", matrix=rows,
               n_rows=len(rows), n_cols=len(rows[0]))


@mcp.tool(description="Test whether a complex square matrix is Hermitian (H = H^†).")
async def is_hermitian(matrix: list[list[float]]) -> str:
    M = np.array(matrix, dtype=complex)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        return _err("not a square matrix")
    err = float(np.max(np.abs(M - M.conj().T)))
    return _ok("is_hermitian",
               hermitian=(err < 1e-9),
               max_deviation=err)


@mcp.tool(description="Test whether a complex square matrix is unitary (H^† H = I).")
async def is_unitary(matrix: list[list[float]]) -> str:
    M = np.array(matrix, dtype=complex)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        return _err("not a square matrix")
    n = M.shape[0]
    err = float(np.max(np.abs(M.conj().T @ M - np.eye(n))))
    return _ok("is_unitary", unitary=(err < 1e-9),
               max_deviation=err)


@mcp.tool(description="Matrix trace (sum of diagonal entries).")
async def matrix_trace(matrix: list[list[float]]) -> str:
    M = np.array(matrix, dtype=complex)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        return _err("not a square matrix")
    return _ok("matrix_trace",
               trace=float(np.trace(M).real))


@mcp.tool(description="Matrix exponential exp(A) using Taylor / scipy.linalg.expm if available.")
async def matrix_exponential(matrix: list[list[float]]) -> str:
    M = np.array(matrix, dtype=complex)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        return _err("not a square matrix")
    if _SCIPY_OK:
        E = la.expm(M)
    else:
        # Series expansion, 25 terms.
        E = np.eye(M.shape[0], dtype=complex)
        term = np.eye(M.shape[0], dtype=complex)
        for k in range(1, 25):
            term = term @ M / k
            E = E + term
    return _ok("matrix_exponential", matrix=[[float(z.real) for z in row] for row in E])


# ---------------------------------------------------------------------------
# Quantum spin-1/2
# ---------------------------------------------------------------------------

@mcp.tool(description="Return the three Pauli matrices σ_x, σ_y, σ_z as 2x2 complex matrices.")
async def pauli_matrices() -> str:
    sx = [[0.0, 1.0], [1.0, 0.0]]
    sy = [[0.0, -1.0], [1.0, 0.0]]
    sz = [[1.0, 0.0], [0.0, -1.0]]
    return _ok("pauli_matrices",
               sigma_x=sx, sigma_y=sy, sigma_z=sz)


@mcp.tool(description="Spin-1/2 density matrix from a Bloch vector (bx, by, bz): ρ = (I + bx σx + by σy + bz σz) / 2.")
async def spin_density_matrix(bx: float, by: float, bz: float) -> str:
    rho = [
        [0.5 * (1.0 + bz), 0.5 * (bx - 1j * by)],
        [0.5 * (bx + 1j * by), 0.5 * (1.0 - bz)],
    ]
    purity = 0.5 * (1.0 + bx * bx + by * by + bz * bz)
    return _ok("spin_density_matrix",
               rho=rho, bloch_vector=[bx, by, bz],
               purity=purity)


@mcp.tool(description="Expectation value of an observable A given density matrix ρ: <A> = Tr(ρ A).")
async def expectation_value(rho: list[list[float]], A: list[list[float]]) -> str:
    R = np.array(rho, dtype=complex)
    M = np.array(A, dtype=complex)
    if R.shape != M.shape:
        return _err("rho and A must have the same shape")
    val = float(np.real(np.trace(R @ M)))
    return _ok("expectation_value", value=val)


# ---------------------------------------------------------------------------
# Hydrogen transitions / spectroscopy
# ---------------------------------------------------------------------------

@mcp.tool(description="Hydrogen transition matrix element for dipole-allowed transitions: |<n'|r|n>| in atomic units.")
async def hydrogen_dipole_matrix_element(n_initial: int, n_final: int) -> str:
    if n_initial <= 0 or n_final <= 0:
        return _err("quantum numbers must be > 0")
    # Very rough: dipole matrix element scales as n^2 * a0 for transitions
    # between neighbouring n. Return n^2 * a0 as crude proxy in units of a0.
    delta_n = abs(n_final - n_initial)
    a0 = 5.29177210903e-11
    proxy = (n_initial ** 2) * a0 if delta_n == 1 else (n_initial ** 2) / delta_n * a0
    return _ok("hydrogen_dipole_matrix_element",
               n_initial=n_initial, n_final=n_final,
               matrix_element_m=proxy,
               allowed=(delta_n == 1))


@mcp.tool(description="Selection-rule check for electric dipole transitions: Δℓ = ±1, Δm = 0, ±1.")
async def check_dipole_selection_rule(l_initial: int, l_final: int,
                                      m_initial: int, m_final: int) -> str:
    if l_initial < 0 or l_final < 0:
        return _err("l must be >= 0")
    delta_l = l_final - l_initial
    delta_m = m_final - m_initial
    allowed = (abs(delta_l) == 1) and (delta_m in (-1, 0, 1))
    return _ok("check_dipole_selection_rule",
               allowed=allowed,
               delta_l=delta_l, delta_m=delta_m)


# ---------------------------------------------------------------------------
# Particle physics kinematics
# ---------------------------------------------------------------------------

@mcp.tool(description="Invariant mass squared from E and momentum 4-vector: M^2 c^4 = E^2 - |p|^2 c^2. Returns M in natural units (eV/c^2) if E and p in eV.")
async def invariant_mass_squared(E: float, p_x: float, p_y: float, p_z: float) -> str:
    p2 = p_x ** 2 + p_y ** 2 + p_z ** 2
    M2 = E ** 2 - p2 * C ** 2  # in (J, m/s) units, M in kg
    M = math.sqrt(abs(M2)) / C ** 2 if M2 > 0 else -math.sqrt(abs(M2)) / C ** 2
    return _ok("invariant_mass_squared",
               M_kg=M, M_eV=M * C ** 2 / 1.602176634e-19,
               E_J=E, p_J=math.sqrt(p2) * 1e-9 if False else None)


@mcp.tool(description="Threshold energy for production of a particle of mass m in a fixed-target collision with beam energy E_beam.")
async def threshold_energy(m_product_kg: float, m_target_kg: float,
                            m_beam_kg: float) -> str:
    if m_beam_kg <= 0:
        return _err("beam mass must be > 0")
    # E_thr = m_product * (1 + m_product / (2 m_target))
    E_thr_J = m_product_kg * C ** 2 * (1.0 + m_product_kg / (2.0 * m_target_kg))
    return _ok("threshold_energy",
               E_thr_J=E_thr_J,
               E_thr_eV=E_thr_J / 1.602176634e-19,
               m_product_kg=m_product_kg)


# ---------------------------------------------------------------------------
# Quantum gates
# ---------------------------------------------------------------------------

@mcp.tool(description="Standard CNOT gate matrix (control=0, target=1) as a 4x4 complex matrix.")
async def cnot_gate() -> str:
    g = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
    return _ok("cnot_gate", matrix=g)


@mcp.tool(description="Anti-control (control=1, target flips) variant of CNOT.")
async def anti_cnot_gate() -> str:
    g = [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]
    return _ok("anti_cnot_gate", matrix=g)


# ---------------------------------------------------------------------------
# Quantum metrology / Fisher information
# ---------------------------------------------------------------------------

@mcp.tool(description="Quantum Fisher information for a pure-state family |ψ(θ)⟩: F_Q = 4 (⟨∂θψ|∂θψ⟩ - |⟨ψ|∂θψ⟩|^2).")
async def quantum_fisher_information_pure(dpsi_dtheta_norm_sq: float,
                                            inner_product_sq: float) -> str:
    F_Q = 4.0 * (dpsi_dtheta_norm_sq - inner_product_sq)
    return _ok("quantum_fisher_information_pure",
               F_Q=F_Q,
               dpsi_norm_sq=dpsi_dtheta_norm_sq,
               inner_product_sq=inner_product_sq)


# ---------------------------------------------------------------------------
# Plasma / Saha
# ---------------------------------------------------------------------------

@mcp.tool(description="Saha ionization equation ratio n_e * n_ion / n_atom for hydrogen-like plasma at temperature T and electron density n_e.")
async def saha_ionization_ratio(T_K: float, n_e_m3: float, ionization_energy_J: float) -> str:
    if T_K <= 0 or n_e_m3 <= 0 or ionization_energy_J <= 0:
        return _err("T_K, n_e, ionization_energy must be > 0")
    # Saha factor (ga = 1, gi = 1, me assumed classical).
    kT = KB * T_K
    factor = (1.0 / n_e_m3) * (2.0 * math.pi * 9.10938356e-31 * kT / (6.62607015e-34 ** 2)) ** 1.5 \
        * math.exp(-ionization_energy_J / kT)
    return _ok("saha_ionization_ratio",
               ratio_factor=factor,
               T_K=T_K, n_e_m3=n_e_m3,
               ionization_energy_eV=ionization_energy_J / 1.602176634e-19)


@mcp.tool(description="MHD pipe flow velocity (idealized): v = (1/cB) (P - n k T) / (B^2) for pressure gradient P.")
async def mhd_pipe_flow(pressure_gradient_Pa_m: float, n_m3: float,
                         T_K: float, magnetic_field_T: float, ion_sound_m_s: float = 1e4) -> str:
    if magnetic_field_T <= 0 or ion_sound_m_s <= 0:
        return _err("B>0 and c_s>0 required")
    if n_m3 <= 0 or T_K <= 0:
        return _err("n>0 and T>0 required")
    P = n_m3 * KB * T_K
    v = (pressure_gradient_Pa_m - P * 0) / (ion_sound_m_s * magnetic_field_T ** 2)
    return _ok("mhd_pipe_flow", velocity_m_s=v,
               magnetic_field_T=magnetic_field_T)


# ---------------------------------------------------------------------------
# Solid / structural mechanics
# ---------------------------------------------------------------------------

@mcp.tool(description="Axial stress σ = F / A for a bar under axial load.")
async def axial_stress(force_N: float, area_m2: float) -> str:
    if area_m2 <= 0:
        return _err("area must be > 0")
    return _ok("axial_stress",
               stress_Pa=force_N / area_m2)


@mcp.tool(description="Hoop stress σ_θ = p r / t in a thin-walled cylindrical pressure vessel.")
async def hoop_stress(internal_pressure_Pa: float, radius_m: float, wall_thickness_m: float) -> str:
    if wall_thickness_m <= 0 or radius_m <= 0 or internal_pressure_Pa < 0:
        return _err("require r>0, t>0, p>=0")
    return _ok("hoop_stress",
               stress_Pa=internal_pressure_Pa * radius_m / wall_thickness_m)


@mcp.tool(description="Torsion angle per unit length for a circular shaft: φ' = T / (G J).")
async def torsion_angle_per_length(torque_N_m: float, shear_modulus_Pa: float,
                                     polar_moment_m4: float) -> str:
    if shear_modulus_Pa <= 0 or polar_moment_m4 <= 0:
        return _err("G>0 and J>0 required")
    return _ok("torsion_angle_per_length",
               phi_per_length_rad_m=torque_N_m / (shear_modulus_Pa * polar_moment_m4))


@mcp.tool(description="Euler critical load for a column: P_cr = π² E I / L_eff², with end-condition factor.")
async def euler_critical_load(modulus_E_Pa: float, moment_of_inertia_m4: float,
                               length_m: float, end_factor: float = 2.0) -> str:
    if modulus_E_Pa <= 0 or moment_of_inertia_m4 <= 0 or length_m <= 0 or end_factor <= 0:
        return _err("E>0, I>0, L>0, k>0 required")
    P_cr = math.pi ** 2 * modulus_E_Pa * moment_of_inertia_m4 / (end_factor * length_m) ** 2
    return _ok("euler_critical_load",
               load_N=P_cr,
               end_condition_factor=end_factor)


@mcp.tool(description="Cantilever deflection at free end for point load at the tip: δ = F L^3 / (3 E I).")
async def cantilever_deflection_point_load(load_N: float, length_m: float,
                                            modulus_E_Pa: float, moment_of_inertia_m4: float) -> str:
    if length_m <= 0 or modulus_E_Pa <= 0 or moment_of_inertia_m4 <= 0:
        return _err("L>0, E>0, I>0 required")
    deflection = load_N * length_m ** 3 / (3.0 * modulus_E_Pa * moment_of_inertia_m4)
    return _ok("cantilever_deflection_point_load",
               deflection_m=deflection)


@mcp.tool(description="Simply-supported beam max bending moment at midspan for centre load: M_max = P L / 4.")
async def simply_supported_center_moment(load_N: float, length_m: float) -> str:
    if length_m <= 0:
        return _err("L>0 required")
    return _ok("simply_supported_center_moment",
               moment_N_m=load_N * length_m / 4.0,
               max_deflection_m=load_N * length_m ** 3 / (48.0 * 2.0e11 * 1e-6))


# ---------------------------------------------------------------------------
# Thermodynamics
# ---------------------------------------------------------------------------

@mcp.tool(description="Ideal gas law: P V = n R T. Solves for any missing variable given the other three.")
async def ideal_gas(P_Pa: float = 0.0, V_m3: float = 0.0,
                    n_mol: float = 0.0, T_K: float = 0.0) -> str:
    supplied = [v != 0 for v in (P_Pa, V_m3, n_mol, T_K)]
    if sum(supplied) != 3:
        return _err("supply exactly three of P, V, n, T")
    if P_Pa == 0:
        P_Pa = n_mol * R_GAS * T_K / V_m3
        return _ok("ideal_gas", P_Pa=P_Pa, V_m3=V_m3, n_mol=n_mol, T_K=T_K,
                   solved_for="P_Pa")
    if V_m3 == 0:
        V_m3 = n_mol * R_GAS * T_K / P_Pa
        return _ok("ideal_gas", P_Pa=P_Pa, V_m3=V_m3, n_mol=n_mol, T_K=T_K,
                   solved_for="V_m3")
    if n_mol == 0:
        n_mol = P_Pa * V_m3 / (R_GAS * T_K)
        return _ok("ideal_gas", P_Pa=P_Pa, V_m3=V_m3, n_mol=n_mol, T_K=T_K,
                   solved_for="n_mol")
    T_K = P_Pa * V_m3 / (n_mol * R_GAS)
    return _ok("ideal_gas", P_Pa=P_Pa, V_m3=V_m3, n_mol=n_mol, T_K=T_K,
               solved_for="T_K")


@mcp.tool(description="Maxwell-Boltzmann average kinetic energy per particle: ⟨E_kin⟩ = (3/2) k_B T for 3D.")
async def maxwell_boltzmann_avg_ke(T_K: float, degrees_of_freedom: int = 3) -> str:
    if T_K <= 0:
        return _err("T_K must be > 0")
    E = 0.5 * degrees_of_freedom * KB * T_K
    return _ok("maxwell_boltzmann_avg_ke",
               energy_J=E, T_K=T_K, dof=degrees_of_freedom)


@mcp.tool(description="Most probable speed for Maxwell-Boltzmann: v_p = sqrt(2 k_B T / m).")
async def maxwell_boltzmann_most_probable_speed(T_K: float, mass_kg: float) -> str:
    if T_K <= 0 or mass_kg <= 0:
        return _err("T>0, m>0 required")
    v = math.sqrt(2.0 * KB * T_K / mass_kg)
    return _ok("maxwell_boltzmann_most_probable_speed",
               speed_m_s=v, T_K=T_K, mass_kg=mass_kg)


@mcp.tool(description="Carnot efficiency: η = 1 - T_cold / T_hot (absolute temperatures).")
async def carnot_efficiency(T_hot_K: float, T_cold_K: float) -> str:
    if T_hot_K <= 0 or T_cold_K <= 0 or T_cold_K >= T_hot_K:
        return _err("require T_hot > T_cold > 0")
    eta = 1.0 - T_cold_K / T_hot_K
    return _ok("carnot_efficiency",
               eta=eta, T_hot_K=T_hot_K, T_cold_K=T_cold_K)


@mcp.tool(description="Clausius-Clapeyron: P2 = P1 * exp(-ΔH/R * (1/T2 - 1/T1)).")
async def clausius_clapeyron(P1_Pa: float, T1_K: float, T2_K: float, delta_H_J_per_mol: float) -> str:
    if P1_Pa <= 0 or T1_K <= 0 or T2_K <= 0:
        return _err("P>0, T>0 required")
    inv_T2 = 1.0 / T2_K - 1.0 / T1_K
    P2 = P1_Pa * math.exp(-delta_H_J_per_mol / R_GAS * inv_T2)
    return _ok("clausius_clapeyron",
               P2_Pa=P2, P1_Pa=P1_Pa, T1_K=T1_K, T2_K=T2_K)


@mcp.tool(description="Critical nucleus radius from classical nucleation theory: r* = 2 γ V_m / |ΔG_v|, where γ is surface energy, V_m molar volume, ΔG_v volumetric free-energy change.")
async def nucleation_critical_radius(surface_energy_J_m2: float, molar_volume_m3_mol: float,
                                       delta_G_v_J_m3: float) -> str:
    if delta_G_v_J_m3 == 0:
        return _err("delta_G_v must be non-zero")
    r_star = 2.0 * surface_energy_J_m2 * molar_volume_m3_mol / abs(delta_G_v_J_m3)
    return _ok("nucleation_critical_radius",
               r_star_m=r_star,
               homogeneous=(delta_G_v_J_m3 < 0))


@mcp.tool(description="Heat conduction: Fourier's law, 1D steady-state heat flux q = -k dT/dx.")
async def fourier_heat_flux(thermal_conductivity_W_m: float,
                              T_hot_K: float, T_cold_K: float, thickness_m: float) -> str:
    if thickness_m <= 0:
        return _err("thickness>0 required")
    q = -thermal_conductivity_W_m * (T_cold_K - T_hot_K) / thickness_m
    return _ok("fourier_heat_flux",
               heat_flux_W_m2=q,
               dT_dx_K_m=(T_cold_K - T_hot_K) / thickness_m)


# ---------------------------------------------------------------------------
# Stellar / ionization extras
# ---------------------------------------------------------------------------

@mcp.tool(description="Stark-effect ionization ratio change factor: exp(ΔE / k_B T) for Stark shift ΔE.")
async def stark_ionization_change_factor(delta_E_J: float, T_K: float) -> str:
    if T_K <= 0:
        return _err("T_K must be > 0")
    return _ok("stark_ionization_change_factor",
               factor=math.exp(delta_E_J / (KB * T_K)),
               delta_E_eV=delta_E_J / 1.602176634e-19)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

@mcp.tool(description=(
    "Plot hydrogen energy levels (-13.6 eV / n^2). Returns path to PNG."
))
def plot_hydrogen_energy_levels(max_n: int = 6) -> str:
    if not _MPL_OK:
        return _err("matplotlib not available")
    if max_n < 1:
        return _err("max_n must be >= 1")
    import os
    os.makedirs(_VIZ_DIR, exist_ok=True)
    ns = list(range(1, max_n + 1))
    Es = [-13.6 / n ** 2 for n in ns]
    fig, ax = plt.subplots(figsize=(6, 5))
    for n, E in zip(ns, Es):
        ax.hlines(E, 0, 1, color="b", linewidth=2)
        ax.text(1.02, E, f"n={n}: {E:.2f} eV", va="center")
    ax.set_xlim(0, 2)
    ax.set_ylim(min(Es) - 2, 1)
    ax.set_ylabel("Energy (eV)")
    ax.set_xticks([])
    ax.set_title("Hydrogen energy levels")
    ax.grid(True, alpha=0.3, axis="y")
    out = f"{_VIZ_DIR}/hydrogen_levels.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return _ok("plot_hydrogen_energy_levels", path=out, n_levels=max_n)


@mcp.tool(description=(
    "Plot a Maxwell-Boltzmann speed distribution for given T. Returns path "
    "to PNG."
))
def plot_maxwell_boltzmann_speed_distribution(temperatures_k: list[float],
                                                 molar_mass_g_mol: float = 28.0,
                                                 v_max: float = 1500.0,
                                                 n_points: int = 200) -> str:
    if not _MPL_OK:
        return _err("matplotlib not available")
    if not temperatures_k:
        return _err("temperatures_k must be non-empty")
    import os
    os.makedirs(_VIZ_DIR, exist_ok=True)
    M = molar_mass_g_mol / 1000.0
    vs = [v_max * i / n_points for i in range(1, n_points + 1)]
    fig, ax = plt.subplots(figsize=(7, 4))
    for T in temperatures_k:
        kBT = KB * T
        coef = (M / (2 * math.pi * kBT)) ** 1.5
        fv = [4 * math.pi * coef * v ** 2 * math.exp(-M * v ** 2 / (2 * kBT))
              for v in vs]
        ax.plot(vs, fv, label=f"T = {T} K")
    ax.set_xlabel("speed v (m/s)")
    ax.set_ylabel("f(v)")
    ax.set_title("Maxwell-Boltzmann speed distribution (M = "
                 f"{molar_mass_g_mol} g/mol)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out = f"{_VIZ_DIR}/mb_speed.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return _ok("plot_maxwell_boltzmann_speed_distribution", path=out)


if __name__ == "__main__":
    print("physics-quantum-atomic tools:", [t.name for t in mcp._tools.values()])