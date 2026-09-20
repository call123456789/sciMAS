#!/usr/bin/env python3
"""Condensed matter and solid-state physics MCP server.

Tools ported from SciAgentGYM-main/toolkits/physics/condensed_matter_physics/
— covering Ising / Heisenberg spin models, Anderson localization,
topological SSH insulators, Hubbard strongly-correlated systems,
semiconductor drift-diffusion (Scharfetter-Gummel), and basic tight-binding
band structures.

All implementations are stdlib + numpy. Every tool returns a JSON
string. Tool naming is snake_case.
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

mcp = MCPServer("physics-condensed-matter")

KB = 1.380649e-23  # Boltzmann constant (J/K)
EV_TO_J = 1.602176634e-19


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

_VIZ_DIR = "/tmp/scimas_physics_condensed_matter"


# ---------------------------------------------------------------------------
# 1D Ising model
# ---------------------------------------------------------------------------

@mcp.tool(description="1D Ising model partition function (open chain): Z = 2 * cosh(N*beta*J) + ... shortcut: Z_N = lambda_+^N + lambda_-^N where lambda_± = exp(βJ) ± exp(-βJ). For simplicity, the closed-form for N spins with nearest-neighbour coupling J is returned via the transfer matrix.")
async def ising_1d_partition(N: int, J: float, T_K: float) -> str:
    if N <= 0 or T_K <= 0:
        return _err("N>0 and T_K>0 required")
    beta = 1.0 / (KB * T_K)
    lam_plus = math.exp(beta * J) + math.exp(-beta * J)
    lam_minus = math.exp(beta * J) - math.exp(-beta * J)
    Z = lam_plus ** N + lam_minus ** N
    # Per-spin free energy.
    F_per_spin = -KB * T_K * math.log(Z) / N
    return _ok("ising_1d_partition",
               N=N, J=J, T_K=T_K,
               partition_function=Z,
               free_energy_per_spin_J=F_per_spin)


@mcp.tool(description="Mean-field magnetization for a 2D / 3D Ising ferromagnet above T_c: m ≈ sqrt(1 - T/T_c). Below T_c, m is the solution of m = tanh(beta * (z*J*m + h)).")
async def ising_mean_field_magnetization(T_K: float, T_c_K: float, J: float, z: int = 4,
                                          external_field_J: float = 0.0) -> str:
    if z <= 0 or T_c_K <= 0 or T_K <= 0:
        return _err("z>0, T_c>0, T>0 required")
    if T_K >= T_c_K:
        # Near T_c with h = 0: m ≈ sqrt(1 - T/T_c) (classical).
        m = math.sqrt(max(0.0, 1.0 - T_K / T_c_K))
        return _ok("ising_mean_field_magnetization",
                   T_K=T_K, T_c_K=T_c_K,
                   magnetization_per_spin=m,
                   phase="paramagnetic")
    # Below T_c, solve self-consistently.
    beta = 1.0 / (KB * T_K)
    h_eff = z * J + external_field_J
    m = 1.0  # initial guess
    for _ in range(200):
        arg = beta * h_eff * m
        arg = max(-500.0, min(500.0, arg))
        new_m = math.tanh(arg)
        if abs(new_m - m) < 1e-9:
            break
        m = new_m
    return _ok("ising_mean_field_magnetization",
               T_K=T_K, T_c_K=T_c_K,
               magnetization_per_spin=m,
               phase="ferromagnetic",
               iterations=200)


# ---------------------------------------------------------------------------
# Heisenberg model
# ---------------------------------------------------------------------------

@mcp.tool(description="Heisenberg chain magnetization m(T) for spin-1/2 1D chain (Bethe ansatz exact at h=0): m = 0 for T>0 strictly, but J/T scaling shown for reference.")
async def heisenberg_magnetization_vs_T(J: float, T_K: float, B_field_T: float = 0.0) -> str:
    """Approximate: at T=0, m=1/2 (saturated). Thermal reduction via
    mean-field-like expression: m = 1/2 * B(J/T) where B is the Brillouin
    function."""
    if T_K <= 0:
        return _err("T_K must be > 0")
    x = J / (KB * T_K) if T_K > 0 else math.inf
    # Brillouin function B_J(x) for spin-1/2: B = tanh(x)/...
    # Use simple Brillouin 1/2: B = tanh(x).
    b = math.tanh(x)
    m = 0.5 * b
    return _ok("heisenberg_magnetization_vs_T",
               J=J, T_K=T_K, B_field_T=B_field_T,
               magnetization_per_spin=m,
               brillouin_b=b)


@mcp.tool(description="Antiferromagnetic Heisenberg: staggered magnetization proxy = tanh(|J|/kT) (rough).")
async def heisenberg_staggered_magnetization(J: float, T_K: float) -> str:
    if T_K <= 0:
        return _err("T_K must be > 0")
    x = abs(J) / (KB * T_K)
    m_staggered = math.tanh(x)
    return _ok("heisenberg_staggered_magnetization",
               J=J, T_K=T_K, staggered_magnetization=m_staggered)


# ---------------------------------------------------------------------------
# Anderson localization
# ---------------------------------------------------------------------------

@mcp.tool(description="Anderson 1D tight-binding Hamiltonian diagonalization: H = -t * (|i><i+1| + h.c.) + W * random_i * |i><i|. Returns eigenvalues, density of states, and inverse participation ratio.")
async def anderson_1d(N: int, hopping_t: float, disorder_W: float,
                       seed: int = 42) -> str:
    if N < 4:
        return _err("N must be >= 4")
    if hopping_t <= 0 or disorder_W < 0:
        return _err("hopping>0, W>=0 required")
    rng = np.random.default_rng(seed)
    diag = -hopping_t * np.ones(N - 1)
    H = np.diag(diag, 1) + np.diag(diag, -1)
    H += disorder_W * np.diag(rng.uniform(-0.5, 0.5, size=N))
    eigvals, eigvecs = np.linalg.eigh(H)
    # Inverse participation ratio.
    ipr = np.sum(np.abs(eigvecs) ** 4, axis=0)
    return _ok("anderson_1d",
               N=N, hopping_t=hopping_t, disorder_W=disorder_W,
               eigenvalues_min=float(eigvals.min()),
               eigenvalues_max=float(eigvals.max()),
               mean_ipr=float(ipr.mean()),
               localized=(float(ipr.mean()) > 0.1))


@mcp.tool(description="Anderson 1D localization length estimate ξ ≈ 105 * t / W for W in the metallic limit (rough scaling).")
async def anderson_localization_length(hopping_t: float, disorder_W: float) -> str:
    if hopping_t <= 0 or disorder_W < 0:
        return _err("hopping>0, W>=0 required")
    if disorder_W == 0:
        return _ok("anderson_localization_length",
                   xi_sites=float("inf"), hopping_t=hopping_t, disorder_W=disorder_W)
    xi = 105.0 * hopping_t / disorder_W
    return _ok("anderson_localization_length",
               xi_sites=xi, hopping_t=hopping_t, disorder_W=disorder_W)


# ---------------------------------------------------------------------------
# SSH topological insulator
# ---------------------------------------------------------------------------

@mcp.tool(description="SSH model Hamiltonian gap (closed in topological phase): gap = |v - w| for uniform hopping.")
async def ssh_gap(v: float, w: float) -> str:
    return _ok("ssh_gap",
               v=v, w=w,
               gap=abs(v - w),
               topological=(w > v))


@mcp.tool(description="SSH winding number in the 1D Brillouin zone: ν = (1/2π) ∮ d k (∂ arg(h_x + i h_y) / ∂ k). For uniform hopping, returns 1 if w > v else 0.")
async def ssh_winding_number(v: float, w: float) -> str:
    nu = 1 if w > v else 0
    return _ok("ssh_winding_number",
               v=v, w=w, winding_number=nu,
               topological_phase=(nu == 1))


@mcp.tool(description="SSH edge-state energy for finite chain with uniform hopping.")
async def ssh_edge_state_energy(v: float, w: float, N: int) -> str:
    if N < 2:
        return _err("N>=2 required")
    diag_inner = np.full(N - 1, w)
    H = np.diag(diag_inner, 1) + np.diag(diag_inner, -1)
    H[0, 0] = 0  # edge perturbation
    H[-1, -1] = 0
    # Re-apply alternating pattern: v,w,v,w,...
    for i in range(N - 1):
        hop = v if i % 2 == 0 else w
        H[i, i + 1] = hop
        H[i + 1, i] = hop
    eigvals = np.linalg.eigvalsh(H)
    bulk_gap = abs(v - w)
    # Edge states are within the bulk gap.
    if w > v:
        edge_e = float(eigvals[N // 2])
    else:
        edge_e = float("nan")
    return _ok("ssh_edge_state_energy",
               N=N, v=v, w=w,
               bulk_gap=bulk_gap,
               edge_state_energy_eV=edge_e,
               eigenvalues=[float(e) for e in eigvals])


# ---------------------------------------------------------------------------
# Hubbard strongly-correlated
# ---------------------------------------------------------------------------

@mcp.tool(description="Hubbard model 2-site double-occupancy proxy: ⟨n↑ n↓⟩ = U / (U + 4 t) for half-filling, U/t large.")
async def hubbard_double_occupancy(U: float, t: float) -> str:
    if t <= 0:
        return _err("t>0 required")
    n_up_n_down = U / (U + 4.0 * t)
    return _ok("hubbard_double_occupancy",
               U=U, t=t,
               double_occupancy=n_up_n_down,
               mott_insulator=(U > 4.0 * t))


@mcp.tool(description="Hubbard charge gap proxy: Δ = U - 4 t * effective kinetic term, simplified as U_eff = U * (1 - 4 t/U) for U > 4t.")
async def hubbard_charge_gap(U: float, t: float) -> str:
    if t <= 0:
        return _err("t>0 required")
    if U > 4.0 * t:
        gap = U - 4.0 * t
    else:
        gap = 0.0
    return _ok("hubbard_charge_gap",
               U=U, t=t, charge_gap_proxy=gap,
               insulator=(gap > 0))


# ---------------------------------------------------------------------------
# Semiconductor Scharfetter-Gummel
# ---------------------------------------------------------------------------

@mcp.tool(description="Bernoulli function B(x) = x / (exp(x) - 1) for |x| large, with safe-asymptote handling.")
async def bernoulli_function(x: float) -> str:
    if abs(x) < 1e-6:
        b = 1.0 - x / 2.0 + x ** 2 / 12.0
    else:
        try:
            b = x / (math.exp(x) - 1.0)
        except OverflowError:
            b = 0.0 if x > 0 else -x
    return _ok("bernoulli_function", x=x, B_x=b)


@mcp.tool(description="Thermal voltage V_T = k_B T / q at temperature T.")
async def thermal_voltage(T_K: float) -> str:
    if T_K <= 0:
        return _err("T_K must be > 0")
    q = 1.602176634e-19
    V_T = KB * T_K / q
    return _ok("thermal_voltage",
               T_K=T_K, V_T_volts=V_T, V_T_mV=V_T * 1000.0)


@mcp.tool(description="Scharfetter-Gummel discretized current between two nodes (n_i, n_j) for electrons, mobility mu_n, electric field V_j - V_i, grid spacing h.")
async def scharfetter_gummel_electron_current(n_i: float, n_j: float,
                                                V_i_V: float, V_j_V: float,
                                                h_m: float, mu_n_m2_V_s: float,
                                                T_K: float = 300.0) -> str:
    q = 1.602176634e-19
    if n_i <= 0 or n_j <= 0 or h_m <= 0 or mu_n_m2_V_s < 0 or T_K <= 0:
        return _err("n_i,n_j>0, h>0, mu>=0, T_K>0")
    V_T = KB * T_K / q
    dV = V_j_V - V_i_V
    # Bernoulli function of -dV/V_T.
    arg = -dV / V_T
    if abs(arg) < 1e-6:
        B_arg = 1.0 - arg / 2.0 + arg ** 2 / 12.0
        B_marg = 1.0 + arg / 2.0 + arg ** 2 / 12.0
    else:
        if arg > 500:
            B_arg = 0.0
            B_marg = -arg
        else:
            B_arg = arg / (math.exp(arg) - 1.0)
            B_marg = arg + B_arg
    J_n = (q * mu_n_m2_V_s / h_m) * V_T * (n_i * B_arg - n_j * B_marg)
    return _ok("scharfetter_gummel_electron_current",
               current_density_A_m2=J_n, V_T_V=V_T,
               bernoulli_arg=arg, n_i=n_i, n_j=n_j)


# ---------------------------------------------------------------------------
# Tight-binding band structure
# ---------------------------------------------------------------------------

@mcp.tool(description="1D tight-binding dispersion E(k) = ε0 - 2 t cos(k a) for k in [-π/a, π/a]. Returns energy at given k.")
async def tight_binding_1d(epsilon_0: float, hopping_t: float,
                            lattice_a: float, k: float) -> str:
    if lattice_a <= 0:
        return _err("lattice_a>0 required")
    E = epsilon_0 - 2.0 * hopping_t * math.cos(k * lattice_a)
    return _ok("tight_binding_1d",
               k=k, energy=E,
               bandwidth=4.0 * abs(hopping_t))


@mcp.tool(description="2D square-lattice tight-binding dispersion: E(kx, ky) = ε0 - 2 t (cos(kx a) + cos(ky a)).")
async def tight_binding_2d_square(epsilon_0: float, hopping_t: float,
                                   lattice_a: float, kx: float, ky: float) -> str:
    if lattice_a <= 0:
        return _err("lattice_a>0 required")
    E = epsilon_0 - 2.0 * hopping_t * (math.cos(kx * lattice_a) + math.cos(ky * lattice_a))
    return _ok("tight_binding_2d_square",
               kx=kx, ky=ky, energy=E,
               bandwidth=8.0 * abs(hopping_t))


@mcp.tool(description="Generate a diamond lattice bond-angle configuration: returns angles between nearest-neighbour bonds (ideal 109.47°).")
async def diamond_bond_angles(n_unit_cells: int = 1) -> str:
    """Ideal diamond structure bond angle is arccos(-1/3) ≈ 109.47°."""
    if n_unit_cells <= 0:
        return _err("n_unit_cells>0 required")
    ideal = math.degrees(math.acos(-1.0 / 3.0))
    return _ok("diamond_bond_angles",
               ideal_angle_deg=ideal,
               n_unit_cells=n_unit_cells,
               per_cell_bonds=4,
               total_bonds=8 * n_unit_cells)


# ---------------------------------------------------------------------------
# Magnetization & magnetic materials
# ---------------------------------------------------------------------------

@mcp.tool(description="Curie-Weiss magnetic susceptibility proxy: chi = C / (T - T_c).")
async def curie_weiss_susceptibility(Curie_const: float, T_K: float, T_c_K: float) -> str:
    if T_K <= 0:
        return _err("T_K must be > 0")
    denom = T_K - T_c_K
    if abs(denom) < 1e-9:
        return _err("denominator T - T_c is essentially zero")
    chi = Curie_const / denom
    return _ok("curie_weiss_susceptibility",
               chi=chi, T_K=T_K, T_c_K=T_c_K, Curie_const=Curie_const)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

@mcp.tool(description=(
    "Plot a 1-D tight-binding band structure E(k) vs k. Returns path to "
    "saved PNG."
))
def plot_tight_binding_1d(k_points: list[float], energies: list[list[float]],
                            labels: list[str] = None) -> str:
    if not _MPL_OK:
        return _err("matplotlib not available")
    if not energies or len(k_points) != len(energies):
        return _err("k_points and energies length mismatch")
    import os
    os.makedirs(_VIZ_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for band_idx, band in enumerate(energies):
        if len(band) != len(k_points):
            return _err(f"band {band_idx} length mismatch")
        lbl = labels[band_idx] if labels and band_idx < len(labels) else f"band {band_idx}"
        ax.plot(k_points, band, label=lbl)
    ax.set_xlabel("k (a.u.)")
    ax.set_ylabel("E (eV)")
    ax.set_title("1-D tight-binding band structure")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out = f"{_VIZ_DIR}/band1d.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return _ok("plot_tight_binding_1d", path=out)


@mcp.tool(description=(
    "Plot magnetisation vs temperature (Curie-Weiss χ(T)). Returns path to "
    "saved PNG."
))
def plot_magnetisation_vs_temperature(temperatures: list[float],
                                        magnetisation: list[float],
                                        T_c: float = 1.0) -> str:
    if not _MPL_OK:
        return _err("matplotlib not available")
    if len(temperatures) != len(magnetisation):
        return _err("length mismatch")
    import os
    os.makedirs(_VIZ_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(temperatures, magnetisation, "b-")
    ax.axvline(T_c, color="r", linestyle="--", label=f"T_c = {T_c}")
    ax.set_xlabel("T (K)")
    ax.set_ylabel("M (a.u.)")
    ax.set_title("Magnetisation vs temperature")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out = f"{_VIZ_DIR}/magnetisation.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return _ok("plot_magnetisation_vs_temperature", path=out)


if __name__ == "__main__":
    import asyncio
    asyncio.run(mcp.run_stdio_async())
