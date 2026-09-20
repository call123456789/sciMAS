#!/usr/bin/env python3
"""Waves, acoustics, fluid dynamics, and Stark spectroscopy MCP server.

Tools ported from SciAgentGYM-main/toolkits/physics/{acoustics,fluid_dynamics}/
— covering sound pressure level (SPL) algebra, Doppler ultrasound,
wetting / contact angle hysteresis, fluid statics, and wastewater
hydraulic network design.

All implementations are stdlib + numpy only. Every tool returns a JSON
string. Tool naming is snake_case.
"""

from __future__ import annotations

import json
import math

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("physics-waves-fluid")

SPEED_OF_SOUND_AIR = 343.0   # m/s
SPEED_OF_SOUND_WATER = 1480.0  # m/s
G = 9.80665
PI = math.pi


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

_VIZ_DIR = "/tmp/scimas_physics_waves_fluid"


# ---------------------------------------------------------------------------
# Sound pressure level (SPL) algebra
# ---------------------------------------------------------------------------

@mcp.tool(description="Convert sound pressure to SPL in decibels: SPL = 20*log10(p / p_ref). Default p_ref = 20 µPa.")
async def spl_from_pressure(pressure_Pa: float, p_ref_Pa: float = 20e-6) -> str:
    if p_ref_Pa <= 0:
        return _err("p_ref must be > 0")
    if pressure_Pa <= 0:
        return _err("pressure must be > 0")
    return _ok("spl_from_pressure",
               pressure_Pa=pressure_Pa, p_ref_Pa=p_ref_Pa,
               spl_dB=20.0 * math.log10(pressure_Pa / p_ref_Pa))


@mcp.tool(description="Convert SPL (dB) back to pressure in Pa.")
async def pressure_from_spl(spl_dB: float, p_ref_Pa: float = 20e-6) -> str:
    return _ok("pressure_from_spl",
               spl_dB=spl_dB, p_ref_Pa=p_ref_Pa,
               pressure_Pa=p_ref_Pa * 10 ** (spl_dB / 20.0))


@mcp.tool(description="SPL increase when adding two incoherent sources: L_sum = 10*log10(10^(L1/10) + 10^(L2/10)).")
async def spl_add_two(L1_dB: float, L2_dB: float) -> str:
    p1sq = 10 ** (L1_dB / 10.0)
    p2sq = 10 ** (L2_dB / 10.0)
    L = 10.0 * math.log10(p1sq + p2sq)
    return _ok("spl_add_two",
               L1_dB=L1_dB, L2_dB=L2_dB, L_total_dB=L,
               delta_dB=L - max(L1_dB, L2_dB))


@mcp.tool(description="Total SPL from N incoherent sources, all expressed in dB.")
async def spl_total(L_dB_list: list[float]) -> str:
    if not L_dB_list:
        return _err("L_dB_list must be non-empty")
    p2_sum = sum(10 ** (L / 10.0) for L in L_dB_list)
    L_total = 10.0 * math.log10(p2_sum)
    return _ok("spl_total",
               inputs=L_dB_list, L_total_dB=L_total,
               n_sources=len(L_dB_list))


@mcp.tool(description="Point-source SPL at distance r, given SPL at reference distance r0 and inverse-square spreading: L = L0 - 20*log10(r/r0).")
async def spl_at_distance(L0_dB: float, r0_m: float, r_m: float) -> str:
    if r0_m <= 0 or r_m <= 0:
        return _err("distances must be > 0")
    return _ok("spl_at_distance",
               L0_dB=L0_dB, r0_m=r0_m, r_m=r_m,
               L_dB=L0_dB - 20.0 * math.log10(r_m / r0_m))


# ---------------------------------------------------------------------------
# Doppler / ultrasound
# ---------------------------------------------------------------------------

@mcp.tool(description="Doppler-shifted frequency from moving observer / source. Returns f_observed.")
async def doppler_shift(f_source_Hz: float, v_source_m_s: float = 0.0,
                         v_observer_m_s: float = 0.0,
                         speed_of_sound_m_s: float = SPEED_OF_SOUND_AIR,
                         toward: bool = True) -> str:
    if f_source_Hz <= 0 or speed_of_sound_m_s <= 0:
        return _err("f_source>0 and c>0 required")
    sign = 1.0 if toward else -1.0
    # Source moving toward observer → wavelength compressed.
    v_s = sign * v_source_m_s
    v_o = sign * v_observer_m_s
    f_obs = f_source_Hz * (speed_of_sound_m_s + v_o) / (speed_of_sound_m_s - v_s)
    return _ok("doppler_shift",
               f_source_Hz=f_source_Hz,
               f_observed_Hz=f_obs,
               shift_Hz=f_obs - f_source_Hz)


@mcp.tool(description="Blood flow velocity from Doppler ultrasound shift: v = (c * delta_f) / (2 * f_source * cos(theta)).")
async def doppler_blood_velocity(delta_f_Hz: float, f_source_Hz: float,
                                  angle_deg: float = 0.0,
                                  speed_of_sound_m_s: float = SPEED_OF_SOUND_WATER) -> str:
    if f_source_Hz <= 0:
        return _err("f_source must be > 0")
    cos_t = math.cos(math.radians(angle_deg))
    if cos_t == 0:
        return _err("incidence angle of 90° (cos=0) undefined")
    v = speed_of_sound_m_s * delta_f_Hz / (2.0 * f_source_Hz * cos_t)
    return _ok("doppler_blood_velocity",
               delta_f_Hz=delta_f_Hz, f_source_Hz=f_source_Hz,
               angle_deg=angle_deg, velocity_m_s=v)


# ---------------------------------------------------------------------------
# Stark spectroscopy (atomic hydrogen)
# ---------------------------------------------------------------------------

@mcp.tool(description="Hydrogen atom Stark effect first-order energy shift: delta_E = 3/2 * n * k * e * a0 * E_electric, with k the principal quantum number n_k degeneracy. Returns delta_E in eV for given n, m, E.")
async def stark_first_order_shift(n: int, m: int, electric_field_V_m: float) -> str:
    """Linear Stark effect. For n=1,2 hydrogen, the first-order shift is
    delta_E = 3/2 * n * k * e * a0 * E, where k = m is the magnetic quantum
    number (in atomic units). For higher n, additional degeneracy gives
    a richer spectrum."""
    if n <= 0 or abs(m) > n:
        return _err("require n>=1, |m|<=n")
    a0 = 5.29177210903e-11      # Bohr radius (m)
    e = 1.602176634e-19          # elementary charge (C)
    delta_E_J = 1.5 * n * m * e * a0 * electric_field_V_m
    delta_E_eV = delta_E_J / e
    return _ok("stark_first_order_shift",
               n=n, m=m, electric_field_V_m=electric_field_V_m,
               delta_E_eV=delta_E_eV, delta_E_J=delta_E_J)


@mcp.tool(description="Hydrogen transition wavelength from n_initial → n_final (Rydberg formula): 1/lambda = R_H (1/n_i^2 - 1/n_f^2).")
async def hydrogen_transition_wavelength(n_initial: int, n_final: int) -> str:
    if n_initial <= 0 or n_final <= 0:
        return _err("quantum numbers must be > 0")
    R_H = 1.0973731568508e7      # Rydberg constant (1/m)
    inv_lambda = R_H * (1.0 / n_initial ** 2 - 1.0 / n_final ** 2)
    if inv_lambda <= 0:
        return _err("absorption transition requires n_initial < n_final")
    lam = 1.0 / inv_lambda
    series = ""
    if n_final == 1: series = "Lyman (UV)"
    elif n_final == 2: series = "Balmer (visible)"
    elif n_final == 3: series = "Paschen (IR)"
    elif n_final == 4: series = "Brackett (IR)"
    elif n_final == 5: series = "Pfund (IR)"
    return _ok("hydrogen_transition_wavelength",
               n_initial=n_initial, n_final=n_final,
               wavelength_m=lam, series=series)


@mcp.tool(description="Rabi frequency for a two-level atom driven by a field of amplitude E: Omega_R = d * E / hbar, where d is the dipole moment.")
async def rabi_frequency(dipole_moment_C_m: float, electric_field_V_m: float) -> str:
    if dipole_moment_C_m == 0:
        return _err("dipole moment must be non-zero")
    HBAR = 1.054571817e-34
    Omega = dipole_moment_C_m * electric_field_V_m / HBAR
    return _ok("rabi_frequency",
               dipole_moment_C_m=dipole_moment_C_m,
               electric_field_V_m=electric_field_V_m,
               rabi_freq_rad_s=Omega,
               rabi_freq_Hz=Omega / (2.0 * math.pi))


# ---------------------------------------------------------------------------
# Fluid statics
# ---------------------------------------------------------------------------

@mcp.tool(description="Hydrostatic pressure at height h below a free surface of density rho: P = P_atm + rho*g*h.")
async def hydrostatic_pressure(depth_m: float, density_kg_m3: float,
                                g: float = G, p_atm_Pa: float = 101325.0) -> str:
    if depth_m < 0 or density_kg_m3 <= 0:
        return _err("depth>=0, density>0 required")
    P = p_atm_Pa + density_kg_m3 * g * depth_m
    return _ok("hydrostatic_pressure",
               depth_m=depth_m, density_kg_m3=density_kg_m3,
               gauge_Pa=density_kg_m3 * g * depth_m,
               absolute_Pa=P)


@mcp.tool(description="Tilt angle of a free liquid surface in a container accelerating horizontally with a.")
async def fluid_tilt_angle(acceleration_m_s2: float, g: float = G) -> str:
    if g <= 0:
        return _err("g>0 required")
    if acceleration_m_s2 == 0:
        return _ok("fluid_tilt_angle", angle_deg=0.0,
                   acceleration_m_s2=acceleration_m_s2)
    return _ok("fluid_tilt_angle",
               angle_deg=math.degrees(math.atan(acceleration_m_s2 / g)),
               acceleration_m_s2=acceleration_m_s2)


@mcp.tool(description="Buoyant force on a submerged volume V of fluid density rho: F_b = rho * g * V.")
async def buoyant_force(volume_m3: float, fluid_density_kg_m3: float,
                         g: float = G) -> str:
    if volume_m3 <= 0 or fluid_density_kg_m3 <= 0:
        return _err("volume>0, density>0 required")
    return _ok("buoyant_force",
               volume_m3=volume_m3,
               fluid_density_kg_m3=fluid_density_kg_m3,
               force_N=fluid_density_kg_m3 * g * volume_m3)


@mcp.tool(description="Continuity equation for incompressible flow: A1*v1 = A2*v2.")
async def continuity_equation(A1_m2: float, v1_m_s: float, A2_m2: float) -> str:
    if A1_m2 <= 0 or A2_m2 <= 0:
        return _err("areas must be > 0")
    v2 = A1_m2 * v1_m_s / A2_m2
    return _ok("continuity_equation",
               A1_m2=A1_m2, v1_m_s=v1_m_s,
               A2_m2=A2_m2, v2_m_s=v2)


# ---------------------------------------------------------------------------
# Surface wetting
# ---------------------------------------------------------------------------

@mcp.tool(description="Contact-angle hysteresis: delta = advancing - receding.")
async def contact_angle_hysteresis(advancing_deg: float, receding_deg: float) -> str:
    if not (0 <= advancing_deg <= 180) or not (0 <= receding_deg <= 180):
        return _err("contact angles must be in [0, 180]")
    return _ok("contact_angle_hysteresis",
               advancing_deg=advancing_deg,
               receding_deg=receding_deg,
               hysteresis_deg=advancing_deg - receding_deg)


@mcp.tool(description="Wenzel roughness from contact-angle hysteresis: r ≈ 1 + (hysteresis_deg) / 60 (heuristic).")
async def wenzel_roughness_estimate(advancing_deg: float, receding_deg: float) -> str:
    if not (0 <= advancing_deg <= 180) or not (0 <= receding_deg <= 180):
        return _err("contact angles must be in [0, 180]")
    h = advancing_deg - receding_deg
    r = 1.0 + h / 60.0
    return _ok("wenzel_roughness_estimate",
               hysteresis_deg=h, roughness_ratio=r)


@mcp.tool(description="Young-Dupré surface energy from contact angle: gamma_sv = gamma_sl + gamma_lv*cos(theta); solve for missing component given two of three.")
async def young_dupre(gamma_sv_mN_m: float = 0.0, gamma_sl_mN_m: float = 0.0,
                       gamma_lv_mN_m: float = 0.0, contact_angle_deg: float = 0.0) -> str:
    if not (0 <= contact_angle_deg <= 180):
        return _err("contact_angle_deg must be in [0, 180]")
    cos_t = math.cos(math.radians(contact_angle_deg))
    n_supplied = sum(v != 0 for v in (gamma_sv_mN_m, gamma_sl_mN_m, gamma_lv_mN_m))
    if n_supplied != 2:
        return _err("supply exactly two of gamma_sv, gamma_sl, gamma_lv (mN/m)")
    if gamma_sv_mN_m == 0:
        gamma_sv_mN_m = gamma_sl_mN_m + gamma_lv_mN_m * cos_t
        return _ok("young_dupre",
                   gamma_sv_mN_m=gamma_sv_mN_m,
                   gamma_sl_mN_m=gamma_sl_mN_m,
                   gamma_lv_mN_m=gamma_lv_mN_m,
                   contact_angle_deg=contact_angle_deg,
                   solved_for="gamma_sv")
    if gamma_sl_mN_m == 0:
        gamma_sl_mN_m = gamma_sv_mN_m - gamma_lv_mN_m * cos_t
        return _ok("young_dupre",
                   gamma_sv_mN_m=gamma_sv_mN_m,
                   gamma_sl_mN_m=gamma_sl_mN_m,
                   gamma_lv_mN_m=gamma_lv_mN_m,
                   contact_angle_deg=contact_angle_deg,
                   solved_for="gamma_sl")
    gamma_lv_mN_m = (gamma_sv_mN_m - gamma_sl_mN_m) / cos_t if cos_t != 0 else None
    return _ok("young_dupre",
               gamma_sv_mN_m=gamma_sv_mN_m,
               gamma_sl_mN_m=gamma_sl_mN_m,
               gamma_lv_mN_m=gamma_lv_mN_m,
               contact_angle_deg=contact_angle_deg,
               solved_for="gamma_lv")


# ---------------------------------------------------------------------------
# Wastewater / hydraulics
# ---------------------------------------------------------------------------

@mcp.tool(description="Manning equation for open-channel flow velocity: v = (1/n) * R^(2/3) * S^(1/2).")
async def manning_velocity(manning_n: float, hydraulic_radius_m: float,
                            slope: float) -> str:
    if manning_n <= 0 or hydraulic_radius_m <= 0 or slope < 0:
        return _err("manning_n>0, R>0, S>=0 required")
    v = (1.0 / manning_n) * hydraulic_radius_m ** (2.0 / 3.0) * math.sqrt(slope)
    return _ok("manning_velocity",
               v_m_s=v,
               manning_n=manning_n,
               hydraulic_radius_m=hydraulic_radius_m,
               slope=slope)


@mcp.tool(description="Design pipe diameter from peak flow, slope, and Manning's n (circular pipe, full flow): D = (Q * n / (0.312 * S^0.5))^(3/8). Approximate for circular full pipe.")
async def design_pipe_diameter(flow_m3_s: float, manning_n: float, slope: float) -> str:
    if flow_m3_s <= 0 or manning_n <= 0 or slope <= 0:
        return _err("flow>0, n>0, slope>0 required")
    D = (flow_m3_s * manning_n / (0.312 * slope ** 0.5)) ** (3.0 / 8.0)
    return _ok("design_pipe_diameter",
               diameter_m=D, flow_m3_s=flow_m3_s,
               manning_n=manning_n, slope=slope)


@mcp.tool(description="Velocity from continuity: v = Q / A for a full pipe of diameter D.")
async def pipe_velocity(flow_m3_s: float, diameter_m: float) -> str:
    if flow_m3_s <= 0 or diameter_m <= 0:
        return _err("flow>0, D>0 required")
    A = PI * (diameter_m / 2.0) ** 2
    return _ok("pipe_velocity",
               area_m2=A, velocity_m_s=flow_m3_s / A)


@mcp.tool(description="Total variation coefficient for a peak-flow factor design: K_total = sum of subarea contributions weighted by pipe length.")
async def total_variation_coefficient(areas: list[dict]) -> str:
    """`areas` = [{area_ha: float, k_sub: float, pipe_length_m: float}]"""
    if not areas:
        return _err("areas must be non-empty")
    K = 0.0
    for entry in areas:
        a = float(entry.get("area_ha", 0.0))
        k = float(entry.get("k_sub", 0.0))
        L = float(entry.get("pipe_length_m", 0.0))
        if a < 0 or k < 0 or L < 0:
            return _err("non-negative area, k, length required")
        K += k * L
    return _ok("total_variation_coefficient",
               total_variation_coefficient=K,
               n_areas=len(areas))


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

@mcp.tool(description=(
    "Plot SPL (dB) vs distance on log scale. Returns path to saved PNG."
))
def plot_spl_vs_distance(distances_m: list[float], source_spl_db: float,
                          n_sources: int = 1) -> str:
    if not _MPL_OK:
        return _err("matplotlib not available")
    import os
    os.makedirs(_VIZ_DIR, exist_ok=True)
    distances = [d for d in distances_m if d > 0]
    if not distances:
        return _err("distances must be > 0")
    spls = [source_spl_db - 20 * math.log10(d) - 3 for d in distances]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(distances, spls, "b-")
    ax.set_xscale("log")
    ax.set_xlabel("distance r (m, log scale)")
    ax.set_ylabel("SPL (dB)")
    ax.set_title(f"SPL vs distance (n_sources = {n_sources})")
    ax.grid(True, alpha=0.3, which="both")
    out = f"{_VIZ_DIR}/spl.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return _ok("plot_spl_vs_distance", path=out)


@mcp.tool(description=(
    "Plot a Doppler velocity profile: received frequency vs source velocity. "
    "Returns path to saved PNG."
))
def plot_doppler_profile(source_speeds: list[float], f0: float,
                           v_sound: float = 343.0) -> str:
    if not _MPL_OK:
        return _err("matplotlib not available")
    import os
    os.makedirs(_VIZ_DIR, exist_ok=True)
    f_received = [f0 * v_sound / (v_sound - v) if (v_sound - v) != 0 else 0
                  for v in source_speeds]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(source_speeds, f_received, "b-")
    ax.axvline(v_sound, color="r", linestyle="--", label="speed of sound")
    ax.set_xlabel("source speed v (m/s)")
    ax.set_ylabel("received f (Hz)")
    ax.set_title(f"Doppler-shifted frequency (f₀ = {f0} Hz)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out = f"{_VIZ_DIR}/doppler.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return _ok("plot_doppler_profile", path=out)


if __name__ == "__main__":
    import asyncio
    asyncio.run(mcp.run_stdio_async())
