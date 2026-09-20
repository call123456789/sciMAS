#!/usr/bin/env python3
"""Electromagnetism and optics MCP server.

Tools ported from SciAgentGYM-main/toolkits/physics/{electromagnetism,optics}/
— covering DC circuits (series / parallel / Wheatstone bridge), magnetic
fields and materials, Maxwell equations, photon energy / colour, optical
wave propagation, and thin-film interference.

All implementations are stdlib + numpy only. Every tool returns a JSON
string. Tool naming is snake_case.
"""

from __future__ import annotations

import json
import math

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("physics-electromagnetism")

C = 2.99792458e8          # speed of light (m/s)
H_PLANCK = 6.62607015e-34  # Planck constant (J s)
EV_TO_J = 1.602176634e-19 # electron-volt to joule
NM_TO_M = 1e-9            # nanometre to metre
MU0 = 4.0 * math.pi * 1e-7 # vacuum permeability (N / A^2)


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

_VIZ_DIR = "/tmp/scimas_physics_electromagnetism"


# ---------------------------------------------------------------------------
# DC circuits
# ---------------------------------------------------------------------------

@mcp.tool(description="Series combination of N resistors.")
async def series_resistance(resistors_ohm: list[float]) -> str:
    if not resistors_ohm:
        return _err("at least one resistor required")
    if any(r < 0 for r in resistors_ohm):
        return _err("negative resistance not allowed")
    return _ok("series_resistance",
               resistors_ohm=list(resistors_ohm),
               total_ohm=sum(resistors_ohm))


@mcp.tool(description="Parallel combination of N resistors.")
async def parallel_resistance(resistors_ohm: list[float]) -> str:
    if not resistors_ohm:
        return _err("at least one resistor required")
    if any(r <= 0 for r in resistors_ohm):
        return _err("all resistors must be > 0 for parallel")
    inv = sum(1.0 / r for r in resistors_ohm)
    return _ok("parallel_resistance",
               resistors_ohm=list(resistors_ohm),
               total_ohm=1.0 / inv)


@mcp.tool(description="Wheatstone bridge: balance condition R1/R2 = R3/R4; report balance residual given bridge resistance R_galv and supply V.")
async def wheatstone_bridge(r1_ohm: float, r2_ohm: float, r3_ohm: float, r4_ohm: float,
                            v_supply_V: float = 0.0, r_galv_ohm: float = 0.0) -> str:
    if any(r <= 0 for r in (r1_ohm, r2_ohm, r3_ohm, r4_ohm)):
        return _err("all four resistors must be > 0")
    # Balance condition: R1/R2 = R3/R4 → R1*R4 = R2*R3.
    lhs = r1_ohm * r4_ohm
    rhs = r2_ohm * r3_ohm
    balance = abs(lhs - rhs) < 1e-12
    out: dict = {"status": "ok", "tool": "wheatstone_bridge",
                 "balanced": balance, "r1_r4": lhs, "r2_r3": rhs}
    if balance:
        out["v_galv_V"] = 0.0
    elif v_supply_V > 0 and r_galv_ohm > 0:
        # V_galv = V * (R1/(R1+R2) - R3/(R3+R4)) * R_galv / (R_galv + R_bridge)
        # Approx: ignore R_galv loading for simplicity.
        out["v_galv_V_approx"] = v_supply_V * (
            r1_ohm / (r1_ohm + r2_ohm) - r3_ohm / (r3_ohm + r4_ohm)
        )
    return json.dumps(out, ensure_ascii=False)


@mcp.tool(description="Ohm's law: given two of V, I, R compute the third.")
async def ohms_law(voltage_V: float = 0.0, current_A: float = 0.0,
                   resistance_ohm: float = 0.0) -> str:
    supplied = [v != 0 for v in (voltage_V, current_A, resistance_ohm)]
    if sum(supplied) != 2:
        return _err("supply exactly two of voltage_V, current_A, resistance_ohm")
    if resistance_ohm and current_A:
        V = resistance_ohm * current_A
        return _ok("ohms_law", voltage_V=V,
                   current_A=current_A, resistance_ohm=resistance_ohm,
                   solved_for="voltage_V")
    if resistance_ohm and voltage_V:
        I = voltage_V / resistance_ohm
        return _ok("ohms_law", voltage_V=voltage_V,
                   current_A=I, resistance_ohm=resistance_ohm,
                   solved_for="current_A")
    # V and I given → R.
    if current_A == 0:
        return _err("division by zero current")
    R = voltage_V / current_A
    return _ok("ohms_law", voltage_V=voltage_V,
               current_A=current_A, resistance_ohm=R,
               solved_for="resistance_ohm")


# ---------------------------------------------------------------------------
# Magnetic fields
# ---------------------------------------------------------------------------

@mcp.tool(description="Magnetic field at distance r from a long straight wire carrying current I: B = mu_0 * I / (2*pi*r).")
async def infinite_wire_field(current_A: float, distance_m: float) -> str:
    if current_A == 0:
        return _err("current must be non-zero")
    if distance_m <= 0:
        return _err("distance must be > 0")
    B = MU0 * current_A / (2.0 * math.pi * distance_m)
    return _ok("infinite_wire_field",
               current_A=current_A, distance_m=distance_m,
               field_T=B)


@mcp.tool(description="Solenoid axial field at its centre: B = mu_0 * n * I, with n = N / L.")
async def solenoid_field(turns: int, length_m: float, current_A: float) -> str:
    if turns <= 0 or length_m <= 0:
        return _err("turns>0 and length>0 required")
    n = turns / length_m
    B = MU0 * n * current_A
    return _ok("solenoid_field",
               turns_per_meter=n, current_A=current_A,
               field_T=B)


@mcp.tool(description="Force between two parallel wires of length L carrying currents I1 and I2 separated by distance d.")
async def parallel_wire_force(current1_A: float, current2_A: float,
                              separation_m: float, length_m: float) -> str:
    if separation_m <= 0 or length_m <= 0:
        return _err("separation>0 and length>0 required")
    if current1_A == 0 or current2_A == 0:
        return _err("currents must be non-zero")
    F_per_len = MU0 * abs(current1_A * current2_A) / (2.0 * math.pi * separation_m)
    F = F_per_len * length_m
    attractive = (current1_A * current2_A) > 0
    return _ok("parallel_wire_force",
               force_N=F, force_per_length_N_m=F_per_len,
               attractive=attractive)


@mcp.tool(description="Magnetic moment of a current loop: mu = I * A.")
async def magnetic_moment_loop(current_A: float, area_m2: float, turns: int = 1) -> str:
    if turns <= 0:
        return _err("turns>0 required")
    mu = current_A * area_m2 * turns
    return _ok("magnetic_moment_loop",
               current_A=current_A, area_m2=area_m2, turns=turns,
               moment_A_m2=mu)


@mcp.tool(description="Relative permeability mu_r from B-H curve at a given H field: mu_r = B / (mu_0 * H).")
async def relative_permeability(b_field_T: float, h_field_A_m: float) -> str:
    if h_field_A_m <= 0:
        return _err("H must be > 0")
    mu_r = b_field_T / (MU0 * h_field_A_m)
    return _ok("relative_permeability",
               b_field_T=b_field_T, h_field_A_m=h_field_A_m,
               mu_r=mu_r, classification=(
                   "ferromagnetic" if mu_r > 100 else
                   "ferrimagnetic" if mu_r > 10 else
                   "paramagnetic" if mu_r > 1.0 else
                   "diamagnetic" if mu_r > 0 else
                   "unknown"
               ))


# ---------------------------------------------------------------------------
# Electrostatics
# ---------------------------------------------------------------------------

@mcp.tool(description="Capacitor energy: U = 0.5 * C * V^2.")
async def capacitor_energy(capacitance_F: float, voltage_V: float) -> str:
    if capacitance_F < 0:
        return _err("capacitance must be non-negative")
    return _ok("capacitor_energy",
               capacitance_F=capacitance_F, voltage_V=voltage_V,
               energy_J=0.5 * capacitance_F * voltage_V ** 2)


@mcp.tool(description="Capacitor charging energy from a source: same as 0.5*C*V^2; also returns charge stored Q = C*V.")
async def capacitor_charge(capacitance_F: float, voltage_V: float) -> str:
    if capacitance_F < 0:
        return _err("capacitance must be non-negative")
    Q = capacitance_F * voltage_V
    return _ok("capacitor_charge",
               capacitance_F=capacitance_F, voltage_V=voltage_V,
               charge_C=Q, energy_J=0.5 * Q * voltage_V)


@mcp.tool(description="Force on a charge in an electric field: F = q * E.")
async def electric_force_on_charge(charge_C: float, electric_field_V_m: float) -> str:
    return _ok("electric_force_on_charge",
               charge_C=charge_C, electric_field_V_m=electric_field_V_m,
               force_N=charge_C * electric_field_V_m)


# ---------------------------------------------------------------------------
# Optics / photon energy
# ---------------------------------------------------------------------------

@mcp.tool(description="Photon energy in joules and eV from wavelength in metres.")
async def photon_energy_from_wavelength(wavelength_m: float) -> str:
    if wavelength_m <= 0:
        return _err("wavelength must be > 0")
    E_J = H_PLANCK * C / wavelength_m
    E_eV = E_J / EV_TO_J
    return _ok("photon_energy_from_wavelength",
               wavelength_m=wavelength_m,
               energy_J=E_J, energy_eV=E_eV)


@mcp.tool(description="Photon energy in joules and eV from frequency in Hz.")
async def photon_energy_from_frequency(frequency_Hz: float) -> str:
    if frequency_Hz <= 0:
        return _err("frequency must be > 0")
    E_J = H_PLANCK * frequency_Hz
    E_eV = E_J / EV_TO_J
    return _ok("photon_energy_from_frequency",
               frequency_Hz=frequency_Hz, energy_J=E_J, energy_eV=E_eV)


@mcp.tool(description="Approximate visible-light colour from a wavelength in metres (380–780 nm).")
async def wavelength_to_color(wavelength_m: float) -> str:
    nm = wavelength_m / NM_TO_M
    if nm < 380 or nm > 780:
        return _err("wavelength outside the visible range (380–780 nm)",
                    wavelength_nm=nm)
    if nm < 440: color = "violet"
    elif nm < 490: color = "blue"
    elif nm < 510: color = "cyan"
    elif nm < 580: color = "green"
    elif nm < 645: color = "yellow"
    elif nm < 700: color = "orange"
    else: color = "red"
    return _ok("wavelength_to_color",
               wavelength_nm=nm, color=color)


@mcp.tool(description="Fresnel reflectance at normal incidence between two dielectric media: R = ((n1-n2)/(n1+n2))^2.")
async def fresnel_reflectance(n1: float, n2: float) -> str:
    if n1 <= 0 or n2 <= 0:
        return _err("refractive indices must be > 0")
    r = (n1 - n2) / (n1 + n2)
    return _ok("fresnel_reflectance",
               n1=n1, n2=n2, amplitude=r,
               reflectance=r * r, transmittance=1.0 - r * r)


@mcp.tool(description="Snell's law: sin(theta2) = (n1/n2) * sin(theta1). Returns theta2 and total-internal-reflection flag if n1 > n2.")
async def snells_law(n1: float, n2: float, theta1_deg: float) -> str:
    if n1 <= 0 or n2 <= 0:
        return _err("refractive indices must be > 0")
    theta1 = math.radians(theta1_deg)
    s = (n1 / n2) * math.sin(theta1)
    if s >= 1:
        return _ok("snells_law",
                   n1=n1, n2=n2, theta1_deg=theta1_deg,
                   total_internal_reflection=True, sin_theta2=s)
    theta2 = math.asin(s)
    return _ok("snells_law",
               n1=n1, n2=n2, theta1_deg=theta1_deg,
               theta2_deg=math.degrees(theta2),
               total_internal_reflection=False)


@mcp.tool(description="Thin-film interference: optical path difference 2*n*d*cos(theta_r).")
async def thin_film_optical_path_difference(n_film: float, thickness_m: float,
                                              incidence_angle_deg: float = 0.0,
                                              n_substrate: float = 1.0) -> str:
    if n_film <= 0 or thickness_m < 0 or n_substrate <= 0:
        return _err("n>0, d>=0, n_substrate>0 required")
    theta_i = math.radians(incidence_angle_deg)
    # Snell inside the film.
    sin_t = math.sin(theta_i) / n_film
    if sin_t >= 1:
        return _err("evanescent inside film — incidence beyond critical angle")
    cos_t = math.sqrt(1.0 - sin_t * sin_t)
    opd = 2.0 * n_film * thickness_m * cos_t
    return _ok("thin_film_optical_path_difference",
               opd_m=opd,
               cos_theta_film=cos_t)


@mcp.tool(description="Thin-film constructive / destructive interference wavelength for normal incidence: constructive at 2*n*d = m*lambda; destructive at (m+1/2)*lambda. Returns first 3 wavelengths for each.")
async def thin_film_interference_wavelengths(n_film: float, thickness_m: float,
                                              order_max: int = 5) -> str:
    if n_film <= 0 or thickness_m <= 0 or order_max <= 0:
        return _err("n>0, d>0, order_max>0 required")
    constructive = []
    destructive = []
    for m in range(1, order_max + 1):
        lam_c = 2.0 * n_film * thickness_m / m
        if 380e-9 <= lam_c <= 780e-9:
            constructive.append({"order": m, "wavelength_m": lam_c})
        lam_d = 2.0 * n_film * thickness_m / (m + 0.5)
        if 380e-9 <= lam_d <= 780e-9:
            destructive.append({"order": m + 0.5, "wavelength_m": lam_d})
    return _ok("thin_film_interference_wavelengths",
               n_film=n_film, thickness_m=thickness_m,
               constructive=constructive, destructive=destructive)


@mcp.tool(description="Reflectance spectrum of a thin film on a substrate (normal incidence): R(lambda) = (r01^2 + r12^2 + 2 r01 r12 cos(2 phi)) / (1 + (r01 r12)^2 + 2 r01 r12 cos(2 phi)), with phi = 2 pi n d / lambda.")
async def thin_film_reflectance_spectrum(n_film: float, thickness_m: float,
                                          n_substrate: float,
                                          wavelengths_m: list[float]) -> str:
    if n_film <= 0 or thickness_m < 0 or n_substrate <= 0:
        return _err("n>0, d>=0, n_substrate>0 required")
    if not wavelengths_m:
        return _err("wavelengths_m must be non-empty")
    n_air = 1.0
    r01 = (n_air - n_film) / (n_air + n_film)
    r12 = (n_film - n_substrate) / (n_film + n_substrate)
    spectrum = []
    for lam in wavelengths_m:
        if lam <= 0:
            return _err("all wavelengths must be > 0")
        phi = 2.0 * math.pi * n_film * thickness_m / lam
        cos2 = math.cos(2.0 * phi)
        num = r01 ** 2 + r12 ** 2 + 2.0 * r01 * r12 * cos2
        den = 1.0 + (r01 * r12) ** 2 + 2.0 * r01 * r12 * cos2
        R = num / den
        spectrum.append({"wavelength_m": lam, "reflectance": R})
    return _ok("thin_film_reflectance_spectrum",
               n_film=n_film, thickness_m=thickness_m, n_substrate=n_substrate,
               spectrum=spectrum)


@mcp.tool(description="Minimum film thickness for antireflection coating (n_film ≈ sqrt(n_substrate)).")
async def minimum_antireflection_thickness(n_substrate: float, wavelength_m: float) -> str:
    if n_substrate <= 0 or wavelength_m <= 0:
        return _err("n_substrate>0 and wavelength>0 required")
    n_film = math.sqrt(n_substrate)
    d = wavelength_m / (4.0 * n_film)
    return _ok("minimum_antireflection_thickness",
               n_substrate=n_substrate, wavelength_m=wavelength_m,
               n_film_ideal=n_film, thickness_m=d)


# ---------------------------------------------------------------------------
# Optics / diffraction
# ---------------------------------------------------------------------------

@mcp.tool(description="Single-slit first minimum angle: sin(theta) = lambda / a.")
async def single_slit_min_angle(wavelength_m: float, slit_width_m: float) -> str:
    if wavelength_m <= 0 or slit_width_m <= 0:
        return _err("wavelength>0 and slit_width>0 required")
    if wavelength_m > slit_width_m:
        return _err("wavelength must be < slit width for a real angle")
    s = wavelength_m / slit_width_m
    return _ok("single_slit_min_angle",
               sin_theta=s, theta_deg=math.degrees(math.asin(s)))


@mcp.tool(description="Double-slit interference maxima: sin(theta_m) = m * lambda / d.")
async def double_slit_max_angle(wavelength_m: float, slit_separation_m: float,
                                 order: int) -> str:
    if wavelength_m <= 0 or slit_separation_m <= 0 or order <= 0:
        return _err("wavelength>0, d>0, order>0 required")
    s = order * wavelength_m / slit_separation_m
    if s >= 1:
        return _err("order beyond observable angle", sin_theta=s)
    return _ok("double_slit_max_angle",
               sin_theta=s, theta_deg=math.degrees(math.asin(s)),
               order=order)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

@mcp.tool(description=(
    "Plot B-field magnitude vs distance for an infinite wire. Returns path "
    "to saved PNG."
))
def plot_b_field_vs_distance(distances_m: list[float], currents_a: list[float],
                                mu0: float = 4e-7 * math.pi) -> str:
    if not _MPL_OK:
        return _err("matplotlib not available")
    import os
    os.makedirs(_VIZ_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for I in currents_a:
        B = [mu0 * I / (2 * math.pi * d) if d > 0 else 0 for d in distances_m]
        ax.plot(distances_m, B, label=f"I = {I} A")
    ax.set_xlabel("distance r (m)")
    ax.set_ylabel("B (T)")
    ax.set_title("Infinite-wire B-field vs distance")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out = f"{_VIZ_DIR}/b_field.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return _ok("plot_b_field_vs_distance", path=out)


@mcp.tool(description=(
    "Plot a thin-film reflectance spectrum (R vs wavelength) from arrays. "
    "Returns path to saved PNG."
))
def plot_thin_film_spectrum(wavelengths_nm: list[float],
                              reflectance: list[float],
                              title: str = "Thin-film reflectance") -> str:
    if not _MPL_OK:
        return _err("matplotlib not available")
    if len(wavelengths_nm) != len(reflectance):
        return _err("wavelengths and reflectance length mismatch")
    import os
    os.makedirs(_VIZ_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(wavelengths_nm, reflectance, "b-")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("reflectance")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    out = f"{_VIZ_DIR}/thin_film.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return _ok("plot_thin_film_spectrum", path=out, n=len(wavelengths_nm))


if __name__ == "__main__":
    import asyncio
    asyncio.run(mcp.run_stdio_async())
