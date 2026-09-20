#!/usr/bin/env python3
"""Classical mechanics MCP server.

Tools ported from SciAgentGYM-main/toolkits/physics/mechanics/ — covering
Newtonian kinematics & dynamics, energy methods, statics, rotational
motion, pulley / connected-mass problems, vibration (driven + damped),
Lagrangian / Hamiltonian building blocks, and special relativity.

All implementations are stdlib + numpy only. Every tool returns a JSON
string. Tool naming is snake_case.
"""

from __future__ import annotations

import json
import math

import numpy as np

try:
    from scipy.integrate import quad  # noqa: F401  (for action integrals)
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("physics-classical")

# Physical constants (SI).
G = 6.67430e-11           # gravitational constant (m^3 kg^-1 s^-2)
C = 2.99792458e8          # speed of light (m/s)
HBAR = 1.054571817e-34    # reduced Planck constant (J s)


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

_VIZ_DIR = "/tmp/scimas_physics_classical"


# ---------------------------------------------------------------------------
# Kinematics / dynamics primitives
# ---------------------------------------------------------------------------

@mcp.tool(description="Constant-acceleration displacement: s = s0 + v0*t + 0.5*a*t^2.")
async def kinematics_displacement(s0: float, v0: float, a: float, t: float) -> str:
    if t < 0:
        return _err("time t must be non-negative", t=t)
    s = s0 + v0 * t + 0.5 * a * t * t
    v = v0 + a * t
    return _ok("kinematics_displacement",
               s0=s0, v0=v0, a=a, t=t, s=s, v_final=v)


@mcp.tool(description="Projectile motion (flat ground): range, time-of-flight, peak height from launch speed and angle.")
async def projectile_motion(v0: float, angle_deg: float, g: float = 9.80665) -> str:
    if v0 < 0:
        return _err("v0 must be non-negative", v0=v0)
    if not 0 <= angle_deg <= 90:
        return _err("angle_deg must be in [0, 90]", angle_deg=angle_deg)
    if g <= 0:
        return _err("g must be positive", g=g)
    theta = math.radians(angle_deg)
    t_flight = 2.0 * v0 * math.sin(theta) / g
    h_peak = (v0 * math.sin(theta)) ** 2 / (2.0 * g)
    range_ = (v0 ** 2) * math.sin(2.0 * theta) / g
    return _ok("projectile_motion",
               v0=v0, angle_deg=angle_deg, g=g,
               t_flight_s=t_flight, h_peak_m=h_peak, range_m=range_)


@mcp.tool(description="Apply Newton's second law: given net force and mass, return acceleration; or given acceleration and mass, return force.")
async def newtons_second_law(force_N: float = 0.0, mass_kg: float = 0.0, acceleration_m_s2: float = 0.0) -> str:
    """Two of {force, mass, acceleration} must be supplied; the third is computed.

    Negative values are rejected for mass. If exactly two are supplied
    (the third is zero), the missing one is computed. If three are
    supplied, we return an error to disambiguate.
    """
    supplied = [v != 0 for v in (force_N, mass_kg, acceleration_m_s2)]
    if sum(supplied) < 2:
        return _err("at least two of force_N, mass_kg, acceleration_m_s2 must be supplied")
    if mass_kg < 0:
        return _err("mass_kg must be non-negative")
    if sum(supplied) == 3:
        return _err("exactly two of force/mass/accel must be supplied")
    if mass_kg == 0:
        # given F, a -> m
        if acceleration_m_s2 == 0:
            return _err("division by zero mass and zero acceleration")
        m = force_N / acceleration_m_s2
        return _ok("newtons_second_law", force_N=force_N,
                   mass_kg=m, acceleration_m_s2=acceleration_m_s2,
                   solved_for="mass_kg")
    if force_N == 0:
        F = mass_kg * acceleration_m_s2
        return _ok("newtons_second_law", force_N=F,
                   mass_kg=mass_kg, acceleration_m_s2=acceleration_m_s2,
                   solved_for="force_N")
    a = force_N / mass_kg
    return _ok("newtons_second_law", force_N=force_N,
               mass_kg=mass_kg, acceleration_m_s2=a,
               solved_for="acceleration_m_s2")


# ---------------------------------------------------------------------------
# Energy & work
# ---------------------------------------------------------------------------

@mcp.tool(description="Kinetic energy of a point mass: KE = 0.5 * m * v^2.")
async def kinetic_energy(mass_kg: float, speed_m_s: float) -> str:
    if mass_kg < 0:
        return _err("mass_kg must be non-negative")
    ke = 0.5 * mass_kg * speed_m_s ** 2
    return _ok("kinetic_energy", mass_kg=mass_kg,
               speed_m_s=speed_m_s, energy_J=ke)


@mcp.tool(description="Gravitational potential energy near Earth's surface: U = m*g*h.")
async def gravitational_potential_energy(mass_kg: float, height_m: float, g: float = 9.80665) -> str:
    if mass_kg < 0:
        return _err("mass_kg must be non-negative")
    u = mass_kg * g * height_m
    return _ok("gravitational_potential_energy",
               mass_kg=mass_kg, height_m=height_m, g=g, energy_J=u)


@mcp.tool(description="Spring potential energy: U = 0.5 * k * x^2.")
async def spring_potential_energy(spring_constant_N_m: float, displacement_m: float) -> str:
    u = 0.5 * spring_constant_N_m * displacement_m ** 2
    return _ok("spring_potential_energy",
               spring_constant_N_m=spring_constant_N_m,
               displacement_m=displacement_m, energy_J=u)


@mcp.tool(description="Conservation of mechanical energy: given initial KE + PE and final KE, compute final PE (or vice versa).")
async def energy_conservation(ke_initial_J: float, pe_initial_J: float,
                              ke_final_J: float = 0.0, pe_final_J: float = 0.0) -> str:
    """At least one of ke_final_J / pe_final_J must be supplied (positive)."""
    if ke_final_J != 0 and pe_final_J != 0:
        return _err("supply exactly one of ke_final_J or pe_final_J")
    total_initial = ke_initial_J + pe_initial_J
    if ke_final_J != 0:
        pe_final_J = total_initial - ke_final_J
        return _ok("energy_conservation",
                   ke_initial_J=ke_initial_J, pe_initial_J=pe_initial_J,
                   ke_final_J=ke_final_J, pe_final_J=pe_final_J,
                   total_J=total_initial, solved_for="pe_final_J")
    pe_final_J = 0.0
    ke_final_J = total_initial
    return _ok("energy_conservation",
               ke_initial_J=ke_initial_J, pe_initial_J=pe_initial_J,
               ke_final_J=ke_final_J, pe_final_J=pe_final_J,
               total_J=total_initial, solved_for="ke_final_J")


# ---------------------------------------------------------------------------
# Statics / equilibrium
# ---------------------------------------------------------------------------

@mcp.tool(description="Torque = r * F * sin(theta).")
async def torque(moment_arm_m: float, force_N: float, angle_deg: float = 90.0) -> str:
    theta = math.radians(angle_deg)
    tau = moment_arm_m * force_N * math.sin(theta)
    return _ok("torque",
               moment_arm_m=moment_arm_m, force_N=force_N, angle_deg=angle_deg,
               torque_N_m=tau)


@mcp.tool(description="Static equilibrium: solve sum(F_x)=0, sum(F_y)=0 for up to 3 unknown magnitudes.")
async def static_equilibrium(forces: list[dict]) -> str:
    """`forces` is a list of {name, mag_N, angle_deg}; unknowns have mag_N=None."""
    fx, fy = 0.0, 0.0
    known_fx, known_fy = 0.0, 0.0
    unknowns: list[dict] = []
    for entry in forces:
        name = str(entry.get("name", "F"))
        mag = entry.get("mag_N")
        ang = math.radians(float(entry.get("angle_deg", 0.0)))
        cos_a, sin_a = math.cos(ang), math.sin(ang)
        if mag is None:
            unknowns.append({"name": name, "cos": cos_a, "sin": sin_a})
        else:
            known_fx += float(mag) * cos_a
            known_fy += float(mag) * sin_a
    if len(unknowns) > 2:
        return _err("at most 2 unknowns supported (2D equilibrium)")
    if len(unknowns) == 0:
        net = math.hypot(known_fx, known_fy)
        return _ok("static_equilibrium",
                   net_force_N=net, fx_N=known_fx, fy_N=known_fy,
                   unknowns=[])
    if len(unknowns) == 1:
        # 1 unknown, must be along a known direction.
        u = unknowns[0]
        # choose whichever component is non-trivial.
        if abs(u["cos"]) > abs(u["sin"]):
            mag = -known_fx / u["cos"]
            fy_check = known_fy
        else:
            mag = -known_fy / u["sin"]
            fy_check = known_fx
        u["mag_N"] = mag
        return _ok("static_equilibrium",
                   unknowns=[{k: v for k, v in u.items() if k in ("name", "mag_N")}],
                   residual_N=math.hypot(fx + mag * u["cos"],
                                          fy + mag * u["sin"]),
                   net_residual_N=math.hypot(fy_check, 0.0))
    # 2 unknowns: solve 2x2 linear system.
    a, b = unknowns
    det = a["cos"] * b["sin"] - b["cos"] * a["sin"]
    if abs(det) < 1e-12:
        return _err("force directions are linearly dependent")
    ma = (-known_fx * b["sin"] + known_fy * b["cos"]) / det
    mb = (known_fx * a["sin"] - known_fy * a["cos"]) / det
    a["mag_N"] = ma
    b["mag_N"] = mb
    return _ok("static_equilibrium",
               unknowns=[{k: v for k, v in u.items() if k in ("name", "mag_N")}
                         for u in unknowns],
               residual_N=math.hypot(known_fx + ma * a["cos"] + mb * b["cos"],
                                     known_fy + ma * a["sin"] + mb * b["sin"]))


# ---------------------------------------------------------------------------
# Circular / rotational motion
# ---------------------------------------------------------------------------

@mcp.tool(description="Centripetal force: F = m * v^2 / r (also returns angular velocity if r given).")
async def centripetal_force(mass_kg: float, speed_m_s: float, radius_m: float) -> str:
    if mass_kg < 0 or speed_m_s < 0 or radius_m <= 0:
        return _err("mass>=0, speed>=0, radius>0 required")
    f = mass_kg * speed_m_s ** 2 / radius_m
    omega = speed_m_s / radius_m
    period = 2.0 * math.pi / omega
    return _ok("centripetal_force",
               force_N=f, omega_rad_s=omega, period_s=period)


@mcp.tool(description="Conical pendulum: ball on string of length L moving in horizontal circle of radius r, find period and string tension.")
async def conical_pendulum(string_length_m: float, radius_m: float,
                           mass_kg: float, g: float = 9.80665) -> str:
    if string_length_m <= 0 or radius_m <= 0 or mass_kg < 0:
        return _err("L>0, r>0, m>=0 required")
    if radius_m >= string_length_m:
        return _err("radius must be < string_length")
    omega = math.sqrt(g / (math.sqrt(string_length_m ** 2 - radius_m ** 2) ** 3 / string_length_m ** 2))
    # Simplified: omega = sqrt(g / (L * cos(theta))), cos(theta) = sqrt(L^2-r^2)/L
    cos_t = math.sqrt(max(0.0, 1 - (radius_m / string_length_m) ** 2))
    omega = math.sqrt(g / (string_length_m * cos_t))
    period = 2.0 * math.pi / omega
    tension = mass_kg * g / cos_t
    return _ok("conical_pendulum",
               omega_rad_s=omega, period_s=period, tension_N=tension,
               half_angle_deg=math.degrees(math.acos(cos_t)))


@mcp.tool(description="Two-mass Atwood machine: masses m1 and m2 on a frictionless pulley, find acceleration and tension.")
async def atwood_machine(m1_kg: float, m2_kg: float, g: float = 9.80665) -> str:
    if m1_kg <= 0 or m2_kg <= 0:
        return _err("both masses must be positive")
    a = (m1_kg - m2_kg) * g / (m1_kg + m2_kg)
    T = 2.0 * m1_kg * m2_kg * g / (m1_kg + m2_kg)
    return _ok("atwood_machine",
               m1_kg=m1_kg, m2_kg=m2_kg, acceleration_m_s2=a,
               tension_N=T, heavier_side=("m1" if m1_kg > m2_kg else "m2"))


@mcp.tool(description="Inclined plane with friction: mass m on incline angle theta, friction coefficient mu. Find acceleration and friction force.")
async def inclined_plane(mass_kg: float, angle_deg: float, mu: float = 0.0,
                         g: float = 9.80665) -> str:
    if mass_kg <= 0:
        return _err("mass must be positive")
    theta = math.radians(angle_deg)
    g_along = g * math.sin(theta)
    g_normal = g * math.cos(theta)
    max_static = mu * mass_kg * g_normal
    if abs(g_along) <= max_static:
        a = 0.0
        friction = mass_kg * g_along
        state = "static"
    else:
        friction = max_static if g_along > 0 else -max_static
        a = (mass_kg * g_along - friction) / mass_kg
        state = "kinetic"
    return _ok("inclined_plane",
               acceleration_m_s2=a, friction_N=friction,
               state=state, g_along_m_s2=g_along,
               g_normal_m_s2=g_normal)


# ---------------------------------------------------------------------------
# Vibration (driven + damped)
# ---------------------------------------------------------------------------

@mcp.tool(description="Natural frequency and period of a mass-spring oscillator: omega_0 = sqrt(k/m).")
async def mass_spring_frequency(mass_kg: float, spring_constant_N_m: float) -> str:
    if mass_kg <= 0 or spring_constant_N_m <= 0:
        return _err("mass>0 and k>0 required")
    omega = math.sqrt(spring_constant_N_m / mass_kg)
    return _ok("mass_spring_frequency",
               omega_rad_s=omega, freq_Hz=omega / (2 * math.pi),
               period_s=2 * math.pi / omega)


@mcp.tool(description="Damped oscillator quality factor and decay envelope time constant tau = 2m / c for viscous damping coefficient c.")
async def damping_coefficient_to_quality(mass_kg: float, damping_coefficient: float,
                                          spring_constant_N_m: float) -> str:
    if mass_kg <= 0 or spring_constant_N_m <= 0 or damping_coefficient < 0:
        return _err("m>0, k>0, c>=0 required")
    omega0 = math.sqrt(spring_constant_N_m / mass_kg)
    tau = 2 * mass_kg / damping_coefficient if damping_coefficient > 0 else math.inf
    Q = omega0 * mass_kg / damping_coefficient if damping_coefficient > 0 else math.inf
    return _ok("damping_coefficient_to_quality",
               omega0_rad_s=omega0, tau_s=tau, Q=Q)


@mcp.tool(description="Steady-state amplitude of a driven damped oscillator: A = F0 / sqrt((k - m w^2)^2 + (c w)^2).")
async def driven_oscillator_amplitude(mass_kg: float, spring_constant_N_m: float,
                                       damping: float, drive_freq_Hz: float,
                                       drive_amplitude_N: float) -> str:
    if mass_kg <= 0 or spring_constant_N_m <= 0:
        return _err("mass>0, k>0 required")
    if drive_amplitude_N == 0:
        return _ok("driven_oscillator_amplitude", amplitude_m=0.0)
    k = spring_constant_N_m
    w = 2 * math.pi * drive_freq_Hz
    denom = math.sqrt((k - mass_kg * w * w) ** 2 + (damping * w) ** 2)
    if denom == 0:
        return _err("resonance denominator is zero (pure resonance)")
    A = drive_amplitude_N / denom
    return _ok("driven_oscillator_amplitude",
               amplitude_m=A, denom=denom,
               resonance_freq_Hz=math.sqrt(k / mass_kg) / (2 * math.pi))


# ---------------------------------------------------------------------------
# Lagrangian / Hamiltonian
# ---------------------------------------------------------------------------

@mcp.tool(description="Lagrangian L = T - V given kinetic energy T and potential energy V (scalar values).")
async def lagrangian(kinetic_J: float, potential_J: float) -> str:
    L = kinetic_J - potential_J
    return _ok("lagrangian", kinetic_J=kinetic_J,
               potential_J=potential_J, lagrangian_J=L)


@mcp.tool(description="Generalized momentum p = m * v.")
async def generalized_momentum(mass_kg: float, velocity_m_s: float) -> str:
    if mass_kg < 0:
        return _err("mass must be non-negative")
    return _ok("generalized_momentum",
               mass_kg=mass_kg, velocity_m_s=velocity_m_s,
               momentum_kg_m_s=mass_kg * velocity_m_s)


@mcp.tool(description="Hamiltonian H = T + V (for natural systems where L = T - V).")
async def hamiltonian(kinetic_J: float, potential_J: float) -> str:
    return _ok("hamiltonian", kinetic_J=kinetic_J,
               potential_J=potential_J, hamiltonian_J=kinetic_J + potential_J)


# ---------------------------------------------------------------------------
# Special relativity
# ---------------------------------------------------------------------------

@mcp.tool(description="Lorentz factor gamma = 1 / sqrt(1 - v^2/c^2). Returns gamma and 1/gamma.")
async def lorentz_factor(speed_m_s: float) -> str:
    if abs(speed_m_s) >= C:
        return _err(f"speed must be < c ({C} m/s)", speed_m_s=speed_m_s)
    beta = speed_m_s / C
    gamma = 1.0 / math.sqrt(1.0 - beta * beta)
    return _ok("lorentz_factor",
               speed_m_s=speed_m_s, beta=beta,
               gamma=gamma, beta_gamma=beta * gamma)


@mcp.tool(description="Time dilation: dilated interval = proper interval * gamma.")
async def time_dilation(proper_time_s: float, speed_m_s: float) -> str:
    if abs(speed_m_s) >= C:
        return _err("speed must be < c")
    gamma = 1.0 / math.sqrt(1.0 - (speed_m_s / C) ** 2)
    dilated = proper_time_s * gamma
    return _ok("time_dilation",
               proper_time_s=proper_time_s, speed_m_s=speed_m_s,
               gamma=gamma, dilated_time_s=dilated)


@mcp.tool(description="Length contraction: contracted length = proper length / gamma.")
async def length_contraction(proper_length_m: float, speed_m_s: float) -> str:
    if abs(speed_m_s) >= C:
        return _err("speed must be < c")
    gamma = 1.0 / math.sqrt(1.0 - (speed_m_s / C) ** 2)
    contracted = proper_length_m / gamma
    return _ok("length_contraction",
               proper_length_m=proper_length_m, speed_m_s=speed_m_s,
               gamma=gamma, contracted_length_m=contracted)


@mcp.tool(description="Relativistic momentum and total energy: p = gamma*m*v, E = gamma*m*c^2, E0 = m*c^2.")
async def relativistic_energy_momentum(mass_kg: float, speed_m_s: float) -> str:
    if mass_kg < 0:
        return _err("mass must be non-negative")
    if abs(speed_m_s) >= C:
        return _err("speed must be < c")
    gamma = 1.0 / math.sqrt(1.0 - (speed_m_s / C) ** 2)
    E0 = mass_kg * C ** 2
    E = gamma * E0
    p = gamma * mass_kg * speed_m_s
    KE = (gamma - 1.0) * E0
    return _ok("relativistic_energy_momentum",
               gamma=gamma, rest_energy_J=E0,
               total_energy_J=E, momentum_kg_m_s=p,
               kinetic_energy_J=KE)


# ---------------------------------------------------------------------------
# Sliding on a slider-crank / connected systems helpers
# ---------------------------------------------------------------------------

@mcp.tool(description="Slider-crank position/velocity: x(θ)=r*cos(θ)+sqrt(L^2 - r^2*sin^2(θ)). Returns x, dx/dθ, and numerical dx/dω for given ω.")
async def slider_crank_kinematics(crank_r_m: float, connecting_rod_L_m: float,
                                   angle_deg: float, omega_rad_s: float = 0.0) -> str:
    if crank_r_m <= 0 or connecting_rod_L_m < crank_r_m:
        return _err("require L > r > 0")
    theta = math.radians(angle_deg)
    s = math.sqrt(max(0.0, connecting_rod_L_m ** 2 - (crank_r_m * math.sin(theta)) ** 2))
    x = crank_r_m * math.cos(theta) + s
    dx_dtheta = -crank_r_m * math.sin(theta) - (crank_r_m ** 2 * math.sin(theta) * math.cos(theta)) / s
    v = dx_dtheta * omega_rad_s
    return _ok("slider_crank_kinematics",
               x_m=x, dx_dtheta_rad=dx_dtheta,
               velocity_m_s=v, omega_rad_s=omega_rad_s)


# ---------------------------------------------------------------------------
# Convenience: ke + pe ↔ speed + height
# ---------------------------------------------------------------------------

@mcp.tool(description="Final speed from energy conservation: drop mass m from height h, return v at ground (no air resistance).")
async def final_speed_from_drop(mass_kg: float, height_m: float, g: float = 9.80665) -> str:
    if mass_kg <= 0 or height_m < 0:
        return _err("m>0, h>=0 required")
    v = math.sqrt(2.0 * g * height_m)
    return _ok("final_speed_from_drop",
               mass_kg=mass_kg, height_m=height_m,
               final_speed_m_s=v, kinetic_energy_J=0.5 * mass_kg * v ** 2)


@mcp.tool(description="Pendulum period (small-angle approximation): T = 2π sqrt(L/g).")
async def pendulum_period(string_length_m: float, g: float = 9.80665) -> str:
    if string_length_m <= 0 or g <= 0:
        return _err("L>0 and g>0 required")
    T = 2.0 * math.pi * math.sqrt(string_length_m / g)
    return _ok("pendulum_period",
               string_length_m=string_length_m, g=g,
               period_s=T, freq_Hz=1.0 / T)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

@mcp.tool(description=(
    "Plot a 1-D trajectory (x vs t and v vs t) from sample arrays. "
    "Returns path to saved PNG (matplotlib Agg backend)."
))
def plot_trajectory(t: list[float], x: list[float],
                      v: list[float] = None,
                      save_name: str = "trajectory.png") -> str:
    if not _MPL_OK:
        return _err("matplotlib not available")
    if len(t) != len(x) or len(t) < 2:
        return _err("t and x must match in length (≥ 2)")
    import os
    os.makedirs(_VIZ_DIR, exist_ok=True)
    fig, axes = plt.subplots(2 if v is not None else 1, 1, figsize=(7, 6), sharex=True)
    if v is None:
        axes = [axes]
    axes[0].plot(t, x, "b-", linewidth=1.5)
    axes[0].set_ylabel("x(t)")
    axes[0].set_title("Position trajectory")
    axes[0].grid(True, alpha=0.3)
    if v is not None:
        if len(v) != len(t):
            return _err("v length must match t")
        axes[1].plot(t, v, "r-", linewidth=1.5)
        axes[1].set_ylabel("v(t)")
        axes[1].set_xlabel("t (s)")
        axes[1].grid(True, alpha=0.3)
    else:
        axes[0].set_xlabel("t (s)")
    out = f"{_VIZ_DIR}/{save_name}"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return _ok("plot_trajectory", path=out, n=len(t))


@mcp.tool(description=(
    "Plot kinetic, potential, and total mechanical energy over time. "
    "Returns path to saved PNG."
))
def plot_energy_vs_time(t: list[float], ke: list[float],
                          pe: list[float]) -> str:
    if not _MPL_OK:
        return _err("matplotlib not available")
    if not (len(t) == len(ke) == len(pe)) or len(t) < 2:
        return _err("t, ke, pe must match in length (≥ 2)")
    import os
    os.makedirs(_VIZ_DIR, exist_ok=True)
    total = [k + p for k, p in zip(ke, pe)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, ke, label="KE", color="tab:red")
    ax.plot(t, pe, label="PE", color="tab:blue")
    ax.plot(t, total, label="Total", color="black", linestyle="--")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("Energy (J)")
    ax.set_title("Mechanical energy vs time")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out = f"{_VIZ_DIR}/energy.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return _ok("plot_energy_vs_time", path=out, n=len(t))


if __name__ == "__main__":
    import asyncio
    asyncio.run(mcp.run_stdio_async())
