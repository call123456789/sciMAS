#!/usr/bin/env python3
"""Environmental chemistry MCP server.

Tools ported from SciAgentGYM-main/toolkits/chemistry/environmental_chemistry/.
All implementations are pure-Python (no external chemistry libraries).

Tools
-----
- dissolved_oxygen_winkler: Winkler iodometric titration → DO (mg/L).
- bod5_from_titration: BOD5 from day-0 and day-5 titration data.
- waste_calorific_value: weighted-average calorific value of a waste
  composition (MJ/kg on a dry basis).
"""

from __future__ import annotations

import json
import math

import numpy as np
from scipy import stats

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("chemistry-environmental")


@mcp.tool(
    description=(
        "Calculate dissolved oxygen (DO, mg/L) from a Winkler iodometric "
        "titration. Inputs: volume of sample in mL, normality of Na2S2O3 "
        "titrant (mol/L), and volume of Na2S2O3 used (mL). The factor 8 "
        "is the standard mg-of-O2 per meq conversion (1 mL of 1 N "
        "thiosulfate ≡ 8 mg O2)."
    )
)
async def dissolved_oxygen_winkler(
    sample_volume_mL: float,
    thiosulfate_normality: float,
    thiosulfate_volume_mL: float,
) -> str:
    if sample_volume_mL <= 0:
        return json.dumps({"error": "sample_volume_mL must be > 0"})
    do_mg_per_L = (thiosulfate_normality * thiosulfate_volume_mL * 8 * 1000) / sample_volume_mL
    return json.dumps(
        {
            "sample_volume_mL": sample_volume_mL,
            "thiosulfate_normality": thiosulfate_normality,
            "thiosulfate_volume_mL": thiosulfate_volume_mL,
            "dissolved_oxygen_mg_per_L": round(do_mg_per_L, 4),
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Calculate BOD5 (mg/L) from Winkler titration data on day 0 and "
        "day 5, with a parallel blank. Formula: "
        "BOD5 = ((D1 - D2) - (B1 - B2)) * f, where D1/D2 are the "
        "sample DO on day 0 / day 5 (mg/L), B1/B2 are the blank DO on "
        "day 0 / day 5 (mg/L), and f is the dilution factor "
        "(sample_volume / diluted_volume)."
    )
)
async def bod5_from_titration(
    sample_do_day0_mg_per_L: float,
    sample_do_day5_mg_per_L: float,
    blank_do_day0_mg_per_L: float,
    blank_do_day5_mg_per_L: float,
    dilution_factor: float = 1.0,
) -> str:
    if dilution_factor <= 0:
        return json.dumps({"error": "dilution_factor must be > 0"})
    depletion_sample = sample_do_day0_mg_per_L - sample_do_day5_mg_per_L
    depletion_blank = blank_do_day0_mg_per_L - blank_do_day5_mg_per_L
    bod5 = (depletion_sample - depletion_blank) * dilution_factor
    return json.dumps(
        {
            "depletion_sample_mg_per_L": round(depletion_sample, 4),
            "depletion_blank_mg_per_L": round(depletion_blank, 4),
            "dilution_factor": dilution_factor,
            "bod5_mg_per_L": round(bod5, 4),
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Compute the weighted-average calorific value (MJ/kg, dry basis) "
        "of a municipal-solid-waste composition. Input: list of "
        "{component, mass_percent, calorific_value_MJ_per_kg} dicts. "
        "Validates that the mass percentages sum to ~100%."
    )
)
async def waste_calorific_value(
    components: list[dict],
) -> str:
    if not components:
        return json.dumps({"error": "components must be a non-empty list"})

    total_pct = 0.0
    weighted = 0.0
    per_component = []
    for entry in components:
        try:
            name = str(entry["component"])
            pct = float(entry["mass_percent"])
            cv = float(entry["calorific_value_MJ_per_kg"])
        except (KeyError, TypeError, ValueError) as exc:
            return json.dumps({"error": f"bad component entry: {exc}"})
        total_pct += pct
        weighted += (pct / 100.0) * cv
        per_component.append(
            {
                "component": name,
                "mass_percent": pct,
                "calorific_value_MJ_per_kg": cv,
                "contribution_MJ_per_kg": round((pct / 100.0) * cv, 6),
            }
        )

    return json.dumps(
        {
            "components": per_component,
            "total_mass_percent": round(total_pct, 4),
            "calorific_value_MJ_per_kg": round(weighted, 4),
            "warning": (
                None
                if abs(total_pct - 100.0) < 0.5
                else f"mass_percent sum is {total_pct:.2f}, not ~100"
            ),
        },
        ensure_ascii=False,
    )


# ---- Wave 1: stdlib-only ports from SciAgentGYM environmental_chemistry/ ----

# BOD5 / DO family
# -----------------


@mcp.tool(
    description=(
        "Oxygen depletion over an incubation period: drop in dissolved "
        "oxygen from initial (day 0) to final (day N), mg/L. "
        "depletion = DO_day0 - DO_dayN."
    )
)
async def calculate_oxygen_depletion(
    initial_do_mg_per_L: float,
    final_do_mg_per_L: float,
) -> str:
    return json.dumps(
        {
            "initial_do_mg_per_L": initial_do_mg_per_L,
            "final_do_mg_per_L": final_do_mg_per_L,
            "depletion_mg_per_L": round(initial_do_mg_per_L - final_do_mg_per_L, 4),
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Apply a dilution factor to a concentration measurement. "
        "Used to back-correct BOD5 from a diluted sample. "
        "corrected = measured * factor."
    )
)
async def apply_dilution_factor(
    measured_value: float,
    dilution_factor: float,
) -> str:
    if dilution_factor <= 0:
        return json.dumps({"error": "dilution_factor must be > 0"})
    return json.dumps(
        {
            "measured_value": measured_value,
            "dilution_factor": dilution_factor,
            "corrected_value": round(measured_value * dilution_factor, 4),
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Validate a BOD5 measurement against typical-range bounds (0–10000 "
        "mg/L) and that DO depletion is non-negative. Returns pass/fail "
        "with reasons."
    )
)
async def validate_bod5_measurement(bod5_mg_per_L: float) -> str:
    reasons = []
    if bod5_mg_per_L < 0:
        reasons.append("BOD5 is negative — measurement error")
    if bod5_mg_per_L > 10000:
        reasons.append("BOD5 unrealistically high (>10000 mg/L)")
    return json.dumps(
        {
            "bod5_mg_per_L": bod5_mg_per_L,
            "is_valid": not reasons,
            "issues": reasons,
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Compute BOD5 for a list of samples, each with day-0 and day-5 "
        "DO readings plus a blank. Input: list of {sample_id, "
        "do_day0_mg_per_L, do_day5_mg_per_L, blank_day0_mg_per_L, "
        "blank_day5_mg_per_L, dilution_factor}. Returns per-sample BOD5 "
        "plus the mean across the batch."
    )
)
async def batch_calculate_bod5(samples: list[dict]) -> str:
    if not samples:
        return json.dumps({"error": "samples must be non-empty"})
    results = []
    for s in samples:
        try:
            sid = str(s["sample_id"])
            d0 = float(s["do_day0_mg_per_L"])
            d5 = float(s["do_day5_mg_per_L"])
            b0 = float(s["blank_day0_mg_per_L"])
            b5 = float(s["blank_day5_mg_per_L"])
            f = float(s.get("dilution_factor", 1.0))
        except (KeyError, TypeError, ValueError) as exc:
            return json.dumps({"error": f"bad sample entry: {exc}"})
        depletion_s = d0 - d5
        depletion_b = b0 - b5
        bod5 = (depletion_s - depletion_b) * f
        results.append(
            {
                "sample_id": sid,
                "bod5_mg_per_L": round(bod5, 4),
                "depletion_sample_mg_per_L": round(depletion_s, 4),
            }
        )
    mean_bod5 = sum(r["bod5_mg_per_L"] for r in results) / len(results)
    return json.dumps(
        {"results": results, "mean_bod5_mg_per_L": round(mean_bod5, 4)},
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Heuristic optimal-dilution estimator for BOD5. Aim: target "
        "~50% DO depletion (~4 mg/L drop out of 8 mg/L saturated). "
        "Returns suggested dilution factor = expected_bod5 / 4."
    )
)
async def estimate_optimal_dilution(
    expected_bod5_mg_per_L: float,
    target_depletion_mg_per_L: float = 4.0,
) -> str:
    if target_depletion_mg_per_L <= 0:
        return json.dumps({"error": "target_depletion_mg_per_L must be > 0"})
    f = expected_bod5_mg_per_L / target_depletion_mg_per_L
    return json.dumps(
        {
            "expected_bod5_mg_per_L": expected_bod5_mg_per_L,
            "target_depletion_mg_per_L": target_depletion_mg_per_L,
            "suggested_dilution_factor": round(f, 3),
            "note": (
                "Round to nearest convenient value (1, 2, 5, 10, 20, 50, "
                "100, 200)."
            ),
        },
        ensure_ascii=False,
    )


# Waste calorific family
# ----------------------


@mcp.tool(
    description=(
        "Validate a single waste-component entry: mass_percent in 0-100, "
        "non-negative calorific value."
    )
)
async def validate_waste_component(component: dict) -> str:
    issues = []
    try:
        name = str(component["component"])
        pct = float(component["mass_percent"])
        cv = float(component["calorific_value_MJ_per_kg"])
    except (KeyError, TypeError, ValueError) as exc:
        return json.dumps({"is_valid": False, "issues": [f"bad entry: {exc}"]})
    if not (0 <= pct <= 100):
        issues.append(f"mass_percent {pct} outside 0-100")
    if cv < 0:
        issues.append(f"calorific_value_MJ_per_kg {cv} is negative")
    return json.dumps(
        {"component": name, "is_valid": not issues, "issues": issues},
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Calorific contribution of one waste component = "
        "(mass_percent / 100) * calorific_value_MJ_per_kg."
    )
)
async def calculate_component_contribution(
    mass_percent: float,
    calorific_value_MJ_per_kg: float,
) -> str:
    return json.dumps(
        {
            "mass_percent": mass_percent,
            "calorific_value_MJ_per_kg": calorific_value_MJ_per_kg,
            "contribution_MJ_per_kg": round(
                (mass_percent / 100.0) * calorific_value_MJ_per_kg, 6
            ),
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Sum a list of contributions (MJ/kg) into a single total."
    )
)
async def sum_contributions(contributions: list[float]) -> str:
    if not contributions:
        return json.dumps({"error": "contributions must be non-empty"})
    total = float(sum(contributions))
    return json.dumps(
        {"n": len(contributions), "total": round(total, 6)},
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Validate that a list of mass_percent values sums to ~100% "
        "(tolerance 0.5%)."
    )
)
async def validate_total_percentage(mass_percents: list[float]) -> str:
    if not mass_percents:
        return json.dumps({"is_valid": False, "issues": ["empty list"]})
    total = float(sum(mass_percents))
    issues = []
    if abs(total - 100.0) > 0.5:
        issues.append(f"sum is {total:.4f}, expected ~100")
    return json.dumps(
        {"total_percent": round(total, 4), "is_valid": not issues, "issues": issues},
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Analyze a waste composition: validate entries, sum percentages, "
        "compute weighted calorific value. Input: list of "
        "{component, mass_percent, calorific_value_MJ_per_kg} dicts."
    )
)
async def analyze_waste_composition(components: list[dict]) -> str:
    if not components:
        return json.dumps({"error": "components must be non-empty"})
    total_pct = 0.0
    weighted = 0.0
    validated = []
    invalid_count = 0
    for entry in components:
        try:
            name = str(entry["component"])
            pct = float(entry["mass_percent"])
            cv = float(entry["calorific_value_MJ_per_kg"])
        except (KeyError, TypeError, ValueError) as exc:
            return json.dumps({"error": f"bad entry: {exc}"})
        if pct < 0 or cv < 0:
            invalid_count += 1
            continue
        total_pct += pct
        weighted += (pct / 100.0) * cv
        validated.append(
            {
                "component": name,
                "mass_percent": pct,
                "calorific_value_MJ_per_kg": cv,
            }
        )
    return json.dumps(
        {
            "n_components": len(validated),
            "n_invalid": invalid_count,
            "total_mass_percent": round(total_pct, 4),
            "calorific_value_MJ_per_kg": round(weighted, 4),
            "components": validated,
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Load and parse a JSON file from disk. Returns the parsed JSON "
        "object on success, error message on failure."
    )
)
async def load_json_file(filepath: str) -> str:
    import os
    if not os.path.exists(filepath):
        return json.dumps({"error": f"file not found: {filepath}"})
    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        return json.dumps({"error": str(exc)})
    return json.dumps({"filepath": filepath, "data": data}, ensure_ascii=False)


@mcp.tool(
    description=(
        "Save a calculation result (dict) to a JSON report file. "
        "Returns the absolute file path on success."
    )
)
async def save_calculation_report(
    report: dict,
    output_dir: str = "./runs",
    label: str = "calculation",
) -> str:
    import os
    from datetime import datetime
    os.makedirs(output_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_label = "".join(c for c in label if c.isalnum() or c in ("-", "_"))
    path = os.path.join(output_dir, f"{safe_label}-{stamp}.json")
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
    except OSError as exc:
        return json.dumps({"error": str(exc)})
    return json.dumps(
        {"saved_to": os.path.abspath(path), "label": label},
        ensure_ascii=False,
    )


# Nitrogen family
# ---------------


@mcp.tool(
    description=(
        "Total nitrogen (mg/L) = TKN + NO3-N + NO2-N + NH3-N. Pass the "
        "four nitrogen fractions in mg/L."
    )
)
async def calculate_total_nitrogen(
    tkn_mg_per_L: float,
    nitrate_n_mg_per_L: float,
    nitrite_n_mg_per_L: float,
    ammonia_n_mg_per_L: float,
) -> str:
    tn = tkn_mg_per_L + nitrate_n_mg_per_L + nitrite_n_mg_per_L + ammonia_n_mg_per_L
    return json.dumps(
        {
            "tkn_mg_per_L": tkn_mg_per_L,
            "nitrate_n_mg_per_L": nitrate_n_mg_per_L,
            "nitrite_n_mg_per_L": nitrite_n_mg_per_L,
            "ammonia_n_mg_per_L": ammonia_n_mg_per_L,
            "total_nitrogen_mg_per_L": round(tn, 4),
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Removal efficiency (%) for a treatment stage: "
        "100 * (influent - effluent) / influent. Returns 0 if influent "
        "is zero (avoids div-by-zero)."
    )
)
async def calculate_removal_efficiency(
    influent_mg_per_L: float,
    effluent_mg_per_L: float,
) -> str:
    if influent_mg_per_L == 0:
        eff = 0.0
    else:
        eff = 100.0 * (influent_mg_per_L - effluent_mg_per_L) / influent_mg_per_L
    return json.dumps(
        {
            "influent_mg_per_L": influent_mg_per_L,
            "effluent_mg_per_L": effluent_mg_per_L,
            "removal_efficiency_pct": round(eff, 4),
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Mass balance for a unit operation: check whether "
        "influent - effluent - loss is approximately 0 (within "
        "tolerance_pct). Returns residual and imbalance percentage."
    )
)
async def calculate_mass_balance(
    influent_mg_per_L: float,
    effluent_mg_per_L: float,
    loss_mg_per_L: float,
    tolerance_pct: float = 5.0,
) -> str:
    balance = influent_mg_per_L - effluent_mg_per_L - loss_mg_per_L
    imbalance_pct = (
        100.0 * abs(balance) / influent_mg_per_L if influent_mg_per_L else 0.0
    )
    return json.dumps(
        {
            "influent_mg_per_L": influent_mg_per_L,
            "effluent_mg_per_L": effluent_mg_per_L,
            "loss_mg_per_L": loss_mg_per_L,
            "residual_mg_per_L": round(balance, 4),
            "imbalance_pct": round(imbalance_pct, 4),
            "within_tolerance": imbalance_pct <= tolerance_pct,
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Analyze a nitrogen removal train: input TN, list of intermediate "
        "stages. Returns total removal plus per-stage removal efficiency. "
        "Input stages: list of {stage, tn_mg_per_L}."
    )
)
async def analyze_nitrogen_removal(
    influent_tn_mg_per_L: float,
    stages: list[dict],
) -> str:
    if not stages:
        return json.dumps({"error": "stages must be non-empty"})
    try:
        ordered = [
            {"stage": str(s["stage"]), "tn_mg_per_L": float(s["tn_mg_per_L"])}
            for s in stages
        ]
    except (KeyError, TypeError, ValueError) as exc:
        return json.dumps({"error": f"bad stage entry: {exc}"})
    final_eff = ordered[-1]["tn_mg_per_L"]
    total_removal_pct = 100.0 * (
        (influent_tn_mg_per_L - final_eff) / influent_tn_mg_per_L
        if influent_tn_mg_per_L
        else 0.0
    )
    per_stage = []
    prev = influent_tn_mg_per_L
    for s in ordered:
        if prev == 0:
            stage_eff = 0.0
        else:
            stage_eff = 100.0 * (prev - s["tn_mg_per_L"]) / prev
        per_stage.append(
            {
                "stage": s["stage"],
                "tn_mg_per_L": s["tn_mg_per_L"],
                "stage_removal_pct": round(stage_eff, 4),
            }
        )
        prev = s["tn_mg_per_L"]
    return json.dumps(
        {
            "influent_tn_mg_per_L": influent_tn_mg_per_L,
            "effluent_tn_mg_per_L": final_eff,
            "total_removal_pct": round(total_removal_pct, 4),
            "per_stage": per_stage,
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Heuristic evaluation of a nitrification-denitrification process. "
        "Inputs: influent NH3-N, effluent NH3-N and NO3-N (mg/L), DO in "
        "the aeration tank (mg/L), and pH. Returns DO/pH check status "
        "and expected pathway (complete vs partial nitrification)."
    )
)
async def evaluate_nitrification_denitrification(
    influent_ammonia_n_mg_per_L: float,
    effluent_ammonia_n_mg_per_L: float,
    effluent_nitrate_n_mg_per_L: float,
    aeration_DO_mg_per_L: float,
    pH: float,
) -> str:
    issues = []
    if aeration_DO_mg_per_L < 1.5:
        issues.append("DO too low for nitrification (<1.5 mg/L)")
    if aeration_DO_mg_per_L > 4.0:
        issues.append("DO may inhibit denitrification (>4 mg/L)")
    if pH < 6.5:
        issues.append("pH too low for nitrification (<6.5)")
    if pH > 9.0:
        issues.append("pH too high for nitrification (>9.0)")
    if influent_ammonia_n_mg_per_L == 0:
        nh3_removal = 0.0
    else:
        nh3_removal = 100.0 * (
            influent_ammonia_n_mg_per_L - effluent_ammonia_n_mg_per_L
        ) / influent_ammonia_n_mg_per_L
    if nh3_removal > 90:
        pathway = "complete nitrification"
    elif nh3_removal > 50:
        pathway = "partial nitrification"
    else:
        pathway = "incomplete / failing"
    return json.dumps(
        {
            "influent_ammonia_n_mg_per_L": influent_ammonia_n_mg_per_L,
            "effluent_ammonia_n_mg_per_L": effluent_ammonia_n_mg_per_L,
            "effluent_nitrate_n_mg_per_L": effluent_nitrate_n_mg_per_L,
            "aeration_DO_mg_per_L": aeration_DO_mg_per_L,
            "pH": pH,
            "ammonia_removal_pct": round(nh3_removal, 4),
            "pathway": pathway,
            "issues": issues,
        },
        ensure_ascii=False,
    )


# ---- Wave 2: numpy / scipy ports ----


@mcp.tool(
    description=(
        "First-order decay model: C(t) = C0 · exp(-k·t). Returns the "
        "concentration at query time t (mg/L or as supplied) plus the "
        "time at which C reaches C/2."
    )
)
async def first_order_decay_model(
    initial_conc: float,
    rate_constant_per_time: float,
    time: float,
) -> str:
    if initial_conc <= 0 or rate_constant_per_time < 0:
        return json.dumps({"error": "initial_conc > 0, k >= 0"})
    c_t = initial_conc * np.exp(-rate_constant_per_time * time)
    t_half = math.log(2) / rate_constant_per_time if rate_constant_per_time > 0 else None
    return json.dumps(
        {
            "initial_conc": initial_conc,
            "k": rate_constant_per_time,
            "time": time,
            "C_t": round(c_t, 6),
            "t_half": round(t_half, 6) if t_half is not None else None,
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Centered moving average smoother. For each y[i] returns the "
        "mean of y[i-window:i+window+1]. Useful for noise reduction "
        "in sensor traces. window must be a positive integer."
    )
)
async def moving_average(
    y_values: list[float],
    window: int = 3,
) -> str:
    if window < 1 or not y_values:
        return json.dumps({"error": "window >= 1 and non-empty y_values"})
    y = np.asarray(y_values, dtype=float)
    if window > len(y):
        return json.dumps({"error": "window > length of y_values"})
    kernel = np.ones(2 * window + 1) / (2 * window + 1)
    smoothed = np.convolve(y, kernel, mode="valid")
    pad_left = window
    pad_right = len(y) - len(smoothed) - window
    full = np.concatenate(
        [
            np.full(pad_left, smoothed[0]),
            smoothed,
            np.full(pad_right, smoothed[-1]),
        ]
    )
    return json.dumps(
        {"window": window, "smoothed_y": full.tolist()}, ensure_ascii=False
    )


@mcp.tool(
    description=(
        "Welch's two-sample t-test (independent, unequal-variance). "
        "Returns t-statistic, two-sided p-value, and the means/std/"
        "n of each group."
    )
)
async def t_test_two_sample(group_a: list[float], group_b: list[float]) -> str:
    if len(group_a) < 2 or len(group_b) < 2:
        return json.dumps({"error": "each group needs >= 2 samples"})
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    res = stats.ttest_ind(a, b, equal_var=False)
    return json.dumps(
        {
            "group_a_n": int(a.size),
            "group_a_mean": float(a.mean()),
            "group_a_std": float(a.std(ddof=1)),
            "group_b_n": int(b.size),
            "group_b_mean": float(b.mean()),
            "group_b_std": float(b.std(ddof=1)),
            "t_statistic": float(res.statistic),
            "p_value": float(res.pvalue),
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Normal-distribution PDF value at x given mean mu and std "
        "sigma. Also returns the CDF (cumulative probability up to "
        "x) and the standardized z-score."
    )
)
async def normal_distribution_density(
    x: float, mu: float = 0.0, sigma: float = 1.0
) -> str:
    if sigma <= 0:
        return json.dumps({"error": "sigma must be > 0"})
    pdf = stats.norm.pdf(x, loc=mu, scale=sigma)
    cdf = stats.norm.cdf(x, loc=mu, scale=sigma)
    z = (x - mu) / sigma
    return json.dumps(
        {
            "x": x,
            "mu": mu,
            "sigma": sigma,
            "pdf": float(pdf),
            "cdf": float(cdf),
            "z": z,
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Ordinary-least-squares linear regression of y on x. Returns "
        "slope, intercept, R², residuals, and standard errors of the "
        "estimates."
    )
)
async def linear_regression_with_residuals(
    x_values: list[float],
    y_values: list[float],
) -> str:
    if len(x_values) != len(y_values) or len(x_values) < 3:
        return json.dumps({"error": "need at least 3 paired x,y values"})
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    res = stats.linregress(x, y)
    y_fit = res.slope * x + res.intercept
    residuals = (y - y_fit).tolist()
    return json.dumps(
        {
            "n": int(x.size),
            "slope": float(res.slope),
            "intercept": float(res.intercept),
            "R_squared": float(res.rvalue**2),
            "slope_stderr": float(res.stderr),
            "intercept_stderr": float(res.intercept_stderr),
            "residuals": residuals,
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Dissolved-oxygen saturation curve (mg/L) vs temperature (°C) "
        "for water at sea level, using the Benson-Krause (1984) "
        "simplified equation: DO_sat = exp(-1.55503 + 0.027206·T + "
        "0.0187813·T² - 0.00011081·T³ + 0.0000017746·T⁴) · "
        "(1 - 0.0015·(S)) where S is salinity (psu)."
    )
)
async def dissolved_oxygen_saturation_curve(
    temperatures_C: list[float],
    salinity_psu: float = 0.0,
) -> str:
    if salinity_psu < 0 or salinity_psu > 40:
        return json.dumps({"error": "salinity_psu must be in 0-40"})
    if not temperatures_C:
        return json.dumps({"error": "temperatures_C must be non-empty"})
    T = np.asarray(temperatures_C, dtype=float)
    do = np.exp(
        -1.55503
        + 0.027206 * T
        + 0.0187813 * T**2
        - 0.00011081 * T**3
        + 0.0000017746 * T**4
    ) * (1.0 - 0.0015 * salinity_psu)
    return json.dumps(
        {
            "temperatures_C": T.tolist(),
            "salinity_psu": salinity_psu,
            "DO_saturation_mg_per_L": [round(float(v), 4) for v in do],
        },
        ensure_ascii=False,
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(mcp.run_stdio_async())
