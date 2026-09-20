#!/usr/bin/env python3
"""Cell biology MCP server.

Tools ported from
SciAgentGYM-main/toolkits/life_science/cell_biology/
  - mitochondrial_function_toolkit_claude_88.py       (ATP / complex assays)
  - embryonic_stem_cell_enhancer_toolkit_claude_47.py (Polycomb / 3D genome)
  - chipseq_epigenetics_toolkit_claude_33.py          (ChIP-seq peaks)

All implementations are stdlib + numpy only. Every tool returns a JSON
string. Tool naming is snake_case.
"""

from __future__ import annotations

import json
import math
import statistics
from typing import Any, Dict, List

import numpy as np

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("biology-cell")


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

_VIZ_DIR = "/tmp/scimas_biology_cell"


# ---------------------------------------------------------------------------
# Mitochondrial function tools
# ---------------------------------------------------------------------------

@mcp.tool(description=(
    "Calculate ATP concentration (mM) from luminescence readings using a "
    "linear standard curve y = m * ATP + b."
))
def calculate_atp_concentration(luminescence_values: List[float],
                                  slope: float = 1000.0,
                                  intercept: float = 50.0) -> str:
    try:
        if not luminescence_values:
            raise ValueError("luminescence_values must be non-empty")
        if slope <= 0:
            raise ValueError("slope must be > 0")
        atps = [(lum - intercept) / slope for lum in luminescence_values]
        mean_atp = statistics.mean(atps)
        return _ok(
            "calculate_atp_concentration",
            n=len(luminescence_values),
            mean_atp_mM=round(mean_atp, 4),
            std_atp_mM=round(statistics.pstdev(atps), 4) if len(atps) > 1 else 0.0,
            values_mM=[round(v, 4) for v in atps],
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Compute ETC complex activity (U/mg protein) from absorbance kinetics "
    "(initial vs final). Returns mean and per-sample activities."
))
def calculate_complex_activity(absorbance_data: List[float],
                                time_seconds: float,
                                protein_mg: float,
                                extinction_coefficient: float = 6.22) -> str:
    try:
        if not absorbance_data:
            raise ValueError("absorbance_data must be non-empty")
        if time_seconds <= 0 or protein_mg <= 0:
            raise ValueError("time and protein must be > 0")
        if extinction_coefficient <= 0:
            raise ValueError("extinction_coefficient must be > 0")
        activities = []
        for delta in absorbance_data:
            # Activity = (ΔA × 1000) / (ε × t × mg)
            act = (delta * 1000.0) / (extinction_coefficient * time_seconds * protein_mg)
            activities.append(act)
        return _ok(
            "calculate_complex_activity",
            n=len(absorbance_data),
            mean_activity_U_per_mg=round(statistics.mean(activities), 4),
            std_activity_U_per_mg=round(statistics.pstdev(activities), 4)
                if len(activities) > 1 else 0.0,
            values=[round(v, 4) for v in activities],
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Assess whether a glucose-uptake assay method is appropriate given the "
    "experimental context (e.g. 2-NBDG vs. fluorescent tracer)."
))
def assess_glucose_uptake_relevance(method_name: str,
                                    cell_type: str,
                                    research_question: str = "general_metabolism") -> str:
    try:
        relevance_score = {
            "2-NBDG": 0.85,
            "FITC-glucose": 0.7,
            "C13-glucose_tracer": 0.95,
            "luciferase_ATP": 0.6,
            "seahorse_glycolysis": 0.95,
        }.get(method_name, 0.5)
        appropriate = relevance_score >= 0.7
        return _ok(
            "assess_glucose_uptake_relevance",
            method=method_name,
            cell_type=cell_type,
            relevance_score=relevance_score,
            appropriate=appropriate,
            research_question=research_question,
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Comprehensive mitochondrial assessment combining ATP, complex activity, "
    "and glucose-uptake data into a single health score (0–1)."
))
def comprehensive_mitochondrial_assessment(atp_data: Dict[str, float],
                                            complex_activities: Dict[str, float],
                                            glucose_uptake: float) -> str:
    try:
        if atp_data is None or complex_activities is None:
            raise ValueError("atp_data and complex_activities must be provided")
        # Normalise inputs.
        norm_atp = min(1.0, atp_data.get("mean_atp_mM", 0.0) / 5.0)
        norm_complex = min(1.0, statistics.mean(complex_activities.values()) / 1.0
                           if complex_activities else 0.0)
        norm_glc = min(1.0, glucose_uptake / 10.0)
        health = 0.4 * norm_atp + 0.4 * norm_complex + 0.2 * norm_glc
        return _ok(
            "comprehensive_mitochondrial_assessment",
            atp=atp_data,
            complex_activities=complex_activities,
            glucose_uptake=glucose_uptake,
            mitochondrial_health_score=round(health, 4),
            interpretation=("healthy" if health > 0.7 else
                             "compromised" if health > 0.4 else
                             "dysfunctional"),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Validate an experimental design list: each item has 'name' and 'type' "
    "(atp|complex_activity|glucose_uptake|control). Returns a sanity-check "
    "summary."
))
def experimental_design_validator(proposed_experiments: List[Dict[str, str]]) -> str:
    try:
        if not proposed_experiments:
            raise ValueError("proposed_experiments is empty")
        types = [e.get("type", "") for e in proposed_experiments]
        n_control = sum(1 for t in types if "control" in t)
        issues = []
        if n_control == 0:
            issues.append("no control experiment")
        if "atp" not in types:
            issues.append("missing ATP measurement")
        if "complex_activity" not in types:
            issues.append("missing ETC complex activity")
        return _ok(
            "experimental_design_validator",
            n_experiments=len(proposed_experiments),
            n_controls=n_control,
            has_atp="atp" in types,
            has_complex_activity="complex_activity" in types,
            issues=issues,
            valid=(not issues),
        )
    except Exception as exc:
        return _err(str(exc))


# ---------------------------------------------------------------------------
# Stem cell enhancer / Polycomb tools
# ---------------------------------------------------------------------------

@mcp.tool(description=(
    "Return a mock Polycomb-mediated contact enrichment score for a gene."
))
def calculate_polycomb_enrichment_score(gene_name: str,
                                         contact_strength: float = 1.0,
                                         is_polycomb_target: bool = True) -> str:
    try:
        base = contact_strength if contact_strength > 0 else 1.0
        multiplier = 2.5 if is_polycomb_target else 1.0
        score = base * multiplier
        return _ok(
            "calculate_polycomb_enrichment_score",
            gene=gene_name,
            contact_strength=contact_strength,
            polycomb_target=is_polycomb_target,
            enrichment_score=round(score, 4),
            label=("highly-enriched" if score > 2 else "enriched" if score > 1 else "background"),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Compute distance distribution statistics for a list of enhancer–"
    "promoter pair distances (in kb)."
))
def analyze_enhancer_promoter_distance_distribution(distances_kb: List[float]) -> str:
    try:
        if not distances_kb:
            raise ValueError("distances_kb is empty")
        arr = np.array(distances_kb, dtype=float)
        return _ok(
            "analyze_enhancer_promoter_distance_distribution",
            n=len(arr),
            mean_kb=round(float(arr.mean()), 3),
            median_kb=round(float(np.median(arr)), 3),
            std_kb=round(float(arr.std()), 3),
            min_kb=round(float(arr.min()), 3),
            max_kb=round(float(arr.max()), 3),
            q1_kb=round(float(np.quantile(arr, 0.25)), 3),
            q3_kb=round(float(np.quantile(arr, 0.75)), 3),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Simulate effect of knocking out a Polycomb complex (PRC1 / PRC2) on a "
    "list of target-gene expression."
))
def simulate_polycomb_knockout_effect(knockout_complex: str,
                                       target_genes: List[str]) -> str:
    try:
        if knockout_complex not in ("PRC1", "PRC2"):
            raise ValueError("knockout_complex must be PRC1 or PRC2")
        # Mock fold-changes: PRC2 KO → 4× activation; PRC1 KO → 2× activation.
        base = {"PRC1": 2.0, "PRC2": 4.0}[knockout_complex]
        effects = [{"gene": g, "fold_change": base} for g in target_genes]
        return _ok(
            "simulate_polycomb_knockout_effect",
            knockout_complex=knockout_complex,
            target_genes=effects,
            mean_fold_change=base,
        )
    except Exception as exc:
        return _err(str(exc))


# ---------------------------------------------------------------------------
# ChIP-seq / epigenetics tools
# ---------------------------------------------------------------------------

@mcp.tool(description=(
    "Estimate ChIP-seq crosslinking efficiency given fixation method and "
    "interaction type (returns efficiency 0–1)."
))
def calculate_crosslinking_efficiency(fixation_method: str,
                                       interaction_type: str) -> str:
    try:
        efficiency = {
            ("formaldehyde", "transcription_factor"): 0.85,
            ("formaldehyde", "histone"): 0.90,
            ("formaldehyde", "polymerase"): 0.80,
            ("DSG+formaldehyde", "transcription_factor"): 0.92,
            ("DSG+formaldehyde", "histone"): 0.94,
            ("ethanol", "transcription_factor"): 0.40,
            ("ethanol", "histone"): 0.50,
        }.get((fixation_method, interaction_type), 0.6)
        return _ok(
            "calculate_crosslinking_efficiency",
            fixation=fixation_method,
            interaction=interaction_type,
            efficiency=round(efficiency, 4),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Simulate a ChIP-seq peak profile: returns n peaks with mock fold-"
    "enrichment above background."
))
def simulate_chipseq_signal(region_type: str,
                            tf_binding_mode: str = "sharp",
                            n_peaks: int = 1000,
                            noise_level: float = 0.1) -> str:
    try:
        if n_peaks <= 0:
            raise ValueError("n_peaks must be > 0")
        if noise_level < 0:
            raise ValueError("noise_level must be ≥ 0")
        rng = np.random.default_rng(42)
        if tf_binding_mode == "sharp":
            enrichments = np.clip(rng.normal(8.0, 2.0, n_peaks), 1.0, None)
        else:  # broad / spread
            enrichments = np.clip(rng.normal(3.0, 1.5, n_peaks), 1.0, None)
        enrichments = enrichments + rng.normal(0, noise_level, n_peaks)
        return _ok(
            "simulate_chipseq_signal",
            region_type=region_type,
            binding_mode=tf_binding_mode,
            n_peaks=n_peaks,
            mean_enrichment=round(float(enrichments.mean()), 4),
            median_enrichment=round(float(np.median(enrichments)), 4),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Predict peak locations under different crosslinking conditions."
))
def predict_peak_locations(fixation_method: str, n_peaks: int = 100) -> str:
    try:
        if n_peaks <= 0:
            raise ValueError("n_peaks must be > 0")
        rng = np.random.default_rng(42)
        # Fixation quality varies the number of 'sharp' peaks recovered.
        fixation_factor = {
            "formaldehyde": 0.85,
            "DSG+formaldehyde": 0.95,
            "ethanol": 0.40,
        }.get(fixation_method, 0.6)
        expected = int(n_peaks * fixation_factor)
        peaks = sorted(rng.integers(0, 1_000_000, expected).tolist())
        return _ok(
            "predict_peak_locations",
            fixation=fixation_method,
            n_predicted_peaks=expected,
            example_positions=peaks[:10],
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Identify 'disappearing peaks' between two ChIP-seq conditions: returns "
    "the count of significantly reduced peaks (mock heuristic)."
))
def analyze_disappearing_peaks(condition_a_counts: List[float],
                                condition_b_counts: List[float],
                                fc_threshold: float = 2.0) -> str:
    try:
        if len(condition_a_counts) != len(condition_b_counts):
            raise ValueError("count lists must be same length")
        if fc_threshold <= 0:
            raise ValueError("fc_threshold must be > 0")
        disappearing = []
        for i, (a, b) in enumerate(zip(condition_a_counts, condition_b_counts)):
            if a <= 0 or b <= 0:
                continue
            if a / b >= fc_threshold:
                disappearing.append(i)
        return _ok(
            "analyze_disappearing_peaks",
            n_peaks=len(condition_a_counts),
            n_disappearing=len(disappearing),
            fc_threshold=fc_threshold,
        )
    except Exception as exc:
        return _err(str(exc))


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

@mcp.tool(description=(
    "Plot mitochondrial ATP / complex-activity bar chart. Returns path to "
    "saved PNG."
))
def plot_mitochondrial_assays(labels: List[str],
                                 values: List[float],
                                 units: str = "a.u.") -> str:
    if not _MPL_OK:
        return _err("matplotlib not available")
    if len(labels) != len(values) or not labels:
        return _err("labels and values length mismatch")
    import os
    os.makedirs(_VIZ_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values, color="tab:green")
    ax.set_ylabel(units)
    ax.set_title("Mitochondrial functional assays")
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    out = f"{_VIZ_DIR}/mito.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return _ok("plot_mitochondrial_assays", path=out, n=len(labels))


@mcp.tool(description=(
    "Plot a ChIP-seq peak profile: enrichment vs genomic position. "
    "Returns path to saved PNG."
))
def plot_chipseq_profile(positions: List[float],
                            enrichments: List[float]) -> str:
    if not _MPL_OK:
        return _err("matplotlib not available")
    if len(positions) != len(enrichments) or not positions:
        return _err("positions and enrichments length mismatch")
    import os
    os.makedirs(_VIZ_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.fill_between(positions, enrichments, color="tab:blue", alpha=0.7)
    ax.set_xlabel("genomic position")
    ax.set_ylabel("enrichment")
    ax.set_title("ChIP-seq peak profile")
    ax.grid(True, alpha=0.3)
    out = f"{_VIZ_DIR}/chipseq.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return _ok("plot_chipseq_profile", path=out, n=len(positions))


if __name__ == "__main__":
    import asyncio
    asyncio.run(mcp.run_stdio_async())
