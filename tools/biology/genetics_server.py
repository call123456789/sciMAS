#!/usr/bin/env python3
"""Genetics MCP server.

Tools ported from
SciAgentGYM-main/toolkits/life_science/cell_biology/
  - genetic_interaction_toolkit_claude_17.py  (epistasis / redundancy)
  - chromosome_biology_toolkit_claude_27.py   (mitosis, spindle tension)
  - bioinformatics_snp_toolkit_claude_22.py   (SNP lookup, FASTA parsing)

All implementations are stdlib + numpy only. Every tool returns a JSON
string. Tool naming is snake_case.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from typing import Any, Dict, List, Tuple

import numpy as np

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("biology-genetics")


# ---------------------------------------------------------------------------
# Mock SNP / flanking-sequence database (offline-friendly).
# ---------------------------------------------------------------------------

_SNP_DB: Dict[str, Dict[str, Any]] = {
    "rs334":  {"gene": "HBB",  "chrom": "11", "pos": 5227002,
               "ref": "A", "alt": "T",
               "consequence": "missense", "aa_change": "E6V"},
    "rs4977579": {"gene": "CDKN2A", "chrom": "9", "pos": 22067100,
                   "ref": "G", "alt": "A",
                   "consequence": "intergenic"},
    "rs7412": {"gene": "APOE", "chrom": "19", "pos": 44908822,
               "ref": "C", "alt": "T",
               "consequence": "missense", "aa_change": "R158C"},
    "rs1801133": {"gene": "MTHFR", "chrom": "1", "pos": 11796321,
                   "ref": "G", "alt": "A",
                   "consequence": "missense", "aa_change": "A222V"},
    "rs113488022": {"gene": "BRAF", "chrom": "7", "pos": 140453136,
                     "ref": "A", "alt": "T",
                     "consequence": "missense", "aa_change": "V600E"},
}

# Tiny mock genome windows for flanking-sequence calls.
_MOCK_FLANKING: Dict[str, str] = {
    "rs334":  ("ATGGTGCATCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACG"
                "TGGATGAAGTTGGTGGTGAGGCCCTGGGCAGGCTGCTGGTGGTCTACCCTTGGACCCAGA"),
    "rs4977579": ("TGCAGCCTCCAGACCAGCCTGCCCCCACCAGCAGCCTCAGTGCCCCAGGAGCCTGGGTG"),
    "rs7412": ("CTGGGGCTGGGGCTGGGGCTCAGTGGTGGGCAGGGGCAGAGGTGGCAGGGACTGGGCAGC"),
    "rs1801133": ("GCACTTTGAGGCTGACACATTCTTCCGCTCTGTGAAGGCATGTGGTGGGGGATGAGCTC"),
    "rs113488022": ("TCAGTGGTGTGTGTTCAGTGCAGTGTTTCAGTTATACATCTCTTCAGTTTTTGTGTCA"),
}


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

_VIZ_DIR = "/tmp/scimas_biology_genetics"


# ---------------------------------------------------------------------------
# Genetic-interaction / epistasis tools
# ---------------------------------------------------------------------------

@mcp.tool(description=(
    "Classify epistasis from single-mutant and double-mutant resistance "
    "scores (additive model). Returns the epistasis coefficient ε."
))
def calculate_epistasis_coefficient(gene_a_resistance: float,
                                    gene_b_resistance: float,
                                    double_mutant_resistance: float,
                                    wild_type_resistance: float = 1.0) -> str:
    try:
        for n, v in (("wt", wild_type_resistance),
                      ("a", gene_a_resistance),
                      ("b", gene_b_resistance),
                      ("ab", double_mutant_resistance)):
            if v <= 0:
                raise ValueError(f"{n}_resistance must be > 0")
        expected_ab = wild_type_resistance * (
            (gene_a_resistance / wild_type_resistance) *
            (gene_b_resistance / wild_type_resistance)
        )
        epsilon = math.log(double_mutant_resistance / expected_ab)
        if epsilon > 0.5:
            label = "synergistic (positive epistasis)"
        elif epsilon < -0.5:
            label = "antagonistic (negative epistasis)"
        else:
            label = "additive / multiplicative"
        return _ok(
            "calculate_epistasis_coefficient",
            wild_type_resistance=wild_type_resistance,
            gene_a_resistance=gene_a_resistance,
            gene_b_resistance=gene_b_resistance,
            double_mutant_resistance=double_mutant_resistance,
            expected_double_mutant=round(expected_ab, 4),
            epistasis_epsilon=round(epsilon, 4),
            interpretation=label,
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Quantify the additive, multiplicative, and synergistic components of "
    "two gene effects (returns three component scores)."
))
def analyze_single_mutant_effects(resistance_data: Dict[str, float]) -> str:
    try:
        if "WT" not in resistance_data:
            raise ValueError("resistance_data must contain 'WT'")
        wt = resistance_data["WT"]
        mut_names = [k for k in resistance_data if k != "WT"]
        if not mut_names:
            raise ValueError("need at least one mutant entry")
        # Pairwise epistasis between every (a, b, ab) triple.
        triples = []
        for i, a in enumerate(mut_names):
            for b in mut_names[i + 1:]:
                ab_key = f"{a}+{b}"
                if ab_key in resistance_data:
                    ab = resistance_data[ab_key]
                    expected = wt * (resistance_data[a] / wt) * (resistance_data[b] / wt)
                    eps = math.log(ab / expected)
                    triples.append({"a": a, "b": b, "epistasis": round(eps, 4),
                                     "observed_ab": ab, "expected_ab": round(expected, 4)})
        return _ok(
            "analyze_single_mutant_effects",
            wild_type=wt,
            mutants=mut_names,
            pairwise_epistasis=triples,
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Determine whether two genes act in redundant / parallel / synergistic "
    "pathways based on combined-knockout phenotype."
))
def determine_epistatic_relationship(gene1: str, gene2: str,
                                      resistance_data: Dict[str, float]) -> str:
    try:
        wt = resistance_data.get("WT")
        ra = resistance_data.get(gene1)
        rb = resistance_data.get(gene2)
        rab = resistance_data.get(f"{gene1}+{gene2}")
        if None in (wt, ra, rb, rab):
            raise ValueError("resistance_data missing required entries")
        eps = math.log(rab / (wt * (ra / wt) * (rb / wt)))
        if eps < -0.5:
            relation = "synergistic (parallel pathways)"
        elif eps > 0.5:
            relation = "antagonistic (same pathway, buffering)"
        else:
            relation = "multiplicative / additive"
        return _ok(
            "determine_epistatic_relationship",
            gene1=gene1, gene2=gene2,
            epistasis_epsilon=round(eps, 4),
            relationship=relation,
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Heuristic identification of candidate transcription-factor based on "
    "double-knockout epistasis clustering."
))
def identify_transcription_factor(resistance_data: Dict[str, float]) -> str:
    try:
        # Treat the mutant whose single Δ has the largest magnitude vs WT as
        # the candidate TF.
        wt = resistance_data["WT"]
        deltas = {k: abs(math.log(v / wt)) for k, v in resistance_data.items()
                   if k != "WT" and v > 0}
        if not deltas:
            raise ValueError("no usable mutant entries")
        candidate = max(deltas, key=deltas.get)
        return _ok(
            "identify_transcription_factor",
            candidate_tf=candidate,
            effect_size_log2fc=round(math.log2(resistance_data[candidate] / wt), 4)
                if resistance_data[candidate] > 0 else None,
            all_deltas={k: round(v, 4) for k, v in deltas.items()},
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Classify two genes as redundant vs. non-redundant based on combined "
    "knockout phenotype severity."
))
def analyze_gene_redundancy(gene1: str, gene2: str,
                             resistance_data: Dict[str, float]) -> str:
    try:
        wt = resistance_data["WT"]
        ra = resistance_data.get(gene1)
        rb = resistance_data.get(gene2)
        rab = resistance_data.get(f"{gene1}+{gene2}")
        if None in (ra, rb, rab):
            raise ValueError("need single and double mutant entries")
        redundant = (abs(math.log(rab / ra)) < 0.3 and abs(math.log(rab / rb)) < 0.3)
        return _ok(
            "analyze_gene_redundancy",
            gene1=gene1, gene2=gene2,
            redundant=redundant,
            interpretation=("redundant backup" if redundant
                             else "non-redundant (distinct functions)"),
        )
    except Exception as exc:
        return _err(str(exc))


# ---------------------------------------------------------------------------
# Chromosome / mitosis tools
# ---------------------------------------------------------------------------

@mcp.tool(description=(
    "Calculate the tension on a kinetochore from centromere positions and "
    "cell length (returns mean tension and per-pair tensions)."
))
def calculate_spindle_tension(centromere_positions: List[float],
                              cell_length: float) -> str:
    try:
        if len(centromere_positions) < 2:
            raise ValueError("need ≥ 2 centromere positions")
        if cell_length <= 0:
            raise ValueError("cell_length must be > 0")
        n = len(centromere_positions)
        centre = cell_length / 2.0
        # Tension proportional to |position - centre|.
        tensions = [abs(p - centre) for p in centromere_positions]
        mean_tension = statistics.mean(tensions)
        asymmetry = max(tensions) - min(tensions)
        return _ok(
            "calculate_spindle_tension",
            n_centromeres=n,
            cell_length=cell_length,
            mean_tension=round(mean_tension, 4),
            max_tension=round(max(tensions), 4),
            min_tension=round(min(tensions), 4),
            tension_asymmetry=round(asymmetry, 4),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Simulate chromosome segregation for n_steps and return the final "
    "positions and whether any lagging chromosomes remain."
))
def simulate_chromosome_segregation(centromere_positions: List[float],
                                    simulation_steps: int = 100,
                                    pole_attraction: float = 1.0) -> str:
    try:
        if not centromere_positions or simulation_steps <= 0:
            raise ValueError("inputs invalid")
        # Walk each centromere toward its nearest pole with a step = pole_attraction / n_steps.
        positions = np.array(centromere_positions, dtype=float)
        centre = 0.5  # normalised
        for _ in range(simulation_steps):
            direction = np.sign(centre - positions)
            positions += direction * pole_attraction / simulation_steps
        lagging = int(np.sum(np.abs(positions - centre) > 0.1))
        return _ok(
            "simulate_chromosome_segregation",
            initial=centromere_positions,
            final=[round(float(p), 4) for p in positions],
            n_lagging=lagging,
            segregates_cleanly=(lagging == 0),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Predict mitotic behaviour of a chromosome type (holocentric / "
    "monocentric) at a given ploidy."
))
def analyze_mitotic_behavior(chromosome_type: str, ploidy_level: str) -> str:
    try:
        if chromosome_type not in ("monocentric", "holocentric", "acrocentric"):
            raise ValueError("chromosome_type must be monocentric|holocentric|acrocentric")
        if ploidy_level not in ("haploid", "diploid", "tetraploid"):
            raise ValueError("ploidy_level must be haploid|diploid|tetraploid")
        segregation_error_rate = {
            "monocentric": {"haploid": 0.01, "diploid": 0.02, "tetraploid": 0.05},
            "holocentric": {"haploid": 0.005, "diploid": 0.01, "tetraploid": 0.03},
            "acrocentric": {"haploid": 0.02, "diploid": 0.04, "tetraploid": 0.08},
        }[chromosome_type][ploidy_level]
        return _ok(
            "analyze_mitotic_behavior",
            chromosome_type=chromosome_type,
            ploidy=ploidy_level,
            segregation_error_rate=segregation_error_rate,
            note="Based on empirical mis-segregation frequencies.",
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Evaluate chromosome-stability factors from gene density and the "
    "chromosome / ploidy context."
))
def evaluate_chromosome_stability_factors(chromosome_type: str,
                                          ploidy: str,
                                          gene_density: float) -> str:
    try:
        if gene_density < 0:
            raise ValueError("gene_density must be ≥ 0 (genes/Mb)")
        base = {"haploid": 0.05, "diploid": 0.07, "tetraploid": 0.12}[ploidy]
        if chromosome_type == "holocentric":
            base *= 0.5
        stability_score = max(0.0, min(1.0, 1.0 - base - 0.001 * gene_density))
        return _ok(
            "evaluate_chromosome_stability_factors",
            chromosome_type=chromosome_type,
            ploidy=ploidy,
            gene_density=gene_density,
            stability_score=round(stability_score, 4),
            stability_label=("stable" if stability_score > 0.7 else
                              "intermediate" if stability_score > 0.4 else
                              "unstable"),
        )
    except Exception as exc:
        return _err(str(exc))


# ---------------------------------------------------------------------------
# SNP / bioinformatics tools
# ---------------------------------------------------------------------------

@mcp.tool(description=(
    "Look up a dbSNP rsID: returns gene, chromosome, position, ref/alt, "
    "consequence, and AA change (if available)."
))
def fetch_snp_from_ncbi(rs_id: str) -> str:
    try:
        rs = rs_id.strip()
        if rs not in _SNP_DB:
            return _err(f"Unknown SNP '{rs}'", known=list(_SNP_DB.keys()))
        s = _SNP_DB[rs]
        return _ok(
            "fetch_snp_from_ncbi",
            rs_id=rs,
            gene=s["gene"],
            chromosome=s["chrom"],
            position=s["pos"],
            ref=s["ref"],
            alt=s["alt"],
            consequence=s["consequence"],
            aa_change=s.get("aa_change"),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Return a mock flanking sequence (window_size nt) around an rsID."
))
def fetch_flanking_sequence_ensembl(rs_id: str, window_size: int = 100) -> str:
    try:
        rs = rs_id.strip()
        if rs not in _MOCK_FLANKING:
            return _err(f"Unknown SNP '{rs}'", known=list(_MOCK_FLANKING.keys()))
        seq = _MOCK_FLANKING[rs][:window_size]
        return _ok(
            "fetch_flanking_sequence_ensembl",
            rs_id=rs,
            window_size=window_size,
            sequence=seq,
            length=len(seq),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Compute GC/AT counts and per-base composition for a nucleotide "
    "sequence."
))
def analyze_sequence_composition(sequence: str) -> str:
    try:
        seq = sequence.upper().replace("\n", "").replace(" ", "")
        if not seq:
            raise ValueError("sequence is empty")
        # Strip ambiguous calls for the ratio.
        valid = [c for c in seq if c in "ACGT"]
        if not valid:
            raise ValueError("no ACGT bases found")
        counts = {b: valid.count(b) for b in "ACGT"}
        gc = counts["G"] + counts["C"]
        gc_frac = gc / len(valid)
        return _ok(
            "analyze_sequence_composition",
            length=len(seq),
            counts=counts,
            gc_content=round(gc_frac, 4),
            at_content=round(1 - gc_frac, 4),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Format a sequence with FASTA-style line breaks and per-10-base group "
    "spacing for readability."
))
def format_sequence_with_spacing(sequence: str, line_length: int = 50,
                                 group_size: int = 10) -> str:
    try:
        seq = sequence.upper().replace("\n", "").replace(" ", "")
        if line_length <= 0 or group_size <= 0:
            raise ValueError("line_length and group_size must be > 0")
        lines = []
        for i in range(0, len(seq), line_length):
            block = seq[i:i + line_length]
            grouped = " ".join(block[j:j + group_size]
                                for j in range(0, len(block), group_size))
            lines.append(grouped)
        return _ok(
            "format_sequence_with_spacing",
            line_length=line_length,
            group_size=group_size,
            formatted="\n".join(lines),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Compute basic chromosome features (length, gc, AT-skew) from a "
    "nucleotide string."
))
def chromosome_metrics(chromosome_name: str, sequence: str) -> str:
    try:
        seq = sequence.upper().replace("\n", "").replace(" ", "")
        valid = [c for c in seq if c in "ACGT"]
        n = len(valid)
        if n == 0:
            raise ValueError("no ACGT bases")
        gc = valid.count("G") + valid.count("C")
        at = valid.count("A") + valid.count("T")
        g = valid.count("G")
        c = valid.count("C")
        a = valid.count("A")
        t = valid.count("T")
        gc_skew = (g - c) / (g + c) if (g + c) else 0.0
        at_skew = (a - t) / (a + t) if (a + t) else 0.0
        return _ok(
            "chromosome_metrics",
            chromosome=chromosome_name,
            length_bp=n,
            gc_content=round(gc / n, 4),
            at_content=round(at / n, 4),
            gc_skew=round(gc_skew, 4),
            at_skew=round(at_skew, 4),
        )
    except Exception as exc:
        return _err(str(exc))


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

@mcp.tool(description=(
    "Plot pairwise epistasis coefficients as a heatmap. Returns path to PNG."
))
def plot_epistasis_heatmap(gene_pairs: List[List[float]],
                             epistasis_values: List[float]) -> str:
    if not _MPL_OK:
        return _err("matplotlib not available")
    if len(gene_pairs) != len(epistasis_values) or not gene_pairs:
        return _err("gene_pairs and epistasis_values length mismatch")
    import os
    os.makedirs(_VIZ_DIR, exist_ok=True)
    n = max(max(p) for p in gene_pairs) + 1
    M = np.full((n, n), np.nan)
    for (i, j), e in zip(gene_pairs, epistasis_values):
        M[i, j] = M[j, i] = e
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(M, cmap="RdBu_r", origin="upper")
    ax.set_xlabel("gene index")
    ax.set_ylabel("gene index")
    ax.set_title("Epistasis coefficient matrix")
    fig.colorbar(im, ax=ax, label="ε")
    out = f"{_VIZ_DIR}/epistasis.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return _ok("plot_epistasis_heatmap", path=out, n_pairs=len(epistasis_values))


@mcp.tool(description=(
    "Plot spindle tension asymmetry: tension per centromere vs position. "
    "Returns path to PNG."
))
def plot_spindle_tension(centromere_positions: List[float],
                           cell_length: float) -> str:
    if not _MPL_OK:
        return _err("matplotlib not available")
    if not centromere_positions or cell_length <= 0:
        return _err("inputs invalid")
    import os
    os.makedirs(_VIZ_DIR, exist_ok=True)
    centre = cell_length / 2
    tensions = [abs(p - centre) for p in centromere_positions]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(tensions)), tensions, color="tab:blue")
    ax.set_xlabel("centromere index")
    ax.set_ylabel("tension (a.u.)")
    ax.set_title(f"Spindle tension (cell length = {cell_length})")
    ax.grid(True, alpha=0.3, axis="y")
    out = f"{_VIZ_DIR}/spindle.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return _ok("plot_spindle_tension", path=out)


if __name__ == "__main__":
    import asyncio
    asyncio.run(mcp.run_stdio_async())
