#!/usr/bin/env python3
"""Structural biology MCP server.

Tools ported from
SciAgentGYM-main/toolkits/life_science/structural_biology/
  - rna_structure_toolkit_0160002.py                (RNA secondary structure)
  - structural_biology_toolkit_claude_24.py        (mutation / binding-site)
  - structural_biology_toolkit_0090002.py          (PDB parsing helpers)
SciAgentGYM-main/toolkits/life_science/protein_structure_analysis/
  - sars_cov2_molecular_biology_toolkit_claude_25.py (nsp complexes)

All implementations are stdlib + numpy only. Every tool returns a JSON
string. Tool naming is snake_case.
"""

from __future__ import annotations

import json
import math
import statistics
from typing import Any, Dict, List, Tuple

import numpy as np

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("biology-structural")


# ---------------------------------------------------------------------------
# Reference tables
# ---------------------------------------------------------------------------

# BLOSUM62-subset for substitution scoring (just identity-aware heuristic).
_AA_HYDROPATHY: Dict[str, float] = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

_AA_VOLUME: Dict[str, float] = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5,
    "Q": 143.8, "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7,
    "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7,
    "S": 89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0,
}

_AA_CHARGE: Dict[str, float] = {
    "R": 1.0, "K": 1.0, "H": 0.5,
    "D": -1.0, "E": -1.0,
    "A": 0.0, "N": 0.0, "C": 0.0, "Q": 0.0, "G": 0.0,
    "I": 0.0, "L": 0.0, "M": 0.0, "F": 0.0, "P": 0.0,
    "S": 0.0, "T": 0.0, "W": 0.0, "Y": 0.0, "V": 0.0,
}

# Mock SARS-CoV-2 nsp database.
_NSP_DB: Dict[str, Dict[str, Any]] = {
    "nsp1":  {"function": "host mRNA degradation", "interactors": ["40S ribosomal subunit"]},
    "nsp3":  {"function": "PLpro protease", "interactors": ["nsp4", "nsp6"]},
    "nsp5":  {"function": "3CLpro main protease", "interactors": ["nsp12", "nsp13"]},
    "nsp12": {"function": "RNA-dependent RNA polymerase", "interactors": ["nsp7", "nsp8", "nsp13"]},
    "nsp13": {"function": "helicase", "interactors": ["nsp12"]},
    "nsp15": {"function": "endoRNAse", "interactors": ["nsp10"]},
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

_VIZ_DIR = "/tmp/scimas_biology_structural"


# ---------------------------------------------------------------------------
# RNA structure tools
# ---------------------------------------------------------------------------

@mcp.tool(description=(
    "Parse and validate an RNA sequence: returns length, nucleotide "
    "counts, and whether all bases are valid (ACGU)."
))
def parse_rna_sequence(sequence: str, validate: bool = True) -> str:
    try:
        seq = sequence.upper().replace("\n", "").replace(" ", "").replace("T", "U")
        invalid = [i for i, c in enumerate(seq) if c not in "ACGU"]
        if validate and invalid:
            raise ValueError(f"invalid RNA bases at positions {invalid[:10]} "
                             f"(showing first 10)")
        counts = {b: seq.count(b) for b in "ACGU"}
        return _ok(
            "parse_rna_sequence",
            length=len(seq),
            counts=counts,
            gc_content=round((counts["G"] + counts["C"]) / max(1, len(seq)), 4),
            valid=(not invalid) if validate else None,
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Detect simple Watson-Crick base pairs (A-U, G-C) in a sequence using "
    "an O(n^2) check for nested stems (min_stem_length ≥ 3)."
))
def detect_base_pairs(sequence: str, min_stem_length: int = 3) -> str:
    try:
        seq = sequence.upper().replace("T", "U")
        pairs: List[Tuple[int, int]] = []
        wc = {"A": "U", "U": "A", "G": "C", "C": "G"}
        for i in range(len(seq)):
            for j in range(i + min_stem_length + 2, len(seq)):
                if wc.get(seq[i]) == seq[j]:
                    pairs.append((i, j))
        return _ok(
            "detect_base_pairs",
            n_pairs=len(pairs),
            min_stem_length=min_stem_length,
            pairs=pairs[:20],
            has_many_pairs=len(pairs) > 5,
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Compute RNA structure complexity: number of pairs, mean stem length, "
    "and approximate free-energy estimate (-kcal/mol heuristic)."
))
def calculate_structure_complexity(pairs: List[Tuple[int, int]]) -> str:
    try:
        if not pairs:
            return _ok("calculate_structure_complexity",
                       n_pairs=0, mean_stem=0.0, complexity_score=0.0,
                       approximate_dG_kcal=0.0)
        stems = [j - i - 1 for (i, j) in pairs]
        mean_stem = statistics.mean(stems)
        # Heuristic free-energy estimate: -1.5 kcal/mol per stacked pair.
        dG = -1.5 * len(pairs)
        complexity = mean_stem / max(1, max(stems))
        return _ok(
            "calculate_structure_complexity",
            n_pairs=len(pairs),
            mean_stem_length=round(mean_stem, 3),
            complexity_score=round(complexity, 3),
            approximate_dG_kcal_per_mol=round(dG, 3),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Classify an RNA sequence into rRNA / tRNA / mRNA / miRNA based on "
    "length heuristic."
))
def classify_rna_type(sequence: str) -> str:
    try:
        seq = sequence.upper().replace("T", "U").replace("\n", "")
        n = len(seq)
        if n < 30:
            rna_type = "miRNA/small RNA"
        elif n < 100:
            rna_type = "tRNA"
        elif n < 500:
            rna_type = "small rRNA / snRNA"
        else:
            rna_type = "mRNA / large rRNA"
        return _ok(
            "classify_rna_type",
            length=n,
            rna_type=rna_type,
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Predict catalytic activity (ribozyme-like) from sequence length and "
    "GC content heuristic. Returns a 0–1 score."
))
def predict_catalytic_activity(sequence: str) -> str:
    try:
        seq = sequence.upper().replace("T", "U").replace("\n", "")
        gc = (seq.count("G") + seq.count("C")) / max(1, len(seq))
        score = min(1.0, 0.5 * gc + 0.005 * len(seq) / 100.0)
        return _ok(
            "predict_catalytic_activity",
            length=len(seq),
            gc_content=round(gc, 4),
            catalytic_score=round(score, 4),
            likely_catalytic=score > 0.5,
        )
    except Exception as exc:
        return _err(str(exc))


# ---------------------------------------------------------------------------
# Mutation / binding-site tools
# ---------------------------------------------------------------------------

@mcp.tool(description=(
    "Return biochemical properties (hydropathy, volume, charge) for a "
    "single-letter amino acid code."
))
def get_amino_acid_properties(residue: str) -> str:
    try:
        r = residue.strip().upper()
        if len(r) != 1 or r not in _AA_HYDROPATHY:
            raise ValueError("residue must be a single-letter AA code")
        return _ok(
            "get_amino_acid_properties",
            residue=r,
            hydropathy=_AA_HYDROPATHY[r],
            volume_A3=_AA_VOLUME[r],
            charge=_AA_CHARGE[r],
            hydrophobic=_AA_HYDROPATHY[r] > 1.0,
            charged=(_AA_CHARGE[r] != 0),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Heuristic mutation-effect score: Δhydropathy, Δvolume, Δcharge. "
    "Higher absolute score = larger predicted functional impact."
))
def calculate_mutation_effect_score(original: str, mutant: str) -> str:
    try:
        if original not in _AA_HYDROPATHY or mutant not in _AA_HYDROPATHY:
            raise ValueError("unknown amino acid code")
        d_hyd = _AA_HYDROPATHY[mutant] - _AA_HYDROPATHY[original]
        d_vol = _AA_VOLUME[mutant] - _AA_VOLUME[original]
        d_chg = _AA_CHARGE[mutant] - _AA_CHARGE[original]
        impact = abs(d_hyd) * 0.4 + abs(d_vol) * 0.01 + abs(d_chg) * 1.0
        return _ok(
            "calculate_mutation_effect_score",
            original=original,
            mutant=mutant,
            delta_hydropathy=round(d_hyd, 3),
            delta_volume=round(d_vol, 3),
            delta_charge=round(d_chg, 3),
            impact_score=round(impact, 3),
            impactful=impact > 1.0,
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Score a binding-site residue at a given position by combining "
    "hydropathy, charge, and a position-weighting factor."
))
def analyze_binding_site_residue(residue: str, position: int) -> str:
    try:
        if residue not in _AA_HYDROPATHY:
            raise ValueError("unknown amino acid")
        pos_weight = 1.0 + 0.1 * abs(position)
        score = (_AA_HYDROPATHY[residue] * 0.3 +
                 abs(_AA_CHARGE[residue]) * 0.5 +
                 _AA_VOLUME[residue] * 0.005) * pos_weight
        return _ok(
            "analyze_binding_site_residue",
            residue=residue,
            position=position,
            score=round(score, 3),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Rank a list of candidate point-mutations by predicted functional "
    "impact (largest impact first)."
))
def compare_mutation_candidates(mutations: List[str]) -> str:
    try:
        if not mutations:
            raise ValueError("mutations list is empty")
        scored = []
        for mut in mutations:
            # Mutation format: "X123Y" (original, position, mutant).
            if len(mut) < 4 or not mut[1:-1].isdigit():
                raise ValueError(f"bad mutation format: {mut}")
            orig = mut[0].upper()
            new = mut[-1].upper()
            if orig not in _AA_HYDROPATHY or new not in _AA_HYDROPATHY:
                raise ValueError(f"unknown AA in {mut}")
            impact = (abs(_AA_HYDROPATHY[new] - _AA_HYDROPATHY[orig]) * 0.4 +
                      abs(_AA_VOLUME[new] - _AA_VOLUME[orig]) * 0.01 +
                      abs(_AA_CHARGE[new] - _AA_CHARGE[orig]) * 1.0)
            scored.append({"mutation": mut, "impact": round(impact, 3)})
        scored.sort(key=lambda x: x["impact"], reverse=True)
        return _ok(
            "compare_mutation_candidates",
            ranked=scored,
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Compute mean and spread of hydropathy / charge / volume for a list of "
    "active-site residues."
))
def analyze_active_site_composition(residues: List[str]) -> str:
    try:
        if not residues:
            raise ValueError("residues list is empty")
        for r in residues:
            if r not in _AA_HYDROPATHY:
                raise ValueError(f"unknown residue {r}")
        hydro = [_AA_HYDROPATHY[r] for r in residues]
        charge = [_AA_CHARGE[r] for r in residues]
        volume = [_AA_VOLUME[r] for r in residues]
        return _ok(
            "analyze_active_site_composition",
            residues=residues,
            mean_hydropathy=round(statistics.mean(hydro), 3),
            mean_charge=round(statistics.mean(charge), 3),
            mean_volume=round(statistics.mean(volume), 3),
            n_charged=sum(1 for c in charge if c != 0),
            n_hydrophobic=sum(1 for h in hydro if h > 1.0),
        )
    except Exception as exc:
        return _err(str(exc))


# ---------------------------------------------------------------------------
# PDB parsing / composition helpers (no network, mock-friendly).
# ---------------------------------------------------------------------------

@mcp.tool(description=(
    "Parse a PDB text block and return atom / residue counts. Designed for "
    "small files (no real network fetching)."
))
def parse_pdb_structure(pdb_text: str, remove_hydrogen: bool = True) -> str:
    try:
        atoms = []
        residues = set()
        chains = set()
        hetatms = 0
        for line in pdb_text.splitlines():
            record = line[:6].strip()
            if record in ("ATOM", "HETATM"):
                if remove_hydrogen and len(line) > 12 and line[12] == "H" and \
                        len(line) > 76 and line[76] == "H":
                    continue
                if record == "HETATM":
                    hetatms += 1
                atoms.append(record)
                if len(line) > 21:
                    residues.add(line[17:20].strip())
                if len(line) > 21:
                    chains.add(line[21])
        return _ok(
            "parse_pdb_structure",
            n_atoms=len(atoms),
            n_hetatms=hetatms,
            n_unique_residues=len(residues),
            n_chains=len(chains),
            residues=sorted(residues)[:25],
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Classify a PDB residue as protein / ligand / water / ion based on its "
    "name and HETATM flag."
))
def classify_residue_type(residue_name: str, is_hetatm: bool = False) -> str:
    try:
        r = residue_name.strip().upper()
        if r in ("HOH", "WAT"):
            cls = "water"
        elif r in ("NA", "K", "CL", "MG", "CA", "ZN", "FE"):
            cls = "ion"
        elif is_hetatm:
            cls = "ligand"
        else:
            cls = "protein"
        return _ok(
            "classify_residue_type",
            residue=r,
            residue_class=cls,
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Count ligands vs. protein residues in a PDB text block."
))
def count_ligand_chains(pdb_text: str) -> str:
    try:
        ligands = 0
        protein = 0
        waters = 0
        for line in pdb_text.splitlines():
            rec = line[:6].strip()
            if rec != "ATOM" and rec != "HETATM":
                continue
            name = line[17:20].strip().upper()
            if rec == "HETATM":
                if name in ("HOH", "WAT"):
                    waters += 1
                else:
                    ligands += 1
            else:
                protein += 1
        return _ok(
            "count_ligand_chains",
            n_protein_atoms=protein,
            n_ligand_atoms=ligands,
            n_water_atoms=waters,
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Build a per-chain residue / atom composition summary from a PDB text "
    "block."
))
def analyze_structure_composition(pdb_text: str) -> str:
    try:
        summary: Dict[str, Dict[str, int]] = {}
        for line in pdb_text.splitlines():
            rec = line[:6].strip()
            if rec not in ("ATOM", "HETATM"):
                continue
            chain = line[21] if len(line) > 21 else "?"
            res = line[17:20].strip()
            summary.setdefault(chain, {"atoms": 0, "residues": set()})
            summary[chain]["atoms"] += 1
            summary[chain]["residues"].add(res)
        out = {c: {"atoms": v["atoms"], "n_residues": len(v["residues"])}
               for c, v in summary.items()}
        return _ok("analyze_structure_composition", chains=out)
    except Exception as exc:
        return _err(str(exc))


# ---------------------------------------------------------------------------
# SARS-CoV-2 nsp helpers
# ---------------------------------------------------------------------------

@mcp.tool(description=(
    "Look up SARS-CoV-2 non-structural protein (nsp) function and known "
    "interactors."
))
def query_nsp_protein_info(protein_name: str) -> str:
    try:
        p = protein_name.strip().lower()
        if p not in _NSP_DB:
            return _err(f"Unknown nsp '{protein_name}'",
                        known=list(_NSP_DB.keys()))
        info = _NSP_DB[p]
        return _ok(
            "query_nsp_protein_info",
            protein=p,
            function=info["function"],
            interactors=info["interactors"],
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Predict whether two nsps form a complex (mock heuristic: shared "
    "interactor or polymerase complex)."
))
def analyze_protein_complex_formation(protein1: str, protein2: str) -> str:
    try:
        p1, p2 = protein1.strip().lower(), protein2.strip().lower()
        if p1 not in _NSP_DB or p2 not in _NSP_DB:
            raise ValueError("unknown nsp")
        i1 = set(_NSP_DB[p1]["interactors"])
        i2 = set(_NSP_DB[p2]["interactors"])
        shared = i1 & i2
        # Direct interactors each other.
        direct = p2 in i1 or p1 in i2
        forms_complex = bool(shared) or direct
        return _ok(
            "analyze_protein_complex_formation",
            protein1=p1,
            protein2=p2,
            shared_interactors=sorted(shared),
            direct_interaction=direct,
            forms_complex=forms_complex,
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Comprehensive nsp analysis for a list of target proteins: returns "
    "function and combined interactor set."
))
def comprehensive_nsp_analysis(target_proteins: List[str]) -> str:
    try:
        results = []
        all_interactors = set()
        for p in target_proteins:
            pl = p.lower()
            if pl not in _NSP_DB:
                raise ValueError(f"unknown nsp {p}")
            info = _NSP_DB[pl]
            all_interactors.update(info["interactors"])
            results.append({"protein": pl, "function": info["function"]})
        return _ok(
            "comprehensive_nsp_analysis",
            target_proteins=results,
            combined_interactors=sorted(all_interactors),
        )
    except Exception as exc:
        return _err(str(exc))


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

@mcp.tool(description=(
    "Plot a mutation-impact ranked bar chart. Returns path to saved PNG."
))
def plot_mutation_impact_ranking(mutations: List[str],
                                    impacts: List[float]) -> str:
    if not _MPL_OK:
        return _err("matplotlib not available")
    if len(mutations) != len(impacts) or not mutations:
        return _err("mutations and impacts length mismatch")
    import os
    os.makedirs(_VIZ_DIR, exist_ok=True)
    order = sorted(range(len(impacts)), key=lambda i: -impacts[i])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([mutations[i] for i in order], [impacts[i] for i in order],
            color="tab:purple")
    ax.set_ylabel("predicted impact")
    ax.set_title("Mutation candidates ranked by impact")
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    out = f"{_VIZ_DIR}/mutation_impact.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return _ok("plot_mutation_impact_ranking", path=out, n=len(mutations))


@mcp.tool(description=(
    "Plot an RNA secondary-structure arc diagram from a base-pair list. "
    "Returns path to saved PNG."
))
def plot_rna_arc_diagram(sequence: str,
                              pairs: List[Tuple[int, int]]) -> str:
    if not _MPL_OK:
        return _err("matplotlib not available")
    if not sequence:
        return _err("sequence empty")
    import os
    os.makedirs(_VIZ_DIR, exist_ok=True)
    n = len(sequence)
    fig, ax = plt.subplots(figsize=(10, 3))
    xs = list(range(n))
    ys = [0] * n
    ax.plot(xs, ys, "o", color="tab:blue", markersize=4)
    for i, j in pairs:
        if 0 <= i < n and 0 <= j < n:
            arc_y = (j - i) / 2
            ax.plot([i, i, j, j], [0, arc_y, arc_y, 0], "k-", alpha=0.5)
    ax.set_xlabel("position")
    ax.set_yticks([])
    ax.set_title(f"RNA arc diagram ({len(pairs)} pairs, length {n})")
    ax.grid(True, alpha=0.3, axis="x")
    out = f"{_VIZ_DIR}/rna_arc.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return _ok("plot_rna_arc_diagram", path=out, n_pairs=len(pairs))


if __name__ == "__main__":
    import asyncio
    asyncio.run(mcp.run_stdio_async())
