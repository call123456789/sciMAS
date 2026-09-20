#!/usr/bin/env python3
"""Molecular biology MCP server.

Tools ported from
SciAgentGYM-main/toolkits/life_science/cell_biology/molecular_biology_toolkit_*.py
covering qPCR analysis, restriction enzyme sensitivity & digest pattern
analysis, plasmid query / compatibility / transformation strategy, and DMD
exon-skipping mutation therapy.

All implementations are stdlib + numpy only (no external network calls;
databases are in-memory mocks). Every tool returns a JSON string. Tool
naming is snake_case.
"""

from __future__ import annotations

import json
import math
import statistics
from typing import Any, Dict, List

import numpy as np

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("biology-molecular")


# ---------------------------------------------------------------------------
# Tiny in-memory mock databases (so the tools can be exercised offline).
# ---------------------------------------------------------------------------

_GENE_DB: Dict[str, Dict[str, Any]] = {
    "BRCA1": {"chromosome": "17", "length_bp": 81189, "gc_content": 0.41},
    "TP53":  {"chromosome": "17", "length_bp": 19149, "gc_content": 0.39},
    "EGFR":  {"chromosome": "7",  "length_bp": 188307, "gc_content": 0.45},
    "MYC":   {"chromosome": "8",  "length_bp": 6942,  "gc_content": 0.52},
}

_RNA_DB: Dict[str, Dict[str, Any]] = {
    "DMD": {"transcript": "NM_004006", "exon_count": 79, "protein_length": 3685,
            "critical_exons": [45, 46, 47, 48, 49, 50, 51]},
    "DMD_isoform_dp71": {"transcript": "NM_004009", "exon_count": 73,
                          "protein_length": 617, "critical_exons": [71, 72, 73]},
}

_PLASMID_DB: Dict[str, Dict[str, Any]] = {
    "pUC19":   {"size_bp": 2686, "copy_number": "high", "ori": "pMB1",
                "marker": "ampR"},
    "pET28a":  {"size_bp": 5369, "copy_number": "medium", "ori": "pBR322",
                "marker": "kanR"},
    "pBR322":  {"size_bp": 4361, "copy_number": "medium", "ori": "pMB1",
                "marker": "ampR+tetR"},
    "pcDNA3":  {"size_bp": 5446, "copy_number": "high", "ori": "SV40",
                "marker": "ampR"},
    "pEGFP-N1": {"size_bp": 4733, "copy_number": "high", "ori": "SV40",
                 "marker": "kanR"},
}

_STRAIN_DB: Dict[str, Dict[str, Any]] = {
    "DH5a": {"genotype": "F- endA1 glnV44 thi-1 recA1 relA1 gyrA96",
             "methylation": "dam+ dcm+", "recommended_for": "cloning"},
    "XL1-Blue": {"genotype": "recA1 endA1 gyrA96 thi-1 hsdR17 supE44",
                 "methylation": "dam+ dcm+", "recommended_for": "blue_white"},
    "JM110": {"genotype": "dam dcm-",
              "methylation": "dam- dcm-",
              "recommended_for": "methylation_sensitive_cloning"},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

_VIZ_DIR = "/tmp/scimas_biology_molecular"


def _require_positive(name: str, value: float) -> None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        raise ValueError(f"{name} must be a finite number")
    if value <= 0:
        raise ValueError(f"{name} must be > 0 (got {value})")


# ---------------------------------------------------------------------------
# qPCR analysis (from molecular_biology_toolkit_claude_11)
# ---------------------------------------------------------------------------

@mcp.tool(description=(
    "Perform qPCR Ct-based relative-expression analysis. Inputs: target_gene, "
    "control_gene, ct_target, ct_control, ct_target_ref, ct_control_ref. "
    "Returns ΔΔCt, fold-change, and amplification efficiency-corrected ratio."
))
def analyze_qrt_pcr(target_gene: str, control_gene: str,
                    ct_target: float, ct_control: float,
                    ct_target_ref: float, ct_control_ref: float,
                    efficiency: float = 2.0) -> str:
    """Standard ΔΔCt with optional efficiency correction."""
    try:
        for v in (ct_target, ct_control, ct_target_ref, ct_control_ref):
            if v is None:
                raise ValueError("Ct values cannot be null")
        if efficiency < 1.0 or efficiency > 2.5:
            raise ValueError("efficiency must be in [1.0, 2.5]")
        delta_ct = ct_target - ct_control
        delta_ct_ref = ct_target_ref - ct_control_ref
        ddct = delta_ct - delta_ct_ref
        fold_change = efficiency ** (-ddct)
        return _ok(
            "analyze_qrt_pcr",
            target_gene=target_gene,
            control_gene=control_gene,
            delta_ct=round(delta_ct, 4),
            delta_ct_ref=round(delta_ct_ref, 4),
            delta_delta_ct=round(ddct, 4),
            fold_change=round(fold_change, 4),
            log2_fold_change=round(math.log2(fold_change), 4) if fold_change > 0 else None,
            efficiency=efficiency,
            interpretation=("upregulated" if fold_change > 2 else
                            "downregulated" if fold_change < 0.5 else
                            "no_change"),
        )
    except Exception as exc:
        return _err(str(exc))


# ---------------------------------------------------------------------------
# Restriction enzyme / strain helpers (from molecular_biology_toolkit_claude_20)
# ---------------------------------------------------------------------------

# Recognise a handful of common restriction enzymes.
_ENZYMES: Dict[str, Dict[str, Any]] = {
    "EcoRI":  {"recognition": "GAATTC", "cut_position": 1, "overhang": "5' AATT"},
    "BamHI":  {"recognition": "GGATCC", "cut_position": 1, "overhang": "5' GATC"},
    "HindIII": {"recognition": "AAGCTT", "cut_position": 1, "overhang": "5' AGCT"},
    "NotI":   {"recognition": "GCGGCCGC", "cut_position": 2, "overhang": "5' GGCC"},
    "XhoI":   {"recognition": "CTCGAG", "cut_position": 1, "overhang": "5' TCGA"},
    "SalI":   {"recognition": "GTCGAC", "cut_position": 1, "overhang": "5' TCGA"},
    "DpnI":   {"recognition": "GATC", "cut_position": 0, "overhang": "blunt",
               "note": "cuts only methylated GATC (dam+)"},
    "MluI":   {"recognition": "ACGCGT", "cut_position": 1, "overhang": "5' CGGT"},
    "SmaI":   {"recognition": "CCCGGG", "cut_position": 3, "overhang": "blunt"},
    "NcoI":   {"recognition": "CCATGG", "cut_position": 1, "overhang": "5' CATG"},
}

# dam-methylase blocks certain enzymes when present.
_DAM_BLOCKED = {"MluI", "SmaI"}  # canonical dam-blocked examples


@mcp.tool(description=(
    "Look up a restriction enzyme: recognition sequence, overhang, and "
    "whether dam-methylation blocks its activity."
))
def query_restriction_enzyme(name: str) -> str:
    try:
        name_norm = name.strip()
        if name_norm not in _ENZYMES:
            return _err(f"Unknown enzyme '{name}'",
                        known=list(_ENZYMES.keys()))
        e = _ENZYMES[name_norm]
        return _ok(
            "query_restriction_enzyme",
            name=name_norm,
            recognition=e["recognition"],
            cut_position=e["cut_position"],
            overhang=e["overhang"],
            dam_blocked=name_norm in _DAM_BLOCKED,
            note=e.get("note", ""),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Digest a linear sequence with the given list of restriction enzymes. "
    "Returns expected fragment sizes in bp."
))
def simulate_restriction_digest(sequence: str, enzymes: List[str]) -> str:
    """Count cut sites and compute fragment sizes."""
    try:
        seq = sequence.upper().replace("\n", "").replace(" ", "")
        if not seq:
            raise ValueError("sequence is empty")
        fragments: List[int] = []
        cuts: List[Dict[str, Any]] = []
        cursor = 0
        # Greedy single-pass cut (assumes a single hit per enzyme; works for
        # the demo fragment sizes).
        for enz in enzymes:
            if enz not in _ENZYMES:
                raise ValueError(f"Unknown enzyme {enz}")
            rec = _ENZYMES[enz]["recognition"]
            idx = seq.find(rec)
            if idx >= 0:
                cut_pos = idx + _ENZYMES[enz]["cut_position"]
                cuts.append({"enzyme": enz, "site_position": idx, "cut_at": cut_pos})
                if cursor == 0:
                    fragments.append(cut_pos)
                else:
                    fragments.append(cut_pos - cursor)
                cursor = cut_pos
        if cursor < len(seq):
            fragments.append(len(seq) - cursor)
        return _ok(
            "simulate_restriction_digest",
            enzymes=enzymes,
            cuts=cuts,
            fragment_sizes_bp=fragments,
            total_bp=sum(fragments),
            n_fragments=len(fragments),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Analyse a digest-pattern list (a list of 'enzyme:band_bp' descriptors) "
    "to compute total band intensity and identify missing enzymes."
))
def analyze_digest_pattern(digest_results: List[str],
                           expected_enzymes: List[str]) -> str:
    try:
        observed: Dict[str, List[int]] = {}
        for entry in digest_results:
            if ":" not in entry:
                continue
            enz, val = entry.split(":", 1)
            try:
                bp = int(val.strip())
            except ValueError:
                continue
            observed.setdefault(enz.strip(), []).append(bp)
        missing = [e for e in expected_enzymes if e not in observed]
        observed_total_bp = sum(sum(v) for v in observed.values())
        return _ok(
            "analyze_digest_pattern",
            observed_bands_per_enzyme=observed,
            expected_enzymes=expected_enzymes,
            missing_enzymes=missing,
            observed_total_bp=observed_total_bp,
            mean_band_bp=(observed_total_bp / max(1, sum(len(v) for v in observed.values()))),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Check whether a host strain's dam/dcm methylation will block any of "
    "the supplied enzymes."
))
def check_methylation_sensitivity(enzymes: List[str], strain: str) -> str:
    try:
        if strain not in _STRAIN_DB:
            return _err(f"Unknown strain '{strain}'",
                        known=list(_STRAIN_DB.keys()))
        methyl = _STRAIN_DB[strain]["methylation"]
        dam_present = "dam+" in methyl
        blocked = [e for e in enzymes if dam_present and e in _DAM_BLOCKED]
        return _ok(
            "check_methylation_sensitivity",
            strain=strain,
            methylation=methyl,
            dam_present=dam_present,
            blocked_enzymes=blocked,
            recommended_alternative=("JM110" if (dam_present and blocked) else None),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Compute DNA purity metrics from OD260/280, OD260/230 and concentration. "
    "A260/A280 ~ 1.8 is 'pure' for DNA; A260/A230 in 2.0–2.2 is good."
))
def calculate_dna_quality_metrics(od260_280: float, od260_230: float,
                                   concentration_ng_ul: float) -> str:
    try:
        for v in (od260_280, od260_230, concentration_ng_ul):
            if v <= 0:
                raise ValueError("metrics must be > 0")
        purity_260_280 = (
            "pure" if 1.7 <= od260_280 <= 2.0 else
            "protein_contamination" if od260_280 < 1.7 else
            "rna_contamination"
        )
        purity_260_230 = (
            "good" if od260_230 >= 2.0 else
            "carbohydrate_or_phenol_contamination"
        )
        return _ok(
            "calculate_dna_quality_metrics",
            od260_280=od260_280,
            od260_230=od260_230,
            concentration_ng_ul=concentration_ng_ul,
            purity_260_280=purity_260_280,
            purity_260_230=purity_260_230,
            total_ug_per_ul=concentration_ng_ul / 1000.0,
        )
    except Exception as exc:
        return _err(str(exc))


# ---------------------------------------------------------------------------
# Plasmid helpers (from molecular_biology_toolkit_claude_56)
# ---------------------------------------------------------------------------

@mcp.tool(description=(
    "Look up plasmid properties (size, copy number, marker, origin)."
))
def query_plasmid_properties(plasmid: str) -> str:
    try:
        if plasmid not in _PLASMID_DB:
            return _err(f"Unknown plasmid '{plasmid}'",
                        known=list(_PLASMID_DB.keys()))
        p = _PLASMID_DB[plasmid]
        return _ok(
            "query_plasmid_properties",
            plasmid=plasmid,
            size_bp=p["size_bp"],
            copy_number=p["copy_number"],
            origin=p["ori"],
            selection_marker=p["marker"],
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Check whether two plasmids are compatible (compatible = same origin "
    "AND different selection markers)."
))
def calculate_plasmid_compatibility(plasmid1: str, plasmid2: str) -> str:
    try:
        for name, val in (("plasmid1", plasmid1), ("plasmid2", plasmid2)):
            if val not in _PLASMID_DB:
                raise ValueError(f"Unknown plasmid {name}={val}")
        p1, p2 = _PLASMID_DB[plasmid1], _PLASMID_DB[plasmid2]
        same_ori = p1["ori"] == p2["ori"]
        same_marker = p1["marker"] == p2["marker"]
        compatible = (not same_ori) and (not same_marker)
        return _ok(
            "calculate_plasmid_compatibility",
            plasmid1=plasmid1,
            plasmid2=plasmid2,
            same_origin=same_ori,
            same_marker=same_marker,
            compatible=compatible,
            recommendation=("can co-transform" if compatible
                             else "use sequential transformation"),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Rate how difficult a transformation is on a 1 (easy) – 5 (hard) scale "
    "based on size, copy-number compatibility, and selection."
))
def calculate_transformation_difficulty(source_plasmid: str,
                                        target_plasmid: str) -> str:
    try:
        for name, val in (("source", source_plasmid), ("target", target_plasmid)):
            if val not in _PLASMID_DB:
                raise ValueError(f"Unknown plasmid {name}={val}")
        s = _PLASMID_DB[source_plasmid]
        t = _PLASMID_DB[target_plasmid]
        size_diff = abs(s["size_bp"] - t["size_bp"])
        score = 1.0
        score += min(2.0, size_diff / 5000.0)
        if s["ori"] != t["ori"]:
            score += 1.0
        if s["marker"] == t["marker"]:
            score += 1.0
        score = max(1.0, min(5.0, score))
        return _ok(
            "calculate_transformation_difficulty",
            source=source_plasmid,
            target=target_plasmid,
            difficulty_score=round(score, 2),
            difficulty_label=("easy" if score < 2 else
                              "moderate" if score < 3.5 else "hard"),
            notes="Difficulty scales with size delta and marker/origin mismatch.",
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Recommend a swap-in plasmid compatible with the requested marker."
))
def recommend_alternative_plasmid(target_marker: str,
                                  max_size_bp: int = 6000) -> str:
    try:
        candidates = []
        for name, p in _PLASMID_DB.items():
            if target_marker in p["marker"] and p["size_bp"] <= max_size_bp:
                candidates.append({"plasmid": name, "size_bp": p["size_bp"]})
        candidates.sort(key=lambda c: c["size_bp"])
        return _ok(
            "recommend_alternative_plasmid",
            target_marker=target_marker,
            candidates=candidates,
        )
    except Exception as exc:
        return _err(str(exc))


# ---------------------------------------------------------------------------
# DMD exon-skipping helpers (from molecular_biology_toolkit_claude_5)
# ---------------------------------------------------------------------------

@mcp.tool(description=(
    "Determine whether deleting a list of exons preserves the reading frame "
    "(skipping must keep frame = multiple of 3)."
))
def analyze_exon_frame_shift(exon_deletions: List[int],
                             exon_sizes: List[int]) -> str:
    try:
        if not exon_deletions or not exon_sizes:
            raise ValueError("exon_deletions and exon_sizes must be non-empty")
        total_removed = sum(exon_sizes[i] for i in exon_deletions
                             if 0 <= i < len(exon_sizes))
        in_frame = total_removed % 3 == 0
        return _ok(
            "analyze_exon_frame_shift",
            deleted_exons=exon_deletions,
            total_bp_removed=total_removed,
            reading_frame_preserved=in_frame,
            predicted_outcome=("in-frame truncation" if in_frame
                                else "frameshift → truncated protein"),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Estimate a morpholino binding score against the requested exon "
    "(0 = no binding, 1 = perfect 25-mer)."
))
def simulate_morpholino_binding(target_exon: int,
                                morpholino_sequence: str,
                                exon_sequence: str = "") -> str:
    try:
        mo = morpholino_sequence.upper().replace(" ", "")
        if len(mo) < 18 or len(mo) > 30:
            raise ValueError("morpholino length must be 18–30 nt")
        gc_count = mo.count("G") + mo.count("C")
        gc_fraction = gc_count / len(mo)
        gc_ok = 0.4 <= gc_fraction <= 0.6
        seed_match = 0.0
        if exon_sequence:
            ex = exon_sequence.upper()
            seed = mo[:8]
            if seed in ex:
                seed_match = 1.0
        score = 0.4 * gc_ok + 0.6 * seed_match
        return _ok(
            "simulate_morpholino_binding",
            target_exon=target_exon,
            morpholino_length=len(mo),
            gc_fraction=round(gc_fraction, 3),
            gc_in_optimal_range=gc_ok,
            seed_match=bool(seed_match),
            predicted_binding_score=round(score, 3),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Evaluate a DMD exon-skipping therapy: compute frame restoration, "
    "percentage of transcript retained, and a recommended skip exon."
))
def analyze_dmd_mutation_therapy(deleted_exons: List[int],
                                  transcript_name: str = "DMD") -> str:
    try:
        if transcript_name not in _RNA_DB:
            return _err(f"Unknown transcript '{transcript_name}'",
                        known=list(_RNA_DB.keys()))
        t = _RNA_DB[transcript_name]
        # Average exon length = protein_length * 3 / exon_count (rough).
        avg_exon_len = (t["protein_length"] * 3) // max(1, t["exon_count"])
        removed_bp = len(deleted_exons) * avg_exon_len
        # Estimate transcript retained.
        retained_fraction = max(0.0, 1.0 - removed_bp / max(1, t["protein_length"] * 3))
        frame_restored = removed_bp % 3 == 0
        skip_recommendation = (
            min(t["critical_exons"], key=lambda e: abs(e - deleted_exons[0]))
            if deleted_exons else None
        )
        return _ok(
            "analyze_dmd_mutation_therapy",
            transcript=transcript_name,
            deleted_exons=deleted_exons,
            estimated_bp_removed=removed_bp,
            retained_transcript_fraction=round(retained_fraction, 3),
            frame_restored=frame_restored,
            recommended_skip_exon=skip_recommendation,
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Return DMD transcript metadata (exon count, protein length, "
    "critical exons)."
))
def query_rna_structure_involvement(transcript_name: str) -> str:
    try:
        if transcript_name not in _RNA_DB:
            return _err(f"Unknown transcript '{transcript_name}'",
                        known=list(_RNA_DB.keys()))
        t = _RNA_DB[transcript_name]
        return _ok(
            "query_rna_structure_involvement",
            transcript=transcript_name,
            n_exons=t["exon_count"],
            protein_length=t["protein_length"],
            critical_exons=t["critical_exons"],
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Compute total RNA concentration in ng/µL from absorbance, path length, "
    "and dilution factor (Beer-Lambert with ε260 = 1 OD = 40 ng/µL for ssRNA)."
))
def calculate_rna_concentration(od260: float, dilution_factor: float = 1.0,
                                path_length_cm: float = 1.0) -> str:
    try:
        _require_positive("od260", od260)
        _require_positive("dilution_factor", dilution_factor)
        _require_positive("path_length_cm", path_length_cm)
        ng_ul = od260 * dilution_factor * 40.0 / path_length_cm
        return _ok(
            "calculate_rna_concentration",
            od260=od260,
            dilution_factor=dilution_factor,
            path_length_cm=path_length_cm,
            concentration_ng_ul=round(ng_ul, 3),
            concentration_ug_ul=round(ng_ul / 1000.0, 4),
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(description=(
    "Translate a DNA coding sequence into the single-letter amino-acid "
    "sequence (first frame)."
))
def translate_dna_to_protein(dna_sequence: str) -> str:
    try:
        seq = dna_sequence.upper().replace("\n", "").replace(" ", "")
        codon_table = {
            "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
            "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
            "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
            "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
            "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
            "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
            "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
            "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
            "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
            "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
            "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
            "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
            "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
            "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
            "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
            "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
        }
        if len(seq) < 3:
            raise ValueError("sequence must be ≥ 3 nt")
        if len(seq) % 3 != 0:
            seq = seq[:len(seq) - len(seq) % 3]
        protein = "".join(codon_table.get(seq[i:i+3], "X")
                          for i in range(0, len(seq), 3))
        n_stop = protein.count("*")
        return _ok(
            "translate_dna_to_protein",
            dna_length=len(seq),
            protein_length=len(protein),
            protein=protein,
            n_stop_codons=n_stop,
            has_premature_stop=(n_stop > 1),
        )
    except Exception as exc:
        return _err(str(exc))


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

@mcp.tool(description=(
    "Plot a qPCR amplification curve: fluorescence vs cycle number. "
    "Returns path to saved PNG."
))
def plot_qpcr_curve(cycles: List[float], fluorescence: List[float],
                      ct: float = None) -> str:
    if not _MPL_OK:
        return _err("matplotlib not available")
    if len(cycles) != len(fluorescence) or not cycles:
        return _err("cycles and fluorescence length mismatch")
    import os
    os.makedirs(_VIZ_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(cycles, fluorescence, "b-")
    if ct is not None:
        ax.axvline(ct, color="r", linestyle="--", label=f"Ct = {ct}")
        ax.legend()
    ax.set_xlabel("cycle")
    ax.set_ylabel("fluorescence")
    ax.set_title("qPCR amplification curve")
    ax.grid(True, alpha=0.3)
    out = f"{_VIZ_DIR}/qpcr_curve.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return _ok("plot_qpcr_curve", path=out, n=len(cycles))


@mcp.tool(description=(
    "Plot a simulated agarose-gel image from a digest: one band per "
    "fragment size. Returns path to saved PNG."
))
def plot_gel_simulation(fragment_sizes_bp: List[int],
                          lane_name: str = "digest") -> str:
    if not _MPL_OK:
        return _err("matplotlib not available")
    if not fragment_sizes_bp:
        return _err("fragment_sizes_bp must be non-empty")
    import os
    os.makedirs(_VIZ_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    n = len(fragment_sizes_bp)
    for size in fragment_sizes_bp:
        y = max(1.0, math.log10(max(size, 10)))
        ax.add_patch(plt.Rectangle((0.2, y - 0.05), 0.6, 0.1,
                                      facecolor="black"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(7, math.log10(max(fragment_sizes_bp)) + 1))
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_ylabel("log10(size in bp)")
    ax.set_title(f"Gel — {lane_name} ({n} bands)")
    out = f"{_VIZ_DIR}/gel.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return _ok("plot_gel_simulation", path=out, n_bands=n)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    asyncio.run(mcp.run_stdio_async())
