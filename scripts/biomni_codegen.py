#!/usr/bin/env python3
"""Generate sciMAS MCP servers + skills from BiOMNI tool descriptions.

For each Phase-1 category the generator:

1. Parses ``BiOMNI/biomni/tool/tool_description/<category>.py`` and
   extracts every ``{name, description, required/optional_parameters}``
   entry. Validates that each named tool actually exists in
   ``BiOMNI/biomni/tool/<category>.py`` (catches the description/
   function drift like ``bioimaging.create_registration_visualization``).
2. Emits ``tools/biomni/<category>_server.py`` — one MCP server per
   category, one wrapper per tool. Wrappers are explicit-signature
   (not ``**kwargs``) so the MCP layer sees real parameter schemas.
   Every wrapper routes execution through
   ``common.call_with_error_boundary`` so a missing dep or a broken
   network call surfaces as JSON, not a crashed stdio loop.
3. Emits one or more ``scimas_skills/biomni-<...>/SKILL.md`` files
   per category. Bundling is configured in ``PHASE_1_BUNDLES`` below
   so no single skill exceeds ~8 tools (the user's "3–8 tools per
   skill" guidance).
4. Writes ``tools/biomni/_manifest.json`` mapping
   ``category → server_name, role_target, tool_names``. The
   orchestrator reads this so Phase-1 wiring is data-driven rather
   than hand-edited in ``orchestrator.py``.

Usage::

    python scripts/biomni_codegen.py --phase 1
    python scripts/biomni_codegen.py --phase 1 --categories biochemistry literature

The generator never imports BiOMNI at runtime — it only reads the
``.py`` source files — so it works on machines where BiOMNI is not
installed.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Same resolver the generated servers use at runtime, so the generator and
# the code it emits can never disagree about where BiOMNI lives.
from tools.biomni.common import biomni_tool_dir  # noqa: E402

SERVER_DIR = REPO / "tools" / "biomni"
SKILL_DIR = REPO / "scimas_skills"
MANIFEST = SERVER_DIR / "_manifest.json"


# ----------------------------------------------------------------------------
# BiOMNI description-type → Python / JSON-RPC type mapping.
#
# MCP exposes a fixed JSON-RPC schema (str/int/float/bool/list/dict), so
# anything numpy/pandas-shaped collapses to ``list`` / ``dict`` and is
# coerced in the wrapper body.
# ----------------------------------------------------------------------------
TYPE_MAP = {
    "str": "str",
    "string": "str",
    "int": "int",
    "integer": "int",
    "float": "float",
    "number": "float",
    "bool": "bool",
    "boolean": "bool",
    "list": "list",
    "list of str": "list[str]",
    "list of list": "list[list[float]]",
    "list of dict": "list[dict]",
    "list or str": "Union[list, str]",
    "list or numpy.ndarray": "list",
    "list or list of dict": "Union[list, list[dict]]",
    "numpy.ndarray": "list",
    "dict": "dict",
    "dict[str, Any]": "dict",
    "dict[str, str]": "dict",
    "dict[str, list or numpy.ndarray]": "dict",
    "pandas.DataFrame": "str",  # CSV/TSV path
    "Path": "str",
    "path": "str",
    "file": "str",
}


def map_type(biomni_type: str) -> str:
    return TYPE_MAP.get(biomni_type, "str")


def python_default(raw: object, py_type: str) -> str:
    """Render a default-value literal for the wrapper signature."""
    if raw is None:
        return ""  # no default
    if isinstance(raw, bool):
        return "True" if raw else "False"
    if isinstance(raw, (int, float)):
        return repr(raw)
    if isinstance(raw, str):
        if not raw:
            return '""'
        # repr() escapes properly; replace only the wrapping.
        return repr(raw)
    if isinstance(raw, list):
        return repr(raw)
    if isinstance(raw, dict):
        return repr(raw)
    return repr(raw)


@dataclass
class Param:
    name: str
    biomni_type: str
    py_type: str
    description: str
    required: bool
    default_repr: str = ""

    def signature(self) -> str:
        colon = ": " + self.py_type
        if self.required:
            return f"{self.name}{colon}"
        if self.default_repr:
            return f"{self.name}{colon} = {self.default_repr}"
        return f"{self.name}{colon} = None"


@dataclass
class Tool:
    name: str
    description: str
    params: list[Param] = field(default_factory=list)


# ----------------------------------------------------------------------------
# BiOMNI description loader (AST literal eval — no BiOMNI import needed).
# ----------------------------------------------------------------------------

def load_description(category: str) -> list[Tool]:
    path = biomni_tool_dir() / "tool_description" / f"{category}.py"
    if not path.exists():
        raise FileNotFoundError(f"missing description file: {path}")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    description_obj: Optional[ast.AST] = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "description":
                    description_obj = node.value
                    break
        if description_obj is not None:
            break
    if description_obj is None:
        raise ValueError(f"no `description = [...]` in {path}")
    raw_list = ast.literal_eval(description_obj)
    tools: list[Tool] = []
    for entry in raw_list:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "")).strip()
        if not name:
            continue
        description = str(entry.get("description", "")).strip()
        params: list[Param] = []
        for group in ("required_parameters", "optional_parameters"):
            for raw in entry.get(group, []) or []:
                if not isinstance(raw, dict):
                    continue
                pname = str(raw.get("name", "")).strip()
                if not pname:
                    continue
                ptype = str(raw.get("type", "str")).strip()
                required = group == "required_parameters"
                params.append(
                    Param(
                        name=pname,
                        biomni_type=ptype,
                        py_type=map_type(ptype),
                        description=str(raw.get("description", "")).strip(),
                        required=required,
                        default_repr=python_default(
                            raw.get("default"),
                            map_type(ptype),
                        ) if not required else "",
                    )
                )
        tools.append(Tool(name=name, description=description, params=params))
    return tools


def function_names(category: str) -> set[str]:
    """Top-level function names defined in tool/<category>.py."""
    path = biomni_tool_dir() / f"{category}.py"
    if not path.exists():
        return set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            names.add(node.name)
    return names


# ----------------------------------------------------------------------------
# Wrapper generator. The wrapper signature uses the description's
# parameter list; the body lazy-loads BiOMNI via ``common`` and
# invokes the function with JSON-safe coercion.
# ----------------------------------------------------------------------------

WRAPPER_TEMPLATE = '''\
@mcp.tool(description={description_repr})
def {tool_name}({params_signature}) -> str:
    """BiOMNI wrapper: {tool_name} from {category}."""
    try:
        from tools.biomni.common import (
            call_with_error_boundary,
            coerce_array,
            coerce_dataframe,
            load_biomni_module,
            normalize_output_dir,
            require_positive,
            truncate,
        )
        import json as _json

        mod = load_biomni_module("{category}")
        fn = getattr(mod, "{tool_name}")
{body}
        return call_with_error_boundary(
            "{tool_name}",
            fn,
{arg_kw_pairs}        )
    except Exception as exc:
        from tools.biomni.common import err
        return err(str(exc), tool="{tool_name}")
'''


def generate_wrapper(category: str, tool: Tool) -> str:
    sig_parts = []
    arg_pairs = []
    body_lines = []
    for param in tool.params:
        sig_parts.append(param.signature())
        argname = param.name
        btype = param.biomni_type
        if btype in ("numpy.ndarray", "list or numpy.ndarray"):
            if "list of list" in btype or "list[list" in (param.py_type or ""):
                body_lines.append(
                    f"        {argname}_arr = coerce_array({argname}, ndim=2)"
                )
                arg_pairs.append((argname, f"{argname}_arr"))
            else:
                body_lines.append(
                    f"        {argname}_arr = coerce_array({argname}, ndim=1)"
                )
                arg_pairs.append((argname, f"{argname}_arr"))
        elif btype == "pandas.DataFrame":
            body_lines.append(
                f"        {argname}_df = coerce_dataframe({argname}, name={argname!r})"
            )
            arg_pairs.append((argname, f"{argname}_df"))
        elif argname in {"output_dir", "output_dir_path"}:
            body_lines.append(
                f"        {argname}_norm = normalize_output_dir({argname}, {tool.name!r})"
            )
            arg_pairs.append((argname, f"{argname}_norm"))
        else:
            arg_pairs.append((argname, argname))

    description_repr = repr(tool.description or "(no description)")

    arg_kw = "".join(
        f"            {k}={v},\n" for k, v in arg_pairs
    )

    body = "\n".join(body_lines) if body_lines else "        pass"
    if not body_lines:
        body = "        pass"

    return WRAPPER_TEMPLATE.format(
        description_repr=description_repr,
        tool_name=tool.name,
        params_signature=", ".join(sig_parts) if sig_parts else "",
        category=category,
        body=body,
        arg_kw_pairs=arg_kw,
    )


def server_imports(category: str) -> str:
    return (
        '"""Auto-generated by scripts/biomni_codegen.py.\n\n'
        'MCP server wrapping the BiOMNI ``' + category + '`` module.\n'
        'Every wrapper routes through ``tools.biomni.common`` so missing\n'
        'BiOMNI dependencies surface as a JSON error rather than crashing\n'
        'the MCP stdio loop.\n'
        '"""\n\n'
        'from __future__ import annotations\n\n'
        'import asyncio\nimport sys\nfrom pathlib import Path\n\n'
        '# Make the sciMAS repo root importable when this file is launched\n'
        '# as a script via ``python tools/biomni/<cat>_server.py`` — without\n'
        '# this the ``from tools.biomni.common import ...`` lines in every\n'
        '# wrapper would fail with ModuleNotFoundError because sys.path[0]\n'
        '# would point at tools/biomni/ instead of the repo root.\n'
        '_REPO_ROOT = Path(__file__).resolve().parent.parent.parent\n'
        'if str(_REPO_ROOT) not in sys.path:\n'
        '    sys.path.insert(0, str(_REPO_ROOT))\n\n'
        'from mcp.server.mcpserver import MCPServer\n\n'
        f'mcp = MCPServer("biomni-{category}")\n\n'
    )


def server_footer() -> str:
    return (
        '\n\nif __name__ == "__main__":\n'
        '    asyncio.run(mcp.run_stdio_async())\n'
    )


def render_server(category: str, tools: list[Tool]) -> str:
    parts = [server_imports(category)]
    for tool in tools:
        parts.append(generate_wrapper(category, tool))
    parts.append(server_footer())
    return "".join(parts)


# ----------------------------------------------------------------------------
# Skill bundling. Hand-curated per category so each skill stays small.
# ----------------------------------------------------------------------------

@dataclass
class Bundle:
    skill_id: str
    title: str
    description: str
    role: str
    server: str
    tools: list[str]
    body: str = ""


PHASE_1_BUNDLES: dict[str, list[Bundle]] = {
    "biochemistry": [
        Bundle(
            skill_id="biomni-biochemistry-cd-kinetics",
            title="CD spectroscopy & enzyme kinetics",
            description="Use for circular-dichroism secondary-structure / thermal-melting analysis and for Michaelis-Menten / dose-response fits of protease and enzyme time-course fluorescence data.",
            role="molecular-biologist",
            server="biomni-biochemistry",
            tools=[
                "analyze_circular_dichroism_spectra",
                "analyze_protease_kinetics",
                "analyze_enzyme_kinetics_assay",
            ],
            body=(
                "## When to use\n"
                "- User has CD spectra (wavelength vs CD signal) and wants protein / "
                "nucleic-acid secondary-structure classification or Tm.\n"
                "- User has fluorescence vs time data at multiple substrate concentrations "
                "and wants Vmax / Km / kcat from Michaelis-Menten.\n"
                "- User has dose-response data with modulators and wants IC50 / Hill slope.\n\n"
                "## Limitations\n"
                "- Output is a research-log string; numbers are not pre-parsed — "
                "downstream steps must grep ``Km:`` / ``Vmax:`` / ``IC50:`` from the log.\n"
                "- Files are written under ``/tmp/scimas_biomni/<tool>/`` by default; "
                "pass ``output_dir=\"\"`` to accept the default or supply a writable path.\n"
                "- ``analyze_enzyme_kinetics_assay`` synthesizes a time-course internally; "
                "do NOT use it to fit real experimental time-courses — use "
                "``analyze_protease_kinetics`` for that."
            ),
        ),
        Bundle(
            skill_id="biomni-biochemistry-rna-secondary",
            title="RNA secondary-structure features",
            description="Use for dot-bracket → stems / loops / paired-fraction analysis; optional nearest-neighbor free-energy if the RNA sequence is supplied.",
            role="molecular-biologist",
            server="biomni-biochemistry",
            tools=["analyze_rna_secondary_structure_features"],
            body=(
                "## When to use\n"
                "- User has a dot-bracket RNA structure (and optionally the matching "
                "sequence) and wants stems, loops, paired-fraction, or nearest-neighbor "
                "free energy.\n\n"
                "## Limitations\n"
                "- Only ``()``, ``[]``, ``{}``, and ``.`` are accepted brackets; mismatched "
                "or unbalanced input returns an error string, not a JSON envelope.\n"
                "- Energy parameters are simplified (AU=-0.9, GC=-2.1, GU=-0.5 kcal/mol); "
                "treat as approximate."
            ),
        ),
        Bundle(
            skill_id="biomni-biochemistry-protein-conservation",
            title="Protein conservation analysis",
            description="Use for multi-sequence-alignment + phylogenetic tree + position-wise conservation across a small protein set (BiOMNI uses Biopython / MUSCLE).",
            role="molecular-biologist",
            server="biomni-biochemistry",
            tools=["analyze_protein_conservation"],
            body=(
                "## When to use\n"
                "- User has 2+ protein sequences (plain or FASTA) and wants alignment + "
                "tree + per-position conservation scores.\n\n"
                "## Limitations\n"
                "- Requires Biopython; falls back to a padding-based alignment if MUSCLE "
                "is not on PATH, but the fallback is not biologically meaningful.\n"
                "- Returns a research log string, not a JSON alignment record."
            ),
        ),
    ],
    "literature": [
        Bundle(
            skill_id="biomni-literature-arxiv-pubmed",
            title="arXiv & PubMed lookups",
            description="Use for arXiv preprint or PubMed bibliographic lookups when the existing OpenAlex ``search_literature`` is insufficient (e.g. user wants a specific preprint or MeSH-constrained PubMed query).",
            role="literature-searcher",
            server="biomni-literature",
            tools=["query_arxiv", "query_pubmed"],
            body=(
                "## When to use\n"
                "- OpenAlex lacks coverage of a recent arXiv preprint or a PubMed-only record.\n"
                "- User wants raw title / author / abstract / DOI strings in arXiv or "
                "PubMed format rather than OpenAlex normalized records.\n\n"
                "## Limitations\n"
                "- Uses BiOMNI's HTTP wrappers; rate-limit and availability reflect the "
                "BiOMNI service, not sciMAS.\n"
                "- The arXiv / PubMed result is a research-log string, not a structured "
                "JSON list of papers — parse carefully before comparing to OpenAlex output."
            ),
        ),
        Bundle(
            skill_id="biomni-literature-web-extract",
            title="URL & PDF content extraction",
            description="Use to fetch and extract plain text from a URL or PDF (e.g. extract figure caption text from an open-access PDF).",
            role="literature-searcher",
            server="biomni-literature",
            tools=["extract_url_content", "extract_pdf_content"],
            body=(
                "## When to use\n"
                "- User has a specific URL / PDF and wants the raw text content.\n\n"
                "## Limitations\n"
                "- Heavy lifting by BiOMNI helpers; needs ``requests`` and a PDF "
                "extractor (``pymupdf`` / ``PyPDF2``). Returns JSON error if missing.\n"
                "- No citation parsing — combine with the OpenAlex or arXiv skill when "
                "the URL is a known paper."
            ),
        ),
    ],
    "protocols": [
        Bundle(
            skill_id="biomni-protocols-wetlab-reference",
            title="Wet-lab protocol reference",
            description="Use to look up a wet-lab protocol from protocols.io by keyword or ID and read the step-by-step procedure.",
            role="molecular-biologist",
            server="biomni-protocols",
            tools=[
                "search_protocols",
                "get_protocol_details",
                "list_local_protocols",
                "read_local_protocol",
            ],
            body=(
                "## When to use\n"
                "- User wants a published wet-lab protocol (PCR setup, Western blot, "
                "transfection, etc.) and either a text search or a numeric protocol ID.\n"
                "- User has locally cached protocols and wants them indexed / read.\n\n"
                "## Limitations\n"
                "- ``search_protocols`` / ``get_protocol_details`` hit the protocols.io "
                "API; network availability determines success.\n"
                "- ``list_local_protocols`` / ``read_local_protocol`` look under a local "
                "directory configured by BiOMNI; if the directory is empty they return "
                "a JSON error rather than a fake list."
            ),
        ),
    ],
    "database": [
        Bundle(
            skill_id="biomni-database-uniprot-pdb-pubchem",
            title="UniProt / PDB / PubChem lookups",
            description="Use to fetch protein, structure, or compound metadata by ID from UniProt, RCSB PDB, or PubChem.",
            role="pharma-data-specialist",
            server="biomni-database",
            tools=["query_uniprot", "query_pdb", "query_pubchem"],
            body=(
                "## When to use\n"
                "- UniProt accession → canonical sequence, organism, function annotations.\n"
                "- PDB ID → entry metadata, polymer entity list, experimental method.\n"
                "- Compound name / SMILES / CID → PubChem properties.\n\n"
                "## Limitations\n"
                "- Results are research-log strings, not JSON records.\n"
                "- Network availability reflects BiOMNI's REST wrappers, not sciMAS."
            ),
        ),
        Bundle(
            skill_id="biomni-database-chembl-openfda-clinicaltrials",
            title="ChEMBL / OpenFDA / ClinicalTrials lookups",
            description="Use to fetch bioactivity, adverse-event, or trial metadata by drug or trial ID.",
            role="pharma-data-specialist",
            server="biomni-database",
            tools=["query_chembl", "query_openfda", "query_clinicaltrials"],
            body=(
                "## When to use\n"
                "- ChEMBL compound / target queries for known bioactivity values.\n"
                "- OpenFDA drug adverse-event counts / label sections / recalls.\n"
                "- ClinicalTrials.gov NCT ID → trial design, endpoints, status.\n\n"
                "## Limitations\n"
                "- Returns research-log strings.\n"
                "- Heavy use may trigger rate limits on the underlying services."
            ),
        ),
    ],
    "pharmacology": [
        Bundle(
            skill_id="biomni-pharmacology-rdkit-properties",
            title="RDKit physicochemical property calculation",
            description="Use to compute drug-likeness / physicochemical properties from a SMILES string using RDKit only.",
            role="drug-discovery-scientist",
            server="biomni-pharmacology",
            tools=["calculate_physicochemical_properties"],
            body=(
                "## When to use\n"
                "- User has a SMILES string and wants MW, logP, HBA/HBD, TPSA, "
                "rotatable bonds, ring counts, etc.\n\n"
                "## Limitations\n"
                "- Requires RDKit; returns JSON error if RDKit is unavailable.\n"
                "- For ADMET / binding-affinity predictions use ``DeepPurpose``-backed "
                "tools instead (Phase 2)."
            ),
        ),
        Bundle(
            skill_id="biomni-pharmacology-radiolabel-dosimetry",
            title="Radiolabeled biodistribution / dosimetry",
            description="Use to fit radiolabeled-antibody biodistribution time-courses and estimate alpha-particle radiotherapy dosimetry.",
            role="drug-discovery-scientist",
            server="biomni-pharmacology",
            tools=[
                "analyze_radiolabeled_antibody_biodistribution",
                "estimate_alpha_particle_radiotherapy_dosimetry",
            ],
            body=(
                "## When to use\n"
                "- User has time-resolved tissue-uptake data (%ID/g vs hours) for a "
                "radiolabeled antibody and wants mono-/bi-exponential fit parameters.\n"
                "- User has tumor-absorbed-dose / time pairs and wants alpha-emitter "
                "dosimetry (e.g. ``At-211``, ``Pb-212``).\n\n"
                "## Limitations\n"
                "- Pure numpy/scipy fits; no compartment-model identifiability check."
            ),
        ),
        Bundle(
            skill_id="biomni-pharmacology-tumor-inhibition",
            title="Xenograft tumor-growth inhibition analysis",
            description="Use to fit xenograft tumor-growth curves and compute treatment / control ratios, TGI%, and growth-delay metrics.",
            role="drug-discovery-scientist",
            server="biomni-pharmacology",
            tools=["analyze_xenograft_tumor_growth_inhibition"],
            body=(
                "## When to use\n"
                "- User has tumor-volume (mm³) time-courses for vehicle / treatment "
                "groups and wants TGI%, AUC ratio, or time-to-progression.\n\n"
                "## Limitations\n"
                "- Returns a research-log string; parse to extract numeric metrics."
            ),
        ),
    ],
}


def render_skill(bundle: Bundle) -> str:
    frontmatter = (
        "---\n"
        f"name: {bundle.skill_id}\n"
        f"description: {bundle.description}\n"
        f"x-scimas-role: {bundle.role}\n"
        f"x-scimas-server: {bundle.server}\n"
        "x-scimas-tools:\n"
    )
    for t in bundle.tools:
        frontmatter += f"  - {t}\n"
    frontmatter += "---\n"
    body = f"# {bundle.title}\n\n" + bundle.body.strip() + "\n"
    return frontmatter + "\n" + body


# ----------------------------------------------------------------------------
# Orchestrator wiring. Phase-1 maps BiOMNI servers onto existing
# sciMAS roles — we do NOT introduce new role names. The mapping is
# written into _manifest.json so orchestrator.py can read it.
# ----------------------------------------------------------------------------

ROLE_TARGETS = {
    "biochemistry": "molecular-biologist",
    "literature": "literature-searcher",
    "protocols": "molecular-biologist",
    "database": "pharma-data-specialist",
    "pharmacology": "drug-discovery-scientist",
    # Phase 2.
    "molecular_biology": "molecular-biologist",
    "genetics": "geneticist",
    "synthetic_biology": "molecular-biologist",
    "systems_biology": "structural-biologist",
    # Phase 3.
    "genomics": "geneticist",
    "bioimaging": "cell-biologist",
    "cell_biology": "cell-biologist",
    "cancer_biology": "cell-biologist",
    "immunology": "cell-biologist",
}


# Hand-curated Phase-2 bundling. Phase-2 categories depend on BioPython,
# pandas, requests, and (for genetics) torch — all of which are in
# environment.yml. Most tools actually run end-to-end in the current
# environment; a few that need external CLI binaries such as BLAST+ return
# a JSON error if the binary is missing.
PHASE_2_BUNDLES: dict[str, list[Bundle]] = {
    "molecular_biology": [
        Bundle(
            skill_id="biomni-molbio-plasmid-orf",
            title="Plasmid & ORF annotation",
            description="Use to annotate an open-reading frame on a sequence, design primers for a region, or characterize a plasmid sequence (linear or circular).",
            role="molecular-biologist",
            server="biomni-molecular_biology",
            tools=[
                "annotate_open_reading_frames",
                "annotate_plasmid",
                "find_restriction_sites",
                "find_restriction_enzymes",
            ],
            body=(
                "## When to use\n"
                "- User has a DNA sequence and wants ORF / start-codon annotation.\n"
                "- User has a circular plasmid map (GenBank / FASTA) and wants a "
                "feature table.\n"
                "- User wants to enumerate restriction-enzyme cut sites on a given "
                "sequence, or pick enzymes that cut a given sequence.\n\n"
                "## Limitations\n"
                "- Uses Biopython Restriction / SeqIO; circular / linear flag matters.\n"
                "- ``find_restriction_enzymes`` returns Biopython's REBASE subset, not "
                "the full commercial catalog."
            ),
        ),
        Bundle(
            skill_id="biomni-molbio-pcr-cloning",
            title="PCR / cloning workflow helpers",
            description="Use to design primers / oligos for a PCR, Golden Gate assembly, or knockout sgRNA; simulate a PCR on a linear / circular template; simulate a restriction digest.",
            role="molecular-biologist",
            server="biomni-molecular_biology",
            tools=[
                "pcr_simple",
                "digest_sequence",
                "design_primer",
                "design_verification_primers",
                "design_golden_gate_oligos",
                "golden_gate_assembly",
            ],
            body=(
                "## When to use\n"
                "- User has template + primer pair and wants the simulated PCR product.\n"
                "- User has a multi-fragment Golden Gate assembly and wants junction "
                "oligo design.\n"
                "- User wants primer Tm / hairpin / dimer-aware design for a target "
                "region.\n\n"
                "## Limitations\n"
                "- ``pcr_simple`` is a strict 3' / 5' identity match — does not model "
                "mismatch / wobble bases.\n"
                "- ``golden_gate_assembly`` assumes the standard Type IIS overhang set "
                "(BsaI / BsmBI)."
            ),
        ),
        Bundle(
            skill_id="biomni-molbio-alignment-mutation",
            title="Sequence alignment & mutation analysis",
            description="Use to align a long sequence against short probes, retrieve a gene coding sequence from NCBI Entrez, find sequence mutations between a query and a reference, or get a plasmid sequence from AddGene.",
            role="molecular-biologist",
            server="biomni-molecular_biology",
            tools=[
                "align_sequences",
                "find_sequence_mutations",
                "get_gene_coding_sequence",
                "get_plasmid_sequence",
            ],
            body=(
                "## When to use\n"
                "- User has a long sequence + a list of short probes and wants the "
                "locations of each probe.\n"
                "- User has two sequences (query vs reference) and wants a mutation "
                "table (SNPs / indels).\n"
                "- User wants a NCBI gene coding sequence (NCBI Entrez).\n"
                "- User wants an AddGene plasmid full sequence by ID.\n\n"
                "## Limitations\n"
                "- ``get_gene_coding_sequence`` / ``get_plasmid_sequence`` hit live "
                "HTTP endpoints; require network access and may rate-limit."
            ),
        ),
        Bundle(
            skill_id="biomni-molbio-knockout-protocols",
            title="CRISPR knockout design & wet-lab protocols",
            description="Use to design a knockout sgRNA for a target locus or fetch a wet-lab protocol (oligo annealing, Golden Gate assembly, bacterial transformation).",
            role="molecular-biologist",
            server="biomni-molecular_biology",
            tools=[
                "design_knockout_sgrna",
                "get_oligo_annealing_protocol",
                "get_golden_gate_assembly_protocol",
                "get_bacterial_transformation_protocol",
            ],
            body=(
                "## When to use\n"
                "- User wants a sgRNA designed against a target gene (PAM = NGG).\n"
                "- User wants a step-by-step protocol for oligo annealing, Golden "
                "Gate assembly, or bacterial transformation.\n\n"
                "## Limitations\n"
                "- Protocol helpers return static protocol text; not the latest "
                "vendor-specific procedure.\n"
                "- ``design_knockout_sgrna`` scores off-targets heuristically; for "
                "production CRISPR use a CRISPOR-style tool."
            ),
        ),
    ],
    "genetics": [
        Bundle(
            skill_id="biomni-genetics-crispr-editing",
            title="CRISPR / Cas9 outcome analysis",
            description="Use to analyze the outcomes of a Cas9 edit (original vs edited sequence + guide RNA), classify indel type, or simulate a CRISPR HDR repair template.",
            role="geneticist",
            server="biomni-genetics",
            tools=[
                "analyze_cas9_mutation_outcomes",
                "analyze_crispr_genome_editing",
                "identify_transcription_factor_binding_sites",
            ],
            body=(
                "## When to use\n"
                "- User has Cas9-edited sequence vs WT and wants indel classification.\n"
                "- User has a guide RNA + optional HDR repair template and wants a "
                "predicted edited outcome.\n\n"
                "## Limitations\n"
                "- ``analyze_cas9_mutation_outcomes`` does not call NHEJ vs MMEJ — "
                "only the net indel spectrum.\n"
                "- ``identify_transcription_factor_binding_sites`` uses a PWM "
                "approximation; not a ChIP-seq replacement."
            ),
        ),
        Bundle(
            skill_id="biomni-genetics-phylogeny-demography",
            title="Phylogeny & demographic inference",
            description="Use to build a protein phylogeny from aligned sequences, or to simulate a demographic history under a specified effective population size / time schedule.",
            role="geneticist",
            server="biomni-genetics",
            tools=[
                "analyze_protein_phylogeny",
                "simulate_demographic_history",
            ],
            body=(
                "## When to use\n"
                "- User has multiple aligned protein sequences and wants a tree.\n"
                "- User has population-size / generation schedule and wants a "
                "simulated allele-frequency trajectory.\n\n"
                "## Limitations\n"
                "- Phylogeny uses Biopython NJ; no bootstrap support.\n"
                "- Demographic simulator uses a custom coalescent approximation; not "
                "msprime."
            ),
        ),
        Bundle(
            skill_id="biomni-genetics-prediction-pcr",
            title="Genomic prediction & in-silico PCR",
            description="Use to fit a genomic prediction model (GBLUP-style) on a phenotype + genotype matrix, or to simulate a PCR + gel electrophoresis.",
            role="geneticist",
            server="biomni-genetics",
            tools=[
                "fit_genomic_prediction_model",
                "perform_pcr_and_gel_electrophoresis",
                "liftover_coordinates",
            ],
            body=(
                "## When to use\n"
                "- User has a SNP genotype matrix + phenotype column and wants a GBLUP "
                "prediction.\n"
                "- User has primer pair + template and wants a simulated PCR + gel "
                "image.\n"
                "- User has hg19 coordinates and wants hg38 (liftover).\n\n"
                "## Limitations\n"
                "- GBLUP is a single-kernel ridge regression; no multi-kernel / Bayes "
                "models.\n"
                "- Liftover needs an external chain file at the path BiOMNI expects."
            ),
        ),
    ],
    "synthetic_biology": [
        Bundle(
            skill_id="biomni-synbio-codon-design",
            title="Codon & genome design",
            description="Use to optimize codons of a coding sequence for a target host, design a therapeutic-delivery bacterial genome, or analyze bacterial growth rate from an OD600 time-course.",
            role="molecular-biologist",
            server="biomni-synthetic_biology",
            tools=[
                "optimize_codons_for_heterologous_expression",
                "engineer_bacterial_genome_for_therapeutic_delivery",
                "analyze_bacterial_growth_rate",
            ],
            body=(
                "## When to use\n"
                "- User has a coding sequence and a target host (CAI optimization).\n"
                "- User has a bacterial FASTA + a list of genetic parts and wants an "
                "engineered construct.\n"
                "- User has OD600 vs time and wants growth-rate fit (μ, lag, "
                "carrying capacity).\n\n"
                "## Limitations\n"
                "- Codon optimization uses simple CAI; no mRNA-structure / ribosomal "
                "queueing constraints.\n"
                "- Growth-rate fit uses a simple logistic; no diauxic shift."
            ),
        ),
        Bundle(
            skill_id="biomni-synbio-network-bifurcation",
            title="Gene-circuit & bifurcation simulation",
            description="Use to simulate a gene circuit with growth feedback, scan a bifurcation diagram over a parameter sweep, or analyze barcode-sequencing data for lineage tracing.",
            role="molecular-biologist",
            server="biomni-synthetic_biology",
            tools=[
                "simulate_gene_circuit_with_growth_feedback",
                "analyze_bifurcation_diagram",
                "analyze_barcode_sequencing_data",
                "create_biochemical_network_sbml_model",
                "identify_fas_functional_domains",
            ],
            body=(
                "## When to use\n"
                "- User has a gene-circuit ODE and wants a growth-coupled simulation.\n"
                "- User has a 1D parameter sweep and wants the bifurcation diagram.\n"
                "- User has barcode count table and wants lineage / clonal-frequency "
                "analysis.\n"
                "- User wants an SBML model for a biochemical network.\n"
                "- User has a FAS protein sequence and wants functional-domain "
                "annotation.\n\n"
                "## Limitations\n"
                "- SBML emission requires ``libsbml`` (not installed by default — "
                "wrapper returns JSON error if missing).\n"
                "- Bifurcation diagram uses a coarse parameter grid; no continuation "
                "method (pseudo-arclength)."
            ),
        ),
    ],
    "systems_biology": [
        Bundle(
            skill_id="biomni-sysbio-fba-cobra",
            title="Flux-balance analysis (FBA)",
            description="Use to run a flux-balance analysis on a COBRA SBML model with custom constraints / objective; returns per-reaction flux + objective value.",
            role="structural-biologist",
            server="biomni-systems_biology",
            tools=[
                "perform_flux_balance_analysis",
                "compare_protein_structures",
            ],
            body=(
                "## When to use\n"
                "- User has a COBRA-format SBML model and wants FBA.\n"
                "- User has two PDB files and wants a per-residue RMSD / contact map "
                "comparison.\n\n"
                "## Limitations\n"
                "- FBA requires COBRApy; not installed by default — wrapper returns "
                "JSON error if missing.\n"
                "- Protein comparison does a coarse alignment; not TM-align."
            ),
        ),
        Bundle(
            skill_id="biomni-sysbio-signaling-dynamics",
            title="Signaling & metabolic dynamics",
            description="Use to simulate a protein-signaling ODE network, a metabolic perturbation, a protein-dimerization network, or the renin-angiotensin system.",
            role="structural-biologist",
            server="biomni-systems_biology",
            tools=[
                "simulate_protein_signaling_network",
                "simulate_metabolic_network_perturbation",
                "model_protein_dimerization_network",
                "simulate_renin_angiotensin_system_dynamics",
            ],
            body=(
                "## When to use\n"
                "- User has a signaling ODE topology and wants a time-course.\n"
                "- User has a metabolic network stoichiometry and wants a perturbation "
                "response.\n"
                "- User has monomer concentrations + pairwise affinities and wants a "
                "dimerization network prediction.\n"
                "- User wants the RAS / blood-pressure ODE simulated.\n\n"
                "## Limitations\n"
                "- Uses scipy.integrate.solve_ivp; stiff systems may need manual "
                "method selection.\n"
                "- Metabolic perturbation is a stoichiometric + rate-law hybrid; not a "
                "full kinetic model."
            ),
        ),
    ],
}


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def validate_tools(
    category: str, tools: list[Tool]
) -> tuple[list[Tool], list[str]]:
    fns = function_names(category)
    keep: list[Tool] = []
    missing: list[str] = []
    for tool in tools:
        if tool.name in fns:
            keep.append(tool)
        else:
            missing.append(tool.name)
    return keep, missing


def run(categories: Iterable[str], *, force: bool = False) -> dict:
    SERVER_DIR.mkdir(parents=True, exist_ok=True)
    SKILL_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    issues: list[str] = []
    for category in categories:
        try:
            tools = load_description(category)
        except FileNotFoundError as exc:
            issues.append(str(exc))
            continue
        keep_all, missing = validate_tools(category, tools)
        if missing:
            issues.append(
                f"{category}: {len(missing)} description(s) with no matching "
                f"function: {missing[:5]}{'...' if len(missing) > 5 else ''}"
            )

        # Look up bundles for this category across every phase. A category
        # that appears in multiple phases (unlikely) gets the union.
        bundles: list[Bundle] = []
        for phase_bundles in PHASES.values():
            bundles.extend(phase_bundles.get(category, []))
        bundled_names: set[str] = set()
        for b in bundles:
            bundled_names.update(b.tools)

        unused = [t.name for t in keep_all if t.name not in bundled_names]
        if unused:
            issues.append(
                f"{category}: {len(unused)} description(s) with no skill bundle "
                f"(skipped): {unused[:5]}{'...' if len(unused) > 5 else ''}"
            )

        keep = [t for t in keep_all if t.name in bundled_names]
        out_path = SERVER_DIR / f"{category}_server.py"
        if out_path.exists() and not force:
            text = out_path.read_text(encoding="utf-8")
            if "biomni" not in text or "load_biomni_module" not in text:
                issues.append(f"{out_path}: refusing to overwrite non-generated file")
                continue
        out_path.write_text(render_server(category, keep), encoding="utf-8")

        for bundle in bundles:
            skill_path = SKILL_DIR / bundle.skill_id / "SKILL.md"
            skill_path.parent.mkdir(parents=True, exist_ok=True)
            skill_path.write_text(render_skill(bundle), encoding="utf-8")

        manifest[category] = {
            "server": f"biomni-{category}",
            "role_target": ROLE_TARGETS.get(category, "generalist"),
            "tools": [tool.name for tool in keep],
            "skills": [b.skill_id for b in bundles],
            "skipped_unbundled": unused,
        }

    # Merge with the existing manifest instead of overwriting so
    # ``--phase 3`` (run after ``--phase 1`` and ``--phase 2``) preserves
    # the earlier entries. ``orchestrator.py`` loads this file as the
    # single source of truth, so a category that was migrated in an
    # earlier phase must stay in the manifest across later runs.
    existing_manifest: dict[str, dict] = {}
    if MANIFEST.exists():
        try:
            existing_manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing_manifest = {}
    merged: dict[str, dict] = dict(existing_manifest)
    merged.update(manifest)
    MANIFEST.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    # "no matching function" means BiOMNI's description registry drifted from
    # the implementation (e.g. bioimaging.create_registration_visualization).
    # The codegen drops the dangling description automatically, so we report
    # this as a warning rather than a hard error — the user just needs to
    # know the description list is incomplete.
    real_errors = [msg for msg in issues if "refusing to overwrite" in msg]
    warnings = [msg for msg in issues if msg not in real_errors]
    return {
        "manifest": merged,
        "categories": list(manifest),
        "errors": real_errors,
        "warnings": warnings,
    }


# Phase-3 bundling. These categories pull in heavy scientific stacks
# (scanpy, scvi-tools, SimpleITK, cellpose, flowkit, cobra, libsbml,
# esm, gget, gseapy, ...) that are NOT in environment.yml. Wrappers
# are generated normally but every tool call will surface as a JSON
# ``ImportError`` until the user adds the corresponding ``pip`` line.
# Bundling still matters: it lets skill-routed runs expose only the
# tools the current step needs, even when some are unavailable.
PHASE_3_BUNDLES: dict[str, list[Bundle]] = {
    "genomics": [
        Bundle(
            skill_id="biomni-genomics-scrna-celltype",
            title="scRNA-seq annotation & embedding",
            description="Use to run cell-type annotation on a single-cell RNA-seq AnnData (CellTypist / scANVI / UCE / IMAP + interpretation), batch-corrected embeddings (scVI / Harmony / STATE), or run gene-set enrichment on the marker list.",
            role="geneticist",
            server="biomni-genomics",
            tools=[
                "annotate_celltype_scRNA",
                "annotate_celltype_with_panhumanpy",
                "create_scvi_embeddings_scRNA",
                "create_harmony_embeddings_scRNA",
                "get_uce_embeddings_scRNA",
                "map_to_ima_interpret_scRNA",
                "unsupervised_celltype_transfer_between_scRNA_datasets",
                "generate_embeddings_with_state",
            ],
            body=(
                "## When to use\n"
                "- User has a single-cell AnnData and wants cell-type labels "
                "(CellTypist / scANVI / UCE / IMAP) or batch-corrected "
                "embeddings (scVI / Harmony / STATE).\n"
                "- User wants to transfer labels from a reference atlas to a "
                "query dataset.\n\n"
                "## Limitations\n"
                "- Requires ``scanpy`` + a CellTypist / scANVI / UCE checkpoint "
                "(NOT installed by default — wrapper returns JSON error if "
                "missing).\n"
                "- Heavy GPU recommended for scVI / UCE embeddings; CPU runs are "
                "very slow.\n"
                "- Returns a research-log string; embedding vectors must be "
                "parsed out if needed downstream."
            ),
        ),
        Bundle(
            skill_id="biomni-genomics-gsea-archs4",
            title="Gene-set enrichment & ARCHS4",
            description="Use to run a gene-set enrichment analysis on a ranked gene list (Enrichr / GSEA), or fetch ARCHS4 co-expression / expression-by-gene signatures.",
            role="geneticist",
            server="biomni-genomics",
            tools=[
                "gene_set_enrichment_analysis",
                "get_gene_set_enrichment_analysis_supported_database_list",
                "get_rna_seq_archs4",
            ],
            body=(
                "## When to use\n"
                "- User has a ranked gene list and wants Enrichr / GSEA pathway "
                "enrichment.\n"
                "- User wants the supported Enrichr database list to pick the "
                "right library.\n"
                "- User wants ARCHS4 co-expression / sample-by-gene matrix.\n\n"
                "## Limitations\n"
                "- Requires ``gseapy`` (NOT installed by default).\n"
                "- Returns a research-log string; pathway p-values must be parsed."
            ),
        ),
        Bundle(
            skill_id="biomni-genomics-chipseq-motifs",
            title="ChIP-seq peak calling & motif analysis",
            description="Use to call peaks on ChIP-seq aligned reads (MACS2), find enriched motifs (HOMER), or analyze a genomic region's overlap with a known annotation set.",
            role="geneticist",
            server="biomni-genomics",
            tools=[
                "perform_chipseq_peak_calling_with_macs2",
                "find_enriched_motifs_with_homer",
                "analyze_genomic_region_overlap",
                "analyze_chromatin_interactions",
            ],
            body=(
                "## When to use\n"
                "- User has aligned ChIP-seq reads (BAM) and wants MACS2 peaks.\n"
                "- User has a peak set and wants HOMER motif enrichment.\n"
                "- User has a genomic region BED and wants overlap with a "
                "reference annotation.\n"
                "- User has Hi-C / ChIA-PET data and wants loop / TAD "
                "annotation.\n\n"
                "## Limitations\n"
                "- MACS2 + HOMER must be installed system-wide (NOT in "
                "environment.yml).\n"
                "- Chromatin-interaction tools may require ``cooltools`` / "
                "``HiCExplorer``."
            ),
        ),
        Bundle(
            skill_id="biomni-genomics-comparative-protein-embed",
            title="Comparative genomics & protein-language embeddings",
            description="Use to run comparative-genomics / haplotype analysis on a multi-species alignment, run interspecies gene-name conversion, or generate protein / transcript embeddings with ESM / TranscriptFormer.",
            role="geneticist",
            server="biomni-genomics",
            tools=[
                "analyze_comparative_genomics_and_haplotypes",
                "interspecies_gene_conversion",
                "generate_gene_embeddings_with_ESM_models",
                "generate_transcriptformer_embeddings",
                "detect_and_annotate_somatic_mutations",
                "detect_and_characterize_structural_variations",
            ],
            body=(
                "## When to use\n"
                "- User has a multi-species alignment and wants a haplotype / "
                "selection scan.\n"
                "- User has mouse / rat / zebrafish gene symbols and wants the "
                "human ortholog.\n"
                "- User has a protein sequence and wants ESM / TranscriptFormer "
                "embeddings.\n"
                "- User has a VCF and wants somatic-mutation annotation / "
                "structural-variant characterization.\n\n"
                "## Limitations\n"
                "- Protein / transcript embeddings require ``esm`` / "
                "``transcriptformer`` + a downloaded checkpoint (GB-scale).\n"
                "- Variant callers assume a reference genome; the actual fasta "
                "path is environment-dependent."
            ),
        ),
    ],
    "bioimaging": [
        Bundle(
            skill_id="biomni-bioimaging-registration",
            title="Medical-image registration (rigid / affine / deformable)",
            description="Use to run rigid / affine / deformable registration on a pair of medical images, batch-register a folder of moving images, or compute a similarity metric between two images.",
            role="cell-biologist",
            server="biomni-bioimaging",
            tools=[
                "quick_rigid_registration",
                "quick_affine_registration",
                "quick_deformable_registration",
                "batch_register_images",
                "calculate_similarity_metrics",
            ],
            body=(
                "## When to use\n"
                "- User has a moving + fixed image pair and wants rigid / affine / "
                "deformable registration.\n"
                "- User has a folder of moving images and a fixed image and "
                "wants batch registration.\n"
                "- User has two aligned images and wants a similarity metric "
                "(NCC / MI / SSIM).\n\n"
                "## Limitations\n"
                "- Requires ``SimpleITK`` + ``nibabel`` (NOT installed by default).\n"
                "- Deformable registration is slow on CPU; a GPU build of "
                "SimpleITK is preferred.\n"
                "- ``create_registration_visualization`` is described but has no "
                "matching implementation, so the codegen skips it."
            ),
        ),
        Bundle(
            skill_id="biomni-bioimaging-nnunet",
            title="nnU-Net segmentation pipeline",
            description="Use to split a multi-channel microscopy volume into per-channel images, prepare the input folder for nnU-Net, run nnU-Net inference, and render the resulting segmentation overlay.",
            role="cell-biologist",
            server="biomni-bioimaging",
            tools=[
                "split_modalities",
                "prepare_input_for_nnunet",
                "segment_with_nn_unet",
                "create_segmentation_visualization",
            ],
            body=(
                "## When to use\n"
                "- User has a multi-channel microscopy volume and wants per-"
                "channel splits.\n"
                "- User wants a folder in nnU-Net's expected format.\n"
                "- User has an nnU-Net task set up and wants inference on a new "
                "image.\n"
                "- User has a segmentation mask + image and wants an overlay "
                "figure.\n\n"
                "## Limitations\n"
                "- Requires ``nnunet``, ``SimpleITK``, ``nibabel`` (NOT installed "
                "by default).\n"
                "- nnU-Net inference requires a pre-trained model checkpoint and "
                "the ``nnUNet_results`` env var pointing at it.\n"
                "- ``segment_with_nn_unet`` typically needs a GPU; CPU fallback "
                "is very slow."
            ),
        ),
    ],
    "cell_biology": [
        Bundle(
            skill_id="biomni-cellbio-microscopy-quant",
            title="Microscopy quantification (cell cycle, motility, mitochondria)",
            description="Use to quantify cell-cycle phases from Calcofluor-white microscopy, cluster cell-motility tracks, or analyze mitochondrial morphology & membrane potential from fluorescence images.",
            role="cell-biologist",
            server="biomni-cell_biology",
            tools=[
                "quantify_cell_cycle_phases_from_microscopy",
                "quantify_and_cluster_cell_motility",
                "analyze_mitochondrial_morphology_and_potential",
            ],
            body=(
                "## When to use\n"
                "- User has Calcofluor-stained microscopy images and wants G1 / S "
                "/ G2-M proportions.\n"
                "- User has cell-track time-series and wants motility-cluster "
                "classification.\n"
                "- User has fluorescence images of mitochondria and wants "
                "morphology / membrane-potential quantification.\n\n"
                "## Limitations\n"
                "- Requires ``cellpose`` / ``scikit-image`` (NOT installed by "
                "default).\n"
                "- Classifier is a heuristic random-forest trained on synthetic "
                "features; treat as exploratory."
            ),
        ),
        Bundle(
            skill_id="biomni-cellbio-flow-facs",
            title="Flow cytometry & FACS",
            description="Use to perform simulated FACS cell sorting, or analyze flow-cytometry immunophenotyping panels.",
            role="cell-biologist",
            server="biomni-cell_biology",
            tools=[
                "perform_facs_cell_sorting",
                "analyze_flow_cytometry_immunophenotyping",
            ],
            body=(
                "## When to use\n"
                "- User has marker-expression data and wants a simulated FACS "
                "sort.\n"
                "- User has an immunophenotyping panel and wants per-cluster "
                "abundance.\n\n"
                "## Limitations\n"
                "- Requires ``flowkit`` / ``anndata`` (NOT installed by default).\n"
                "- FACS sort is a simulation; not connected to a physical sorter."
            ),
        ),
    ],
    "cancer_biology": [
        Bundle(
            skill_id="biomni-cancer-genomics-network",
            title="Cancer DDR network & mutation / SV analysis",
            description="Use to model the DNA-damage-response network in a tumor sample, detect / annotate somatic mutations and structural variations from a VCF, run NMF on expression data, or estimate copy-number purity / ploidy.",
            role="cell-biologist",
            server="biomni-cancer_biology",
            tools=[
                "analyze_ddr_network_in_cancer",
                "detect_and_annotate_somatic_mutations",
                "detect_and_characterize_structural_variations",
                "perform_gene_expression_nmf_analysis",
                "analyze_copy_number_purity_ploidy_and_focal_events",
            ],
            body=(
                "## When to use\n"
                "- User has a tumor mutation list and wants DDR-pathway "
                "enrichment.\n"
                "- User has a VCF and wants somatic-mutation annotation + SV "
                "characterization.\n"
                "- User has an expression matrix and wants NMF factor "
                "decomposition.\n"
                "- User has a copy-number profile and wants purity / ploidy / "
                "focal-event estimation.\n\n"
                "## Limitations\n"
                "- Requires ``gseapy`` / ``scanpy`` + variant-caller "
                "dependencies (NOT installed by default).\n"
                "- Results are research-log strings; numeric estimates must be "
                "parsed out."
            ),
        ),
        Bundle(
            skill_id="biomni-cancer-senescence-apoptosis",
            title="Cell senescence & apoptosis scoring",
            description="Use to score senescence and apoptosis signatures from bulk or single-cell expression data.",
            role="cell-biologist",
            server="biomni-cancer_biology",
            tools=[
                "analyze_cell_senescence_and_apoptosis",
            ],
            body=(
                "## When to use\n"
                "- User has an expression matrix and wants senescence / "
                "apoptosis signature scores.\n\n"
                "## Limitations\n"
                "- Requires ``scanpy`` + the underlying signature sets (NOT "
                "installed by default)."
            ),
        ),
    ],
    "immunology": [
        Bundle(
            skill_id="biomni-immuno-atac-motility",
            title="ATAC-seq + immune-cell motility",
            description="Use to run differential accessibility analysis on ATAC-seq peak counts, isolate / purify immune-cell populations in silico, track immune cells under flow, or estimate cell-cycle phase durations from a time-course.",
            role="cell-biologist",
            server="biomni-immunology",
            tools=[
                "analyze_atac_seq_differential_accessibility",
                "isolate_purify_immune_cells",
                "track_immune_cells_under_flow",
                "estimate_cell_cycle_phase_durations",
            ],
            body=(
                "## When to use\n"
                "- User has ATAC-seq peak counts for two conditions and wants "
                "DAR analysis.\n"
                "- User wants an immune-cell subset isolation plan from a "
                "marker panel.\n"
                "- User has immune-cell tracking data under flow and wants "
                "speed / direction stats.\n"
                "- User has a time-course BrdU / EdU staining and wants cell-"
                "cycle phase durations.\n\n"
                "## Limitations\n"
                "- Requires ``scanpy`` / ``diffbind`` / cell-tracking libs (NOT "
                "installed by default).\n"
                "- ``isolate_purify_immune_cells`` is a planning helper; not "
                "connected to a FACS sorter."
            ),
        ),
        Bundle(
            skill_id="biomni-immuno-proliferation-cytokine",
            title="Proliferation, cytokine & histology",
            description="Use to analyze CFSE-based proliferation, intracellular cytokine staining in CD4 T-cells, EBV antibody titers, CNS lesion histology, or immunohistochemistry images.",
            role="cell-biologist",
            server="biomni-immunology",
            tools=[
                "analyze_cfse_cell_proliferation",
                "analyze_cytokine_production_in_cd4_tcells",
                "analyze_ebv_antibody_titers",
                "analyze_cns_lesion_histology",
                "analyze_immunohistochemistry_image",
                "analyze_bacterial_growth_curve",
            ],
            body=(
                "## When to use\n"
                "- User has CFSE dye-dilution data and wants division-index "
                "estimates.\n"
                "- User has intracellular-cytokine staining and wants per-cell-"
                "type cytokine frequency.\n"
                "- User has EBV antibody titer panel and wants recent / past "
                "infection classification.\n"
                "- User has CNS-lesion histology slides and wants lesion "
                "quantification.\n"
                "- User has IHC images and wants marker-positive cell counts.\n"
                "- User has a bacterial growth curve and wants OD-vs-time fit.\n\n"
                "## Limitations\n"
                "- Requires ``flowkit`` / ``scikit-image`` (NOT installed by "
                "default).\n"
                "- Histology / IHC image analysis is heuristic — verify with a "
                "trained pathologist before publication."
            ),
        ),
    ],
}


PHASES: dict[int, dict[str, list[Bundle]]] = {
    1: PHASE_1_BUNDLES,
    2: PHASE_2_BUNDLES,
    3: PHASE_3_BUNDLES,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        type=int,
        default=1,
        help="Migration phase (default: 1).",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="Override category list (default = PHASE_<N> set).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite server files even if they don't look generated.",
    )
    args = parser.parse_args()

    if args.categories:
        cats = list(args.categories)
    elif args.phase in PHASES:
        cats = list(PHASES[args.phase].keys())
    else:
        parser.error(f"unknown phase {args.phase}; valid: {sorted(PHASES)}")

    result = run(cats, force=args.force)
    print(json.dumps(result, indent=2))
    if result["errors"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())