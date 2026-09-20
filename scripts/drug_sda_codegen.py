#!/usr/bin/env python3
"""Generate the DrugSDA-Tool inventory, tool schemas, and skill packages.

The remote SCP-HUB server at ``tools/pharma/_sda_config.py:REMOTE_SERVER_URL``
exposes 81 MCP tools. Three things need to agree on that set:

1. ``tools/pharma/drug_sda_server.py`` — the local stdio wrapper, which
   registers every remote tool. It reads ``_tool_schemas.json``.
2. ``orchestrator.py`` — which needs the tool *names* to build
   ``mcp__<server>__<tool>`` entries in ``--allowedTools``. It reads
   ``_manifest.json`` via ``_load_pharma_inventory()``.
3. ``scimas_skills/pharma-drug-sda-*/SKILL.md`` — the routing layer that
   decides which subset an agent sees for a given task.

This script is the only writer of all three, so there is exactly one source
of truth: the live ``tools/list`` response from the remote server. It is the
much smaller sibling of ``scripts/biomni_codegen.py`` — that one parses
BiOMNI's on-disk descriptions with AST because BiOMNI is a local checkout,
whereas DrugSDA is a remote server we can simply ask.

Usage:
    python scripts/drug_sda_codegen.py            # refresh everything
    python scripts/drug_sda_codegen.py --check     # verify, write nothing
    python scripts/drug_sda_codegen.py --force     # allow shrinking the set

``--check`` exits non-zero when the checked-in artifacts have drifted from
the remote, which is what CI / a curious human wants.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools.pharma._sda_config import REMOTE_SERVER_URL, resolve_api_key  # noqa: E402

PHARMA_DIR = REPO / "tools" / "pharma"
SKILL_DIR = REPO / "scimas_skills"
MANIFEST = PHARMA_DIR / "_manifest.json"
SCHEMAS = PHARMA_DIR / "_tool_schemas.json"

SERVER_NAME = "pharma-drug-sda"

# The remote server hands out a flat namespace spanning four disciplines'
# worth of capability, so one skill per capability keeps each agent step's
# tool list small (repo convention: median 4 tools/skill, max 13, no skill
# over 20). Every remote tool appears in exactly one bundle — `validate()`
# fails the run otherwise.
FILE_IO_NOTE = """
The remote server keeps its own filesystem. Tools that take a `*_file_path`
argument expect a path **on the server**, not a local one. To use a local
structure: call `base64_to_server_file` with the file's base64 content and a
name, then pass the returned server path to the analysis tool, and
`server_file_to_base64` to bring a result back (files must be under 10MB).
"""

BUNDLES: dict[str, dict] = {
    "pharma-drug-sda-mol-descriptors": {
        "role": "drug-discovery-scientist",
        "description": (
            "Use for RDKit molecular descriptors over SMILES lists: basic "
            "properties, partial charges, complexity, drug-likeness, "
            "hydrogen bonding, hydrophobicity, and topology."
        ),
        "tools": [
            "calculate_mol_basic_info",
            "calculate_mol_charge",
            "calculate_mol_complexity",
            "calculate_mol_drug_chemistry",
            "calculate_mol_hbond",
            "calculate_mol_hydrophobicity",
            "calculate_mol_structure_complexity",
            "calculate_mol_topology",
            "is_valid_smiles",
        ],
        "body": """
Compute descriptor sets for candidate molecules. Validate SMILES with
`is_valid_smiles` first and report invalid entries explicitly rather than
dropping them silently.

These are triage metrics. QED, charge, complexity and hydrophobicity
descriptors support comparison and prioritisation; they do not establish
safety, efficacy, or clinical viability.
""",
    },
    "pharma-drug-sda-admet-similarity": {
        "role": "drug-discovery-scientist",
        "description": (
            "Use for ADMET property prediction, disease-associated DLEPS "
            "scoring, Tanimoto similarity to a reference molecule, shared "
            "fragment counts, format conversion, and name-to-SMILES lookup."
        ),
        "tools": [
            "pred_mol_admet",
            "calculate_dleps_score",
            "calculate_morgan_fingerprint_similarity",
            "calculate_common_fragments",
            "convert_smiles_to_format",
            "retrieve_smiles_by_compoundname",
        ],
        "body": """
Predict ADMET profiles and rank candidates against a reference molecule.

`pred_mol_admet` returns over 90 key-value properties per molecule; summarise
the absorption/distribution/metabolism/excretion/toxicity headline values
rather than dumping the full dictionary. `calculate_dleps_score` needs a
disease name and identifies upregulated targets, so state the disease context
in your report.

ADMET predictions are model estimates. Present them as predicted
liabilities to investigate, not measured outcomes.
""",
    },
    "pharma-drug-sda-molecule-generation": {
        "role": "drug-discovery-scientist",
        "description": (
            "Use for generative molecule design: de novo sampling, "
            "molecule-to-molecule optimisation, scaffold R-group and linker "
            "sampling, and peptide sampling."
        ),
        "tools": [
            "reinvent_denovo_sampling",
            "reinvent_mol2mol_sampling",
            "libinvent_rgroup_sampling_by_scaffold",
            "libinvent_rgroup_sampling_by_scaffold_name",
            "linkinvent_linker_sampling_by_warhead_pair_name",
            "linkinvent_linker_sampling_by_warheads",
            "pepinvent_peptide_sampling_by_peptide",
            "pepinvent_peptide_sampling_by_template",
            "get_pepinvent_info",
        ],
        "body": """
Generate candidate molecules with the REINVENT family of models.

Call `get_pepinvent_info` before the pepinvent tools — it lists the preset
templates and amino-acid alphabets that the `*_by_template` variants accept.

Generated molecules are proposals. Validate them (`is_valid_smiles`) and
score them with the descriptor or ADMET skills before reporting, and never
present a generated structure as a synthesised or tested compound.
""",
    },
    "pharma-drug-sda-structure-prediction-design": {
        "role": "structural-biologist",
        "description": (
            "Use to predict protein 3D structure from sequence and to design "
            "protein sequences or binders: ESMFold, Chai-1, Chroma, "
            "ProteinMPNN, and EvoBind2."
        ),
        "tools": [
            "pred_protein_structure_esmfold",
            "chai1_predict",
            "chroma_monomer",
            "chroma_complex",
            "chroma_symmetry",
            "proteinmpnn_tool",
            "evobind_tool",
        ],
        "body": """
Predict structures and design sequences.

`pred_protein_structure_esmfold` and `chai1_predict` take a sequence and
return a PDB path on the remote server. Chroma tools generate de novo
backbones (monomer, complex, symmetric); `proteinmpnn_tool` designs sequences
onto a given backbone and can constrain the design interface;
`evobind_tool` designs linear or cyclic peptide binders from a receptor
sequence.

These are predictions and designs. Report the model used and treat the output
as a hypothesis — a predicted fold is not an experimental structure.
""" + FILE_IO_NOTE,
    },
    "pharma-drug-sda-structure-prep": {
        "role": "structural-biologist",
        "description": (
            "Use to clean up and prepare protein structures: repair missing "
            "atoms and residues, rebuild incomplete models, repack sidechains, "
            "extract chains, and convert CIF to PDB."
        ),
        "tools": [
            "fix_pdb",
            "pulchura_rebuild",
            "pack_sidechains",
            "extract_and_save_chains",
            "extract_pdb_chains",
            "convert_complex_cif_to_pdb",
        ],
        "body": """
Prepare a structure for downstream simulation, docking, or design.

A typical order: `convert_complex_cif_to_pdb` (if needed) →
`fix_pdb` for missing atoms, hydrogens and heterogens →
`pulchura_rebuild` for grossly incomplete backbones →
`pack_sidechains` to restore full-atom sidechains.

`extract_pdb_chains` returns sequences; `extract_and_save_chains` writes chain
files. Report which repairs were applied and what was removed — deleting
heterogens or repairing residues changes what the structure represents.
""" + FILE_IO_NOTE,
    },
    "pharma-drug-sda-pdb-utils": {
        "role": "structural-biologist",
        "description": (
            "Use to inspect a PDB structure or validate a protein sequence: "
            "basic statistics, atom composition, quality metrics, geometric "
            "properties, and sequence physicochemical properties."
        ),
        "tools": [
            "calculate_pdb_basic_info",
            "calculate_pdb_composition_info",
            "calculate_pdb_quality_metrics",
            "calculate_pdb_structural_geometry",
            "calculate_protein_sequence_properties",
            "is_valid_protein_sequence",
        ],
        "body": """
Characterise a structure or sequence before doing anything heavier with it.

Run `is_valid_protein_sequence` before submitting a sequence to any
prediction tool, and `calculate_pdb_quality_metrics` before trusting a
downloaded structure. These are cheap checks that catch most bad inputs.

Cα-based geometry and composition statistics describe the deposited model.
They do not by themselves establish experimental quality.
""" + FILE_IO_NOTE,
    },
    "pharma-drug-sda-retrieval-visualization": {
        "role": "structural-biologist",
        "description": (
            "Use to fetch protein sequences and structures by identifier and "
            "to render protein or molecule images, plus moving files to and "
            "from the remote server."
        ),
        "tools": [
            "retrieve_protein_sequence",
            "retrieve_protein_structure_by_gene_name",
            "retrieve_protein_structure_by_pdb_id",
            "retrieve_protein_structure_by_uniprot_id",
            "visualize_protein",
            "visualize_molecule",
            "base64_to_server_file",
            "server_file_to_base64",
        ],
        "body": """
Retrieve inputs and render figures.

Structure lookups accept a PDB ID, UniProt ID, or gene name and fall back
from `.pdb` to `.cif` automatically — check which format came back before
passing the path on. `retrieve_protein_sequence` takes a gene name or UniProt
ID plus an organism.

This skill also owns the file bridge to the remote server, which the other
pharma-drug-sda skills depend on.
""" + FILE_IO_NOTE,
    },
    "pharma-drug-sda-docking": {
        "role": "computational-chemist",
        "description": (
            "Use to dock molecules and peptides into protein structures and "
            "to detect and score binding pockets: HDOCK, KarmaDock, "
            "QuickVina2-GPU, P2Rank, fpocket, EquiScore, and ProLIF pose "
            "summaries."
        ),
        "tools": [
            "hdock_tool",
            "karmadock_tool",
            "molecule_docking_quickvina_fullprocess",
            "convert_pdb_to_pdbqt_dock",
            "prolif_docking",
            "equiscore_pipeline",
            "equiscore_pocket",
            "equiscore_screen",
            "pred_pocket_prank",
            "fpocket_toolkit",
        ],
        "body": """
Dock ligands or peptides and score the poses.

If the binding site is unknown, detect pockets first with
`pred_pocket_prank` or `fpocket_toolkit`; otherwise docking tools either take
a box or select a site themselves. `convert_pdb_to_pdbqt_dock` prepares a
receptor when a tool needs PDBQT. `prolif_docking` summarises poses as
interaction fingerprints, and `equiscore_*` re-scores them (pocket →
pipeline → screen).

Docking scores rank poses within one run. They are not binding affinities and
do not compare across targets or protocols — say so when reporting a "best"
pose.
""" + FILE_IO_NOTE,
    },
    "pharma-drug-sda-binding-affinity": {
        "role": "computational-chemist",
        "description": (
            "Use to estimate binding affinity and analyse interactions: "
            "Boltz-2 affinity prediction, FoldX stability and interface "
            "analysis, ProLIF interaction fingerprints, and residue numbering "
            "between UniProt, PDB, and tool conventions."
        ),
        "tools": [
            "pred_binding_affinity_boltz2",
            "foldx_tool",
            "analyze_protein_ligand_interactions",
            "interaction_visualizer",
            "prolif_pdb",
            "prolif_protein_protein",
            "residue_mapper",
        ],
        "body": """
Estimate affinity and characterise interfaces.

Use `residue_mapper` whenever a residue number must line up between a
UniProt entry, a PDB file and a tool's internal numbering — mismatched
numbering silently produces wrong mutations and wrong interfaces.

`pred_binding_affinity_boltz2` gives a predicted affinity for a
protein–small-molecule pair; `foldx_tool` covers stability, mutation and
interface analysis; the ProLIF tools return interaction fingerprints for a
complex, a trajectory, or a protein–protein pair.

Predicted affinities are estimates with no experimental error bars. Report
them as such and prefer relative comparisons within one protocol.
""" + FILE_IO_NOTE,
    },
    "pharma-drug-sda-md-mmpbsa": {
        "role": "physical-chemist",
        "description": (
            "Use to set up and run molecular dynamics and end-point free "
            "energy calculations: OpenMM and GROMACS MM/PBSA protein–ligand "
            "and protein–protein workflows, coarse-grained OpenAWSM and GoCa "
            "simulations, BioEmu conformational sampling, and trajectory "
            "extraction."
        ),
        "tools": [
            "protein_openmm_md",
            "prepare_protein_md",
            "prepare_complex",
            "run_mmpbsa",
            "analyze_mmpbsa",
            "gmx_mmpbsa_propro",
            "openmm_extract_frames",
            "prolif_md",
            "openawsem_sim",
            "openawsem_traj_extract",
            "goca_pipeline",
            "run_bioemu",
            "extract_bioemu_structures",
        ],
        "body": """
Run simulations and free-energy calculations. These are the slowest tools in
the set — expect minutes, and treat a timeout as "not finished", not
"failed": do not silently retry with a shorter budget and report the
truncated result as converged.

Protein–ligand path: `prepare_complex` → `run_mmpbsa` → `analyze_mmpbsa`.
Protein–protein path: `prepare_protein_md` → `gmx_mmpbsa_propro`.
`protein_openmm_md` runs a plain OpenMM simulation and returns its run
directory. For conformational ensembles use `run_bioemu`, then
`extract_bioemu_structures`; for coarse-grained work `openawsem_sim` /
`openawsem_traj_extract` or `goca_pipeline`.

MM/PBSA and MM/GBSA end-point estimates are sensitive to the setup and are
not rigorous free energies. State the protocol, the sampling, and the
approximations alongside any number.
""" + FILE_IO_NOTE,
    },
}


async def fetch_remote_tools() -> dict[str, dict]:
    """Return {tool_name: {description, input_schema}} from the remote server."""
    try:
        import httpx2
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(f"mcp/httpx2 not importable: {exc}")

    key = resolve_api_key()
    if not key:
        raise SystemExit(
            "No API key. Set DRUGSDA_API_KEY or SCP_HUB_API_KEY in the "
            "environment, or put it in config/mcp.local.json."
        )

    client = httpx2.AsyncClient(headers={"SCP-HUB-API-KEY": key})
    try:
        async with streamable_http_client(
            url=REMOTE_SERVER_URL, http_client=client
        ) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
    finally:
        await client.aclose()

    return {
        tool.name: {
            "description": tool.description or "",
            "input_schema": tool.input_schema or {},
        }
        for tool in result.tools
    }


def validate(remote: dict[str, dict]) -> list[str]:
    """Check the bundles partition the remote tool set exactly."""
    errors: list[str] = []
    assigned: list[str] = [t for b in BUNDLES.values() for t in b["tools"]]

    seen: set[str] = set()
    dupes = sorted({t for t in assigned if t in seen or seen.add(t)})
    if dupes:
        errors.append(f"tools assigned to more than one bundle: {dupes}")

    missing = sorted(set(assigned) - set(remote))
    if missing:
        errors.append(f"bundled tools the remote does not expose: {missing}")

    unbundled = sorted(set(remote) - set(assigned))
    if unbundled:
        errors.append(
            f"remote tools in no bundle ({len(unbundled)}): {unbundled}"
        )
    return errors


def render_skill(skill_id: str, bundle: dict) -> str:
    tools = "\n".join(f"  - {t}" for t in bundle["tools"])
    return (
        f"---\n"
        f"name: {skill_id}\n"
        f"description: {bundle['description']}\n"
        f"x-scimas-role: {bundle['role']}\n"
        f"x-scimas-server: {SERVER_NAME}\n"
        f"x-scimas-tools:\n{tools}\n"
        f"---\n"
        f"{bundle['body'].strip()}\n"
    )


def build_manifest(remote: dict[str, dict]) -> dict:
    all_tools = sorted(remote)
    return {
        "drug-sda": {
            "server": SERVER_NAME,
            "role_targets": sorted({b["role"] for b in BUNDLES.values()}),
            "tools": all_tools,
            "skills": {
                skill_id: {"role": b["role"], "tools": list(b["tools"])}
                for skill_id, b in BUNDLES.items()
            },
            "skipped_unbundled": [],
        }
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify checked-in artifacts match the remote; write nothing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="write even if the remote set shrank (default: refuse, as a shrink "
        "usually means a wrong endpoint or a broken key)",
    )
    args = parser.parse_args()

    remote = asyncio.run(fetch_remote_tools())
    errors = validate(remote)
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1

    manifest = build_manifest(remote)
    manifest_text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    schemas_text = json.dumps(remote, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    skills = {
        skill_id: render_skill(skill_id, bundle)
        for skill_id, bundle in BUNDLES.items()
    }

    if args.check:
        drift: list[str] = []
        if not MANIFEST.exists() or MANIFEST.read_text(encoding="utf-8") != manifest_text:
            drift.append(str(MANIFEST.relative_to(REPO)))
        if not SCHEMAS.exists() or SCHEMAS.read_text(encoding="utf-8") != schemas_text:
            drift.append(str(SCHEMAS.relative_to(REPO)))
        for skill_id, text in skills.items():
            path = SKILL_DIR / skill_id / "SKILL.md"
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                drift.append(str(path.relative_to(REPO)))
        if drift:
            print("drifted from the remote server:", file=sys.stderr)
            for d in drift:
                print(f"  {d}", file=sys.stderr)
            return 1
        print(f"up to date: {len(remote)} tools, {len(BUNDLES)} skills")
        return 0

    previous = None
    if MANIFEST.exists():
        try:
            previous = json.loads(MANIFEST.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = None
    if previous and not args.force:
        old = set(previous.get("drug-sda", {}).get("tools", []))
        new = set(remote)
        if old - new:
            print(
                f"ERROR: remote no longer exposes {len(old - new)} tool(s) "
                f"present in {MANIFEST.name}: {sorted(old - new)}\n"
                f"Refusing to shrink the inventory — check the endpoint and the "
                f"API key, or pass --force.",
                file=sys.stderr,
            )
            return 1

    PHARMA_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(manifest_text, encoding="utf-8")
    SCHEMAS.write_text(schemas_text, encoding="utf-8")
    for skill_id, text in skills.items():
        path = SKILL_DIR / skill_id / "SKILL.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    print(
        json.dumps(
            {
                "server": SERVER_NAME,
                "tools": len(remote),
                "skills": sorted(skills),
                "manifest": str(MANIFEST.relative_to(REPO)),
                "schemas": str(SCHEMAS.relative_to(REPO)),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
