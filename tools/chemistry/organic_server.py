#!/usr/bin/env python3
"""Organic chemistry MCP server.

Tools ported from SciAgentGYM-main/toolkits/chemistry/organic_chemistry/.
The full SciAgentGYM organic tools use rdkit reaction templates and
external xtb/CREST for QM optimization; those require heavy deps that
are not assumed here. This port provides a deterministic rules-table
predictor covering the most common undergraduate-level transformations.

Tools
-----
- sn_product_predict: SN1 vs SN2 product selector.
- ester_hydrolysis_product: ester → carboxylic acid + alcohol (acid- or
  base-catalyzed).
- amide_hydrolysis_product: amide → carboxylic acid + amine.
"""

from __future__ import annotations

import json
import math
import re

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Crippen
    _RDKIT_OK = True
except ImportError:
    _RDKIT_OK = False

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("chemistry-organic")


def _round_energy(x) -> float:
    """Round a tiny or huge value to ~6 significant figures without
    dropping to 0. Returns the raw float when |x| is in a normal range.
    """
    if x == 0:
        return 0.0
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return float("nan")
    if xf != xf:  # NaN
        return xf
    mag = math.floor(math.log10(abs(xf)))
    if -3 <= mag <= 6:
        return float(round(xf, 6))
    return float(f"{xf:.6g}")

# Functional-group markers. SMILES uses lowercase 'c' (and 'n', 'o',
# 's') exclusively for aromatic atoms; uppercase C is always sp3
# aliphatic. So a case-sensitive 'c' check is the only reliable
# aromatic marker.
def _classify_carbon(smiles: str) -> str:
    if "c1ccccc1" in smiles or "c1ccccc" in smiles:
        return "aromatic"
    if "c" in smiles:  # any lowercase 'c' outside ring-closure digits
        return "aromatic"
    return "aliphatic"


def _count_aliphatic_c_neighbors(smiles: str) -> int:
    """Count C atoms directly bonded to the halogen-bearing carbon.

    Walks back from the first halogen over any (...) branches and
    the surrounding [...] bracket to find the halogen-bearing C, then
    counts: (a) one 'C' if the atom to the immediate left of that
    C/bracket is an aliphatic C, and (b) one 'C' per C inside any
    branch attached to the halogen-bearing C.

    Returns 0 for methyl halides, 1 for primary, 2 for secondary, 3+
    for tertiary. Returns -1 when the substrate contains no halogen
    (no halogen-bearing C to classify).

    This is a teaching-grade heuristic, not a real SMILES parser.
    """
    halogens = ["Cl", "Br", "I", "F"]
    for h in halogens:
        hal_idx = smiles.find(h)
        if hal_idx < 0:
            continue
        branch_count = 0
        i = hal_idx - 1
        walked = False
        # Process branches attached to the halogen-bearing C. In SMILES
        # these are written between the C and the halogen.
        while i >= 0 and smiles[i] == ")":
            walked = True
            depth = 1
            j = i - 1
            while j >= 0 and depth > 0:
                if smiles[j] == ")":
                    depth += 1
                elif smiles[j] == "(":
                    depth -= 1
                j -= 1
            branch = smiles[j + 2 : i]
            branch_count += branch.count("C") + branch.count("c")
            i = j  # j is the char before '('
        # Process a possible [...] bracket surrounding the halogen-bearing C.
        if i >= 0 and smiles[i] == "]":
            walked = True
            depth = 1
            j = i - 1
            while j >= 0 and depth > 0:
                if smiles[j] == "]":
                    depth += 1
                elif smiles[j] == "[":
                    depth -= 1
                j -= 1
            i = j  # char before '['
        # If we walked, `i` is the left-neighbor position. Otherwise `i`
        # is the halogen-bearing C position itself (no left neighbor).
        left_count = 0
        if walked:
            if i >= 0 and smiles[i] == "C":
                left_count = 1
        else:
            if i >= 1 and smiles[i - 1] == "C":
                left_count = 1
        return branch_count + left_count
    return -1


def _is_primary_carbon(smiles: str) -> bool:
    if _classify_carbon(smiles) == "aromatic":
        return False
    n = _count_aliphatic_c_neighbors(smiles)
    # methyl (0) and primary (1) both count as primary
    return 0 <= n <= 1


def _is_tertiary_carbon(smiles: str) -> bool:
    if _classify_carbon(smiles) == "aromatic":
        return False
    n = _count_aliphatic_c_neighbors(smiles)
    return n >= 3


@mcp.tool(
    description=(
        "Predict the dominant product of a nucleophilic substitution "
        "between a nucleophile (Nu-, e.g. OH-, CN-, I-) and a substrate "
        "alkyl halide (R-X). Uses SN1 vs SN2 selection rules: primary "
        "→ SN2, tertiary → SN1, aromatic → no reaction."
    )
)
async def sn_product_predict(
    substrate_smiles: str,
    nucleophile: str,
    solvent: str = "polar_aprotic",
) -> str:
    if not substrate_smiles.strip():
        return json.dumps({"error": "substrate_smiles must be non-empty"})

    carbon_type = _classify_carbon(substrate_smiles)
    if carbon_type == "aromatic":
        mechanism = "no_reaction"
        yield_pct = 0.0
        note = "Aryl halides do not undergo SN1 or SN2; use Pd-catalyzed coupling instead."
    elif _is_tertiary_carbon(substrate_smiles):
        mechanism = "SN1"
        yield_pct = 70.0 if solvent == "polar_protic" else 20.0
        note = "Tertiary substrate favors SN1 (carbocation); use polar protic solvent."
    elif _is_primary_carbon(substrate_smiles):
        mechanism = "SN2"
        yield_pct = 85.0 if solvent == "polar_aprotic" else 30.0
        note = "Primary substrate favors SN2; use polar aprotic solvent (DMSO, acetone)."
    else:
        # secondary: mixture
        mechanism = "SN1+SN2"
        yield_pct = 50.0
        note = "Secondary substrate gives a mixture; solvent and temperature bias the ratio."

    # Strip the halogen from the substrate to make R, append nucleophile.
    product_smiles = substrate_smiles
    for h in ["Cl", "Br", "I", "F"]:
        if h in product_smiles:
            product_smiles = product_smiles.replace(h, "", 1)
            break

    return json.dumps(
        {
            "substrate": substrate_smiles,
            "nucleophile": nucleophile,
            "solvent": solvent,
            "mechanism": mechanism,
            "expected_yield_pct": yield_pct,
            "product_smiles": product_smiles,
            "note": note,
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Predict the products of ester hydrolysis. Returns the "
        "carboxylate (base-catalyzed) or carboxylic acid (acid-catalyzed) "
        "plus the corresponding alcohol. Uses a small SMILES-style "
        "string parser; full SMILES is not required — pass either a "
        "SMILES or a name and the tool will return approximate SMILES."
    )
)
async def ester_hydrolysis_product(
    ester_name: str = "",
    ester_smiles: str = "",
    conditions: str = "acid",
) -> str:
    if conditions not in ("acid", "base"):
        return json.dumps({"error": "conditions must be 'acid' or 'base'"})

    # Known easy cases (most common textbook examples).
    lookup = {
        "ethyl acetate": ("CC(=O)O", "CCO", "CC(=O)OCC"),
        "methyl acetate": ("CC(=O)O", "CO", "CC(=O)OC"),
        "ethyl benzoate": ("O=C(O)c1ccccc1", "CCO", "O=C(OCC)c1ccccc1"),
    }
    key = (ester_name or "").strip().lower()
    if key in lookup:
        acid, alcohol, _ = lookup[key]
        out_acid = acid if conditions == "acid" else acid.replace("O", "[O-]", 1)
    elif ester_smiles:
        # Naive: split on the second 'O' (the alkoxy oxygen) — this is
        # only an approximation for teaching, not a general parser.
        out_acid = ester_smiles.split("O", 1)[0] + "=O"
        out_acid = out_acid if conditions == "acid" else out_acid.replace("=O", "=[O-]")
        alcohol = ester_smiles.split("O", 1)[1] if "O" in ester_smiles else "?"
    else:
        return json.dumps(
            {"error": "Provide ester_name (textbook) or ester_smiles."}
        )

    return json.dumps(
        {
            "ester_name": key or None,
            "ester_smiles": ester_smiles or None,
            "conditions": conditions,
            "carboxylate_smiles": out_acid,
            "alcohol_smiles": alcohol,
            "note": "Approximate; not a general SMILES parser.",
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Predict the products of amide hydrolysis. Returns the "
        "carboxylate (or acid) and the amine. Pass an amide name (from "
        "a small table) or a SMILES-like string."
    )
)
async def amide_hydrolysis_product(
    amide_name: str = "",
    amide_smiles: str = "",
    conditions: str = "acid",
) -> str:
    if conditions not in ("acid", "base"):
        return json.dumps({"error": "conditions must be 'acid' or 'base'"})

    lookup = {
        "acetamide": ("CC(=O)O", "N", "CC(=O)N"),
        "benzamide": ("O=C(O)c1ccccc1", "N", "O=C(N)c1ccccc1"),
    }
    key = (amide_name or "").strip().lower()
    if key in lookup:
        acid, amine, _ = lookup[key]
        out_acid = acid if conditions == "acid" else acid.replace("O", "[O-]", 1)
    elif amide_smiles:
        out_acid = amide_smiles.split("N", 1)[0].rstrip() + "OH"
        out_acid = out_acid if conditions == "acid" else out_acid.replace("OH", "[O-]")
        amine = amide_smiles.split("N", 1)[1] if "N" in amide_smiles else "?"
    else:
        return json.dumps(
            {"error": "Provide amide_name (textbook) or amide_smiles."}
        )

    return json.dumps(
        {
            "amide_name": key or None,
            "amide_smiles": amide_smiles or None,
            "conditions": conditions,
            "carboxylate_smiles": out_acid,
            "amine_smiles": amine,
            "note": "Approximate; not a general SMILES parser.",
        },
        ensure_ascii=False,
    )


# ---- Wave 3: rdkit-dependent ports ----


@mcp.tool(
    description=(
        "Parse a SMILES string with rdkit. Returns the canonical "
        "SMILES, InChI, and InChIKey. Returns an error if the "
        "SMILES is invalid or rdkit is not installed."
    )
)
async def parse_smiles_rdkit(smiles: str) -> str:
    if not _RDKIT_OK:
        return json.dumps({"error": "rdkit not installed"})
    if not smiles.strip():
        return json.dumps({"error": "smiles must be non-empty"})
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return json.dumps({"error": f"invalid SMILES: {smiles}"})
    return json.dumps(
        {
            "input_smiles": smiles,
            "canonical_smiles": Chem.MolToSmiles(mol),
            "inchi": Chem.MolToInchi(mol),
            "inchikey": Chem.MolToInchiKey(mol),
            "n_atoms": mol.GetNumAtoms(),
            "n_heavy_atoms": mol.GetNumHeavyAtoms(),
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Predict reaction products by applying a SMARTS reaction "
        "template to a substrate SMILES. Uses rdkit's "
        "AllChem.ReactionFromSmarts and RunReactants. Returns up to "
        "5 product SMILES (canonical)."
    )
)
async def predict_smarts_product(
    substrate_smiles: str, reaction_smarts: str, max_products: int = 5
) -> str:
    if not _RDKIT_OK:
        return json.dumps({"error": "rdkit not installed"})
    try:
        rxn = AllChem.ReactionFromSmarts(reaction_smarts)
        if rxn is None:
            return json.dumps({"error": "invalid reaction SMARTS"})
        mol = Chem.MolFromSmiles(substrate_smiles)
        if mol is None:
            return json.dumps({"error": f"invalid substrate SMILES: {substrate_smiles}"})
        products_sets = rxn.RunReactants((mol,))
    except Exception as exc:  # rdkit raises C++ exceptions
        return json.dumps({"error": f"reaction failed: {exc}"})

    seen = set()
    products: list[str] = []
    for tup in products_sets:
        for p in tup:
            try:
                smi = Chem.MolToSmiles(p)
            except Exception:
                continue
            if smi in seen:
                continue
            seen.add(smi)
            products.append(smi)
            if len(products) >= max_products:
                break
        if len(products) >= max_products:
            break
    return json.dumps(
        {
            "substrate_smiles": substrate_smiles,
            "reaction_smarts": reaction_smarts,
            "n_products": len(products),
            "product_smiles": products,
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Predict octanol-water logP of a molecule from its SMILES "
        "using rdkit's Crippen MolLogP (Wildman-Crippen atom "
        "contribution method). Requires rdkit."
    )
)
async def predict_logp_rdkit(smiles: str) -> str:
    if not _RDKIT_OK:
        return json.dumps({"error": "rdkit not installed"})
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return json.dumps({"error": f"invalid SMILES: {smiles}"})
    return json.dumps(
        {
            "smiles": smiles,
            "canonical_smiles": Chem.MolToSmiles(mol),
            "logP": round(float(Crippen.MolLogP(mol)), 4),
            "MR": round(float(Crippen.MolMR(mol)), 4),
        },
        ensure_ascii=False,
    )


# ---- Wave 2: SMARTS-driven reaction product predictor ----
# Ported from
# SciAgentGYM-main/toolkits/chemistry/organic_chemistry/
# organic_reaction_simulator_15923.py:generate_product_via_smarts.
# Helper functions (_mol_from_smiles_safe, _has_substructure,
# _has_leaving_group, _pick_organic_product) are inlined here to avoid
# dependency on the SciAgentGYM internals.
# ---------------------------------------------------------------------------

def _org_mol_from_smiles_safe(smiles: str):
    """Best-effort SMILES normalization + rdkit parse."""
    if smiles is None:
        return None
    s = str(smiles).strip()
    s = re.sub(r"(?<!\[)O-(?!\])", "[O-]", s)
    s = re.sub(r"(?<!\[)N-(?!\])", "[N-]", s)
    s = re.sub(r"(?<!\[)S-(?!\])", "[S-]", s)
    try:
        return Chem.MolFromSmiles(s)
    except Exception:
        return None


def _org_has_substructure(mol, smarts: str) -> bool:
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        return False
    return mol.HasSubstructMatch(patt)


def _org_has_leaving_group(mol) -> bool:
    return _org_has_substructure(mol, "[Br,Cl,I]")


def _org_pick_product(pset):
    for p in pset:
        if p is None:
            continue
        try:
            smi = Chem.MolToSmiles(p)
        except Exception:
            continue
        if any(anion in smi for anion in ("[Br-]", "[Cl-]", "[I-]")) or not smi.strip():
            continue
        # rdkit reaction products sometimes carry unusual valence
        # warnings (e.g. CC[O-]C from SN2 on neutral alkoxide); sanitize
        # permissively then re-canonicalize.
        try:
            m = Chem.MolFromSmiles(smi, sanitize=False)
            if m is None:
                continue
            Chem.SanitizeMol(
                m,
                sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES,
            )
            return Chem.MolToSmiles(m, canonical=True)
        except Exception:
            # Fallback: return the raw SMILES even if rdkit won't
            # re-sanitize it. The caller gets a best-effort product.
            return smi
    return None


_ORG_SN2_OXYGEN_SMARTS = [
    "[O-;H0:1].[C;X4:2]-[Br:3]>>[O:1]-[C:2].[Br-:3]",
    "[O-;H0:1].[C;X4:2]-[Cl:3]>>[O:1]-[C:2].[Cl-:3]",
    "[O-;H0:1].[C;X4:2]-[I:3]>>[O:1]-[C:2].[I-:3]",
    "[O;X2;H1:1].[C;X4:2]-[Br:3]>>[O:1]-[C:2].[Br-:3]",
    "[O;X2;H1:1].[C;X4:2]-[Cl:3]>>[O:1]-[C:2].[Cl-:3]",
    "[O;X2;H1:1].[C;X4:2]-[I:3]>>[O:1]-[C:2].[I-:3]",
]


def _org_run_sn2_oxygen(nuc_mol, sub_mol):
    """Try each SN2_oxygen template, returning the first organic product.
    Silently skip templates rdkit rejects (valence / SMARTS errors).
    """
    for smarts in _ORG_SN2_OXYGEN_SMARTS:
        try:
            rxn = AllChem.ReactionFromSmarts(smarts)
            prods = rxn.RunReactants((nuc_mol, sub_mol))
        except Exception:
            continue
        for pset in prods:
            smi = _org_pick_product(pset)
            if smi:
                return smi
    return None


def _org_run_sn2_carbanion(nuc_mol, sub_mol):
    for hal_smart, anion_smart in (
        ("[C-:1].[C:2]-[Br:3]>>[C:1]-[C:2].[Br-:3]", "[Br-]"),
        ("[C-:1].[C:2]-[Cl:3]>>[C:1]-[C:2].[Cl-:3]", "[Cl-]"),
        ("[C-:1].[C:2]-[I:3]>>[C:1]-[C:2].[I-:3]", "[I-]"),
    ):
        try:
            rxn = AllChem.ReactionFromSmarts(hal_smart)
        except Exception:
            continue
        for pset in rxn.RunReactants((nuc_mol, sub_mol)):
            smi = _org_pick_product(pset)
            if smi:
                return smi
    return None


def _org_run_cation_hydration(nuc_mol, sub_mol):
    rxn = AllChem.ReactionFromSmarts("[O;H2:1].[C+:2]>>[O:1]-[C:2]")
    for pset in rxn.RunReactants((nuc_mol, sub_mol)):
        smi = _org_pick_product(pset)
        if smi:
            return smi
    return None


@mcp.tool(
    description=(
        "Predict the organic product of an SN2 / cation-hydration reaction "
        "using rdkit SMARTS templates. Pass a nucleophile SMILES "
        "(e.g. '[O-]C', '[C-]#N', 'O') and a substrate SMILES containing "
        "a halogen leaving group (e.g. 'CCBr'). Returns the canonical "
        "SMILES of the organic product, or null if no template matched."
    )
)
async def generate_product_via_smarts(
    nucleophile_smiles: str,
    substrate_smiles: str,
) -> str:
    if not _RDKIT_OK:
        return json.dumps({"error": "rdkit not installed"})
    nuc = _org_mol_from_smiles_safe(nucleophile_smiles)
    sub = _org_mol_from_smiles_safe(substrate_smiles)
    if nuc is None:
        return json.dumps({"error": f"invalid nucleophile SMILES: {nucleophile_smiles}"})
    if sub is None:
        return json.dumps({"error": f"invalid substrate SMILES: {substrate_smiles}"})

    tried: list[str] = []
    product_smi: str | None = None
    reaction_label: str | None = None

    if _org_has_leaving_group(sub) and (
        _org_has_substructure(nuc, "[O-]")
        or _org_has_substructure(nuc, "[O]")
    ):
        tried.append("SN2_oxygen_anion")
        product_smi = _org_run_sn2_oxygen(nuc, sub)
        reaction_label = "SN2_oxygen"

    if product_smi is None and _org_has_leaving_group(sub) and _org_has_substructure(nuc, "[C-]"):
        tried.append("SN2_carbanion")
        product_smi = _org_run_sn2_carbanion(nuc, sub)
        reaction_label = "SN2_carbanion"

    if product_smi is None and _org_has_substructure(sub, "[C+]") and _org_has_substructure(nuc, "[O]"):
        tried.append("cation_hydration")
        product_smi = _org_run_cation_hydration(nuc, sub)
        reaction_label = "cation_hydration"

    return json.dumps(
        {
            "nucleophile_smiles": nucleophile_smiles,
            "substrate_smiles": substrate_smiles,
            "reaction_tried": tried,
            "reaction_label": reaction_label,
            "product_smiles": product_smi,
        },
        ensure_ascii=False,
    )


# ---- Wave 3: hydrolysis + organic reaction predictor ----
# Ported from SciAgentGYM-main/toolkits/chemistry/organic_chemistry/
# organic_reaction_simulator_15923.py. SMARTS-based product prediction
# with Arrhenius-style half-life approximation. No external text-to-
# SMILES dependency — pass SMILES directly.
# ---------------------------------------------------------------------------


@mcp.tool(
    description=(
        "Predict the product and estimate half-life for hydrolysis of an "
        "organic substrate. Recognized substructures: acetals, esters, "
        "amides, nitriles, alkyl halides. Half-life is approximated from "
        "a base half-life (3600 s) scaled by a pH factor and an "
        "Arrhenius-like temperature factor. Pass a SMILES string."
    )
)
async def predict_hydrolysis_product(
    substrate: str,
    pH: float = 7.0,
    temperature: float = 298.15,
) -> str:
    if not _RDKIT_OK:
        return json.dumps({"error": "rdkit not installed"})
    mol = _org_mol_from_smiles_safe(substrate)
    if mol is None:
        return json.dumps({"error": f"invalid SMILES: {substrate}"})
    if _org_has_substructure(mol, "[CX4]([OX2])([OX2])") or _org_has_substructure(mol, "[CX4]([OX2])([OX2])([OX2])"):
        product_smiles = "CC=O"  # Acetal → aldehyde + alcohol
        mechanism = "acetal_hydrolysis_to_aldehyde_and_alcohol"
    elif _org_has_substructure(mol, "C(=O)O[C,H]"):
        product_smiles = "CC(=O)O"
        mechanism = "ester_hydrolysis_to_carboxylic_acid_and_alcohol"
    elif _org_has_substructure(mol, "C(=O)N"):
        product_smiles = "CC(=O)O"
        mechanism = "amide_hydrolysis_to_carboxylic_acid_and_amine"
    elif _org_has_substructure(mol, "C#N"):
        product_smiles = "CC(=O)O"
        mechanism = "nitrile_hydrolysis_to_carboxylic_acid"
    elif _org_has_substructure(mol, "[CX4][Cl,Br,I]"):
        product_smiles = "CCO"  # placeholder
        mechanism = "alkyl_halide_hydrolysis_substitution"
    else:
        product_smiles = "CCO"
        mechanism = "unrecognized_substructure_default_alcohol"
    # pH factor.
    if pH < 3.0:
        ph_factor = 0.1
        regim = "acid_catalyzed"
    elif pH > 10.0:
        ph_factor = 0.2
        regim = "base_catalyzed"
    else:
        ph_factor = 1.0
        regim = "neutral"
    # Arrhenius-style temperature factor: t1/2 ∝ exp(Ea/R (1/T_ref - 1/T))
    Ea_over_R = 5000.0  # K (Ea ≈ 41.6 kJ/mol — typical for hydrolysis)
    T_ref = 298.15
    temp_factor = math.exp(Ea_over_R * (1.0 / T_ref - 1.0 / float(temperature)))
    half_life = 3600.0 * ph_factor * temp_factor
    return json.dumps(
        {
            "substrate": substrate,
            "pH": float(pH),
            "temperature_K": float(temperature),
            "regime": regim,
            "product_smiles": product_smiles,
            "reaction_mechanism": mechanism,
            "half_life_s": _round_energy(half_life),
        },
        ensure_ascii=False,
    )


@mcp.tool(
    description=(
        "Predict products for a handful of named organic reaction types: "
        "SN2, E2, aldol, Diels–Alder. Uses a small SMILES-based lookup "
        "table for the textbook reactions; non-matching cases return "
        "a generic 'Product_<reaction>' placeholder. Returns products "
        "list with predicted yields and a major_product selection."
    )
)
async def predict_organic_reaction_products(
    reactants: list[str],
    reaction_type: str,
    conditions: dict | None = None,
) -> str:
    if conditions is None:
        conditions = {"temperature": 298.15, "solvent": "neutral", "catalyst": None}
    if not isinstance(reactants, list) or not reactants:
        return json.dumps({"error": "reactants must be a non-empty list of SMILES"})
    if not _RDKIT_OK:
        return json.dumps({"error": "rdkit not installed"})
    # Validate SMILES.
    for smiles in reactants:
        if Chem.MolFromSmiles(smiles, sanitize=False) is None:
            return json.dumps({"error": f"invalid SMILES: {smiles}"})
    rt = reaction_type.strip().upper()
    products: list[str] = []
    yields: list[float] = []
    note = ""
    if rt == "SN2":
        if len(reactants) >= 2:
            nu, sub = reactants[0], reactants[1]
            products = ["Product_SN2"]
            yields = [80.0]
            # Simple string match for textbook cases.
            if "O-" in nu and "Br" in sub and "C" in sub:
                products = ["COC"]
                yields = [95.0]
                note = "Methoxide on methyl bromide → methyl ethyl ether analog."
            elif "CN-" in nu and "Cl" in sub:
                products = ["CCC#N"]
                yields = [90.0]
                note = "Cyanide on alkyl chloride → nitrile."
    elif rt == "E2":
        products = ["Alkene_E2"]
        yields = [70.0]
        note = "E2 elimination — geometry-dependent (anti-periplanar)."
    elif rt == "ALDOL":
        products = ["Beta_hydroxy_carbonyl"]
        yields = [60.0]
        note = "Aldol condensation: enolate + aldehyde → β-hydroxy carbonyl."
    elif rt in ("DIELS-ALDER", "DA", "DIELSALDER"):
        products = ["Cyclohexene_adduct"]
        yields = [85.0]
        note = "[4+2] cycloaddition — stereochemistry per endo rule."
    else:
        products = [f"Product_{rt}"]
        yields = [50.0]
        note = f"Unknown reaction_type {reaction_type!r}; returning placeholder."
    return json.dumps(
        {
            "reactants": reactants,
            "reaction_type": reaction_type,
            "conditions": conditions,
            "products": products,
            "yields_pct": yields,
            "major_product": products[0] if products else None,
            "note": note,
        },
        ensure_ascii=False,
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(mcp.run_stdio_async())
