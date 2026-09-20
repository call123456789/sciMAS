#!/usr/bin/env python3
"""Pharmaceutical drug-discovery MCP server.

The public interface is adapted from MADD's molecule-generation and
property-prediction tools, with local RDKit descriptors added for fast
drug-likeness triage.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("pharma-drug-discovery")

ALERT_COLLECTIONS = Path(__file__).resolve().parent / "data" / "alert_collections.csv"
DEFAULT_MADD_ROOT = Path(os.environ.get("MADD_ROOT", "/Users/a123/Documents/games/MADD-main"))

CASE_ALIASES: dict[str, str] = {
    "alzheimer": "Alzhmr",
    "alzheimer's": "Alzhmr",
    "alzheimers": "Alzhmr",
    "alzheimer disease": "Alzhmr",
    "alzheimer's disease": "Alzhmr",
    "parkinson": "Prkns",
    "parkinson's": "Prkns",
    "parkinsons": "Prkns",
    "parkinson disease": "Prkns",
    "parkinson's disease": "Prkns",
    "multiple sclerosis": "Sklrz",
    "sclerosis": "Sklrz",
    "dyslipidemia": "Dslpdm",
    "acquired drug resistance": "TBLET",
    "drug resistance": "TBLET",
    "resistance": "TBLET",
    "lung cancer": "Cnsr",
    "lung": "Cnsr",
    "random": "RNDM",
    "rndm": "RNDM",
    "RNDM".lower(): "RNDM",
    "Alzhmr".lower(): "Alzhmr",
    "Sklrz".lower(): "Sklrz",
    "Prkns".lower(): "Prkns",
    "Cnsr".lower(): "Cnsr",
    "Dslpdm".lower(): "Dslpdm",
    "TBLET".lower(): "TBLET",
}

_ALERT_PATTERNS: dict[str, list[Any]] | None = None
_BRENK_CATALOG: Any | None = None

LOCAL_MADD_PREDICTORS: dict[str, dict[str, Any]] = {
    "alzheimer_gsk3b_ic50": {
        "description": "Alzheimer/GSK-3 beta IC50 classifier from MADD",
        "path": "infrastructure/generative_models/utils/ic_50_models/alzheimer/alzheimer_clf.pkl",
        "module": "infrastructure.generative_models.utils.ic_50_models.alzheimer.predict_ic50_clf",
        "function": "eval_ic_50_alzheimer",
        "task": "classification",
        "output": "MADD activity label, returned by eval_ic_50_alzheimer",
    },
    "sclerosis_btk_ic50": {
        "description": "Multiple-sclerosis/BTK IC50 classifier from MADD",
        "path": "infrastructure/generative_models/utils/ic_50_models/skleroz_ic50_clf/checkpoints/ic50_btk_clf_102.pkl",
        "module": "infrastructure.generative_models.utils.ic_50_models.skleroz_ic50_clf.scripts.predict_ic50_btk_clf",
        "function": "eval_ic_50_sklrz",
        "task": "classification",
        "output": "MADD activity label, returned by eval_ic_50_sklrz",
    },
    "lung_cancer_kras_ic50": {
        "description": "Lung-cancer/KRAS IC50 classifier from MADD",
        "path": "infrastructure/generative_models/utils/ic_50_models/kras_ic50_prediction/lung_cancer_model.pkl",
        "module": "infrastructure.generative_models.utils.ic_50_models.kras_ic50_prediction.predict_ic50_clf",
        "function": "eval_ic_50_cancer",
        "task": "classification",
        "output": "MADD activity label, returned by eval_ic_50_cancer",
    },
    "parkinson_atp_citrate_ic50": {
        "description": "Parkinson/ATP-citrate-synthase IC50 classifier from MADD",
        "path": "infrastructure/generative_models/utils/ic_50_models/citrate_classif_inference/model_citrate_clf.pkl",
        "module": "infrastructure.generative_models.utils.ic_50_models.citrate_classif_inference.inference_citrate_clf",
        "function": "predict",
        "task": "classification",
        "output": "1 - classifier label, following the original MADD wrapper",
    },
    "drug_resistance_stat3_ic50": {
        "description": "Drug-resistance/STAT3 IC50 classifier from MADD",
        "path": "infrastructure/generative_models/utils/ic_50_models/drug_resis_classif_inference/model_drug_clf.pkl",
        "module": "infrastructure.generative_models.utils.ic_50_models.drug_resis_classif_inference.inference_drug_clf",
        "function": "predict",
        "task": "classification",
        "output": "1 - classifier label, following the original MADD wrapper",
    },
    "dyslipidemia_tyrosine_ic50": {
        "description": "Dyslipidemia/tyrosine-kinase IC50 classifier from MADD",
        "path": "infrastructure/generative_models/utils/ic_50_models/tyrosine_classif_inference/model_tyrosine_clf.pkl",
        "module": "infrastructure.generative_models.utils.ic_50_models.tyrosine_classif_inference.inference_tyrosine_clf",
        "function": "predict",
        "task": "classification",
        "output": "1 - classifier label, following the original MADD wrapper",
    },
    "dyslipidemia_tyrosine_ki": {
        "description": "Dyslipidemia/tyrosine-kinase Ki regressor from MADD",
        "path": "infrastructure/generative_models/utils/ki_models/tyrosine_regression_inference/model_tyrosine_regr.pkl",
        "extra_paths": (
            "infrastructure/generative_models/utils/ki_models/tyrosine_regression_inference/scaler_tyrosine_regr.pkl",
            "infrastructure/generative_models/utils/ki_models/tyrosine_regression_inference/selected_features_tyrosine_regr.csv",
        ),
        "module": "infrastructure.generative_models.utils.ki_models.tyrosine_regression_inference.tyrosine_inference_regr",
        "function": "predict",
        "task": "regression",
        "output": "MADD Ki prediction, returned by tyrosine_inference_regr.predict",
    },
    "bbb_classifier": {
        "description": "Blood-brain barrier classifier from MADD",
        "path": "infrastructure/generative_models/utils/inference_BB_clf/model_BB_clf.pkl",
        "extra_paths": (
            "infrastructure/generative_models/utils/inference_BB_clf/scaler_BB_clf.pkl",
            "infrastructure/generative_models/utils/inference_BB_clf/most_importance_features.pkl",
        ),
        "module": "infrastructure.generative_models.utils.inference_BB_clf.BB_inference",
        "function": "predict",
        "task": "classification",
        "output": "MADD BBB class label",
    },
}

LOCAL_MADD_GENERATORS: dict[str, dict[str, Any]] = {
    "Alzhmr": {
        "description": "Alzheimer disease CVAE generator",
        "weights_dir": "infrastructure/generative_models/autotrain/many_prop_CVAE/weights_8p_alzhmr",
    },
    "Sklrz": {
        "description": "Multiple sclerosis CVAE generator",
        "weights_dir": "infrastructure/generative_models/autotrain/many_prop_CVAE/weights_8p_sklrz",
    },
    "Prkns": {
        "description": "Parkinson disease CVAE generator",
        "weights_dir": "infrastructure/generative_models/autotrain/many_prop_CVAE/weights_parkinson",
    },
    "Cnsr": {
        "description": "Lung cancer CVAE generator",
        "weights_dir": "infrastructure/generative_models/autotrain/many_prop_CVAE/weights_8p_cnsr",
    },
    "Dslpdm": {
        "description": "Dyslipidemia CVAE generator",
        "weights_dir": "infrastructure/generative_models/autotrain/many_prop_CVAE/weights_dislip",
    },
    "TBLET": {
        "description": "Acquired drug resistance CVAE generator",
        "weights_dir": "infrastructure/generative_models/autotrain/many_prop_CVAE/weights_8p_tablet",
    },
    "RNDM": {
        "description": "Random GAN generator",
        "weights_file": "infrastructure/generative_models/GAN/gan_lstm_refactoring/weights/v4_gan_mol_124_0.0003_8k.pkl",
    },
}


def _dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _coerce_str_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        if "\n" in text:
            return [part.strip() for part in text.splitlines() if part.strip()]
        if "," in text and " " not in text:
            return [part.strip() for part in text.split(",") if part.strip()]
        return [part.strip() for part in text.split() if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    raise TypeError(f"{field_name} must be a string or a list of strings")


def _rdkit_modules() -> tuple[dict[str, Any] | None, str | None]:
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, Draw, QED, rdMolDescriptors
        from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
    except ImportError as exc:
        return None, f"RDKit is not installed in this environment: {exc}"

    sascorer = None
    sa_error = None
    try:
        from rdkit.Contrib.SA_Score import sascorer as rdkit_sascorer

        sascorer = rdkit_sascorer
    except ImportError as exc:
        sa_error = str(exc)

    return {
        "Chem": Chem,
        "Crippen": Crippen,
        "Descriptors": Descriptors,
        "Draw": Draw,
        "QED": QED,
        "rdMolDescriptors": rdMolDescriptors,
        "FilterCatalog": FilterCatalog,
        "FilterCatalogParams": FilterCatalogParams,
        "sascorer": sascorer,
        "sa_error": sa_error,
    }, None


def _load_alert_patterns(Chem: Any) -> dict[str, list[Any]]:
    global _ALERT_PATTERNS
    if _ALERT_PATTERNS is not None:
        return _ALERT_PATTERNS

    patterns: dict[str, list[Any]] = {"PAINS": [], "SureChEMBL": [], "Glaxo": []}
    if not ALERT_COLLECTIONS.exists():
        _ALERT_PATTERNS = patterns
        return patterns

    with ALERT_COLLECTIONS.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rule_set = row.get("rule_set_name", "")
            if rule_set not in patterns:
                continue
            smarts = row.get("smarts", "")
            if not smarts:
                continue
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is not None:
                patterns[rule_set].append(pattern)
    _ALERT_PATTERNS = patterns
    return patterns


def _brenk_catalog(FilterCatalogParams: Any, FilterCatalog: Any) -> Any:
    global _BRENK_CATALOG
    if _BRENK_CATALOG is None:
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
        _BRENK_CATALOG = FilterCatalog(params)
    return _BRENK_CATALOG


def _alert_flags(mol: Any, modules: dict[str, Any]) -> dict[str, int]:
    Chem = modules["Chem"]
    patterns = _load_alert_patterns(Chem)
    flags = {
        name: int(any(mol.HasSubstructMatch(pattern) for pattern in rule_patterns))
        for name, rule_patterns in patterns.items()
    }
    catalog = _brenk_catalog(modules["FilterCatalogParams"], modules["FilterCatalog"])
    flags["Brenk"] = int(catalog.HasMatch(mol))
    return flags


def _score_molecule(smiles: str, modules: dict[str, Any], include_alerts: bool) -> dict[str, Any]:
    Chem = modules["Chem"]
    Crippen = modules["Crippen"]
    Descriptors = modules["Descriptors"]
    QED = modules["QED"]
    rdMolDescriptors = modules["rdMolDescriptors"]
    sascorer = modules["sascorer"]

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "input_smiles": smiles,
            "valid": False,
            "error": "invalid SMILES",
        }

    result: dict[str, Any] = {
        "input_smiles": smiles,
        "valid": True,
        "canonical_smiles": Chem.MolToSmiles(mol),
        "molecular_formula": rdMolDescriptors.CalcMolFormula(mol),
        "molecular_weight": round(float(Descriptors.MolWt(mol)), 4),
        "qed": round(float(QED.qed(mol)), 4),
        "logp": round(float(Crippen.MolLogP(mol)), 4),
        "polar_surface_area": round(float(Descriptors.TPSA(mol)), 4),
        "h_bond_donors": int(Descriptors.NumHDonors(mol)),
        "h_bond_acceptors": int(Descriptors.NumHAcceptors(mol)),
        "rotatable_bonds": int(Descriptors.NumRotatableBonds(mol)),
        "aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
    }
    if sascorer is None:
        result["synthetic_accessibility"] = None
        result["synthetic_accessibility_warning"] = modules["sa_error"] or "SA scorer unavailable"
    else:
        result["synthetic_accessibility"] = round(float(sascorer.calculateScore(mol)), 4)

    if include_alerts:
        result["structural_alerts"] = _alert_flags(mol, modules)

    return result


def _resolve_base_url(url: str | None, env_name: str) -> str | None:
    base = (url or os.environ.get(env_name) or "").strip()
    if not base:
        return None
    if not base.startswith(("http://", "https://")):
        base = "http://" + base
    return base.rstrip("/")


def _decode_response(resp: Any) -> Any:
    try:
        payload = resp.json()
    except ValueError:
        return resp.text
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return payload
    return payload


def _post_json(url: str, payload: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
    import requests

    started = time.time()
    try:
        resp = requests.post(
            url,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=timeout_seconds,
        )
    except requests.RequestException as exc:
        return {
            "ok": False,
            "url": url,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": round(time.time() - started, 3),
        }
    return {
        "ok": bool(resp.ok),
        "url": url,
        "status_code": resp.status_code,
        "response": _decode_response(resp),
        "elapsed_seconds": round(time.time() - started, 3),
    }


def _case_code(case: str) -> str | None:
    return CASE_ALIASES.get(case.strip().lower())


def _madd_root(path: str | None = None) -> Path:
    return Path(path).expanduser() if path else DEFAULT_MADD_ROOT


def _path_status(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    info: dict[str, Any] = {
        "relative_path": relative_path,
        "path": str(path),
        "exists": path.exists(),
    }
    if path.exists():
        try:
            stat = path.stat()
            info["size_bytes"] = stat.st_size
        except OSError:
            pass
    return info


def _runtime_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {"python": sys.version.split()[0], "executable": sys.executable}
    for package in ("rdkit", "sklearn", "pandas", "numpy", "joblib", "torch", "lightgbm", "catboost", "fedot"):
        try:
            module = __import__(package)
            versions[package] = getattr(module, "__version__", "installed")
        except Exception:
            versions[package] = None
    return versions


def _madd_subprocess_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
    python_path_parts = [
        str(root),
        str(root / "infrastructure" / "generative_models"),
        env.get("PYTHONPATH", ""),
    ]
    env["PYTHONPATH"] = os.pathsep.join(part for part in python_path_parts if part)
    return env


def _subprocess_json(
    code: str,
    payload: dict[str, Any],
    root: Path,
    python_bin: str | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    executable = python_bin or sys.executable
    started = time.time()
    try:
        proc = subprocess.run(
            [executable, "-c", code],
            input=json.dumps(payload),
            text=True,
            cwd=str(root),
            env=_madd_subprocess_env(root),
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "error": f"local MADD subprocess timed out after {timeout_seconds} seconds",
            "python_bin": executable,
            "elapsed_seconds": round(time.time() - started, 3),
            "stdout_tail": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else "",
        }
    except OSError as exc:
        return {
            "ok": False,
            "error": f"failed to start local MADD subprocess: {type(exc).__name__}: {exc}",
            "python_bin": executable,
            "elapsed_seconds": round(time.time() - started, 3),
        }

    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    parsed: dict[str, Any]
    if stdout:
        last_line = stdout.splitlines()[-1]
        try:
            obj = json.loads(last_line)
            parsed = obj if isinstance(obj, dict) else {"ok": True, "result": obj}
        except json.JSONDecodeError:
            parsed = {"ok": proc.returncode == 0, "stdout": stdout}
    else:
        parsed = {"ok": proc.returncode == 0}

    if proc.returncode != 0:
        parsed["ok"] = False
        parsed.setdefault("error", f"local MADD subprocess exited with code {proc.returncode}")
    parsed["python_bin"] = executable
    parsed["elapsed_seconds"] = round(time.time() - started, 3)
    if stderr:
        parsed["stderr_tail"] = stderr[-2000:]
    if stdout and "stdout" not in parsed and len(stdout.splitlines()) > 1:
        parsed["stdout_tail"] = stdout[-2000:]
    return parsed


@mcp.tool(
    description=(
        "Evaluate SMILES drug-likeness and medicinal-chemistry descriptors "
        "using local RDKit: validity, canonical SMILES, formula, MW, QED, SA, "
        "LogP, TPSA, HBD/HBA, rotatable bonds, aromatic rings, and PAINS/"
        "SureChEMBL/Glaxo/Brenk alerts."
    )
)
async def evaluate_druglikeness(smiles_list: list[str] | str, include_alerts: bool = True) -> str:
    modules, error = _rdkit_modules()
    if error:
        return _dumps({"ok": False, "error": error})
    try:
        smiles = _coerce_str_list(smiles_list, "smiles_list")
    except TypeError as exc:
        return _dumps({"ok": False, "error": str(exc)})
    if not smiles:
        return _dumps({"ok": False, "error": "smiles_list is empty"})

    results = [_score_molecule(item, modules, include_alerts) for item in smiles]
    valid_count = sum(1 for item in results if item.get("valid"))
    return _dumps(
        {
            "ok": True,
            "count": len(results),
            "valid_count": valid_count,
            "invalid_count": len(results) - valid_count,
            "results": results,
        }
    )


@mcp.tool(
    description=(
        "List local MADD checkpoints under a MADD checkout and report which "
        "predictive classifiers/regressors and expected generator weights are present. "
        "This is a diagnostic tool for local MADD model availability."
    )
)
async def list_madd_local_checkpoints(madd_root: str | None = None) -> str:
    root = _madd_root(madd_root)
    predictors: dict[str, Any] = {}
    for key, spec in LOCAL_MADD_PREDICTORS.items():
        files = [_path_status(root, spec["path"])]
        files.extend(_path_status(root, extra) for extra in spec.get("extra_paths", ()))
        predictors[key] = {
            "description": spec["description"],
            "task": spec["task"],
            "output": spec["output"],
            "available": all(item["exists"] for item in files),
            "files": files,
        }

    generators: dict[str, Any] = {}
    for case, spec in LOCAL_MADD_GENERATORS.items():
        files = []
        if "weights_dir" in spec:
            weights_dir = root / spec["weights_dir"]
            files.append(_path_status(root, spec["weights_dir"]))
            files.append(_path_status(root, f"{spec['weights_dir']}/model_weights"))
            files.append(_path_status(root, f"{spec['weights_dir']}/toklen_list.csv"))
            available = weights_dir.exists() and (weights_dir / "model_weights").exists()
        else:
            files.append(_path_status(root, spec["weights_file"]))
            available = files[0]["exists"]
        generators[case] = {
            "description": spec["description"],
            "available": available,
            "files": files,
        }

    return _dumps(
        {
            "ok": True,
            "madd_root": str(root),
            "madd_root_exists": root.exists(),
            "runtime": _runtime_versions(),
            "predictors": predictors,
            "generators": generators,
            "notes": [
                "MADD requirements pin scikit-learn==1.2.2 for these pickle checkpoints.",
                "CVAE disease generators require many_prop_CVAE weight directories with model_weights.",
                "The random GAN generator requires torch and MADD's GAN code.",
            ],
        }
    )


@mcp.tool(
    description=(
        "Run one local MADD predictive checkpoint on SMILES by invoking the "
        "original MADD inference wrapper in a local subprocess. Use "
        "list_madd_local_checkpoints to see valid model_key values. A compatible "
        "MADD Python environment may be supplied with python_bin."
    )
)
async def predict_with_local_madd_checkpoint(
    smiles_list: list[str] | str,
    model_key: str,
    madd_root: str | None = None,
    python_bin: str | None = None,
    timeout_seconds: float = 120.0,
) -> str:
    root = _madd_root(madd_root)
    if not root.exists():
        return _dumps({"ok": False, "error": f"MADD root does not exist: {root}"})
    spec = LOCAL_MADD_PREDICTORS.get(model_key)
    if spec is None:
        return _dumps(
            {
                "ok": False,
                "error": f"Unknown model_key: {model_key}",
                "available_model_keys": sorted(LOCAL_MADD_PREDICTORS),
            }
        )

    missing = [
        status["path"]
        for status in [_path_status(root, spec["path"]), *(_path_status(root, p) for p in spec.get("extra_paths", ()))]
        if not status["exists"]
    ]
    if missing:
        return _dumps({"ok": False, "error": "Required local MADD checkpoint files are missing", "missing": missing})

    modules, error = _rdkit_modules()
    if error:
        return _dumps({"ok": False, "error": error})
    Chem = modules["Chem"]
    try:
        smiles = _coerce_str_list(smiles_list, "smiles_list")
    except TypeError as exc:
        return _dumps({"ok": False, "error": str(exc)})
    valid_inputs: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for index, item in enumerate(smiles):
        mol = Chem.MolFromSmiles(item)
        if mol is None:
            results.append(
                {
                    "index": index,
                    "input_smiles": item,
                    "valid": False,
                    "error": "invalid SMILES",
                }
            )
            continue
        canonical = Chem.MolToSmiles(mol)
        valid_inputs.append({"index": index, "input_smiles": item, "canonical_smiles": canonical})
    if not valid_inputs:
        return _dumps({"ok": False, "error": "No valid SMILES to predict", "results": results})

    code = """
import importlib
import json
import os
import sys

payload = json.loads(sys.stdin.read())
root = payload["madd_root"]
os.chdir(root)
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, "infrastructure", "generative_models"))

try:
    module = importlib.import_module(payload["module"])
    fn = getattr(module, payload["function"])
    values = fn(payload["smiles"])
    if hasattr(values, "tolist"):
        values = values.tolist()
    else:
        values = list(values)
    print(json.dumps({"ok": True, "predictions": values}))
except Exception as exc:
    print(json.dumps({
        "ok": False,
        "error": f"{type(exc).__name__}: {exc}",
    }))
    raise
"""
    response = _subprocess_json(
        code,
        {
            "madd_root": str(root),
            "module": spec["module"],
            "function": spec["function"],
            "smiles": [item["canonical_smiles"] for item in valid_inputs],
        },
        root,
        python_bin,
        timeout_seconds,
    )
    response["model_key"] = model_key
    response["model"] = {
        "description": spec["description"],
        "task": spec["task"],
        "output": spec["output"],
        "path": str(root / spec["path"]),
    }
    response["runtime"] = _runtime_versions()
    if not response.get("ok"):
        response["results"] = results + [
            {
                "index": item["index"],
                "input_smiles": item["input_smiles"],
                "canonical_smiles": item["canonical_smiles"],
                "valid": True,
                "prediction": None,
            }
            for item in valid_inputs
        ]
        response["hint"] = (
            "These MADD pickles were built for scikit-learn==1.2.2. "
            "If loading fails with a tree node dtype mismatch, run this tool "
            "with a python_bin from a MADD-compatible environment."
        )
        return _dumps(response)

    predictions = response.get("predictions", [])
    for item, prediction in zip(valid_inputs, predictions):
        results.append(
            {
                "index": item["index"],
                "input_smiles": item["input_smiles"],
                "canonical_smiles": item["canonical_smiles"],
                "valid": True,
                "prediction": prediction,
            }
        )
    response["results"] = sorted(results, key=lambda item: item["index"])
    return _dumps(response)


@mcp.tool(
    description=(
        "Attempt local molecule generation through MADD's original local "
        "case_generator entry point. Use this when local MADD generator "
        "checkpoints are present; otherwise use generate_molecules_by_case "
        "with URL_GEN."
    )
)
async def generate_molecules_with_local_madd(
    case: str,
    n_samples: int = 1,
    madd_root: str | None = None,
    python_bin: str | None = None,
    cuda: bool = False,
    mean_: float = 0.0,
    std_: float = 1.0,
    timeout_seconds: float = 300.0,
) -> str:
    root = _madd_root(madd_root)
    if not root.exists():
        return _dumps({"ok": False, "error": f"MADD root does not exist: {root}"})
    if n_samples < 1:
        return _dumps({"ok": False, "error": "n_samples must be at least 1"})
    case_code = _case_code(case)
    if case_code is None:
        return _dumps(
            {
                "ok": False,
                "error": "Local predefined generation requires a known MADD case alias",
                "available_cases": sorted(set(CASE_ALIASES.values())),
            }
        )

    generator = LOCAL_MADD_GENERATORS.get(case_code, {})
    if "weights_dir" in generator:
        weights_dir = root / generator["weights_dir"]
        if not weights_dir.exists() or not (weights_dir / "model_weights").exists():
            return _dumps(
                {
                    "ok": False,
                    "error": "Local MADD CVAE generator weights are missing",
                    "case": case,
                    "mapped_case": case_code,
                    "expected_weights_dir": str(weights_dir),
                    "expected_model_weights": str(weights_dir / "model_weights"),
                    "fallback": "Use generate_molecules_by_case with URL_GEN, or restore the missing local weights.",
                }
            )
    elif "weights_file" in generator:
        weights_file = root / generator["weights_file"]
        if not weights_file.exists():
            return _dumps(
                {
                    "ok": False,
                    "error": "Local MADD GAN generator checkpoint is missing",
                    "case": case,
                    "mapped_case": case_code,
                    "expected_weights_file": str(weights_file),
                }
            )

    if case_code == "RNDM":
        code = """
import json
import os
import sys

payload = json.loads(sys.stdin.read())
root = payload["madd_root"]
os.chdir(root)
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, "infrastructure", "generative_models"))

try:
    from GAN.gan_lstm_refactoring.gen import generate
    result = generate(payload["n_samples"])
    print(json.dumps({"ok": True, "result": {"Molecules": result}}))
except Exception as exc:
    print(json.dumps({
        "ok": False,
        "error": f"{type(exc).__name__}: {exc}",
    }))
    raise
"""
    else:
        code = """
import json
import os
import sys

payload = json.loads(sys.stdin.read())
root = payload["madd_root"]
os.chdir(root)
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, "infrastructure", "generative_models"))

try:
    from api_utils import GenData, case_generator
    data = GenData(
        numb_mol=payload["n_samples"],
        cuda=payload["cuda"],
        mean_=payload["mean_"],
        std_=payload["std_"],
        case_=payload["case_code"],
    )
    result = case_generator(data)
    print(json.dumps({"ok": True, "result": result}))
except Exception as exc:
    print(json.dumps({
        "ok": False,
        "error": f"{type(exc).__name__}: {exc}",
    }))
    raise
"""
    response = _subprocess_json(
        code,
        {
            "madd_root": str(root),
            "case_code": case_code,
            "n_samples": int(n_samples),
            "cuda": bool(cuda),
            "mean_": float(mean_),
            "std_": float(std_),
        },
        root,
        python_bin,
        timeout_seconds,
    )
    response["case"] = case
    response["mapped_case"] = case_code
    response["runtime"] = _runtime_versions()
    if not response.get("ok"):
        response["hint"] = (
            "Local MADD generation needs a MADD-compatible environment with torch, "
            "lightgbm/catboost as needed, and the generator checkpoint files shown "
            "by list_madd_local_checkpoints."
        )
    return _dumps(response)


@mcp.tool(
    description=(
        "Render SMILES molecules to PNG files with RDKit. Returns absolute "
        "image paths and per-molecule validity errors."
    )
)
async def draw_molecules(
    smiles_list: list[str] | str,
    output_dir: str = "outputs/pharma/molecules",
    legends: list[str] | str | None = None,
) -> str:
    modules, error = _rdkit_modules()
    if error:
        return _dumps({"ok": False, "error": error})
    Chem = modules["Chem"]
    Draw = modules["Draw"]
    try:
        smiles = _coerce_str_list(smiles_list, "smiles_list")
        legend_values = _coerce_str_list(legends, "legends") if legends else []
    except TypeError as exc:
        return _dumps({"ok": False, "error": str(exc)})
    if not smiles:
        return _dumps({"ok": False, "error": "smiles_list is empty"})

    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    results: list[dict[str, Any]] = []
    for idx, item in enumerate(smiles):
        mol = Chem.MolFromSmiles(item)
        if mol is None:
            results.append({"input_smiles": item, "valid": False, "error": "invalid SMILES"})
            continue
        path = out_dir / f"molecule-{stamp}-{idx + 1}.png"
        legend = legend_values[idx] if idx < len(legend_values) else item
        Draw.MolToFile(mol, str(path), legend=legend)
        results.append(
            {
                "input_smiles": item,
                "valid": True,
                "canonical_smiles": Chem.MolToSmiles(mol),
                "path": str(path),
            }
        )

    return _dumps({"ok": True, "output_dir": str(out_dir), "results": results})


@mcp.tool(
    description=(
        "Generate molecules through a MADD generative service. Predefined "
        "cases include Alzheimer, Parkinson, multiple sclerosis, dyslipidemia, "
        "acquired drug resistance, lung cancer, and random generation. Requires "
        "URL_GEN or an explicit url_gen argument."
    )
)
async def generate_molecules_by_case(
    case: str,
    n_samples: int = 1,
    url_gen: str | None = None,
    cuda: bool = True,
    mean_: float = 0.0,
    std_: float = 1.0,
    timeout_seconds: float = 120.0,
) -> str:
    base_url = _resolve_base_url(url_gen, "URL_GEN")
    if base_url is None:
        return _dumps(
            {
                "ok": False,
                "error": "URL_GEN is not configured. Pass url_gen or set the URL_GEN environment variable.",
            }
        )
    if n_samples < 1:
        return _dumps({"ok": False, "error": "n_samples must be at least 1"})

    predefined = _case_code(case)
    if predefined:
        url = f"{base_url}/case_generator"
        payload = {
            "numb_mol": int(n_samples),
            "cuda": bool(cuda),
            "mean_": float(mean_),
            "std_": float(std_),
            "case_": predefined,
        }
    else:
        url = f"{base_url}/generate_gen_models_by_case"
        payload = {"case": case, "n_samples": int(n_samples)}

    response = _post_json(url, payload, timeout_seconds)
    response["case"] = case
    response["mapped_case"] = predefined
    return _dumps(response)


@mcp.tool(
    description=(
        "Predict molecular properties through a MADD predictive service for "
        "a list of SMILES. Requires URL_PRED or an explicit url_pred argument."
    )
)
async def predict_properties_by_smiles(
    smiles_list: list[str] | str,
    case: str = "no_name_case",
    url_pred: str | None = None,
    timeout_minutes: int = 20,
    request_timeout_seconds: float = 120.0,
) -> str:
    base_url = _resolve_base_url(url_pred, "URL_PRED")
    if base_url is None:
        return _dumps(
            {
                "ok": False,
                "error": "URL_PRED is not configured. Pass url_pred or set the URL_PRED environment variable.",
            }
        )
    try:
        smiles = _coerce_str_list(smiles_list, "smiles_list")
    except TypeError as exc:
        return _dumps({"ok": False, "error": str(exc)})
    if not smiles:
        return _dumps({"ok": False, "error": "smiles_list is empty"})

    payload = {"case": case, "smiles_list": smiles, "timeout": int(timeout_minutes)}
    response = _post_json(f"{base_url}/predict_ml", payload, request_timeout_seconds)
    response["case"] = case
    response["smiles_count"] = len(smiles)
    return _dumps(response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(mcp.run_stdio_async())
