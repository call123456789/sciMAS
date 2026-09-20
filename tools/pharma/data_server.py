#!/usr/bin/env python3
"""Pharmaceutical public-data and dataset-preparation MCP server."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("pharma-data")

VALID_AFFINITY_TYPES = {"Ki", "Kd", "IC50"}


def _dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _coerce_list(value: Any) -> list[str]:
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
        if "," in text:
            return [part.strip() for part in text.split(",") if part.strip()]
        return [text]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _safe_filename(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("_")
    return name or "dataset"


def _load_pandas() -> tuple[Any | None, str | None]:
    try:
        import pandas as pd
    except ImportError as exc:
        return None, f"pandas is not installed: {exc}"
    return pd, None


def _read_table(path: str) -> tuple[Any | None, str | None]:
    pd, error = _load_pandas()
    if error:
        return None, error
    data_path = Path(path).expanduser()
    if not data_path.exists():
        return None, f"Data file not found: {data_path}"
    suffix = data_path.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(data_path), None
        if suffix in {".xls", ".xlsx"}:
            return pd.read_excel(data_path), None
    except Exception as exc:
        return None, f"Failed to read {data_path}: {type(exc).__name__}: {exc}"
    return None, "Unsupported file format. Use CSV, XLS, or XLSX."


def _write_rows(rows: list[dict[str, Any]], output_dir: str, stem: str) -> str:
    pd, error = _load_pandas()
    if error:
        raise RuntimeError(error)
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{_safe_filename(stem)}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def _fetch_uniprot_id(protein_name: str, timeout_seconds: float) -> tuple[str | None, dict[str, Any] | None]:
    import requests

    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": f"{protein_name} AND organism_id:9606",
        "format": "json",
        "size": 1,
        "fields": "accession,id,protein_name",
    }
    try:
        response = requests.get(url, params=params, timeout=timeout_seconds)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        return None, {"error": f"{type(exc).__name__}: {exc}", "source": "UniProt"}
    results = data.get("results") or []
    if not results:
        return None, None
    accession = results[0].get("primaryAccession")
    return accession, results[0]


@mcp.tool(
    description=(
        "Inspect columns, dtypes, row count, and a small sample from a CSV/XLS/XLSX "
        "dataset intended for pharmaceutical modeling."
    )
)
async def inspect_dataset_columns(path: str, sample_rows: int = 5) -> str:
    df, error = _read_table(path)
    if error:
        return _dumps({"ok": False, "error": error})
    sample_count = max(0, min(int(sample_rows), 20))
    return _dumps(
        {
            "ok": True,
            "path": str(Path(path).expanduser()),
            "rows": int(len(df)),
            "columns": list(df.columns),
            "dtypes": {str(col): str(dtype) for col, dtype in df.dtypes.items()},
            "sample": df.head(sample_count).to_dict(orient="records"),
        }
    )


@mcp.tool(
    description=(
        "Remove selected columns from a CSV/XLS/XLSX dataset and save a new "
        "file with a _columns_filtered suffix."
    )
)
async def filter_dataset_columns(
    path: str,
    drop_columns: list[str] | str,
    output_path: str | None = None,
) -> str:
    df, error = _read_table(path)
    if error:
        return _dumps({"ok": False, "error": error})

    data_path = Path(path).expanduser()
    requested = _coerce_list(drop_columns)
    present = [col for col in requested if col in df.columns]
    missing = [col for col in requested if col not in df.columns]
    filtered = df.drop(columns=present)

    out_path = Path(output_path).expanduser() if output_path else data_path.with_name(
        f"{data_path.stem}_columns_filtered{data_path.suffix}"
    )
    try:
        if out_path.suffix.lower() == ".csv":
            filtered.to_csv(out_path, index=False)
        elif out_path.suffix.lower() in {".xls", ".xlsx"}:
            filtered.to_excel(out_path, index=False)
        else:
            return _dumps({"ok": False, "error": "output_path must end in .csv, .xls, or .xlsx"})
    except Exception as exc:
        return _dumps({"ok": False, "error": f"Failed to write {out_path}: {type(exc).__name__}: {exc}"})

    return _dumps(
        {
            "ok": True,
            "input_path": str(data_path),
            "output_path": str(out_path),
            "dropped_columns": present,
            "missing_columns": missing,
            "remaining_columns": list(filtered.columns),
            "rows": int(len(filtered)),
        }
    )


@mcp.tool(
    description=(
        "Fetch ligand affinity records from BindingDB by protein name or UniProt "
        "accession, filter by Ki/Kd/IC50, and save a CSV for downstream modeling."
    )
)
async def fetch_bindingdb_affinity(
    protein_name: str = "",
    uniprot_id: str = "",
    affinity_type: str = "Ki",
    cutoff_nm: int = 10000,
    output_dir: str = "outputs/pharma/data",
    timeout_seconds: float = 120.0,
) -> str:
    import requests

    affinity_type = affinity_type.strip()
    if affinity_type not in VALID_AFFINITY_TYPES:
        return _dumps({"ok": False, "error": f"affinity_type must be one of {sorted(VALID_AFFINITY_TYPES)}"})
    if not protein_name and not uniprot_id:
        return _dumps({"ok": False, "error": "Provide protein_name or uniprot_id"})

    uniprot_record = None
    if not uniprot_id:
        uniprot_id, uniprot_record = _fetch_uniprot_id(protein_name, timeout_seconds)
        if not uniprot_id:
            return _dumps({"ok": False, "error": f"No UniProt accession found for {protein_name}"})

    url = "https://www.bindingdb.org/rest/getLigandsByUniprots"
    params = {
        "uniprot": uniprot_id,
        "cutoff": int(cutoff_nm),
        "response": "application/json",
    }
    try:
        response = requests.get(url, params=params, timeout=timeout_seconds)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        return _dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}", "source": "BindingDB"})
    except ValueError as exc:
        return _dumps({"ok": False, "error": f"BindingDB did not return JSON: {exc}"})

    container = (
        data.get("getLigandsByUniprotsResponse")
        or data.get("getLindsByUniprotsResponse")
        or data
    )
    affinities = container.get("affinities") if isinstance(container, dict) else []
    if not isinstance(affinities, list):
        affinities = []
    rows = [row for row in affinities if str(row.get("affinity_type", "")).strip() == affinity_type]
    path = _write_rows(rows, output_dir, f"bindingdb_{protein_name or uniprot_id}_{affinity_type}")
    return _dumps(
        {
            "ok": True,
            "source": "BindingDB",
            "protein_name": protein_name,
            "uniprot_id": uniprot_id,
            "uniprot_record": uniprot_record,
            "affinity_type": affinity_type,
            "cutoff_nm": int(cutoff_nm),
            "records": len(rows),
            "output_path": path,
            "sample": rows[:5],
        }
    )


@mcp.tool(
    description=(
        "Fetch ChEMBL activity records for a target and affinity type, collect "
        "canonical SMILES plus standard values, and save a CSV."
    )
)
async def fetch_chembl_activities(
    target_name: str,
    target_id: str = "",
    affinity_type: str = "Ki",
    limit: int = 1000,
    output_dir: str = "outputs/pharma/data",
    timeout_seconds: float = 60.0,
) -> str:
    import requests

    if not target_name and not target_id:
        return _dumps({"ok": False, "error": "Provide target_name or target_id"})
    affinity_type = affinity_type.strip()
    if affinity_type not in VALID_AFFINITY_TYPES:
        return _dumps({"ok": False, "error": f"affinity_type must be one of {sorted(VALID_AFFINITY_TYPES)}"})

    base_url = "https://www.ebi.ac.uk/chembl/api/data"
    target_record = None
    if not target_id:
        try:
            target_search = requests.get(
                f"{base_url}/target/search?q={quote(target_name)}&format=json&limit=100",
                timeout=timeout_seconds,
            )
            target_search.raise_for_status()
            targets = target_search.json().get("targets", [])
        except requests.RequestException as exc:
            return _dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}", "source": "ChEMBL target search"})
        if not targets:
            return _dumps({"ok": False, "error": f"Target '{target_name}' not found in ChEMBL"})
        target_record = targets[0]
        target_id = target_record.get("target_chembl_id", "")
        if not target_id:
            return _dumps({"ok": False, "error": "ChEMBL target record did not include target_chembl_id"})

    max_rows = max(1, int(limit))
    activities: list[dict[str, Any]] = []
    offset = 0
    page_limit = min(max_rows, 1000)
    while len(activities) < max_rows:
        params = {
            "target_chembl_id": target_id,
            "standard_type": affinity_type,
            "format": "json",
            "limit": page_limit,
            "offset": offset,
        }
        try:
            response = requests.get(f"{base_url}/activity.json", params=params, timeout=timeout_seconds)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            return _dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}", "source": "ChEMBL activity"})
        batch = data.get("activities", [])
        if not batch:
            break
        activities.extend(batch)
        if not data.get("page_meta", {}).get("next"):
            break
        offset += len(batch)

    rows: list[dict[str, Any]] = []
    for activity in activities[:max_rows]:
        smiles = activity.get("canonical_smiles")
        value = activity.get("standard_value")
        if not smiles or value is None:
            continue
        rows.append(
            {
                "smiles": smiles,
                affinity_type: value,
                "affinity_units": activity.get("standard_units"),
                "molecule_chembl_id": activity.get("molecule_chembl_id"),
                "activity_chembl_id": activity.get("activity_id") or activity.get("activity_chembl_id"),
                "target_chembl_id": target_id,
            }
        )

    if not rows:
        return _dumps({"ok": False, "error": "No usable ChEMBL activity rows found", "target_chembl_id": target_id})

    path = _write_rows(rows, output_dir, f"chembl_{target_name or target_id}_{affinity_type}")
    return _dumps(
        {
            "ok": True,
            "source": "ChEMBL",
            "target_name": target_name,
            "target_chembl_id": target_id,
            "target_record": target_record,
            "affinity_type": affinity_type,
            "records": len(rows),
            "output_path": path,
            "sample": rows[:5],
        }
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(mcp.run_stdio_async())
