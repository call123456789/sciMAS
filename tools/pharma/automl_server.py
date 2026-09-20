#!/usr/bin/env python3
"""MADD AutoML service control MCP server for pharmaceutical workflows."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("pharma-automl")


def _dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _coerce_list(value: Any, field_name: str) -> list[str]:
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
    raise TypeError(f"{field_name} must be a string or list of strings")


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
    try:
        if data_path.suffix.lower() == ".csv":
            return pd.read_csv(data_path), None
        if data_path.suffix.lower() in {".xls", ".xlsx"}:
            return pd.read_excel(data_path), None
    except Exception as exc:
        return None, f"Failed to read {data_path}: {type(exc).__name__}: {exc}"
    return None, "Unsupported file format. Use CSV, XLS, or XLSX."


def _resolve_base_url(server: str, explicit_url: str | None) -> tuple[str | None, str | None]:
    key = server.strip().lower()
    if key in {"pred", "predict", "prediction", "ml"}:
        env_name = "URL_PRED"
    elif key in {"gen", "generate", "generation", "dl"}:
        env_name = "URL_GEN"
    else:
        return None, "server must be 'pred' or 'gen'"

    base = (explicit_url or os.environ.get(env_name) or "").strip()
    if not base:
        return None, f"{env_name} is not configured. Pass url or set {env_name}."
    if not base.startswith(("http://", "https://")):
        base = "http://" + base
    return base.rstrip("/"), None


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


def _server_root(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return base_url.rstrip("/")


def _request(method: str, url: str, payload: dict[str, Any] | None, timeout_seconds: float) -> dict[str, Any]:
    import requests

    started = time.time()
    try:
        if method == "GET":
            resp = requests.get(url, timeout=timeout_seconds)
        else:
            resp = requests.post(
                url,
                data=json.dumps(payload or {}),
                headers={"Content-Type": "application/json"},
                timeout=timeout_seconds,
            )
    except requests.Timeout as exc:
        return {
            "ok": None,
            "status": "timeout",
            "url": url,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": round(time.time() - started, 3),
        }
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


def _training_payload(
    path: str,
    case: str,
    feature_column: list[str] | str,
    target_column: list[str] | str,
    regression_props: list[str] | str | None,
    classification_props: list[str] | str | None,
    description: str,
    timeout_minutes: int,
    min_rows: int,
    max_smiles_length: int,
    fine_tune: bool | None = None,
    n_samples: int | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    df, error = _read_table(path)
    if error:
        return None, {"ok": False, "error": error}

    try:
        feature_cols = _coerce_list(feature_column, "feature_column")
        target_cols = _coerce_list(target_column, "target_column")
        regression = _coerce_list(regression_props, "regression_props")
        classification = _coerce_list(classification_props, "classification_props")
    except TypeError as exc:
        return None, {"ok": False, "error": str(exc)}

    if not feature_cols:
        return None, {"ok": False, "error": "feature_column must contain at least one column"}
    if not target_cols:
        return None, {"ok": False, "error": "target_column must contain at least one column"}
    if not regression and not classification:
        regression = list(target_cols)

    missing = [col for col in [*feature_cols, *target_cols] if col not in df.columns]
    if missing:
        return None, {
            "ok": False,
            "error": "Required columns are missing",
            "missing_columns": missing,
            "available_columns": list(df.columns),
        }
    if len(df) < min_rows:
        return None, {
            "ok": False,
            "error": f"Training dataset is too small: {len(df)} rows, minimum is {min_rows}.",
            "rows": int(len(df)),
        }

    before_rows = len(df)
    smiles_col = feature_cols[0]
    df = df[df[smiles_col].apply(lambda item: isinstance(item, str) and len(item) <= max_smiles_length)].copy()
    if len(df) < min_rows:
        return None, {
            "ok": False,
            "error": "Too few rows remain after filtering invalid or overly long SMILES.",
            "rows_before_filter": int(before_rows),
            "rows_after_filter": int(len(df)),
            "min_rows": int(min_rows),
        }

    payload: dict[str, Any] = {
        "case": case,
        "data": df.to_dict(),
        "target_column": target_cols,
        "feature_column": feature_cols,
        "timeout": int(timeout_minutes),
        "description": description,
        "regression_props": regression,
        "classification_props": classification,
    }
    if fine_tune is not None:
        payload["fine_tune"] = bool(fine_tune)
    if n_samples is not None:
        payload["n_samples"] = int(n_samples)

    metadata = {
        "path": str(Path(path).expanduser()),
        "rows_before_filter": int(before_rows),
        "rows_sent": int(len(df)),
        "rows_removed": int(before_rows - len(df)),
        "feature_column": feature_cols,
        "target_column": target_cols,
        "regression_props": regression,
        "classification_props": classification,
    }
    return payload, metadata


@mcp.tool(
    description=(
        "Check a MADD predictive or generative service state. Use server='pred' "
        "for URL_PRED or server='gen' for URL_GEN."
    )
)
async def madd_server_state(
    server: str = "pred",
    url: str | None = None,
    timeout_seconds: float = 30.0,
) -> str:
    base_url, error = _resolve_base_url(server, url)
    if error:
        return _dumps({"ok": False, "error": error})
    response = _request("GET", f"{_server_root(base_url)}/check_state", None, timeout_seconds)
    payload = response.get("response")
    if isinstance(payload, dict) and "state" in payload:
        response["state"] = payload["state"]
    return _dumps(response)


@mcp.tool(
    description=(
        "Check one model/case in a MADD predictive or generative service state."
    )
)
async def madd_case_state(
    case: str,
    server: str = "pred",
    url: str | None = None,
    timeout_seconds: float = 30.0,
) -> str:
    base_url, error = _resolve_base_url(server, url)
    if error:
        return _dumps({"ok": False, "error": error})
    response = _request("GET", f"{_server_root(base_url)}/check_state", None, timeout_seconds)
    payload = response.get("response")
    state = payload.get("state") if isinstance(payload, dict) else None
    if isinstance(state, dict):
        response["case"] = case
        response["case_state"] = state.get(case)
        if case not in state:
            response["message"] = f"Case '{case}' not found"
    return _dumps(response)


@mcp.tool(
    description=(
        "Submit a tabular dataset to MADD predictive ML training. Validates "
        "columns and SMILES length before POSTing to URL_PRED/train_ml."
    )
)
async def start_madd_ml_training(
    case: str,
    path: str,
    feature_column: list[str] | str = "canonical_smiles",
    target_column: list[str] | str = "docking_score",
    regression_props: list[str] | str | None = None,
    classification_props: list[str] | str | None = None,
    description: str = "",
    url_pred: str | None = None,
    training_timeout_minutes: int = 2,
    request_timeout_seconds: float = 10.0,
    min_rows: int = 300,
    max_smiles_length: int = 200,
) -> str:
    payload, metadata = _training_payload(
        path=path,
        case=case,
        feature_column=feature_column,
        target_column=target_column,
        regression_props=regression_props,
        classification_props=classification_props,
        description=description,
        timeout_minutes=training_timeout_minutes,
        min_rows=min_rows,
        max_smiles_length=max_smiles_length,
    )
    if payload is None:
        return _dumps(metadata)
    base_url, error = _resolve_base_url("pred", url_pred)
    if error:
        return _dumps({"ok": False, "error": error, "validated_dataset": metadata})
    response = _request("POST", f"{base_url}/train_ml", payload, request_timeout_seconds)
    response["case"] = case
    response["validated_dataset"] = metadata
    if response.get("status") == "timeout":
        response["message"] = "The training request timed out locally; check madd_case_state for progress."
    return _dumps(response)


@mcp.tool(
    description=(
        "Submit a tabular dataset to MADD generative-model training. Validates "
        "columns and SMILES length before POSTing to URL_GEN/train_gen_models."
    )
)
async def start_madd_generative_training(
    case: str,
    path: str,
    feature_column: list[str] | str = "smiles",
    target_column: list[str] | str = "docking_score",
    regression_props: list[str] | str | None = None,
    classification_props: list[str] | str | None = None,
    description: str = "",
    url_gen: str | None = None,
    fine_tune: bool = True,
    n_samples: int = 10,
    training_timeout_minutes: int = 2,
    request_timeout_seconds: float = 10.0,
    min_rows: int = 300,
    max_smiles_length: int = 200,
) -> str:
    payload, metadata = _training_payload(
        path=path,
        case=case,
        feature_column=feature_column,
        target_column=target_column,
        regression_props=regression_props,
        classification_props=classification_props,
        description=description,
        timeout_minutes=training_timeout_minutes,
        min_rows=min_rows,
        max_smiles_length=max_smiles_length,
        fine_tune=fine_tune,
        n_samples=n_samples,
    )
    if payload is None:
        return _dumps(metadata)
    base_url, error = _resolve_base_url("gen", url_gen)
    if error:
        return _dumps({"ok": False, "error": error, "validated_dataset": metadata})
    response = _request("POST", f"{base_url}/train_gen_models", payload, request_timeout_seconds)
    response["case"] = case
    response["validated_dataset"] = metadata
    if response.get("status") == "timeout":
        response["message"] = "The training request timed out locally; check madd_case_state for progress."
    return _dumps(response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(mcp.run_stdio_async())
