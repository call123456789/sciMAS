from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from tools.pharma import drug_discovery_server as server


def _madd_root(tmp_path: Path) -> Path:
    root = tmp_path / "MADD-main"
    weights = (
        root
        / "infrastructure"
        / "generative_models"
        / "autotrain"
        / "many_prop_CVAE"
        / "weights_8p_cnsr"
    )
    weights.mkdir(parents=True)
    (weights / "model_weights").write_text("checkpoint", encoding="utf-8")
    return root


def test_local_madd_generation_preflight_failure_skips_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SCIMAS_ENABLE_LOCAL_MADD_GENERATION", "1")
    root = _madd_root(tmp_path)
    calls = []

    def fake_subprocess_json(code, payload, root_arg, python_bin, timeout_seconds):
        calls.append(
            {
                "payload": payload,
                "timeout_seconds": timeout_seconds,
                "is_generation": "case_generator(data)" in code,
            }
        )
        return {
            "ok": False,
            "error": "local MADD subprocess timed out after 20.0 seconds",
        }

    monkeypatch.setattr(server, "_subprocess_json", fake_subprocess_json)

    payload = asyncio.run(
        server.generate_molecules_with_local_madd(
            case="lung cancer",
            madd_root=str(root),
            preflight_timeout_seconds=7.0,
        )
    )
    data = json.loads(payload)

    assert data["ok"] is False
    assert data["mapped_case"] == "Cnsr"
    assert "preflight failed" in data["error"]
    assert data["preflight"]["error"].startswith("local MADD subprocess timed out")
    assert calls == [
        {
            "payload": {
                "madd_root": str(root),
                "case_code": "Cnsr",
                "cuda": False,
            },
            "timeout_seconds": 7.0,
            "is_generation": False,
        }
    ]


def test_local_madd_generation_runs_only_after_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SCIMAS_ENABLE_LOCAL_MADD_GENERATION", "1")
    root = _madd_root(tmp_path)
    timeouts = []

    def fake_subprocess_json(code, payload, root_arg, python_bin, timeout_seconds):
        timeouts.append(timeout_seconds)
        if "case_generator(data)" not in code:
            return {"ok": True, "runtime": {"entrypoint": "api_utils.case_generator"}}
        return {"ok": True, "result": {"Molecules": ["C"]}}

    monkeypatch.setattr(server, "_subprocess_json", fake_subprocess_json)

    payload = asyncio.run(
        server.generate_molecules_with_local_madd(
            case="lung cancer",
            madd_root=str(root),
        )
    )
    data = json.loads(payload)

    assert data["ok"] is True
    assert data["result"] == {"Molecules": ["C"]}
    assert data["preflight"]["runtime"]["entrypoint"] == "api_utils.case_generator"
    assert timeouts == [
        server.DEFAULT_LOCAL_MADD_PREFLIGHT_TIMEOUT,
        server.DEFAULT_LOCAL_MADD_GENERATION_TIMEOUT,
    ]


def test_local_madd_generation_is_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SCIMAS_ENABLE_LOCAL_MADD_GENERATION", raising=False)

    payload = asyncio.run(
        server.generate_molecules_with_local_madd(case="lung cancer")
    )
    data = json.loads(payload)

    assert data["ok"] is False
    assert "disabled" in data["error"].lower()
