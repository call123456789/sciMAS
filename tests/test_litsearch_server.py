from __future__ import annotations

import json

import pytest
import requests

from tools.web import litsearch_server as litsearch


class _Response:
    def __init__(self, status: int, payload=None, text: str = "") -> None:
        self.status_code = status
        self._payload = payload
        self.text = text

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def _reset_exhausted_keys():
    litsearch._EXHAUSTED_API_KEYS.clear()
    yield
    litsearch._EXHAUSTED_API_KEYS.clear()


def test_config_keys_are_ordered_deduplicated_and_preferred_over_env(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "openalex.local.json"
    path.write_text(
        json.dumps(
            {
                "api_keys": [" primary ", "secondary", "primary", ""],
                "api_key": "legacy-third",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(litsearch, "LOCAL_CONFIG_PATH", path)
    monkeypatch.setenv("OPENALEX_API_KEY", "environment-key")

    assert litsearch._openalex_api_keys() == [
        "primary",
        "secondary",
        "legacy-third",
    ]


def test_legacy_environment_key_is_used_when_config_is_missing(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(litsearch, "LOCAL_CONFIG_PATH", tmp_path / "missing.json")
    monkeypatch.setenv("OPENALEX_API_KEYS", "first, second")
    monkeypatch.setenv("OPENALEX_API_KEY", "third")

    assert litsearch._openalex_api_keys() == ["first", "second", "third"]


def test_quota_response_switches_to_second_key_and_remembers_first(
    monkeypatch,
) -> None:
    calls = []

    def fake_get(url, *, params, timeout):
        calls.append(dict(params))
        if params["api_key"] == "first-secret":
            return _Response(429, text="daily usage limit exceeded")
        return _Response(200, payload={"results": [], "meta": {"count": 0}})

    monkeypatch.setattr(litsearch.requests, "get", fake_get)
    data, slot = litsearch._request_openalex_works(
        {"search": "kinase"}, 3.0, ["first-secret", "second-secret"]
    )

    assert data["meta"]["count"] == 0
    assert slot == 2
    assert [call["api_key"] for call in calls] == [
        "first-secret",
        "second-secret",
    ]

    calls.clear()
    _, slot = litsearch._request_openalex_works(
        {"search": "kinase"}, 3.0, ["first-secret", "second-secret"]
    )
    assert slot == 2
    assert [call["api_key"] for call in calls] == ["second-secret"]


def test_server_error_does_not_rotate_keys(monkeypatch) -> None:
    calls = []

    def fake_get(url, *, params, timeout):
        calls.append(dict(params))
        return _Response(500, text="upstream unavailable")

    monkeypatch.setattr(litsearch.requests, "get", fake_get)

    with pytest.raises(RuntimeError, match="HTTP 500"):
        litsearch._request_openalex_works(
            {"search": "kinase"}, 3.0, ["first-secret", "second-secret"]
        )
    assert [call["api_key"] for call in calls] == ["first-secret"]


def test_all_rejected_error_never_exposes_keys(monkeypatch) -> None:
    def fake_get(url, *, params, timeout):
        return _Response(403, text=f"key {params['api_key']} has no credits")

    monkeypatch.setattr(litsearch.requests, "get", fake_get)

    with pytest.raises(RuntimeError) as error:
        litsearch._request_openalex_works(
            {"search": "kinase"}, 3.0, ["first-secret", "second-secret"]
        )
    message = str(error.value)
    assert "all 2 configured API keys" in message
    assert "first-secret" not in message
    assert "second-secret" not in message
