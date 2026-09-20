"""Unit tests for the centralized LLM judge (tests/llm_judge.py).

These tests cover:
  - env-var prefix chain resolution
  - credential fallback to the JSON config / its git-ignored local override
  - the explicit disable flag
  - per-dataset ``enabled`` opt-out
  - JSON-config defaults
  - ``extract_boxed`` edge cases
  - judge + secondary-verify transport via a stubbed httpx client
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import llm_judge
from llm_judge import (
    LLMJudgeConfig,
    extract_boxed,
    is_judge_enabled,
    judge_correct,
    resolve_judge_config,
    secondary_verify,
)


# Config isolation is suite-wide; see tests/conftest.py.


# ---------------------------------------------------------------------------
# Env-var resolution
# ---------------------------------------------------------------------------


def _clear_judge_env(monkeypatch):
    """Wipe every env var the prefix chain might consult."""
    for prefix in ("RESEARCH_CLAW_JUDGE", "JUDGE", "SCIMAS_LLM_JUDGE"):
        for suffix in (
            "_API_KEY", "_KEY", "_API_BASE", "_BASE", "_MODEL", "_MODEL_NAME"
        ):
            monkeypatch.delenv(f"{prefix}{suffix}", raising=False)
    monkeypatch.delenv("RESEARCH_CLAW_JUDGE_DISABLED", raising=False)


def _write_local_creds(tmp_path, monkeypatch, **fields) -> Path:
    """Write a ``llm_judge.local.json`` and point the resolver at it."""
    path = tmp_path / "llm_judge.local.json"
    path.write_text(json.dumps(fields), encoding="utf-8")
    monkeypatch.setattr(llm_judge, "LOCAL_CONFIG_PATH", path)
    return path


def test_resolve_returns_none_when_nothing_configured(monkeypatch):
    _clear_judge_env(monkeypatch)
    cfg = resolve_judge_config()
    assert cfg is None
    assert is_judge_enabled() is False


def test_disabled_flag_short_circuits(monkeypatch):
    _clear_judge_env(monkeypatch)
    monkeypatch.setenv("RESEARCH_CLAW_JUDGE_API_KEY", "sk-abc")
    monkeypatch.setenv("RESEARCH_CLAW_JUDGE_MODEL", "gpt-4.1")
    monkeypatch.setenv("RESEARCH_CLAW_JUDGE_DISABLED", "1")
    assert resolve_judge_config() is None


def test_research_claw_prefix_wins(monkeypatch):
    _clear_judge_env(monkeypatch)
    monkeypatch.setenv("RESEARCH_CLAW_JUDGE_API_KEY", "sk-rcb")
    monkeypatch.setenv("RESEARCH_CLAW_JUDGE_MODEL", "gpt-4.1")
    monkeypatch.setenv("JUDGE_API_KEY", "sk-upstream")
    cfg = resolve_judge_config()
    assert cfg is not None
    assert cfg.api_key == "sk-rcb"
    assert cfg.prefix == "RESEARCH_CLAW_JUDGE"


def test_short_key_suffix_also_recognized(monkeypatch):
    _clear_judge_env(monkeypatch)
    monkeypatch.setenv("SCIMAS_LLM_JUDGE_KEY", "sk-sci")
    monkeypatch.setenv("SCIMAS_LLM_JUDGE_MODEL", "claude-3-7-sonnet")
    cfg = resolve_judge_config()
    assert cfg is not None
    assert cfg.api_key == "sk-sci"
    assert cfg.api_base == "https://api.openai.com/v1"  # default
    assert cfg.prefix == "SCIMAS_LLM_JUDGE"


def test_upstream_fallback_when_rcb_unset(monkeypatch):
    _clear_judge_env(monkeypatch)
    monkeypatch.setenv("JUDGE_API_KEY", "sk-upstream")
    monkeypatch.setenv("JUDGE_MODEL_NAME", "gpt-5.1")
    cfg = resolve_judge_config()
    assert cfg is not None
    assert cfg.api_key == "sk-upstream"
    assert cfg.model == "gpt-5.1"
    assert cfg.prefix == "JUDGE"


def test_incomplete_prefix_with_only_key_is_skipped(monkeypatch):
    """A prefix with a key but no model must yield ``None`` so the next
    prefix can try to supply one."""
    _clear_judge_env(monkeypatch)
    monkeypatch.setenv("RESEARCH_CLAW_JUDGE_API_KEY", "sk-rcb")
    # no _MODEL set for this prefix
    monkeypatch.setenv("JUDGE_API_KEY", "sk-upstream")
    monkeypatch.setenv("JUDGE_MODEL_NAME", "gpt-5.1")
    cfg = resolve_judge_config()
    # Falls back to the JUDGE prefix because the RCB one is incomplete.
    assert cfg is not None
    assert cfg.api_key == "sk-upstream"


def test_per_dataset_disabled(monkeypatch, tmp_path):
    """Writing a custom config with SciAgentGYM.enabled=False should
    suppress that dataset even when credentials are present."""
    _clear_judge_env(monkeypatch)
    monkeypatch.setenv("RESEARCH_CLAW_JUDGE_API_KEY", "sk-rcb")
    monkeypatch.setenv("RESEARCH_CLAW_JUDGE_MODEL", "gpt-4.1")

    cfg_path = tmp_path / "llm_judge.json"
    cfg_path.write_text(
        json.dumps(
            {
                "env_var_prefixes": ["RESEARCH_CLAW_JUDGE"],
                "disabled_env_flag": "RESEARCH_CLAW_JUDGE_DISABLED",
                "default_api_base": "https://api.openai.com/v1",
                "per_dataset": {"SciAgentGYM": {"enabled": False}},
            }
        ),
        encoding="utf-8",
    )

    with mock.patch.object(llm_judge, "CONFIG_PATH", cfg_path):
        cfg = resolve_judge_config(dataset="SciAgentGYM")
        assert cfg is None
        cfg = resolve_judge_config(dataset="ResearchClawBench")
        assert cfg is not None
        assert cfg.api_key == "sk-rcb"


def test_missing_config_file_falls_back_to_in_process_defaults(monkeypatch, tmp_path):
    """If ``config/llm_judge.json`` is absent, the resolver still works
    against the in-process defaults."""
    _clear_judge_env(monkeypatch)
    monkeypatch.setenv("RESEARCH_CLAW_JUDGE_API_KEY", "sk-x")
    monkeypatch.setenv("RESEARCH_CLAW_JUDGE_MODEL", "gpt-x")

    fake = tmp_path / "no_such.json"
    with mock.patch.object(llm_judge, "CONFIG_PATH", fake):
        cfg = resolve_judge_config()
        assert cfg is not None
        assert cfg.api_base == "https://api.openai.com/v1"


# ---------------------------------------------------------------------------
# Credentials from the config file
# ---------------------------------------------------------------------------


def test_config_file_credentials_are_used_when_no_env_var_is_set(
    monkeypatch, tmp_path
):
    """A key in the config must enable the judge with no env vars at all.

    This is the trap that kept the judge off: operators (reasonably) put the
    key in ``config/llm_judge.json``'s ``env_var_layout`` block, which is
    documentation and is never read, so ``resolve_judge_config`` returned
    ``None`` and every answer fell back to the deterministic grader.
    """
    _clear_judge_env(monkeypatch)
    _write_local_creds(
        tmp_path,
        monkeypatch,
        api_key="sk-from-file",
        api_base="https://api.deepseek.com",
        model="deepseek-flash",
    )

    cfg = resolve_judge_config(dataset="SciAgentGYM")

    assert cfg is not None
    assert cfg.api_key == "sk-from-file"
    assert cfg.api_base == "https://api.deepseek.com"
    assert cfg.model == "deepseek-flash"
    assert cfg.prefix == "<config>"
    assert is_judge_enabled(dataset="SciAgentGYM") is True


def test_env_vars_still_win_over_the_config_file(monkeypatch, tmp_path):
    """A deployment that exports a key must not be overridden by a file."""
    _clear_judge_env(monkeypatch)
    _write_local_creds(
        tmp_path, monkeypatch, api_key="sk-from-file", model="file-model"
    )
    monkeypatch.setenv("RESEARCH_CLAW_JUDGE_API_KEY", "sk-from-env")
    monkeypatch.setenv("RESEARCH_CLAW_JUDGE_MODEL", "env-model")

    cfg = resolve_judge_config()

    assert cfg is not None
    assert cfg.api_key == "sk-from-env"
    assert cfg.model == "env-model"
    assert cfg.prefix == "RESEARCH_CLAW_JUDGE"


def test_env_key_without_a_model_falls_through_to_the_config_file(
    monkeypatch, tmp_path
):
    """An incomplete env prefix must not shadow a complete file credential."""
    _clear_judge_env(monkeypatch)
    _write_local_creds(
        tmp_path, monkeypatch, api_key="sk-from-file", model="deepseek-flash"
    )
    monkeypatch.setenv("RESEARCH_CLAW_JUDGE_API_KEY", "sk-orphan-key")

    cfg = resolve_judge_config()

    assert cfg is not None
    assert cfg.api_key == "sk-from-file"
    assert cfg.prefix == "<config>"


def test_env_var_layout_is_never_read_as_a_credential(monkeypatch, tmp_path):
    """The documented trap, asserted directly: prose in ``env_var_layout``
    must not enable the judge."""
    _clear_judge_env(monkeypatch)
    tracked = tmp_path / "llm_judge.json"
    tracked.write_text(
        json.dumps(
            {
                "env_var_layout": {
                    "_API_KEY": "sk-this-is-documentation-not-a-credential",
                    "_API_BASE": "https://api.deepseek.com",
                    "_MODEL": "deepseek-flash",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(llm_judge, "CONFIG_PATH", tracked)

    assert resolve_judge_config() is None


def test_a_config_without_a_key_does_not_enable_the_judge(monkeypatch, tmp_path):
    _clear_judge_env(monkeypatch)
    _write_local_creds(tmp_path, monkeypatch, model="deepseek-flash")

    assert resolve_judge_config() is None


def test_the_local_file_overrides_the_tracked_config(monkeypatch, tmp_path):
    """Shared defaults live in the tracked file; the local file wins."""
    _clear_judge_env(monkeypatch)
    tracked = tmp_path / "llm_judge.json"
    tracked.write_text(
        json.dumps({"default_max_tokens": 5, "default_model": "gpt-4.1"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(llm_judge, "CONFIG_PATH", tracked)
    _write_local_creds(
        tmp_path,
        monkeypatch,
        api_key="sk-local",
        model="deepseek-flash",
        default_max_tokens=2048,
    )

    cfg = resolve_judge_config()

    assert cfg is not None
    assert cfg.model == "deepseek-flash"
    assert cfg.max_tokens == 2048


# ---------------------------------------------------------------------------
# extract_boxed
# ---------------------------------------------------------------------------


def test_extract_boxed_simple():
    assert extract_boxed("Answer: \\boxed{42}") == "42"


def test_extract_boxed_with_nested_braces():
    """Nested ``{}`` must be balanced correctly (recursive descend)."""
    text = "Answer: \\boxed{\\frac{1}{2}}"
    assert extract_boxed(text) == "\\frac{1}{2}"


def test_extract_boxed_returns_last_when_multiple():
    text = "First: \\boxed{A}. Last: \\boxed{B}."
    assert extract_boxed(text) == "B"


def test_extract_boxed_handles_dollar_prefix():
    """SciAgentGYM sometimes wraps ``$...$`` around the boxed content."""
    text = "Answer: $\\boxed{2.6 \\times 10^{5}}$ GeV"
    assert extract_boxed(text) == "2.6 \\times 10^{5}"


def test_extract_boxed_returns_none_on_missing():
    assert extract_boxed("no answer here") is None
    assert extract_boxed("") is None
    assert extract_boxed(None) is None


# ---------------------------------------------------------------------------
# judge_correct / secondary_verify transport
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _stub_chat_payload(text: str) -> dict:
    return {"choices": [{"message": {"content": text}}]}


def test_judge_correct_returns_true_on_zhengque(monkeypatch):
    _clear_judge_env(monkeypatch)
    monkeypatch.setenv("RESEARCH_CLAW_JUDGE_API_KEY", "sk-1")
    monkeypatch.setenv("RESEARCH_CLAW_JUDGE_MODEL", "gpt-4.1")

    captured: dict = {}

    def fake_post(url, headers=None, json=None):
        captured["url"] = url
        captured["body"] = json
        return _FakeResponse(_stub_chat_payload("正确"))

    fake_client = mock.MagicMock()
    fake_client.__enter__ = mock.MagicMock(return_value=fake_client)
    fake_client.__exit__ = mock.MagicMock(return_value=False)
    fake_client.post = fake_post

    import httpx

    with mock.patch.object(httpx, "Client", return_value=fake_client):
        verdict = judge_correct("q", "predicted", "expected")
    assert verdict is True
    assert captured["url"].endswith("/chat/completions")
    assert "predicted" in captured["body"]["messages"][1]["content"]
    assert "expected" in captured["body"]["messages"][1]["content"]


def test_judge_correct_returns_false_on_cuowu(monkeypatch):
    _clear_judge_env(monkeypatch)
    monkeypatch.setenv("RESEARCH_CLAW_JUDGE_API_KEY", "sk-1")
    monkeypatch.setenv("RESEARCH_CLAW_JUDGE_MODEL", "gpt-4.1")

    fake_client = mock.MagicMock()
    fake_client.__enter__ = mock.MagicMock(return_value=fake_client)
    fake_client.__exit__ = mock.MagicMock(return_value=False)
    fake_client.post = mock.MagicMock(
        return_value=_FakeResponse(_stub_chat_payload("错误"))
    )

    import httpx

    with mock.patch.object(httpx, "Client", return_value=fake_client):
        assert judge_correct("q", "p", "e") is False


def test_judge_correct_returns_none_when_disabled(monkeypatch):
    _clear_judge_env(monkeypatch)
    assert judge_correct("q", "p", "e") is None


def test_judge_correct_handles_transport_failure(monkeypatch):
    _clear_judge_env(monkeypatch)
    monkeypatch.setenv("RESEARCH_CLAW_JUDGE_API_KEY", "sk-1")
    monkeypatch.setenv("RESEARCH_CLAW_JUDGE_MODEL", "gpt-4.1")

    fake_client = mock.MagicMock()
    fake_client.__enter__ = mock.MagicMock(return_value=fake_client)
    fake_client.__exit__ = mock.MagicMock(return_value=False)
    fake_client.post = mock.MagicMock(side_effect=RuntimeError("boom"))

    import httpx

    with mock.patch.object(httpx, "Client", return_value=fake_client):
        assert judge_correct("q", "p", "e") is None


def test_judge_correct_parses_english_aliases(monkeypatch):
    _clear_judge_env(monkeypatch)
    monkeypatch.setenv("RESEARCH_CLAW_JUDGE_API_KEY", "sk-1")
    monkeypatch.setenv("RESEARCH_CLAW_JUDGE_MODEL", "gpt-4.1")

    fake_client = mock.MagicMock()
    fake_client.__enter__ = mock.MagicMock(return_value=fake_client)
    fake_client.__exit__ = mock.MagicMock(return_value=False)
    fake_client.post = mock.MagicMock(
        return_value=_FakeResponse(_stub_chat_payload("Correct"))
    )

    import httpx

    with mock.patch.object(httpx, "Client", return_value=fake_client):
        assert judge_correct("q", "p", "e") is True


def test_secondary_verify_parses_match(monkeypatch):
    _clear_judge_env(monkeypatch)
    monkeypatch.setenv("RESEARCH_CLAW_JUDGE_API_KEY", "sk-1")
    monkeypatch.setenv("RESEARCH_CLAW_JUDGE_MODEL", "gpt-4.1")

    fake_client = mock.MagicMock()
    fake_client.__enter__ = mock.MagicMock(return_value=fake_client)
    fake_client.__exit__ = mock.MagicMock(return_value=False)
    fake_client.post = mock.MagicMock(
        return_value=_FakeResponse(_stub_chat_payload("匹配"))
    )

    import httpx

    with mock.patch.object(httpx, "Client", return_value=fake_client):
        assert secondary_verify("actual", "expected") is True


def test_secondary_verify_parses_mismatch(monkeypatch):
    _clear_judge_env(monkeypatch)
    monkeypatch.setenv("RESEARCH_CLAW_JUDGE_API_KEY", "sk-1")
    monkeypatch.setenv("RESEARCH_CLAW_JUDGE_MODEL", "gpt-4.1")

    fake_client = mock.MagicMock()
    fake_client.__enter__ = mock.MagicMock(return_value=fake_client)
    fake_client.__exit__ = mock.MagicMock(return_value=False)
    fake_client.post = mock.MagicMock(
        return_value=_FakeResponse(_stub_chat_payload("不匹配"))
    )

    import httpx

    with mock.patch.object(httpx, "Client", return_value=fake_client):
        assert secondary_verify("actual", "expected") is False


# ---------------------------------------------------------------------------
# Dataclass shape
# ---------------------------------------------------------------------------


def test_llmjudgeconfig_is_frozen():
    cfg = LLMJudgeConfig(
        api_base="https://api.example.com/v1",
        api_key="sk-test",
        model="gpt-test",
    )
    try:
        cfg.model = "other"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("LLMJudgeConfig should be frozen")


def test_dataset_setting_returns_empty_for_unknown():
    assert llm_judge.dataset_setting("NoSuchDataset") == {}


def test_dataset_setting_picks_per_dataset_block(tmp_path):
    cfg_path = tmp_path / "llm_judge.json"
    cfg_path.write_text(
        json.dumps(
            {
                "env_var_prefixes": ["RESEARCH_CLAW_JUDGE"],
                "disabled_env_flag": "RESEARCH_CLAW_JUDGE_DISABLED",
                "per_dataset": {
                    "SciAgentGYM": {
                        "enabled": True,
                        "fallback_to_deterministic": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    with mock.patch.object(llm_judge, "CONFIG_PATH", cfg_path):
        per = llm_judge.dataset_setting("SciAgentGYM")
        assert per["enabled"] is True
        assert per["fallback_to_deterministic"] is True
        assert llm_judge.dataset_setting("Other") == {}


# ---------------------------------------------------------------------------
# Question figures (vision content blocks)
# ---------------------------------------------------------------------------


# 1x1 transparent PNG — the same fixture test_rcb_official_judge uses.
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4"
    "2mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


def _write_png(tmp_path, name: str = "figure.png") -> Path:
    import base64

    path = tmp_path / name
    path.write_bytes(base64.b64decode(_TINY_PNG_B64))
    return path


class _StubHttpError(Exception):
    """The shape ``_call_judge`` reads a status code off."""

    def __init__(self, status: int) -> None:
        super().__init__(f"HTTP {status}")
        self.response = type("_Response", (), {"status_code": status})()


def _length_payload() -> dict:
    return {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}


@contextlib.contextmanager
def _scripted_judge(monkeypatch, responses: list):
    """Drive the judge with a scripted httpx client, capturing request bodies.

    ``responses`` is consumed one entry per POST: a ``str`` becomes a chat
    payload answering with it (``"__length__"`` answers with an empty
    length-truncated completion), an ``Exception`` instance is raised.
    """
    _clear_judge_env(monkeypatch)
    monkeypatch.setenv("RESEARCH_CLAW_JUDGE_API_KEY", "sk-1")
    monkeypatch.setenv("RESEARCH_CLAW_JUDGE_MODEL", "gpt-4.1")
    bodies: list[dict] = []

    def fake_post(url, headers=None, json=None):
        bodies.append(json)
        item = responses[min(len(bodies) - 1, len(responses) - 1)]
        if isinstance(item, Exception):
            raise item
        if item == "__length__":
            return _FakeResponse(_length_payload())
        return _FakeResponse(_stub_chat_payload(item))

    fake_client = mock.MagicMock()
    fake_client.__enter__ = mock.MagicMock(return_value=fake_client)
    fake_client.__exit__ = mock.MagicMock(return_value=False)
    fake_client.post = fake_post

    import httpx

    with mock.patch.object(httpx, "Client", return_value=fake_client):
        yield bodies


def test_question_figures_ride_along_as_vision_blocks(monkeypatch, tmp_path):
    figure = _write_png(tmp_path, "chart.png")

    with _scripted_judge(monkeypatch, ["正确"]) as bodies:
        verdict = judge_correct("q", "p", "e", images=[str(figure)])

    assert verdict is True
    content = bodies[0]["messages"][1]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    # The figure belongs to the question, and the prompt has to say so: unlike
    # ResearchClawBench — where the attached image is the ground-truth target —
    # a judge reading these as the model's own output would grade the chart.
    assert "题目附图" in content[0]["text"]
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert llm_judge.get_last_image_note() == ""


def test_without_images_the_user_content_stays_a_plain_string(monkeypatch):
    """The pre-figures request shape, which the prompt tests assert on."""
    with _scripted_judge(monkeypatch, ["正确"]) as bodies:
        assert judge_correct("q", "p", "e") is True

    assert isinstance(bodies[0]["messages"][1]["content"], str)


def test_vision_call_failure_retries_text_only(monkeypatch, tmp_path):
    """A model without image support 400s; the verdict must still arrive, and
    the notes must not imply the figures were seen."""
    figure = _write_png(tmp_path)
    responses = [_StubHttpError(400), "正确"]

    with _scripted_judge(monkeypatch, responses) as bodies:
        verdict = judge_correct("q", "p", "e", images=[str(figure)])

    assert verdict is True
    assert len(bodies) == 2
    assert isinstance(bodies[0]["messages"][1]["content"], list)
    assert isinstance(bodies[1]["messages"][1]["content"], str)
    note = llm_judge.get_last_image_note()
    assert "not sent" in note and "400" in note
    # The call succeeded, so there is no transport error to report.
    assert llm_judge.get_last_judge_error() == ""


def test_failed_text_only_retry_keeps_both_reasons(monkeypatch, tmp_path):
    figure = _write_png(tmp_path)
    responses = [_StubHttpError(400), RuntimeError("boom")]

    with _scripted_judge(monkeypatch, responses) as bodies:
        verdict = judge_correct("q", "p", "e", images=[str(figure)])

    assert verdict is None
    assert len(bodies) == 2
    note = llm_judge.get_last_image_note()
    assert "not sent" in note and "400" in note
    assert "retry failed" in note
    assert llm_judge.get_last_judge_error()  # the text-only failure, kept apart


def test_oversized_image_is_skipped_and_noted(monkeypatch, tmp_path):
    figure = _write_png(tmp_path)
    monkeypatch.setattr(llm_judge, "MAX_IMAGE_BYTES", 10)
    monkeypatch.setattr(llm_judge, "_HAS_PILLOW", False)

    with _scripted_judge(monkeypatch, ["正确"]) as bodies:
        assert judge_correct("q", "p", "e", images=[str(figure)]) is True

    # No block was built, so the request is the plain text-only one — and the
    # prompt must not announce a figure that is not attached.
    content = bodies[0]["messages"][1]["content"]
    assert isinstance(content, str)
    assert "题目附图" not in content
    assert "not sent" not in llm_judge.get_last_image_note()  # it never got sent
    assert str(figure) in llm_judge.get_last_image_note()


def test_unreadable_image_is_skipped_and_noted(monkeypatch, tmp_path):
    missing = tmp_path / "gone.png"

    with _scripted_judge(monkeypatch, ["正确"]) as bodies:
        assert judge_correct("q", "p", "e", images=[str(missing)]) is True

    assert isinstance(bodies[0]["messages"][1]["content"], str)
    assert "not found" in llm_judge.get_last_image_note()


def test_image_note_is_cleared_on_the_next_call(monkeypatch, tmp_path):
    missing = tmp_path / "gone.png"

    with _scripted_judge(monkeypatch, ["正确"]):
        judge_correct("q", "p", "e", images=[str(missing)])
        assert llm_judge.get_last_image_note()
        judge_correct("q", "p", "e")

    assert llm_judge.get_last_image_note() == ""


def test_length_escalation_resends_the_same_images(monkeypatch, tmp_path):
    """The escalation retry re-sends the same body; the figures must not be
    re-encoded or dropped on the way."""
    figure = _write_png(tmp_path)

    with _scripted_judge(monkeypatch, ["__length__", "正确"]) as bodies:
        assert judge_correct("q", "p", "e", images=[str(figure)]) is True

    assert len(bodies) >= 2
    first = bodies[0]["messages"][1]["content"]
    assert all(body["messages"][1]["content"] == first for body in bodies)


def test_secondary_verify_never_attaches_images(monkeypatch, tmp_path):
    """It is a text-only leaf-value rescue; figures have no place in it."""
    figure = _write_png(tmp_path)

    with _scripted_judge(monkeypatch, ["匹配"]) as bodies:
        assert secondary_verify("1.4", "1.4") is True

    assert isinstance(bodies[0]["messages"][1]["content"], str)
    assert str(figure) not in llm_judge.get_last_image_note()


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))