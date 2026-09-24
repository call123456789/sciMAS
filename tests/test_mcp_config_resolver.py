"""Tests for ClaudeRunner._resolve_mcp_config — the runtime mcp.json rewriter.

Run with:
    cd "$(git rev-parse --show-toplevel)"
    python -m pytest tests/test_mcp_config_resolver.py -v
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from claude_runner import ClaudeRunner  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_ambient_no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep SCIMAS_MCP_NO_PROXY from leaking in off the developer's shell.

    Several tests below assert "nothing was rewritten"; an exported
    SCIMAS_MCP_NO_PROXY would rewrite every config and fail them for a
    reason unrelated to the code under test.
    """
    monkeypatch.delenv("SCIMAS_MCP_NO_PROXY", raising=False)


def _write_config(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_resolves_bare_python_to_sys_executable(tmp_path: Path) -> None:
    """Bare `python` / `python3` commands become sys.executable."""
    src = tmp_path / "mcp.json"
    _write_config(src, {
        "mcpServers": {
            "foo": {"command": "python", "args": ["srv.py"]},
            "bar": {"command": "python3", "args": ["srv.py"]},
        }
    })
    runner = ClaudeRunner(mcp_config_path=str(src))
    assert runner.mcp_config_path is not None
    assert runner.mcp_config_path.endswith(".resolved.json")
    resolved = json.loads(Path(runner.mcp_config_path).read_text())
    for name in ("foo", "bar"):
        assert resolved["mcpServers"][name]["command"] == sys.executable


def test_strips_env_path_overrides(tmp_path: Path) -> None:
    """env.PATH overrides are dropped (they bake in machine-specific paths)."""
    src = tmp_path / "mcp.json"
    _write_config(src, {
        "mcpServers": {
            "x": {"command": "python", "args": ["srv.py"],
                  "env": {"PATH": "/opt/anaconda3/envs/scimas/bin:/usr/bin"}},
            "y": {"command": "python", "args": ["srv.py"],
                  "env": {"FOO": "bar", "PATH": "/opt/broken"}},
        }
    })
    runner = ClaudeRunner(mcp_config_path=str(src))
    resolved = json.loads(Path(runner.mcp_config_path).read_text())
    # x had only PATH, so env should be gone entirely.
    assert "env" not in resolved["mcpServers"]["x"]
    # y had FOO too — FOO survives, PATH goes.
    assert resolved["mcpServers"]["y"]["env"] == {"FOO": "bar"}


def test_leaves_absolute_paths_alone(tmp_path: Path) -> None:
    """Hard-coded absolute paths are not rewritten (user knows what they want)."""
    src = tmp_path / "mcp.json"
    abs_py = "/usr/local/bin/python"
    _write_config(src, {
        "mcpServers": {
            "fixed": {"command": abs_py, "args": ["srv.py"]},
        }
    })
    runner = ClaudeRunner(mcp_config_path=str(src))
    resolved = json.loads(Path(runner.mcp_config_path).read_text())
    assert resolved["mcpServers"]["fixed"]["command"] == abs_py


def test_no_rewrites_returns_original_path(tmp_path: Path) -> None:
    """If nothing needed substituting, hand back the original path verbatim."""
    src = tmp_path / "mcp.json"
    _write_config(src, {
        "mcpServers": {
            "ok": {"command": "/already/absolute", "args": ["srv.py"]},
        }
    })
    runner = ClaudeRunner(mcp_config_path=str(src))
    assert runner.mcp_config_path == str(src)


def test_missing_file_returns_input(tmp_path: Path) -> None:
    """Non-existent input → return input path, don't crash."""
    bogus = tmp_path / "nope.json"
    runner = ClaudeRunner(mcp_config_path=str(bogus))
    assert runner.mcp_config_path == str(bogus)


def test_none_mcp_config_returns_none() -> None:
    """mcp_config_path='' opt-out leaves self.mcp_config_path as None."""
    runner = ClaudeRunner(mcp_config_path="")
    assert runner.mcp_config_path is None
    # No MCP in play → allowed_tools defaults to [].
    assert runner.allowed_tools == []


def test_empty_string_is_opt_out_but_none_auto_detects() -> None:
    """'' disables MCP; None falls back to the shipped config/mcp.json.

    The two must not be conflated — a caller forwarding a blank user field
    as '' silently turns MCP off. See _mcp_config_for_job in web_dashboard.
    """
    opt_out = ClaudeRunner(mcp_config_path="")
    assert opt_out.mcp_config_path is None

    auto = ClaudeRunner(mcp_config_path=None)
    shipped = ROOT / "config" / "mcp.json"
    if not shipped.is_file():
        return  # vendored checkout without config/
    assert auto.mcp_config_path is not None
    assert "mcp" in Path(auto.mcp_config_path).name


def test_disabled_mcp_emits_no_mcp_config_flag() -> None:
    """A disabled runner must not put --mcp-config on the command line."""
    runner = ClaudeRunner(binary="claude", mcp_config_path="")
    cmd = runner._build_cmd("PROMPT", None, "stream-json", ["mcp__litsearch__search_literature"])
    assert not any(arg.startswith("--mcp-config") for arg in cmd)


def test_build_cmd_can_override_model_per_call() -> None:
    runner = ClaudeRunner(binary="claude", model="default-model", mcp_config_path="")

    default_cmd = runner._build_cmd("PROMPT", None, "stream-json", [])
    override_cmd = runner._build_cmd(
        "PROMPT", None, "stream-json", [], model="planner-model"
    )

    assert default_cmd[default_cmd.index("--model") + 1] == "default-model"
    assert override_cmd[override_cmd.index("--model") + 1] == "planner-model"


def test_subprocess_env_merges_runner_and_per_call_overrides() -> None:
    runner = ClaudeRunner(
        mcp_config_path="",
        env_overrides={"ANTHROPIC_BASE_URL": "https://runner.example"},
    )

    env = runner._subprocess_env(
        {
            "ANTHROPIC_BASE_URL": "https://call.example",
            "ANTHROPIC_API_KEY": "call-key",
        }
    )

    assert env["ANTHROPIC_BASE_URL"] == "https://call.example"
    assert env["ANTHROPIC_API_KEY"] == "call-key"


def test_scimas_mcp_no_proxy_injects_bypass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCIMAS_MCP_NO_PROXY writes no_proxy/NO_PROXY into every server entry."""
    src = tmp_path / "mcp.json"
    _write_config(src, {
        "mcpServers": {
            "bare": {"command": "python", "args": ["srv.py"]},
            "with_env": {"command": "python", "args": ["srv.py"], "env": {"FOO": "bar"}},
        }
    })
    monkeypatch.setenv("SCIMAS_MCP_NO_PROXY", "*")
    runner = ClaudeRunner(mcp_config_path=str(src))
    resolved = json.loads(Path(runner.mcp_config_path).read_text())
    for name in ("bare", "with_env"):
        env = resolved["mcpServers"][name]["env"]
        assert env["no_proxy"] == "*"
        assert env["NO_PROXY"] == "*"
    # A pre-existing env block is preserved, not clobbered.
    assert resolved["mcpServers"]["with_env"]["env"]["FOO"] == "bar"


def test_scimas_mcp_no_proxy_unset_leaves_config_untouched(tmp_path: Path) -> None:
    """Unset means "inherit the proxy environment as-is" — no injection."""
    src = tmp_path / "mcp.json"
    _write_config(src, {
        "mcpServers": {
            "ok": {"command": "/already/absolute", "args": ["srv.py"]},
        }
    })
    os.environ.pop("SCIMAS_MCP_NO_PROXY", None)
    runner = ClaudeRunner(mcp_config_path=str(src))
    # Nothing rewritten → original path handed back, no .resolved.json.
    assert runner.mcp_config_path == str(src)


def test_real_shipped_mcp_json_resolves(tmp_path: Path) -> None:
    """Round-trip the shipped config/mcp.json and verify it survives parsing."""
    import os
    src = ROOT / "config" / "mcp.json"
    if not src.is_file():
        return  # skip if vendored without the config dir
    # Copy the shipped file into tmp so we don't pollute the repo with .resolved.json.
    scratch = tmp_path / "mcp.json"
    scratch.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    runner = ClaudeRunner(mcp_config_path=str(scratch))
    resolved = json.loads(Path(runner.mcp_config_path).read_text())
    # Every server must end up with an absolute `command`.
    for name, cfg in resolved["mcpServers"].items():
        cmd = cfg.get("command")
        assert cmd and os.path.isabs(cmd), f"{name}: bad command {cmd!r}"
    # No env.PATH overrides survived.
    for name, cfg in resolved["mcpServers"].items():
        env = cfg.get("env") or {}
        assert "PATH" not in env, f"{name}: env.PATH survived"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
