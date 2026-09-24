"""Tests for per-step MCP config narrowing in `ClaudeRunner`.

`claude -p` builds a session's tool registry from the MCP servers already
connected when it emits `init`, and servers still `pending` contribute no
tools. Because it starts servers in config order with bounded concurrency, a
step whose skill routed to a server late in config/mcp.json got "No such tool
available" for tools the server exposed moments later. Narrowing the config to
the servers a step's allowlist actually names removes that race.

Run with:
    cd "$(git rev-parse --show-toplevel)"
    python -m pytest tests/test_mcp_config_narrowing.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from claude_runner import ClaudeRunner  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_ambient_no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep SCIMAS_MCP_NO_PROXY from rewriting configs under the tests."""
    monkeypatch.delenv("SCIMAS_MCP_NO_PROXY", raising=False)


# The interesting shape: the server a step needs sits *after* servers it does
# not, which is exactly what made the real runs lose the race.
SERVERS = {
    "litsearch": {"command": sys.executable, "args": ["tools/web/litsearch_server.py"]},
    "chemistry-analytical": {"command": sys.executable, "args": ["a.py"]},
    "chemistry-computational": {"command": sys.executable, "args": ["c.py"]},
    "chemistry-physical": {"command": sys.executable, "args": ["p.py"]},
    "pharma-drug-sda": {"command": sys.executable, "args": ["d.py"]},
}


@pytest.fixture()
def runner(tmp_path: Path) -> ClaudeRunner:
    src = tmp_path / "mcp.json"
    src.write_text(json.dumps({"mcpServers": SERVERS}, indent=2) + "\n", encoding="utf-8")
    return ClaudeRunner(mcp_config_path=str(src))


# --------------------------------------------------------------------------
# _servers_for_tools — deriving the server set from an allowlist
# --------------------------------------------------------------------------

def test_reads_servers_out_of_qualified_tool_names() -> None:
    servers = ClaudeRunner._servers_for_tools([
        "mcp__chemistry-physical__ideal_gas",
        "mcp__chemistry-physical__ideal_gas_calculation",
        "mcp__pharma-drug-sda__calculate_mol_basic_info",
    ])
    assert servers == {"chemistry-physical", "pharma-drug-sda"}


def test_ignores_builtin_tools_including_scoped_patterns() -> None:
    """`Bash(git:*)` cannot match an mcp__ tool, so it must not defeat narrowing."""
    servers = ClaudeRunner._servers_for_tools([
        "Read",
        "Bash(git:*)",
        "mcp__chemistry-physical__ideal_gas",
    ])
    assert servers == {"chemistry-physical"}


def test_bare_wildcard_is_underivable() -> None:
    """`"*"` is the default when the caller passes no allowlist — keep it whole."""
    assert ClaudeRunner._servers_for_tools(["*"]) is None


def test_wildcard_in_the_server_position_is_underivable() -> None:
    assert ClaudeRunner._servers_for_tools(["mcp__*"]) is None
    assert ClaudeRunner._servers_for_tools(["mcp__*__ideal_gas"]) is None


def test_allowlist_without_mcp_entries_yields_the_empty_set() -> None:
    assert ClaudeRunner._servers_for_tools(["Read", "Bash"]) == set()
    assert ClaudeRunner._servers_for_tools([]) == set()


# --------------------------------------------------------------------------
# _mcp_config_for — which config path a session should be handed
# --------------------------------------------------------------------------

def test_narrows_to_the_named_server_only(runner: ClaudeRunner) -> None:
    path = runner._mcp_config_for(["mcp__chemistry-physical__ideal_gas"])
    assert path is not None
    assert path != runner.mcp_config_path
    narrowed = json.loads(Path(path).read_text())
    assert list(narrowed["mcpServers"]) == ["chemistry-physical"]


def test_narrowed_config_preserves_source_order(runner: ClaudeRunner) -> None:
    """The CLI starts servers in file order, so relative order must survive."""
    path = runner._mcp_config_for([
        "mcp__chemistry-physical__ideal_gas",
        "mcp__chemistry-analytical__something",
    ])
    narrowed = json.loads(Path(path).read_text())
    assert list(narrowed["mcpServers"]) == ["chemistry-analytical", "chemistry-physical"]


def test_narrowed_config_keeps_server_definitions_intact(runner: ClaudeRunner) -> None:
    path = runner._mcp_config_for(["mcp__pharma-drug-sda__x"])
    narrowed = json.loads(Path(path).read_text())
    assert narrowed["mcpServers"]["pharma-drug-sda"] == SERVERS["pharma-drug-sda"]


def test_wildcard_allowlist_keeps_the_full_config(runner: ClaudeRunner) -> None:
    """Regression guard: the no-allowlist default must behave exactly as before."""
    assert runner._mcp_config_for(["*"]) == runner.mcp_config_path


def test_no_mcp_tools_means_no_config_at_all(runner: ClaudeRunner) -> None:
    """A session that cannot call an MCP tool should not spawn any server."""
    assert runner._mcp_config_for(["Read", "Bash"]) is None
    assert runner._mcp_config_for([]) is None


def test_missing_server_is_reported_but_the_rest_still_narrow(
    runner: ClaudeRunner, capsys: pytest.CaptureFixture
) -> None:
    path = runner._mcp_config_for([
        "mcp__chemistry-physical__ideal_gas",
        "mcp__does-not-exist__tool",
    ])
    assert path is not None
    narrowed = json.loads(Path(path).read_text())
    assert list(narrowed["mcpServers"]) == ["chemistry-physical"]
    assert "does-not-exist" in capsys.readouterr().err


def test_allowlist_naming_only_unknown_servers_falls_back(runner: ClaudeRunner) -> None:
    """Never hand the CLI a config with zero servers when tools were requested."""
    assert runner._mcp_config_for(["mcp__nope__tool"]) == runner.mcp_config_path


def test_narrowing_is_cached_per_server_set(runner: ClaudeRunner) -> None:
    first = runner._mcp_config_for(["mcp__chemistry-physical__a"])
    second = runner._mcp_config_for(["mcp__chemistry-physical__b"])
    assert first == second


def test_no_mcp_config_configured_is_a_no_op() -> None:
    runner = ClaudeRunner(mcp_config_path="")
    assert runner.mcp_config_path is None
    assert runner._mcp_config_for(["mcp__chemistry-physical__ideal_gas"]) is None


# --------------------------------------------------------------------------
# _build_cmd — the flag actually handed to the CLI
# --------------------------------------------------------------------------

def _mcp_flag(cmd: list[str]) -> str | None:
    for arg in cmd:
        if arg.startswith("--mcp-config="):
            return arg.split("=", 1)[1]
    return None


def test_build_cmd_points_at_the_narrowed_config(runner: ClaudeRunner) -> None:
    cmd = runner._build_cmd(
        "probe", None, "stream-json", ["mcp__chemistry-physical__ideal_gas"]
    )
    flag = _mcp_flag(cmd)
    assert flag is not None
    narrowed = json.loads(Path(flag).read_text())
    assert list(narrowed["mcpServers"]) == ["chemistry-physical"]


def test_build_cmd_omits_the_flag_when_no_tool_needs_a_server(runner: ClaudeRunner) -> None:
    cmd = runner._build_cmd("probe", None, "stream-json", ["Read"])
    assert _mcp_flag(cmd) is None


def test_build_cmd_uses_the_full_config_for_a_wildcard(runner: ClaudeRunner) -> None:
    cmd = runner._build_cmd("probe", None, "stream-json", ["*"])
    assert _mcp_flag(cmd) == runner.mcp_config_path
