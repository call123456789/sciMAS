"""Claude Code state must not be shared between sciMAS runs.

The CLI keys all of its durable state off one directory
(``$CLAUDE_CONFIG_DIR``, default ``~/.claude``): session transcripts and the
auto-memory store. Every sciMAS run on a machine uses cwd = the repo root, so
they all land in the same bucket as each other *and* as the operator's
interactive sessions. Auto-memory is injected into the system prompt of later
runs, which turns "a run once found the ground-truth file" into "every later
run starts knowing where it is" — the leak these tests pin shut.

What must survive the isolation is the other half of the contract: the run
still authenticates (credentials live in ``settings.json``, which the isolated
directory would not otherwise have) and MCP tools still work.
"""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest

import claude_runner
from claude_runner import ClaudeRunner

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch):
    """Reset the module's process-wide state around every test."""
    monkeypatch.setattr(claude_runner, "_state_seeded", set())
    monkeypatch.setattr(claude_runner, "_state_temp_root", None)
    monkeypatch.setattr(claude_runner, "_state_scope", "default")
    monkeypatch.setattr(claude_runner, "_state_root", None)
    monkeypatch.setattr(claude_runner, "_state_isolated", True)
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("SCIMAS_CLAUDE_STATE", raising=False)
    monkeypatch.delenv("SCIMAS_CLAUDE_STATE_DIR", raising=False)


def _fake_operator_config(tmp_path: Path) -> Path:
    """A stand-in for the deployment's ``~/.claude``."""
    config = tmp_path / "operator-claude"
    config.mkdir()
    (config / "settings.json").write_text(json.dumps({
        "env": {
            "ANTHROPIC_BASE_URL": "https://provider.example",
            "ANTHROPIC_AUTH_TOKEN": "operator-token",
            "ANTHROPIC_MODEL": "provider-model",
        },
        "theme": "dark",
    }))
    (config / ".credentials.json").write_text('{"oauth": "operator"}')
    # Conversation state, which must never be carried over.
    memory = config / "projects" / "-repo-slug" / "memory"
    memory.mkdir(parents=True)
    (memory / "MEMORY.md").write_text("- [answers](../../dataset/gold.json)\n")
    (config / "projects" / "-repo-slug" / "session.jsonl").write_text("{}\n")
    return config


def test_state_isolation_is_on_by_default():
    assert claude_runner.get_claude_state_isolated() is True


def test_subprocess_env_points_at_a_private_state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(_fake_operator_config(tmp_path)))
    env = ClaudeRunner(mcp_config_path="")._subprocess_env()

    state_dir = Path(env["CLAUDE_CONFIG_DIR"])
    assert state_dir.is_dir()
    # Never the operator's store, and never inside the repo (whatever lands in
    # the repo is readable by the next agent, which is the problem itself).
    assert state_dir != tmp_path / "operator-claude"
    assert REPO_ROOT not in state_dir.parents
    assert env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"


def test_state_dir_is_stable_across_a_runs_invocations(tmp_path, monkeypatch):
    """``--resume <session_id>`` re-reads a transcript from this directory.

    Every agent in a run — planner, workers, synthesizer — has to land in the
    same state directory, because they resume each other's sessions.
    """
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(_fake_operator_config(tmp_path)))
    runner = ClaudeRunner(mcp_config_path="")
    claude_runner.start_run_state("job-1")
    dirs = {runner._subprocess_env()["CLAUDE_CONFIG_DIR"] for _ in range(3)}
    assert len(dirs) == 1
    # Only an explicit rotation moves it, and that is the next run's business.
    claude_runner.start_run_state("job-2")
    assert runner._subprocess_env()["CLAUDE_CONFIG_DIR"] not in dirs


def test_two_runs_in_one_process_do_not_share_state(tmp_path, monkeypatch):
    """The dashboard runs jobs back to back, so scope is the boundary."""
    operator = _fake_operator_config(tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(operator))
    runner = ClaudeRunner(mcp_config_path="")

    claude_runner.start_run_state("job-1")
    first = Path(runner._subprocess_env()["CLAUDE_CONFIG_DIR"])
    claude_runner.start_run_state("job-2")
    second = Path(runner._subprocess_env()["CLAUDE_CONFIG_DIR"])

    assert first != second
    assert first.parent == second.parent
    # Both are usable state directories, not just distinct paths.
    for state_dir in (first, second):
        assert state_dir.is_dir()
        assert (state_dir / "settings.json").is_file()


def test_scope_names_are_filesystem_safe(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(_fake_operator_config(tmp_path)))
    claude_runner.start_run_state("DrugQA/2026-09-21 19:04:11")
    state_dir = Path(claude_runner.claude_state_dir())
    assert state_dir.is_dir()
    assert "/" not in state_dir.name and " " not in state_dir.name


def test_credentials_are_carried_over_and_conversation_state_is_not(tmp_path, monkeypatch):
    operator = _fake_operator_config(tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(operator))
    state_dir = Path(ClaudeRunner(mcp_config_path="")._subprocess_env()["CLAUDE_CONFIG_DIR"])

    seeded = json.loads((state_dir / "settings.json").read_text())
    assert seeded["env"]["ANTHROPIC_AUTH_TOKEN"] == "operator-token"
    assert seeded["env"]["ANTHROPIC_BASE_URL"] == "https://provider.example"
    # Only `env`: the rest of the file describes an interactive setup whose
    # files stay behind in the operator's directory.
    assert "theme" not in seeded
    assert stat.S_IMODE((state_dir / "settings.json").stat().st_mode) == 0o600

    # A subscription login is a credential too, and lives outside settings.
    assert json.loads((state_dir / ".credentials.json").read_text()) == {"oauth": "operator"}

    # The leak itself: memory index and transcripts must not follow.
    assert not (state_dir / "projects").exists()
    assert not list(state_dir.rglob("MEMORY.md"))
    assert not list(state_dir.rglob("*.jsonl"))


def test_seeding_survives_a_missing_or_broken_operator_config(tmp_path, monkeypatch):
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "settings.json").write_text("{not json")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(broken))
    state_dir = Path(ClaudeRunner(mcp_config_path="")._subprocess_env()["CLAUDE_CONFIG_DIR"])
    # A new directory with no credentials means "the provider's own auth",
    # which is a legitimate deployment; it must not raise.
    assert state_dir.is_dir()
    assert not (state_dir / "settings.json").exists()

    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "does-not-exist"))
    assert Path(ClaudeRunner(mcp_config_path="")._subprocess_env()["CLAUDE_CONFIG_DIR"]).is_dir()


def test_isolation_can_be_turned_off_for_debugging(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    claude_runner.set_claude_state_isolated(False)
    env = ClaudeRunner(mcp_config_path="")._subprocess_env()
    assert "CLAUDE_CONFIG_DIR" not in env
    assert "CLAUDE_CODE_DISABLE_AUTO_MEMORY" not in env


def test_a_pinned_state_root_is_honoured(tmp_path, monkeypatch):
    """A pinned root is a parent for runs, and it outlives the process."""
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(_fake_operator_config(tmp_path)))
    pinned = tmp_path / "somewhere-else"
    claude_runner.set_claude_state_root(str(pinned))
    claude_runner.start_run_state("job-1")
    env = ClaudeRunner(mcp_config_path="")._subprocess_env()
    state_dir = Path(env["CLAUDE_CONFIG_DIR"])
    assert state_dir.parent == pinned
    assert state_dir.is_dir()
    # Kept on purpose: the operator asked for it to stay for inspection.
    assert claude_runner.get_claude_state_root() == str(pinned)


def test_caller_env_overrides_win_over_isolation(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(_fake_operator_config(tmp_path)))
    runner = ClaudeRunner(mcp_config_path="")
    env = runner._subprocess_env({
        "CLAUDE_CONFIG_DIR": "/explicit/from/caller",
        "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "0",
    })
    assert env["CLAUDE_CONFIG_DIR"] == "/explicit/from/caller"
    assert env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "0"


def test_isolation_does_not_change_the_argv_or_mcp_wiring(tmp_path, monkeypatch):
    """The user-visible worry: MCP tool calls must keep working."""
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(_fake_operator_config(tmp_path)))
    config = tmp_path / "mcp.json"
    config.write_text(json.dumps({"mcpServers": {"demo": {"command": "python", "args": ["s.py"]}}}))
    runner = ClaudeRunner(str(tmp_path / "claude"), mcp_config_path=str(config))
    cmd = runner._build_cmd("prompt", None, "json", runner.allowed_tools)

    assert any(arg.startswith("--mcp-config=") for arg in cmd)
    assert "--allowedTools=*" in cmd
    assert "--dangerously-skip-permissions" in cmd
    # Isolation is an environment change only: it must not add or drop flags.
    assert "CLAUDE_CONFIG_DIR" not in " ".join(cmd)


@pytest.mark.parametrize("output_format", ["json", "stream-json"])
def test_both_spawn_paths_hand_the_child_the_isolated_dir(
    tmp_path, monkeypatch, output_format
):
    """json uses subprocess.run, stream-json uses Popen; both must isolate."""
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(_fake_operator_config(tmp_path)))
    binary = tmp_path / "fake-claude"
    binary.write_text(
        f"#!{sys.executable}\n"
        "import json, os\n"
        "print(json.dumps({'type': 'result', 'result': json.dumps({\n"
        "    'config_dir': os.environ.get('CLAUDE_CONFIG_DIR'),\n"
        "    'auto_memory': os.environ.get('CLAUDE_CODE_DISABLE_AUTO_MEMORY'),\n"
        "})}))\n"
    )
    binary.chmod(0o700)
    runner = ClaudeRunner(binary=str(binary), mcp_config_path="", timeout=10)
    result = runner.run(prompt="test", output_format=output_format)
    payload = json.loads(result.result)

    assert payload["auto_memory"] == "1"
    assert Path(payload["config_dir"]).is_dir()
    assert payload["config_dir"] != str(tmp_path / "operator-claude")
