"""The filesystem boundary: what it builds, and what it actually does.

Two layers, because bubblewrap is Linux-only and this suite also runs on a Mac.
The argv tests stub ``bubblewrap()`` and assert on the mount list; the class at
the end runs a real ``bwrap`` and is skipped where there is none. The second
layer is the one that proves anything — the first only catches a regression in
the recipe.

Note where the real-sandbox layout lives: *not* under ``/tmp``. The recipe
gives ``/tmp`` a private tmpfs, so a fixture built there would be masked by the
very mount it is testing, and ``--chdir`` into it would fail outright. The
fixture therefore mirrors production and builds its tree in the home
directory, which the recipe leaves readable.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent_sandbox
import claude_runner
import sandbox_policy
from claude_runner import ClaudeRunner

BWRAP = shutil.which("bwrap")


@pytest.fixture
def fake_bwrap(monkeypatch):
    """Pretend bubblewrap is installed, so argv is built on any platform."""
    monkeypatch.setattr(agent_sandbox, "bubblewrap", lambda: "/usr/bin/bwrap")
    monkeypatch.delenv(agent_sandbox.BWRAP_ENV, raising=False)
    monkeypatch.setattr(agent_sandbox, "_warned_unavailable", False)
    return "/usr/bin/bwrap"


def _make_layout(root: Path) -> Path:
    """A miniature of the real tree: inputs, answers, past and current runs."""
    (root / "dataset").mkdir(parents=True)
    (root / "dataset" / "questions.json").write_text("question", encoding="utf-8")
    (root / "dataset" / "gold.json").write_text("the answer", encoding="utf-8")
    (root / "runs").mkdir()
    past = root / "tests" / "results" / "past-run"
    past.mkdir(parents=True)
    (past / "per-problem.json").write_text("gold: 42", encoding="utf-8")
    (past / "notes.txt").write_text("the answer is 42", encoding="utf-8")
    current = root / "tests" / "results" / "current-run"
    current.mkdir()
    (current / "out.txt").write_text("mine", encoding="utf-8")
    return root


@pytest.fixture
def layout(tmp_path):
    return _make_layout(tmp_path / "repo")


def _bind(path) -> tuple[str, ...]:
    """A writable bind: source and destination are the same path."""
    return ("--bind", str(path), str(path))


def _mounts(prefix: list[str]) -> list[tuple[str, ...]]:
    """The bwrap mount options, in order, up to where the CLI argv starts."""
    mounts: list[tuple[str, ...]] = []
    index = 1
    while index < len(prefix):
        flag = prefix[index]
        if flag in {"--ro-bind", "--bind"}:
            # Source and destination: both have to be consumed, or the parse
            # shifts by one and stops at the first bind.
            mounts.append((flag, prefix[index + 1], prefix[index + 2]))
            index += 3
        elif flag in {"--tmpfs", "--dev", "--proc", "--chdir"}:
            mounts.append((flag, prefix[index + 1]))
            index += 2
        elif flag in {"--unshare-pid", "--die-with-parent"}:
            mounts.append((flag,))
            index += 1
        else:
            break  # the wrapper ends here; the CLI argv follows
    return mounts


# --------------------------------------------------------------------------
# The recipe
# --------------------------------------------------------------------------


def test_masked_tree_is_re_exposed_after_the_mask(fake_bwrap, layout):
    """The run's own output dir lives inside the masked tree and must survive."""
    prefix = agent_sandbox.build_prefix(
        hide=[layout / "tests" / "results"],
        writable=[layout / "tests" / "results" / "current-run"],
    )
    mounts = _mounts(prefix)
    masked = mounts.index(("--tmpfs", str(layout / "tests" / "results")))
    rebind = mounts.index(_bind(layout / "tests" / "results" / "current-run"))
    # Later mounts win, so the writable bind has to come after the tmpfs.
    assert masked < rebind


def test_mask_source_is_bound_before_it_is_used(fake_bwrap, layout, tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    mask = state / "mask-empty"
    mask.touch()
    prefix = agent_sandbox.build_prefix(
        hide=[layout / "dataset" / "gold.json"],
        writable=[state],
        mask_source=str(mask),
    )
    mounts = _mounts(prefix)
    bind_state = mounts.index(_bind(state))
    use_mask = mounts.index(
        ("--ro-bind", str(mask), str(layout / "dataset" / "gold.json"))
    )
    # The replacement file has to be visible when the mask is applied.
    assert bind_state < use_mask


def test_files_are_masked_with_an_empty_file_dirs_with_a_tmpfs(fake_bwrap, layout):
    empty = layout / "empty"
    empty.touch()
    prefix = agent_sandbox.build_prefix(
        hide=[
            layout / "dataset" / "gold.json",
            layout / "tests" / "results" / "past-run",
        ],
        writable=[],
        mask_source=str(empty),
    )
    mounts = _mounts(prefix)
    assert ("--ro-bind", str(empty), str(layout / "dataset" / "gold.json")) in mounts
    assert ("--tmpfs", str(layout / "tests" / "results" / "past-run")) in mounts
    # /dev/null reads as EACCES, which advertises the boundary; use it only
    # when no empty file is available.
    assert not any("/dev/null" in option for mount in mounts for option in mount)


def test_dev_null_is_only_a_fallback(fake_bwrap, layout):
    prefix = agent_sandbox.build_prefix(hide=[layout / "dataset" / "gold.json"])
    assert (
        "--ro-bind",
        "/dev/null",
        str(layout / "dataset" / "gold.json"),
    ) in _mounts(prefix)


def test_network_namespace_is_never_unshared(fake_bwrap, layout):
    """MCP servers that reach the network must keep working."""
    assert "--unshare-net" not in agent_sandbox.build_prefix(hide=[layout / "dataset"])


def test_tmp_is_private_and_never_masked(fake_bwrap, layout):
    prefix = agent_sandbox.build_prefix(
        hide=["/tmp", str(layout / "runs")], writable=[]
    )
    mounts = _mounts(prefix)
    # Hiding /tmp outright would also hide every mount made after it.
    assert mounts.count(("--tmpfs", "/tmp")) == 1


def test_writable_paths_are_created_when_missing(fake_bwrap, layout):
    missing = layout / "tests" / "results" / "fresh-run"
    assert not missing.exists()
    assert agent_sandbox.build_prefix(hide=[], writable=[missing]) is not None
    assert missing.is_dir()


def test_a_path_in_both_sets_is_re_exposed_inside_the_mask(fake_bwrap, layout):
    """The run's own directory is both hidden (its parent) and writable."""
    target = layout / "tests" / "results" / "current-run"
    tree = layout / "tests" / "results"
    mounts = _mounts(agent_sandbox.build_prefix(hide=[tree, target], writable=[target]))
    assert ("--tmpfs", str(tree)) in mounts
    assert ("--tmpfs", str(target)) not in mounts
    assert mounts.index(("--tmpfs", str(tree))) < mounts.index(_bind(target))


def test_masked_paths_that_do_not_exist_are_skipped(fake_bwrap, layout):
    """A dataset that is not downloaded contributes no mounts, no error."""
    prefix = agent_sandbox.build_prefix(
        hide=[layout / "dataset" / "absent.json", layout / "absent-dir"]
    )
    assert prefix is not None
    mounts = _mounts(prefix)
    assert not any("absent" in option for mount in mounts for option in mount)


# --------------------------------------------------------------------------
# Mode switch
# --------------------------------------------------------------------------


def test_mode_off_builds_nothing(fake_bwrap, layout, monkeypatch):
    monkeypatch.setenv(agent_sandbox.BWRAP_ENV, "off")
    assert agent_sandbox.build_prefix(hide=[layout / "dataset"]) is None


def test_mode_require_is_fatal_without_bubblewrap(layout, monkeypatch):
    monkeypatch.setattr(agent_sandbox, "bubblewrap", lambda: None)
    monkeypatch.setenv(agent_sandbox.BWRAP_ENV, "require")
    with pytest.raises(RuntimeError):
        agent_sandbox.build_prefix(hide=[layout / "dataset"])


def test_mode_auto_warns_once_and_returns_none(layout, monkeypatch, capsys):
    monkeypatch.setattr(agent_sandbox, "bubblewrap", lambda: None)
    monkeypatch.delenv(agent_sandbox.BWRAP_ENV, raising=False)
    monkeypatch.setattr(agent_sandbox, "_warned_unavailable", False)
    assert agent_sandbox.build_prefix(hide=[layout / "dataset"]) is None
    assert "bubblewrap is not available" in capsys.readouterr().err
    assert agent_sandbox.build_prefix(hide=[layout / "dataset"]) is None
    assert capsys.readouterr().err == ""


# --------------------------------------------------------------------------
# Wiring: the boundary reaches the child argv
# --------------------------------------------------------------------------


def _sandboxed_argv(tmp_path: Path, **kwargs) -> list[str]:
    """The argv a runner would spawn, with a stub CLI binary."""
    binary = tmp_path / "claude"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)
    runner = ClaudeRunner(binary=str(binary), mcp_config_path="", **kwargs)
    cmd = runner._build_cmd("hello", None, "json", [], None, None)
    return runner._sandboxed(cmd)


def test_runner_without_a_boundary_spawns_the_cli_directly(tmp_path):
    argv = _sandboxed_argv(tmp_path)
    assert argv[0].endswith("claude")
    assert "bwrap" not in " ".join(argv)


def test_runner_puts_the_boundary_in_front_of_the_cli(fake_bwrap, tmp_path, layout):
    argv = _sandboxed_argv(
        tmp_path,
        sandbox_hide=[str(layout / "dataset" / "gold.json")],
        sandbox_writable=[str(layout / "runs")],
    )
    assert argv[0] == "/usr/bin/bwrap"
    assert argv[1:4] == ["--ro-bind", "/", "/"]
    # The CLI argv is preserved, verbatim, after the wrapper.
    assert any(arg.endswith("/claude") for arg in argv)
    assert argv[-1] == "hello"


def test_settings_file_is_written_somewhere_the_sandbox_can_see(fake_bwrap, tmp_path, layout):
    """The private /tmp would swallow it, and the run would use the wrong key."""
    binary = tmp_path / "claude"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    runner = ClaudeRunner(
        binary=str(binary),
        mcp_config_path="",
        sandbox_hide=[str(layout / "dataset" / "gold.json")],
        sandbox_writable=[str(layout / "runs")],
        state_scope="job-1",
    )
    argv = runner._sandboxed(runner._build_cmd("hello", None, "json", [], None, None))
    bound = [
        mount[1] for mount in _mounts(argv) if mount[0] in ("--bind", "--ro-bind")
    ]

    scratch = runner._scratch_dir()
    assert scratch is not None
    # Bound back in past the tmpfs, or the CLI cannot open what it is told to.
    assert scratch in bound

    with runner._invocation_settings({"ANTHROPIC_BASE_URL": "http://example"}) as path:
        assert path is not None and Path(path).is_file()
        assert str(Path(path).parent).startswith(scratch)


def test_a_runner_that_is_not_sandboxed_uses_the_system_temp(tmp_path, layout):
    runner = ClaudeRunner(
        binary=str(tmp_path / "claude"),
        mcp_config_path="",
    )
    assert runner._scratch_dir() is None


def test_mcp_and_permission_flags_survive_the_prefix(fake_bwrap, tmp_path, layout):
    """The whole point of wrapping argv: the MCP wiring goes through untouched."""
    mcp_config = tmp_path / "mcp.json"
    mcp_config.write_text('{"mcpServers": {}}', encoding="utf-8")
    binary = tmp_path / "claude"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)
    runner = ClaudeRunner(
        binary=str(binary),
        mcp_config_path=str(mcp_config),
        sandbox_hide=[str(layout / "dataset")],
        sandbox_writable=[str(layout / "runs")],
    )
    argv = runner._sandboxed(
        runner._build_cmd("hello", None, "json", runner.allowed_tools, runner.mcp_config_path, None)
    )
    assert argv[0] == "/usr/bin/bwrap"
    assert f"--mcp-config={runner.mcp_config_path}" in argv
    assert "--allowedTools=*" in argv
    assert "--dangerously-skip-permissions" in argv
    # And the CLI itself is still the binary that runs, after the wrapper.
    assert argv.index(str(binary)) > 0


def test_runner_sends_large_prompt_via_stdin_not_bwrap_argv(
    fake_bwrap, tmp_path, layout, monkeypatch
):
    """Large contexts must not count against argv, while MCP flags survive."""
    binary = tmp_path / "claude"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)
    mcp_config = tmp_path / "mcp.json"
    mcp_config.write_text('{"mcpServers": {"demo": {"command": "true"}}}', encoding="utf-8")
    runner = ClaudeRunner(
        binary=str(binary),
        mcp_config_path=str(mcp_config),
        sandbox_hide=[str(layout / "dataset" / "gold.json")],
        sandbox_writable=[str(layout / "runs")],
        timeout=10,
    )
    long_prompt = "review these prior outputs:\n" + ("x" * 250_000)
    captured: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["input"] = kwargs.get("input")
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps({"type": "result", "result": "ok"}),
            stderr="",
        )

    monkeypatch.setattr("claude_runner.subprocess.run", fake_run)

    result = runner.run(prompt=long_prompt, output_format="json")

    cmd = captured["cmd"]
    assert result.result == "ok"
    assert isinstance(cmd, list) and cmd[0] == "/usr/bin/bwrap"
    assert long_prompt not in cmd
    assert captured["input"] == long_prompt
    assert f"--mcp-config={runner.mcp_config_path}" in cmd
    assert "--allowedTools=*" in cmd
    assert str(binary) in cmd
    assert cmd[-1] != long_prompt


def test_state_dir_is_bound_back_in_after_the_private_tmpfs(
    fake_bwrap, tmp_path, layout, monkeypatch
):
    monkeypatch.setattr(claude_runner, "_state_root", str(tmp_path / "state-root"))
    monkeypatch.setattr(claude_runner, "_state_seeded", set())
    argv = _sandboxed_argv(
        tmp_path,
        state_scope="job-1",
        sandbox_hide=[str(layout / "dataset")],
        sandbox_writable=[str(layout / "runs")],
    )
    mounts = _mounts(argv)
    state_dir = tmp_path / "state-root" / "job-1"
    assert _bind(state_dir) in mounts
    assert mounts.index(("--tmpfs", "/tmp")) < mounts.index(_bind(state_dir))
    # Bound back in because it holds the carried-over credentials.
    assert state_dir.is_dir()


def test_two_runners_keep_their_own_state_scope(tmp_path, monkeypatch):
    """Concurrent jobs must not swap state directories."""
    monkeypatch.setattr(claude_runner, "_state_root", str(tmp_path / "state-root"))
    monkeypatch.setattr(claude_runner, "_state_seeded", set())
    first = ClaudeRunner(binary="claude", mcp_config_path="", state_scope="job-a")
    second = ClaudeRunner(binary="claude", mcp_config_path="", state_scope="job-b")
    assert first._state_dir() != second._state_dir()
    assert first._state_dir().endswith("job-a")
    assert second._state_dir().endswith("job-b")


def test_isolation_off_leaves_the_state_dir_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(claude_runner, "_state_root", str(tmp_path / "state-root"))
    monkeypatch.setattr(claude_runner, "_state_seeded", set())
    monkeypatch.setattr(claude_runner, "_state_isolated", False)
    runner = ClaudeRunner(binary="claude", mcp_config_path="", state_scope="job-a")
    assert runner._state_dir() is None
    assert "CLAUDE_CONFIG_DIR" not in runner._subprocess_env()


# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------


def _slug(path: Path) -> str:
    """The same rule claude_runner's state bucket uses."""
    return re.sub(r"[^A-Za-z0-9]", "-", str(path))


def _fake_repo(tmp_path: Path) -> Path:
    """A repo whose dataset layout mirrors the real one, answers and all."""
    root = tmp_path / "repo"
    (root / "dataset" / "gym" / "test_images").mkdir(parents=True)
    (root / "dataset" / "gym" / "test_images" / "q1.png").write_text("fig")
    (root / "dataset" / "refine_merged_multi_questions.json").write_text("{}")
    (root / "dataset" / "refine_merged_single_questions.json").write_text("{}")
    task = root / "dataset" / "ResearchClawBench" / "repo" / "tasks" / "Chemistry_001"
    (task / "target_study" / "images").mkdir(parents=True)
    (task / "target_study" / "checklist.json").write_text("[]")
    (task / "target_study" / "paper.pdf").write_text("paper")
    (task / "target_study" / "images" / "fig1.png").write_text("fig")
    (task / "data").mkdir()
    (task / "data" / "measurement.dat").write_text("1 2 3")
    hf = root / "dataset" / "ResearchClawBench" / "hf_dataset"
    hf.mkdir()
    (hf / "train.arrow").write_text("x")
    # Deployed as CSV rather than parquet, which the loader also accepts.
    biomni = root / "dataset" / "BiomniEval1" / "data"
    biomni.mkdir(parents=True)
    (biomni / "eval1.csv").write_text("prompt,answer\n")
    (root / "tests" / "results" / "old-run").mkdir(parents=True)
    (root / "runs-dashboard").mkdir()
    (root / "runs").mkdir()
    return root


def test_policy_masks_answers_and_keeps_the_agents_inputs(tmp_path):
    root = _fake_repo(tmp_path)
    hidden = set(sandbox_policy.plan(root, mask_shared_state=False).hide)

    for answer_path in (
        "dataset/refine_merged_multi_questions.json",
        "dataset/refine_merged_single_questions.json",
        "dataset/ResearchClawBench/hf_dataset",
        "dataset/ResearchClawBench/repo/tasks/Chemistry_001/target_study/checklist.json",
        "dataset/ResearchClawBench/repo/tasks/Chemistry_001/target_study/paper.pdf",
        "tests/results",
        "runs-dashboard",
    ):
        assert str(root / answer_path) in hidden, answer_path

    # Everything the prompt names by absolute path has to stay readable: an
    # empty file would be a silent wrong answer rather than a visible error.
    for input_path in (
        "dataset/gym/test_images",
        "dataset/gym/test_images/q1.png",
        "dataset/ResearchClawBench/repo/tasks/Chemistry_001/data/measurement.dat",
        "dataset/ResearchClawBench/repo/tasks/Chemistry_001/target_study/images/fig1.png",
    ):
        assert str(root / input_path) not in hidden, input_path


def test_policy_reports_a_pattern_that_stopped_matching(tmp_path):
    root = _fake_repo(tmp_path)
    # The dataset is deployed (its probe is still there) but one of the two
    # question files is gone: that pattern now masks nothing.
    (root / "dataset" / "refine_merged_single_questions.json").unlink()
    boundary = sandbox_policy.plan(root, mask_shared_state=False)
    assert ("SciAgentGYM", "dataset/refine_merged_single_questions.json") in boundary.unmatched
    assert "WARNING" in boundary.render()


def test_policy_skips_a_dataset_that_is_not_deployed(tmp_path):
    boundary = sandbox_policy.plan(_fake_repo(tmp_path), mask_shared_state=False)
    skipped = {group for group, _probe in boundary.skipped}
    assert "MADD" in skipped
    assert "SMDDBench" in skipped
    assert "SciAgentGYM" not in skipped
    assert "not deployed" in boundary.render()


def test_policy_masks_the_whole_biomni_eval1_tree(tmp_path):
    """Which file the loader picks depends on how the dataset was downloaded."""
    root = _fake_repo(tmp_path)
    boundary = sandbox_policy.plan(root, mask_shared_state=False)
    # Deployed here as CSV, from which the loader reads ``answer`` — a boundary
    # that only covered *.parquet would leave it in the open.
    assert str(root / "dataset" / "BiomniEval1") in boundary.hide
    assert "BiomniEval1" not in {group for group, _probe in boundary.skipped}


def test_policy_calls_a_directory_without_its_data_not_deployed(tmp_path):
    """A leftover directory must not read as a dataset whose patterns broke."""
    root = _fake_repo(tmp_path)
    (root / "dataset" / "SMDDBench").mkdir()  # e.g. an interrupted download
    boundary = sandbox_policy.plan(root, mask_shared_state=False)
    assert "SMDDBench" in {group for group, _probe in boundary.skipped}
    assert "SMDDBench" not in {group for group, _pattern in boundary.unmatched}
    assert "WARNING" not in boundary.render()


def test_policy_output_dirs_stay_writable_inside_a_masked_tree(tmp_path):
    root = _fake_repo(tmp_path)
    run_dir = root / "tests" / "results" / "20260921-120000-fresh"
    boundary = sandbox_policy.plan(root, output_dirs=[run_dir], mask_shared_state=False)
    assert str(root / "tests" / "results") in boundary.hide
    assert str(run_dir) in boundary.writable
    assert str(root / "runs") in boundary.writable


def test_shared_state_is_masked_only_while_isolation_is_on(tmp_path, monkeypatch):
    root = _fake_repo(tmp_path)
    config = tmp_path / "claude-config"
    bucket = config / "projects" / _slug(root)
    bucket.mkdir(parents=True)
    (bucket / "MEMORY.md").write_text("the golden answers live in dataset/")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config))

    assert str(bucket) in sandbox_policy.plan(root).hide
    assert str(bucket) not in sandbox_policy.plan(root, mask_shared_state=False).hide


def test_shared_state_covers_the_working_directory_too(tmp_path, monkeypatch):
    """The CLI keys its bucket by cwd, which is not always the repo root."""
    config = tmp_path / "claude-config"
    bucket = config / "projects" / _slug(Path.cwd())
    bucket.mkdir(parents=True)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config))
    assert str(bucket) in sandbox_policy.shared_state_paths(tmp_path / "elsewhere")


def test_shared_state_paths_are_not_repeated(tmp_path, monkeypatch):
    """The two cwds are the same on a normal run; mask the bucket once."""
    config = tmp_path / "claude-config"
    (config / "projects" / _slug(Path.cwd())).mkdir(parents=True)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config))
    paths = sandbox_policy.shared_state_paths(Path.cwd())
    assert paths and len(paths) == len(set(paths))


def test_shared_state_paths_tolerate_a_missing_config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "nope"))
    assert sandbox_policy.shared_state_paths(tmp_path / "repo") == []


# --------------------------------------------------------------------------
# The real thing (Linux only)
# --------------------------------------------------------------------------


@pytest.mark.skipif(BWRAP is None, reason="bubblewrap is Linux-only")
class TestRealSandbox:
    """Run an actual bwrap and look at what the process can reach."""

    @pytest.fixture
    def home_scratch(self):
        """A tree outside /tmp, so the recipe's private tmpfs cannot hide it.

        Anything bound into the sandbox has to live in here, not under
        ``tmp_path``: bwrap creates the missing mount points for a bind, so a
        state directory under ``/tmp`` leaves its parent directories visible
        inside the private tmpfs — the boundary stays correct, but a test that
        asserts ``/tmp`` starts out empty would be measuring its own fixture.
        """
        base = Path(tempfile.mkdtemp(prefix="scimas-sbx-", dir=str(Path.home())))
        try:
            yield base
        finally:
            shutil.rmtree(base, ignore_errors=True)

    @pytest.fixture
    def layout_outside_tmp(self, home_scratch):
        return _make_layout(home_scratch / "repo")

    @pytest.fixture
    def sandboxed(self, layout_outside_tmp, home_scratch):
        state = home_scratch / "state"
        state.mkdir()
        mask = state / "mask-empty"
        mask.touch()
        layout = layout_outside_tmp
        prefix = agent_sandbox.build_prefix(
            hide=[
                layout / "dataset" / "gold.json",
                layout / "tests" / "results" / "past-run",
            ],
            writable=[
                layout / "runs",
                layout / "tests" / "results" / "current-run",
                state,
            ],
            mask_source=str(mask),
            cwd=str(layout),
        )
        assert prefix is not None
        return prefix

    @staticmethod
    def _run(prefix: list[str], script: str, cwd: Path) -> subprocess.CompletedProcess:
        return subprocess.run(
            [*prefix, "/bin/sh", "-c", script],
            capture_output=True,
            text=True,
            cwd=str(cwd),
            timeout=120,
            check=False,
        )

    def test_masked_file_reads_as_empty(self, sandboxed, layout_outside_tmp):
        proc = self._run(sandboxed, "cat dataset/gold.json; echo rc=$?", layout_outside_tmp)
        assert proc.stdout.strip() == "rc=0"
        # A refused read would say so, and tell the agent a boundary exists.
        assert "Permission denied" not in proc.stderr

    def test_masked_file_reports_size_zero(self, sandboxed, layout_outside_tmp):
        proc = self._run(sandboxed, "stat -c %s dataset/gold.json", layout_outside_tmp)
        assert proc.stdout.strip() == "0"

    def test_masked_directory_lists_empty(self, sandboxed, layout_outside_tmp):
        proc = self._run(sandboxed, "ls -A tests/results/past-run", layout_outside_tmp)
        assert proc.stdout.strip() == ""
        assert proc.returncode == 0

    def test_current_run_directory_is_readable_and_writable(
        self, sandboxed, layout_outside_tmp
    ):
        proc = self._run(
            sandboxed,
            "cat tests/results/current-run/out.txt; "
            "echo written > tests/results/current-run/new.txt; echo rc=$?",
            layout_outside_tmp,
        )
        assert "mine" in proc.stdout
        assert "rc=0" in proc.stdout
        # A real bind: the write lands on the host filesystem, so the run's
        # output survives the sandbox.
        written = layout_outside_tmp / "tests" / "results" / "current-run" / "new.txt"
        assert written.read_text().strip() == "written"

    def test_repo_is_read_only_except_the_declared_dirs(
        self, sandboxed, layout_outside_tmp
    ):
        proc = self._run(
            sandboxed,
            "echo x > dataset/questions.json; echo rc=$?; "
            "echo y > runs/ok.txt; echo rc2=$?",
            layout_outside_tmp,
        )
        assert "Read-only file system" in proc.stderr
        assert "rc2=0" in proc.stdout
        assert (layout_outside_tmp / "runs" / "ok.txt").exists()

    def test_inputs_stay_readable(self, sandboxed, layout_outside_tmp):
        proc = self._run(sandboxed, "cat dataset/questions.json", layout_outside_tmp)
        assert proc.stdout.strip() == "question"

    def test_past_run_contents_are_unreachable_by_any_path(
        self, sandboxed, layout_outside_tmp, home_scratch
    ):
        # Enumerating finds nothing, in contrast with the host, where the file
        # is right there — masking hides the contents, not just the one path
        # the prompt would have named. The search is scoped to this test's own
        # trees (a full ``find /`` walks /data and takes minutes here) plus
        # /tmp, which is where a stray copy from a previous run would land.
        proc = self._run(
            sandboxed,
            "cat tests/results/past-run/per-problem.json 2>&1; echo ---; "
            f"find {home_scratch} /tmp -name 'per-problem.json' 2>/dev/null; "
            "echo ---; "
            f"find {home_scratch} /tmp -name 'notes.txt' 2>/dev/null",
            layout_outside_tmp,
        )
        sections = proc.stdout.split("---")
        assert "No such file" in sections[0]
        # Not reachable by absolute path either — masking, not deleting from
        # the listing, is the point. The host-side check keeps this from
        # passing vacuously if the fixture ever stops creating the file.
        host = layout_outside_tmp / "tests" / "results" / "past-run"
        assert (host / "per-problem.json").is_file()
        assert (host / "notes.txt").is_file()
        assert "per-problem.json" not in sections[1]
        assert "notes.txt" not in sections[2]

    def test_tmp_is_private(self, sandboxed, layout_outside_tmp, tmp_path):
        marker = tmp_path / "host-tmp-marker"
        marker.write_text("host")
        proc = self._run(
            sandboxed, f"cat {marker} 2>&1; echo ---; ls -A /tmp", layout_outside_tmp
        )
        sections = proc.stdout.split("---")
        assert "No such file" in sections[0]
        assert sections[1].strip() == ""

    def test_network_namespace_is_shared(self, sandboxed, layout_outside_tmp):
        """No --unshare-net: the literature and web MCP servers keep working."""
        inside = self._run(sandboxed, "readlink /proc/self/ns/net", layout_outside_tmp)
        assert inside.stdout.strip() == os.readlink("/proc/self/ns/net")

    def test_no_bwrap_process_outlives_the_run(self, sandboxed, layout_outside_tmp):
        self._run(sandboxed, "true", layout_outside_tmp)
        listing = subprocess.run(
            ["ps", "-eo", "args"], capture_output=True, text=True, check=False
        )
        # Scoped to this test's tree so unrelated bwrap processes elsewhere on
        # a shared machine cannot make it flaky.
        assert str(layout_outside_tmp) not in listing.stdout

    def test_credentials_reach_the_sandboxed_cli(self, layout_outside_tmp, home_scratch):
        """The one bind that is not about hiding: without it, no auth at all."""
        state = home_scratch / "state"
        state.mkdir()
        (state / "settings.json").write_text('{"env": {"TOKEN": "secret"}}', encoding="utf-8")
        mask = state / "mask-empty"
        mask.touch()
        prefix = agent_sandbox.build_prefix(
            hide=[layout_outside_tmp / "dataset"],
            writable=[state],
            mask_source=str(mask),
            cwd=str(layout_outside_tmp),
        )
        proc = self._run(prefix, f"cat {state / 'settings.json'}", layout_outside_tmp)
        assert "secret" in proc.stdout
