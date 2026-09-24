"""Model selection, credential isolation, and Claude invocation regressions."""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from claude_runner import ClaudeRunner
from model_profiles import load_model_profiles


def _config(tmp_path: Path, **overrides) -> Path:
    data = {
        "models": {
            "planner-a": {
                "label": "Planner A", "provider": "Provider A",
                "model": "model-a", "base_url": "https://a.example/anthropic",
                "api_key": "secret-a",
            },
            "worker-b": {
                "label": "Worker B", "provider": "Provider B",
                "model": "model-b", "base_url": "https://b.example/anthropic",
                "auth_token_env": "TEST_PROVIDER_B_TOKEN",
            },
        },
        "defaults": {"planner": "planner-a", "agent": "worker-b"},
        **overrides,
    }
    path = tmp_path / "models.local.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_profiles_resolve_credentials_without_exposing_them(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PROVIDER_B_TOKEN", "secret-b")
    profiles = load_model_profiles(_config(tmp_path))
    public = json.dumps(profiles.public_data())
    for private in ("secret-a", "secret-b", "TEST_PROVIDER_B_TOKEN", "base_url", "api_key"):
        assert private not in public
    planner = profiles.resolve("planner-a")
    agent = profiles.resolve("worker-b")
    assert planner.env["ANTHROPIC_API_KEY"] == "secret-a"
    assert planner.env["ANTHROPIC_AUTH_TOKEN"] == ""
    assert agent.env["ANTHROPIC_AUTH_TOKEN"] == "secret-b"
    assert agent.env["ANTHROPIC_API_KEY"] == ""
    assert agent.env["ANTHROPIC_BASE_URL"] == "https://b.example/anthropic"
    assert agent.env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "model-b"
    assert "secret" not in repr(planner)
    assert "secret" not in repr(profiles)


@pytest.mark.parametrize("patch", [
    {"model": ""}, {"base_url": "https://secret@provider.example"},
    {"base_url": "https://provider.example?key=secret"},
    {"auth_token": "second-secret"}, {"api_key": None},
    {"env": {"ANTHROPIC_AUTH_TOKEN": "second-secret"}},
    {"env": {"TIMEOUT": 30}}, {"unknown": "secret-value"},
])
def test_invalid_profiles_fail_without_echoing_credentials(tmp_path, patch):
    path = _config(tmp_path)
    data = json.loads(path.read_text())
    data["models"]["planner-a"].update(patch)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError) as error:
        load_model_profiles(path)
    assert "secret" not in str(error.value)


def test_missing_file_env_and_bad_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("SCIMAS_MODELS_CONFIG", raising=False)
    monkeypatch.setattr("model_profiles.DEFAULT_MODEL_CONFIG", tmp_path / "absent.json")
    assert load_model_profiles().public_data()["models"] == []
    with pytest.raises(ValueError, match="not found"):
        load_model_profiles(tmp_path / "absent.json")
    path = _config(tmp_path, defaults={"agent": "unknown"})
    with pytest.raises(ValueError, match="existing model profile"):
        load_model_profiles(path)
    path = _config(tmp_path)
    monkeypatch.setenv("SCIMAS_MODELS_CONFIG", str(path))
    monkeypatch.delenv("TEST_PROVIDER_B_TOKEN", raising=False)
    with pytest.raises(ValueError, match="TEST_PROVIDER_B_TOKEN"):
        load_model_profiles().resolve("worker-b")
    path.write_text('{"models": { "secret-key"')
    with pytest.raises(ValueError, match="line 1") as error:
        load_model_profiles(path)
    assert "secret-key" not in str(error.value)


def test_dashboard_freezes_selection_and_keeps_secrets_server_side(tmp_path, monkeypatch):
    import web_dashboard as dashboard

    profiles = load_model_profiles(_config(tmp_path))
    monkeypatch.setenv("TEST_PROVIDER_B_TOKEN", "secret-b")
    monkeypatch.setitem(dashboard.app.config, "MODEL_PROFILES", profiles)
    monkeypatch.setattr(dashboard, "JOBS", {})
    monkeypatch.setattr(dashboard.threading.Thread, "start", lambda self: None)
    client = dashboard.app.test_client()
    response = client.get("/api/models")
    assert response.status_code == 200
    assert response.json["defaults"] == {"planner": "planner-a", "agent": "worker-b"}
    first = client.post("/api/jobs", json={})
    second = client.post("/api/jobs", json={"planner_profile": "worker-b", "agent_profile": "planner-a"})
    assert first.status_code == second.status_code == 200
    job = dashboard.JOBS[first.json["id"]]
    other_job = dashboard.JOBS[second.json["id"]]
    assert job._model_configs["planner"].model == "model-a"
    assert other_job._model_configs["planner"].model == "model-b"
    monkeypatch.setenv("TEST_PROVIDER_B_TOKEN", "changed-key")
    assert job._model_configs["agent"].env["ANTHROPIC_AUTH_TOKEN"] == "secret-b"
    serialized = response.text + first.text + second.text + client.get("/api/jobs").text
    assert "secret-a" not in serialized and "secret-b" not in serialized
    assert "TEST_PROVIDER_B_TOKEN" not in serialized
    assert "secret" not in repr(job)

    fallback = client.post("/api/jobs", json={"planner_profile": "", "agent_profile": ""})
    assert dashboard.JOBS[fallback.json["id"]]._model_configs == {}
    count = len(dashboard.JOBS)
    for bad in ("unknown", {"api_key": "client-secret"}):
        assert client.post("/api/jobs", json={"planner_profile": bad}).status_code == 400
    monkeypatch.delenv("TEST_PROVIDER_B_TOKEN")
    assert client.post("/api/jobs", json={}).status_code == 400
    assert len(dashboard.JOBS) == count


def test_dashboard_run_dispatches_selected_profiles(tmp_path, monkeypatch):
    import web_dashboard as dashboard
    from plan import ExecutionPlan, RunReport

    profiles = load_model_profiles(_config(tmp_path))
    monkeypatch.setenv("TEST_PROVIDER_B_TOKEN", "secret-b")
    monkeypatch.setitem(dashboard.app.config, "MODEL_PROFILES", profiles)
    received = []

    class Orchestrator:
        def __init__(self, **kwargs):
            received.append(kwargs)

        def run(self, *, problem, **kwargs):
            return RunReport(
                problem=problem, plan=ExecutionPlan(problem_summary="test"),
                runs=[], final_answer="4", output_dir=str(tmp_path / "fake-run"),
            )

    class Runner:
        def __init__(self, **kwargs):
            self._tool_calls = []

    monkeypatch.setattr(dashboard, "SciMASOrchestrator", Orchestrator)
    monkeypatch.setattr(dashboard, "_StreamJsonRunner", Runner)
    job = dashboard.DashboardJob(id="profile-test", config={
        "dataset": dashboard.WRITE_IN_DATASET, "write_in_question": "2 + 2?",
        "output_root": str(tmp_path / "output"),
        "planner_model": "ignored-legacy-model", "agent_base_url": "https://ignored.example",
    })
    dashboard._run_job(job)
    assert job.status == "completed", job.error
    assert received[0]["planner_model"] == "model-a"
    assert received[0]["agent_model"] == "model-b"
    assert received[0]["planner_env_overrides"]["ANTHROPIC_API_KEY"] == "secret-a"
    assert received[0]["agent_env_overrides"]["ANTHROPIC_AUTH_TOKEN"] == "secret-b"
    assert received[0]["agent_env_overrides"]["ANTHROPIC_API_KEY"] == ""
    saved = "".join(p.read_text() for p in (tmp_path / "output").rglob("*.json"))
    assert "secret-a" not in saved and "secret-b" not in saved


@pytest.mark.parametrize("output_format", ["json", "stream-json"])
def test_concurrent_runner_calls_have_private_settings_and_env(tmp_path, monkeypatch, output_format):
    # A local CLI stand-in exercises actual subprocess.run/Popen without API use.
    binary = tmp_path / "fake-claude"
    binary.write_text(f"#!{sys.executable}\n" + '''
import json, os, sys
from pathlib import Path
args = sys.argv
path = Path(args[args.index("--settings") + 1])
settings = json.loads(path.read_text())
payload = {
    "model": args[args.index("--model") + 1],
    "env": {k: os.environ.get(k) for k in settings["env"]},
    "settings": settings, "path": str(path), "mode": path.stat().st_mode & 0o777,
    "argv": args,
}
print(json.dumps({"type": "result", "result": json.dumps(payload)}))
''')
    binary.chmod(0o700)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "inherited-key")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "inherited-token")
    monkeypatch.setenv("TEST_PROVIDER_B_TOKEN", "secret-b")
    profiles = load_model_profiles(_config(tmp_path))
    runner = ClaudeRunner(binary=str(binary), mcp_config_path="", timeout=10)

    def invoke(profile_id):
        profile = profiles.resolve(profile_id)
        return json.loads(runner.run(
            prompt="test", model=profile.model, env_overrides=profile.env,
            output_format=output_format,
        ).result)

    with ThreadPoolExecutor(max_workers=2) as executor:
        a, b = list(executor.map(invoke, ["planner-a", "worker-b"]))
    assert a["model"] == "model-a" and b["model"] == "model-b"
    assert a["env"]["ANTHROPIC_API_KEY"] == "secret-a"
    assert a["env"]["ANTHROPIC_AUTH_TOKEN"] == ""
    assert b["env"]["ANTHROPIC_AUTH_TOKEN"] == "secret-b"
    assert b["env"]["ANTHROPIC_API_KEY"] == ""
    assert a["path"] != b["path"]
    for result in (a, b):
        assert result["env"] == result["settings"]["env"]
        assert result["mode"] == 0o600
        assert not Path(result["path"]).exists()
        assert "secret-" not in json.dumps(result["argv"])
    assert os.environ["ANTHROPIC_API_KEY"] == "inherited-key"
    assert os.environ["ANTHROPIC_AUTH_TOKEN"] == "inherited-token"


def test_settings_file_removed_on_failure(monkeypatch):
    runner = ClaudeRunner(mcp_config_path="")
    monkeypatch.setattr(runner, "_ensure_available", lambda: None)
    paths = []

    def fail(cmd, **kwargs):
        paths.append(Path(cmd[cmd.index("--settings") + 1]))
        assert paths[-1].exists()
        raise RuntimeError("subprocess failed")

    monkeypatch.setattr("claude_runner.subprocess.run", fail)
    with pytest.raises(RuntimeError, match="subprocess failed"):
        runner.run(prompt="test", env_overrides={"ANTHROPIC_API_KEY": "secret"})
    assert paths and not paths[0].exists()
