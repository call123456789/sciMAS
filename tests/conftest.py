"""Suite-wide test isolation.

The judge's config files are real on a working deployment:
``config/llm_judge.json`` ships in the repo and ``config/llm_judge.local.json``
holds a live per-machine credential. Any test that reaches
``resolve_judge_config`` would otherwise read whichever of those the machine
happens to have, with two bad outcomes:

- a test asserting "no judge configured" passes locally and fails on a
  deployment (or the reverse), and
- a test that *thinks* it is exercising the deterministic fallback instead
  makes real, billable requests to the endpoint in the credential.

So the default for every test is a machine with no judge config at all.
Tests that want one patch ``llm_judge.CONFIG_PATH`` /
``llm_judge.LOCAL_CONFIG_PATH`` themselves, which layers over this.

Model profiles have the same shape and the same failure mode:
``config/models.local.json`` is where a deployment names the models it actually
calls, so a test asserting the legacy "no profiles configured" behavior passes
on a developer's machine and picks up the deployment's default profile on the
server — where that profile then overrides the model the test passed in
explicitly. Tests that want profiles load them and install them in
``dashboard.app.config``, which layers over this too.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for path in (str(ROOT), str(ROOT / "tests")):
    if path not in sys.path:
        sys.path.insert(0, path)

import llm_judge  # noqa: E402
import model_profiles  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_judge_config(monkeypatch, tmp_path):
    """Point the judge config at paths that do not exist."""
    monkeypatch.setattr(llm_judge, "CONFIG_PATH", tmp_path / "llm_judge.json")
    monkeypatch.setattr(
        llm_judge, "LOCAL_CONFIG_PATH", tmp_path / "llm_judge.local.json"
    )


@pytest.fixture(autouse=True)
def _forget_learned_judge_budgets():
    """Clear the per-model token-budget memo llm_judge learns while running.

    It is process-global state that outlives a test on purpose (a whole batch
    benefits from it), which makes it exactly the kind of thing that turns a
    test into a function of what ran before it: a test scripted to see the
    judge escalate its budget would instead see one request if an earlier test
    had already taught the memo that this model needs the top rung.
    """
    llm_judge.reset_starting_budget()
    yield
    llm_judge.reset_starting_budget()


@pytest.fixture(autouse=True)
def _isolate_model_profiles(monkeypatch, tmp_path):
    """Point the model-profile config at a path that does not exist.

    ``SCIMAS_MODELS_CONFIG`` is cleared as well: an operator who exports it
    would otherwise bypass the patch and reintroduce the same leak.
    """
    monkeypatch.delenv("SCIMAS_MODELS_CONFIG", raising=False)
    monkeypatch.setattr(
        model_profiles, "DEFAULT_MODEL_CONFIG", tmp_path / "models.local.json"
    )
