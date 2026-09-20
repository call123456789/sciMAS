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


@pytest.fixture(autouse=True)
def _isolate_judge_config(monkeypatch, tmp_path):
    """Point the judge config at paths that do not exist."""
    monkeypatch.setattr(llm_judge, "CONFIG_PATH", tmp_path / "llm_judge.json")
    monkeypatch.setattr(
        llm_judge, "LOCAL_CONFIG_PATH", tmp_path / "llm_judge.local.json"
    )
