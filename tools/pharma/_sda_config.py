"""Shared configuration for the remote DrugSDA-Tool SCP server.

Kept separate from ``drug_sda_server.py`` so the codegen
(``scripts/drug_sda_codegen.py``) can read the URL and API key without
importing the server module, whose import registers every tool.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Git-ignored per-machine credentials; see claude_runner.py:_load_mcp_local_env.
LOCAL_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "mcp.local.json"
)

REMOTE_SERVER_URL = "https://scp.intern-ai.org.cn/api/v1/mcp/2/DrugSDA-Tool"

# Per the repo convention (tools/web/litsearch_server.py), a timeout is a
# tool parameter with a hardcoded default rather than a central setting.
DEFAULT_TIMEOUT_SECONDS = 60.0

# Minutes-long jobs (MD, docking, structure prediction). They get a shorter
# default than the runner's 600s process ceiling on purpose: timing out at the
# tool layer returns a clean error to the agent, whereas overrunning the
# ceiling kills the whole `claude -p` step and loses the step's work.
HEAVY_TIMEOUT_SECONDS = 300.0


def is_heavy(tool_name: str) -> bool:
    """True for tools that routinely run for minutes."""
    return tool_name in HEAVY_TOOLS


HEAVY_TOOLS = frozenset(
    {
        "analyze_mmpbsa",
        "chai1_predict",
        "evobind_tool",
        "foldx_tool",
        "gmx_mmpbsa_propro",
        "goca_pipeline",
        "hdock_tool",
        "karmadock_tool",
        "molecule_docking_quickvina_fullprocess",
        "openawsem_sim",
        "pred_binding_affinity_boltz2",
        "pred_protein_structure_esmfold",
        "prepare_complex",
        "prepare_protein_md",
        "protein_openmm_md",
        "prolif_md",
        "pulchura_rebuild",
        "run_bioemu",
        "run_mmpbsa",
    }
)


def default_timeout_for(tool_name: str) -> float:
    return HEAVY_TIMEOUT_SECONDS if is_heavy(tool_name) else DEFAULT_TIMEOUT_SECONDS


def resolve_api_key() -> str | None:
    """Return the SCP-HUB API key, or None for anonymous access.

    Env-first, matching the existing pharma servers: ``DRUGSDA_API_KEY``
    then ``SCP_HUB_API_KEY``. Only when both are unset does it fall back to
    the git-ignored ``config/mcp.local.json``, mirroring
    ``tools/web/litsearch_server.py:_openalex_api_key`` — an exported
    variable is the documented override and must not lose to a file on the
    machine.
    """
    key = os.environ.get("DRUGSDA_API_KEY") or os.environ.get("SCP_HUB_API_KEY")
    if key:
        return key.strip()
    return _local_config_key()


def _local_config_key() -> str | None:
    """Read ``DRUGSDA_API_KEY`` from the git-ignored ``config/mcp.local.json``.

    The tracked ``config/mcp.json`` carries no credentials. A missing or
    malformed file is not an error: the remote server still answers
    anonymously, just with a lower rate limit.
    """
    try:
        data = json.loads(LOCAL_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    servers = data.get("mcpServers")
    if not isinstance(servers, dict):
        return None
    entry = servers.get("pharma-drug-sda")
    if not isinstance(entry, dict):
        return None
    env = entry.get("env")
    if not isinstance(env, dict):
        return None
    key = env.get("DRUGSDA_API_KEY")
    if not key:
        return None
    return str(key).strip() or None
