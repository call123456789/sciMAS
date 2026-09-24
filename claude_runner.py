"""Thin Python wrapper around the Claude CLI."""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Sequence

import agent_sandbox


@dataclass
class ClaudeResult:
    raw_stdout: str
    raw_json: Dict[str, Any]
    result: str
    session_id: Optional[str]
    total_cost_usd: Optional[float]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    stderr: str = ""
    # Optional: not every output format reports these, and callers building a
    # result by hand should not have to invent them.
    total_tokens: Optional[int] = None
    # Cache writes and cache reads, which the CLI counts separately from
    # `input_tokens` — that field is *uncached* input only. Cost is priced
    # against all three, so a token count that drops these cannot be
    # reconciled with `total_cost_usd`. See `total_input_tokens`.
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None
    started_at: Optional[str] = None
    duration_ms: Optional[int] = None
    # Only populated when stream-json output was requested. Each entry
    # is the short tool name with its parsed input dict (e.g. from an
    # `mcp__<server>__<tool>` use).
    tool_calls: list[Dict[str, Any]] = field(default_factory=list)

    @property
    def total_input_tokens(self) -> Optional[int]:
        """Everything the model read: fresh input + cache writes + reads.

        ``input_tokens`` on its own understates the real input volume by
        whatever came out of the prompt cache, which on a resumed
        multi-agent session is most of it. Returns ``None`` only when the
        CLI reported no input counter at all — a run that reported some of
        them gets the sum of what it did report.
        """
        parts = (
            self.input_tokens,
            self.cache_creation_input_tokens,
            self.cache_read_input_tokens,
        )
        if all(part is None for part in parts):
            return None
        return sum(part or 0 for part in parts)


def _coerce_int(value: Any) -> Optional[int]:
    """Best-effort int, or ``None`` for a missing or unusable counter.

    Counters are absent whenever the provider reports no usage (an unknown
    model, or ``--output-format`` with no JSON at all), and are occasionally
    a string. Neither should raise here: the counter is metadata, and a call
    that produced a good answer should not be failed over its bookkeeping.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_mcp_local_env(src: Path) -> Dict[str, Dict[str, str]]:
    """Read the git-ignored ``config/mcp.local.json`` into per-server env.

    ``config/mcp.json`` is tracked, so it must not carry credentials. A
    machine that needs one (DrugSDA's ``DRUGSDA_API_KEY``) puts it in the
    sibling ``mcp.local.json`` instead::

        {"mcpServers": {"pharma-drug-sda": {"env": {"DRUGSDA_API_KEY": "sk-..."}}}}

    Only ``env`` blocks are merged; commands and args stay exactly as the
    tracked file defines them, so a local file cannot silently repoint a
    server at a different script. A missing or malformed file is normal —
    most deployments have no secrets — and never raises.
    """
    local = src.with_name("mcp.local.json")
    if not local.is_file():
        return {}
    try:
        data = json.loads(local.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(
            f"claude_runner: cannot parse {local}: {exc}; ignoring",
            file=sys.stderr,
        )
        return {}
    servers = data.get("mcpServers")
    if not isinstance(servers, dict):
        return {}
    out: Dict[str, Dict[str, str]] = {}
    for name, cfg in servers.items():
        if not isinstance(cfg, dict):
            continue
        env = cfg.get("env")
        if isinstance(env, dict):
            out[name] = {str(k): str(v) for k, v in env.items()}
    return out


# ---------------------------------------------------------------------------
# Claude Code state isolation
# ---------------------------------------------------------------------------
#
# Every `claude -p` invocation keeps durable state under one directory:
# ``$CLAUDE_CONFIG_DIR`` (default ``~/.claude``), which holds
# ``projects/<cwd-slug>/`` — session transcripts (``*.jsonl``) and the
# auto-memory store (``MEMORY.md`` plus one file per memory). Left alone, that
# directory is shared by every sciMAS run on the machine *and* by the
# operator's own interactive sessions in the same checkout, because the slug is
# derived from the working directory and all of them run with cwd = the repo
# root.
#
# That sharing is a benchmark-integrity bug, not just untidiness. Auto-memory
# is injected into the system prompt of every later run, so a run that found
# the ground-truth file while solving one problem writes a memory pointing at
# it, and the next run starts already knowing where the answers are. Measured
# on this repo's deployment: 35 sessions had read a ground-truth path, 20 had
# pulled gold content back into context, and 24 graded runs across 9 jobs
# inherited it — including Synthesizer runs, whose output *is* the graded
# answer.
#
# So every run gets its own throwaway state directory, and none of them touch
# the operator's store. Four things have to keep working while that happens,
# and each one is the reason for a piece of the code below:
#
#   credentials  deployments keep the API token in ``settings.json``'s ``env``
#                block (not in the sciMAS process environment), and the CLI
#                does *not* fall back to ``~/.claude`` once
#                ``CLAUDE_CONFIG_DIR`` is set — so isolating without carrying
#                the block over breaks every run with an auth error.
#                `_seed_state_dir` copies it.
#   resume       ``--resume <session_id>`` re-reads a transcript from the state
#                directory, so the directory must be stable for the life of one
#                run. It is: state is keyed by run scope, and no caller resumes
#                a session captured by a different run (`start_run_state`).
#   MCP          MCP servers are spawned by the CLI from ``--mcp-config`` and
#                read nothing out of the state directory, so tool calls are
#                unaffected. Verified A/B against the real config/mcp.json
#                servers: ``mcp__litsearch__search_literature`` is available
#                either way.
#   isolation    the boundary is the run, not the process: the dashboard is
#                long-lived and runs jobs back to back, so a process-wide
#                directory would let job N+1's agents read job N's transcripts.
#
# ``SCIMAS_CLAUDE_STATE=shared`` restores the old shared-directory behaviour
# (for debugging an interactive-style run); ``SCIMAS_CLAUDE_STATE_DIR`` pins
# where state goes, which must be *outside* the repo — state inside the repo is
# readable by the next agent, which is the problem being solved.
_state_isolated: bool = (
    os.environ.get("SCIMAS_CLAUDE_STATE", "").strip().lower() != "shared"
)
_state_root: Optional[str] = os.environ.get("SCIMAS_CLAUDE_STATE_DIR") or None
_state_scope: str = "default"
# Set once per process, and only in temp mode: the parent of every per-run
# directory, removed at exit along with everything under it.
_state_temp_root: Optional[str] = None
# Directories already seeded, so credentials are copied once per run.
_state_seeded: set = set()
_state_lock = threading.Lock()
def set_claude_state_isolated(enabled: bool) -> None:
    """Give every run its own Claude Code state dir (default: on).

    Pass ``False`` to inherit the shared ``~/.claude`` store again — that
    re-enables cross-run memory leakage, so it is only for debugging.
    """
    global _state_isolated
    _state_isolated = bool(enabled)


def get_claude_state_isolated() -> bool:
    return _state_isolated


def set_claude_state_root(path: Optional[str]) -> None:
    """Pin the directory that holds each run's state; ``None`` = system temp.

    This is a parent directory, not the state directory itself: each run gets
    ``<root>/<scope>`` under it. The path must be outside the repo — whatever
    lands there is readable by agents running with cwd = the repo root — and
    unlike the temp default it is never deleted, for postmortem inspection.
    """
    global _state_root, _state_temp_root
    _state_root = str(Path(path).expanduser()) if path else None
    _state_temp_root = None


def get_claude_state_root() -> Optional[str]:
    return _state_root


def start_run_state(label: Optional[str] = None) -> str:
    """Start a fresh state scope and return its directory.

    Call once per job/batch before its agents run. Without it, one process
    keeps a single scope, which is correct for a one-run process and a leak
    for a queue of them.
    """
    global _state_scope
    with _state_lock:
        _state_scope = _scope_name(label)
    # Outside the lock: `claude_state_dir` takes it too.
    return claude_state_dir()


def claude_state_dir() -> str:
    """Return the current run's state directory, creating and seeding it once."""
    with _state_lock:
        return _ensure_state_dir(_state_scope)


def state_dir_for(scope: Optional[str]) -> str:
    """The state directory for ``scope``, without touching the current scope.

    For a process that runs jobs concurrently: each job passes its scope to its
    own runners (``ClaudeRunner(state_scope=...)``) instead of racing to set
    one process-wide value, which would let two jobs swap state directories.
    """
    with _state_lock:
        return _ensure_state_dir(_scope_name(scope))


def _ensure_state_dir(scope: str) -> str:
    """Create and seed ``<root>/<scope>`` once. Caller holds ``_state_lock``."""
    path = Path(_state_root) / scope if _state_root else _temp_scope_dir(scope)
    if str(path) not in _state_seeded:
        path.mkdir(parents=True, exist_ok=True)
        _seed_state_dir(path)
        _state_seeded.add(str(path))
    return str(path)


def _temp_scope_dir(scope: str) -> Path:
    """``<per-process temp root>/<scope>``, removed when the process exits."""
    global _state_temp_root
    if _state_temp_root is None:
        _state_temp_root = tempfile.mkdtemp(prefix="scimas-claude-state-")
        atexit.register(shutil.rmtree, _state_temp_root, True)
    return Path(_state_temp_root) / scope


_mask_temp_dir: Optional[str] = None


def _mask_source_file(state_dir: Optional[str] = None) -> tuple[str, str]:
    """``(directory, empty file)`` standing in for masked files.

    bubblewrap needs the replacement to exist when it builds the mounts, and to
    be visible in the sandbox — so it has to sit under a directory that gets
    bound back in past the private ``/tmp``. The state directory is already
    bound for credentials; without state isolation the file gets a directory of
    its own, which the caller binds for the same reason.
    """
    global _mask_temp_dir
    if state_dir:
        directory = Path(state_dir)
    elif _state_isolated:
        directory = Path(claude_state_dir())
    else:
        if _mask_temp_dir is None:
            _mask_temp_dir = tempfile.mkdtemp(prefix="scimas-mask-")
            atexit.register(shutil.rmtree, _mask_temp_dir, True)
        directory = Path(_mask_temp_dir)
    path = directory / "mask-empty"
    if not path.exists():
        path.touch()
    return str(directory), str(path)


def _scope_name(label: Optional[str]) -> str:
    """A filesystem-safe directory name for one job/batch."""
    if not label:
        return "default"
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in str(label))
    return safe.strip("-.")[:64] or "default"


def _state_source_dir() -> Path:
    """The operator's own config dir — where credentials are read from."""
    configured = os.environ.get("CLAUDE_CONFIG_DIR", "").strip()
    return Path(configured).expanduser() if configured else Path.home() / ".claude"


def _seed_state_dir(path: Path) -> None:
    """Copy credentials (and nothing else) into a fresh state directory.

    Only the ``env`` block is taken from ``settings.json``: the rest of that
    file (hooks, plugins, statusline) describes an interactive setup whose
    files live in the original config dir, and copying it would point the
    agent at paths that are not in the new one. Conversation state — memory,
    transcripts, history — is deliberately never copied; that is the leak.
    """
    source = _state_source_dir()
    try:
        raw = json.loads((source / "settings.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raw = None
    if isinstance(raw, dict):
        env = raw.get("env")
        if isinstance(env, dict) and env:
            target = path / "settings.json"
            with target.open("w", encoding="utf-8") as handle:
                target.chmod(0o600)
                json.dump({"env": {str(k): str(v) for k, v in env.items()}}, handle)
    credentials = source / ".credentials.json"
    if credentials.is_file():
        target = path / ".credentials.json"
        try:
            shutil.copyfile(credentials, target)
            target.chmod(0o600)
        except OSError as exc:
            print(
                f"claude_runner: cannot carry credentials from {credentials}: {exc}",
                file=sys.stderr,
            )


class ClaudeRunner:
    def __init__(
        self,
        binary: str = "claude",
        model: Optional[str] = None,
        mcp_config_path: Optional[str] = None,
        allowed_tools: Optional[Sequence[str]] = None,
        permission_mode: str = "bypassPermissions",
        dangerously_skip_permissions: bool = True,
        extra_system_prompt: Optional[str] = None,
        timeout: Optional[float] = 600.0,
        env_overrides: Optional[Dict[str, str]] = None,
        state_scope: Optional[str] = None,
        sandbox_hide: Sequence[str] = (),
        sandbox_writable: Sequence[str] = (),
    ):
        self.binary = binary
        self.model = model
        self.env_overrides = dict(env_overrides or {})
        # This runner's state scope and filesystem boundary. Both fall back to
        # process-wide behaviour (the scope `start_run_state` last set, and no
        # sandbox at all), which is what a one-run process wants. A long-lived
        # process that runs jobs concurrently — the dashboard takes a new job
        # per request — must pass them per runner instead: two jobs sharing one
        # process-wide value hand each other their boundaries, and with them
        # the directories each is allowed to write.
        self.state_scope = str(state_scope) if state_scope else None
        self.sandbox_hide = tuple(str(p) for p in sandbox_hide)
        self.sandbox_writable = tuple(str(p) for p in sandbox_writable)
        # Three-way contract — callers must pick the right one:
        #
        #   None  -> auto-detect config/mcp.json next to this file
        #   ""    -> explicit opt-out, the sentinel `main.py --no-mcp` passes
        #   path  -> use that file
        #
        # ``""`` is *not* interchangeable with ``None``: a caller that
        # forwards a blank user-supplied field as ``""`` silently disables
        # MCP instead of falling back to the default. That is exactly the
        # bug web_dashboard._mcp_config_for_job now guards against, so if
        # you are adding a caller, normalise "user left it blank" to None
        # and reserve "" for a real disable switch.
        if mcp_config_path is None:
            candidate = Path(__file__).resolve().parent / "config" / "mcp.json"
            mcp_config_path = str(candidate) if candidate.exists() else None
        elif mcp_config_path == "":
            mcp_config_path = None  # explicit --no-mcp opt-out
        # The Claude CLI resolves --mcp-config relative to the spawned
        # process's cwd, which can be surprising when sciMAS is invoked from
        # outside the project root. Always pass an absolute path.
        abs_path = os.path.abspath(mcp_config_path) if mcp_config_path else None
        # Rewrite bare `python` / `python3` commands in the config to the
        # running interpreter's absolute path so the same config/mcp.json
        # works on local Mac (scimas env), the remote container (sci env),
        # and any other deployment. Also drops `env.PATH` overrides that
        # baked-in a Mac-only conda path. See `_resolve_mcp_config`.
        self.mcp_config_path = self._resolve_mcp_config(abs_path) if abs_path else None

        # If an MCP config is in play and the caller did not pass an
        # explicit allowlist, default to a permissive "*" so every
        # tool exposed by the configured MCP servers is usable. The
        # user can still override by passing --allowed-tools.
        if allowed_tools is None:
            allowed_tools = ["*"] if self.mcp_config_path else []
        self.allowed_tools: Sequence[str] = list(allowed_tools) if allowed_tools else []
        # Per-allowlist narrowed configs, keyed by the server set. See
        # `_mcp_config_for`; the values are paths on disk.
        self._narrowed_configs: Dict[frozenset, Optional[str]] = {}

        self.permission_mode = permission_mode
        self.dangerously_skip_permissions = dangerously_skip_permissions
        self.extra_system_prompt = extra_system_prompt
        self.timeout = timeout

    @staticmethod
    def _resolve_mcp_config(mcp_config_path: str) -> str:
        """Rewrite ``config/mcp.json`` with this interpreter's absolute path.

        The shipped ``config/mcp.json`` uses bare ``python`` commands so the
        same file works across deployments. Before handing the path to
        ``claude -p --mcp-config=...``, rewrite each server entry:

        - ``command: "python"`` / ``"python3"`` → ``sys.executable`` (the
          interpreter currently running sciMAS, i.e. the conda env's
          ``bin/python``).
        - Drop ``env.PATH`` overrides — these bake in absolute conda paths
          that only exist on one machine and break the subprocess PATH.
        - Force ``no_proxy`` / ``NO_PROXY`` when ``SCIMAS_MCP_NO_PROXY`` is
          set (see below).
        - Merge any per-machine ``env`` values from the git-ignored
          ``config/mcp.local.json`` (see ``_load_mcp_local_env``). This is
          where credentials live, so the tracked file never holds one.

        The resolved file is written next to the original as
        ``<name>.resolved.json``. The original is left untouched so
        ``git diff config/mcp.json`` stays clean. Returns the resolved
        path so the caller can pass it straight to ``--mcp-config=``.

        Proxy note: MCP stdio servers are spawned by ``claude -p`` and
        inherit its environment, including ``HTTPS_PROXY``/``HTTP_PROXY``/
        ``ALL_PROXY``. The repo's own servers (litsearch, ...) call public
        APIs through ``requests``, which honours those variables, so a
        broken proxy on the host takes every one of them down — and unlike
        an agent, a subprocess cannot work around it per-call. Setting
        ``SCIMAS_MCP_NO_PROXY`` (e.g. to ``"*"``, or a comma-separated host
        list) injects the bypass into each server entry. It is opt-in so
        deployments that genuinely need a proxy keep working unchanged.
        """
        src = Path(mcp_config_path)
        if not src.is_file():
            return mcp_config_path
        try:
            data = json.loads(src.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(
                f"claude_runner: cannot parse {src}: {exc}; using as-is",
                file=sys.stderr,
            )
            return mcp_config_path
        servers = data.get("mcpServers")
        if not isinstance(servers, dict):
            return mcp_config_path
        py_abs = sys.executable
        # Opt-in proxy bypass for the spawned servers; unset means "leave the
        # inherited proxy environment exactly as it is".
        mcp_no_proxy = (os.environ.get("SCIMAS_MCP_NO_PROXY") or "").strip()
        local_env = _load_mcp_local_env(src)
        rewritten = False
        for name, cfg in servers.items():
            if not isinstance(cfg, dict):
                continue
            cmd = cfg.get("command")
            if cmd in ("python", "python3"):
                cfg["command"] = py_abs
                rewritten = True
                print(
                    f"claude_runner: mcp[{name}].command {cmd!r} -> {py_abs!r}",
                    file=sys.stderr,
                )
            env = cfg.get("env")
            if isinstance(env, dict) and "PATH" in env:
                env.pop("PATH", None)
                # Drop empty env blocks too — Claude CLI is fine without
                # them and a leftover `env: {}` is just noise.
                if not env:
                    cfg.pop("env", None)
                rewritten = True
                print(
                    f"claude_runner: mcp[{name}]: dropped env.PATH override",
                    file=sys.stderr,
                )
            if mcp_no_proxy:
                env = cfg.get("env")
                if not isinstance(env, dict):
                    env = {}
                    cfg["env"] = env
                # Both spellings: `requests` reads lowercase, `curl` and
                # some libs read either.
                env["no_proxy"] = mcp_no_proxy
                env["NO_PROXY"] = mcp_no_proxy
                rewritten = True
                print(
                    f"claude_runner: mcp[{name}]: no_proxy={mcp_no_proxy!r} "
                    f"(SCIMAS_MCP_NO_PROXY)",
                    file=sys.stderr,
                )
            # Local secrets come last so a per-machine value always wins
            # over anything the tracked file happens to carry.
            overrides = local_env.get(name)
            if overrides:
                env = cfg.get("env")
                if not isinstance(env, dict):
                    env = {}
                    cfg["env"] = env
                env.update(overrides)
                rewritten = True
                print(
                    f"claude_runner: mcp[{name}]: merged {len(overrides)} env "
                    f"value(s) from mcp.local.json",
                    file=sys.stderr,
                )
        if not rewritten:
            # No substitutions were needed; hand the original path back so
            # we don't litter the config directory with a redundant copy.
            return mcp_config_path
        out = src.with_name(src.name + ".resolved.json")
        try:
            out.write_text(
                json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            print(
                f"claude_runner: cannot write {out}: {exc}; using original",
                file=sys.stderr,
            )
            return mcp_config_path
        return str(out)

    def _ensure_available(self) -> None:
        if shutil.which(self.binary) is None:
            raise RuntimeError(
                f"Claude CLI not found: {self.binary}. "
                "Install it or pass --claude-bin with the correct path."
            )

    @staticmethod
    def _coerce_json(value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("```"):
                text = text.removeprefix("```json").removeprefix("```JSON")
                text = text.removeprefix("```").strip()
                if text.endswith("```"):
                    text = text[:-3].strip()
            try:
                parsed = json.loads(text)
                return parsed if isinstance(parsed, dict) else {"result": parsed}
            except json.JSONDecodeError:
                return {}
        return {}

    @staticmethod
    def _usage_counters(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Pull the session/token/cost counters out of a CLI result payload.

        Shared by both output paths so a change lands in one place: ``json``
        puts these at the top level of its single result object while
        ``stream-json`` buries the same object in the terminal ``result``
        event, and the two paths had drifted into copies of the same six
        lookups.

        The cache counters are reported alongside ``input_tokens`` rather
        than folded into it, so a caller can still tell a cache hit from
        fresh context; ``total_input_tokens`` on the result sums them.
        """
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            usage = {}

        def pick(name: str) -> Any:
            # `is not None` rather than truthiness: a genuine 0 (no cache
            # writes on a short call) must not fall through to the nested
            # block and pick up a different counter.
            for source in (payload, usage):
                value = source.get(name)
                if value is not None:
                    return value
            return None

        # Both of these are genuine two-name fallbacks — older CLI releases
        # spelled them differently — not payload-vs-usage lookups.
        total_cost = payload.get("total_cost_usd")
        if total_cost is None:
            total_cost = payload.get("cost_usd")
        session_value = payload.get("session_id")
        if session_value is None:
            session_value = payload.get("sessionId")

        return {
            "session_id": str(session_value) if session_value is not None else None,
            "total_cost_usd": _coerce_float(total_cost),
            "total_tokens": _coerce_int(payload.get("total_tokens")),
            "input_tokens": _coerce_int(pick("input_tokens")),
            "output_tokens": _coerce_int(pick("output_tokens")),
            "cache_creation_input_tokens": _coerce_int(
                pick("cache_creation_input_tokens")
            ),
            "cache_read_input_tokens": _coerce_int(pick("cache_read_input_tokens")),
        }

    @classmethod
    def _build_result(
        cls,
        *,
        raw_stdout: str,
        raw_json: Dict[str, Any],
        result: str,
        stderr: str,
        started_at: str,
        duration_ms: int,
        tool_calls: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> ClaudeResult:
        """Assemble a ``ClaudeResult``, filling the counters from ``raw_json``."""
        return ClaudeResult(
            raw_stdout=raw_stdout,
            raw_json=raw_json,
            result=result,
            stderr=stderr,
            started_at=started_at,
            duration_ms=duration_ms,
            tool_calls=list(tool_calls or []),
            **cls._usage_counters(raw_json),
        )

    @staticmethod
    def _servers_for_tools(effective_tools: Sequence[str]) -> Optional[set[str]]:
        """Which MCP servers a session with this allowlist can actually call.

        Returns ``None`` when the allowlist cannot be narrowed safely, which
        makes the caller fall back to the full config:

        - A bare ``"*"`` — ``ClaudeRunner``'s own default when the caller
          passes no allowlist — permits every tool of every server, so there
          is nothing to narrow to.
        - ``mcp__*`` / ``mcp__*__tool``: the server position is itself a
          wildcard, so the set is not enumerable.
        - An allowlist with no ``mcp__`` entry at all yields the empty set,
          meaning the session can call no MCP tool whatever we attach.

        Otherwise the answer is exact rather than a heuristic: every entry is
        spelled ``mcp__<server>__<tool>``, so the servers a step needs are
        literally written in its allowlist. Built-in entries (``Read``,
        ``Bash(npm run:*)``, ...) cannot match an ``mcp__`` tool and are
        skipped.
        """
        if not effective_tools:
            return set()
        servers: set[str] = set()
        for tool in effective_tools:
            name = str(tool).strip()
            if not name:
                continue
            if not name.startswith("mcp__"):
                if name == "*":
                    return None
                continue
            parts = name.split("__")
            if len(parts) < 2 or not parts[1]:
                return None
            if "*" in parts[1]:
                return None
            servers.add(parts[1])
        return servers

    def _mcp_config_for(self, effective_tools: Sequence[str]) -> Optional[str]:
        """The ``--mcp-config`` to hand a session with this allowlist.

        Why this is not simply ``self.mcp_config_path``: ``claude -p`` builds
        the session's tool registry from the MCP servers that have finished
        connecting at the moment it emits ``init``, and a server still
        ``pending`` contributes *no* tools at all. It starts servers in config
        order with bounded concurrency, so with config/mcp.json's 39 entries
        only the first few are ever ready in time — measured on a 96-core
        host, exactly 2 (``chemistry-computational``, ``litsearch``) were
        connected at init, and the rest were missing no matter what
        ``MCP_TIMEOUT`` was set to.

        That made which runs worked a matter of luck: a step whose skill
        routed to ``chemistry-physical`` (config entry #6) hit "No such tool
        available" even though the server came up fine moments later, while a
        step routing to ``chemistry-computational`` (#3) worked.

        Handing over a config containing only the servers this allowlist
        names puts them at the front of a pool of one or two, so the race is
        gone by construction — and it stops spawning ~37 processes that the
        step could never call into.
        """
        if not self.mcp_config_path:
            return None
        servers = self._servers_for_tools(effective_tools)
        if servers is None:
            return self.mcp_config_path
        if not servers:
            # No MCP tool is callable; attaching servers would only burn
            # startup time on servers the CLI would refuse to expose.
            return None
        return self._narrowed_config(servers)

    def _narrowed_config(self, servers: set[str]) -> Optional[str]:
        """Write (once) a config holding only ``servers``, preserving order."""
        key = frozenset(servers)
        if key in self._narrowed_configs:
            return self._narrowed_configs[key]

        result: Optional[str] = self.mcp_config_path
        try:
            src = Path(self.mcp_config_path)
            data = json.loads(src.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(
                f"claude_runner: cannot narrow {self.mcp_config_path}: {exc}; "
                "using the full config",
                file=sys.stderr,
            )
            self._narrowed_configs[key] = result
            return result

        available = data.get("mcpServers")
        if isinstance(available, dict):
            # Iterate the source dict so the servers keep their relative
            # order — the CLI starts them in file order.
            selected = {name: available[name] for name in available if name in servers}
            missing = sorted(servers - set(available))
            if missing:
                print(
                    "claude_runner: allowlist names server(s) absent from "
                    f"{src.name}: {', '.join(missing)}",
                    file=sys.stderr,
                )
            if selected:
                data["mcpServers"] = selected
                digest = hashlib.sha1(
                    ",".join(sorted(servers)).encode("utf-8")
                ).hexdigest()[:8]
                out = src.with_name(f"{src.stem}.tools-{digest}.json")
                try:
                    # Write-then-rename: concurrent steps may narrow to the
                    # same set, and os.replace keeps every reader seeing a
                    # complete file.
                    tmp = out.with_name(f"{out.name}.{os.getpid()}.tmp")
                    tmp.write_text(
                        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8",
                    )
                    os.replace(tmp, out)
                    result = str(out)
                except OSError as exc:
                    print(
                        f"claude_runner: cannot write {out}: {exc}; using the "
                        "full config",
                        file=sys.stderr,
                    )

        self._narrowed_configs[key] = result
        return result

    def _build_cmd(
        self,
        prompt: str,
        session_id: Optional[str],
        output_format: str,
        allowed_tools: Optional[Sequence[str]] = None,
        mcp_config_path: Optional[str] = None,
        model: Optional[str] = None,
        prompt_in_stdin: bool = False,
    ) -> list[str]:
        # Per-call override falls back to the runner-level default.
        effective_tools = list(allowed_tools) if allowed_tools is not None else list(self.allowed_tools)
        if mcp_config_path is None:
            mcp_config_path = self._mcp_config_for(effective_tools)
        cmd = [self.binary, "-p"]
        if output_format:
            cmd.extend(["--output-format", output_format])
        if session_id:
            cmd.extend(["--resume", session_id])
        effective_model = self.model if model is None else (model.strip() or None)
        if effective_model:
            cmd.extend(["--model", effective_model])
        if mcp_config_path and effective_tools:
            # `--mcp-config` is variadic in recent Claude Code releases.
            # Passing the path as the next argv item can make the final prompt
            # get parsed as another config path. The `--flag=value` form keeps
            # the prompt unambiguous.
            cmd.append(f"--mcp-config={mcp_config_path}")
        if effective_tools:
            # The CLI flag is variadic: each tool must be its own
            # `--allowedTools <name>` argument. Space-joining into a single
            # value silently fails (the model then falls back to built-in
            # tools without telling us).
            for tool in effective_tools:
                # Recent Claude Code releases parse `--allowedTools` as a
                # variadic option. Use --flag=value so the final prompt is not
                # swallowed as another allowed-tool token.
                cmd.append(f"--allowedTools={tool}")
        if self.permission_mode:
            cmd.extend(["--permission-mode", self.permission_mode])
        if self.dangerously_skip_permissions:
            cmd.append("--dangerously-skip-permissions")
        # stream-json requires --verbose in `claude -p` (Claude CLI rejects
        # otherwise with exit 1). Always pair them.
        if output_format == "stream-json":
            cmd.append("--verbose")
        if self.extra_system_prompt:
            cmd.extend(["--append-system-prompt", self.extra_system_prompt])
        if not prompt_in_stdin:
            cmd.append(prompt)
        return cmd

    def _state_dir(self) -> Optional[str]:
        """The config dir this runner's CLI will use; None when not isolated."""
        if not _state_isolated:
            # Debug mode: the shared store, shared on purpose.
            return None
        if self.state_scope:
            return state_dir_for(self.state_scope)
        return claude_state_dir()

    def _sandbox_declared(self) -> bool:
        """Whether this runner's CLI will actually be wrapped.

        Checked without building the prefix, which is the only place a missing
        bubblewrap under ``SCIMAS_SANDBOX=require`` is fatal — here it just
        means no boundary, so a caller can decide where to put a file the CLI
        has to read.
        """
        if not self.sandbox_hide and not self.sandbox_writable:
            return False
        return (
            agent_sandbox.sandbox_mode() != "off"
            and agent_sandbox.bubblewrap() is not None
        )

    def _scratch_dir(self) -> Optional[str]:
        """A writable directory the sandboxed CLI can see.

        ``/tmp`` is a private tmpfs inside the sandbox, so a temporary file the
        harness creates there is invisible to the CLI — while the CLI is told
        to read it by path. The directories bound back in past the tmpfs (the
        state directory, and the mask source's directory when state isolation
        is off) stay visible, so anything the CLI has to read goes in one of
        them. ``None`` means "not sandboxed", where the system temp is right.
        """
        if not self._sandbox_declared():
            return None
        return _mask_source_file(self._state_dir())[0]

    def _sandboxed(self, cmd: list[str]) -> list[str]:
        """Wrap the CLI argv in the filesystem sandbox, when one is declared."""
        hide = self.sandbox_hide
        writable = list(self.sandbox_writable)
        if not hide and not writable:
            return cmd
        state_dir = self._state_dir()
        if state_dir:
            # The sandbox gives /tmp a private tmpfs, and the state directory
            # — the one holding the carried-over credentials — lives there.
            # Without this bind the CLI would start with no auth at all.
            writable.append(state_dir)
        mask_dir, mask_file = _mask_source_file(state_dir)
        if mask_dir not in writable:
            writable.append(mask_dir)
        prefix = agent_sandbox.build_prefix(hide, writable, mask_source=mask_file)
        return cmd if prefix is None else prefix + cmd

    def _subprocess_env(
        self,
        env_overrides: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """Build the environment for one Claude CLI invocation."""
        env = os.environ.copy()
        # Inject the active conda env's bin/ at the front of PATH so MCP
        # servers spawned by the claude CLI subprocess inherit the right
        # Python (with rdkit, matplotlib, etc.).
        conda_prefix = env.get("CONDA_PREFIX", "")
        if conda_prefix and conda_prefix != "/opt/anaconda3":
            scimas_bin = os.path.join(conda_prefix, "bin")
            env["PATH"] = scimas_bin + os.pathsep + env.get("PATH", "")
            env["CONDA_PREFIX"] = conda_prefix
            env["CONDA_DEFAULT_ENV"] = env.get("CONDA_DEFAULT_ENV", "scimas")
        # Keep this run's state out of the shared store. Applied *before* the
        # caller overrides below, so a caller that names its own
        # CLAUDE_CONFIG_DIR — or wants auto-memory back — still wins. See the
        # state-isolation note above.
        state_dir = self._state_dir()
        if state_dir:
            env["CLAUDE_CONFIG_DIR"] = state_dir
            env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] = "1"
        env.update({k: v for k, v in self.env_overrides.items() if v is not None})
        if env_overrides:
            env.update({k: v for k, v in env_overrides.items() if v is not None})
        return env

    @contextmanager
    def _invocation_settings(
        self, env_overrides: Optional[Dict[str, str]] = None,
    ) -> Iterator[Optional[str]]:
        overrides = {
            k: v for k, v in {**self.env_overrides, **(env_overrides or {})}.items()
            if v is not None
        }
        if not overrides:
            yield None
            return
        # CLI --settings beats user/project settings.env. Keep credentials out
        # of argv, use a private file per call, and remove it even on failure.
        # Under a sandbox it has to live where the CLI can still see it — the
        # private tmpfs over /tmp would swallow the default location and the
        # call would silently run with the wrong endpoint or key.
        with tempfile.TemporaryDirectory(
            prefix="scimas-claude-", dir=self._scratch_dir()
        ) as directory:
            path = Path(directory) / "settings.json"
            with path.open("x", encoding="utf-8") as settings_file:
                path.chmod(0o600)
                json.dump({"env": overrides}, settings_file)
            yield str(path)

    def run(
        self,
        prompt: str,
        stdin_text: Optional[str] = None,
        session_id: Optional[str] = None,
        output_format: str = "json",
        allowed_tools: Optional[Sequence[str]] = None,
        model: Optional[str] = None,
        env_overrides: Optional[Dict[str, str]] = None,
    ) -> ClaudeResult:
        self._ensure_available()

        # Resolved here and handed to `_build_cmd` so the narrowing happens
        # once per call.
        effective_tools = (
            list(allowed_tools) if allowed_tools is not None else list(self.allowed_tools)
        )
        mcp_config_path = self._mcp_config_for(effective_tools)

        # Keep task content out of argv. Some dashboard workflows hand a later
        # agent the outputs of several earlier agents, and the bwrap prefix adds
        # hundreds of mount arguments on Linux. Putting the prompt on argv can
        # then exceed ARG_MAX before the Claude CLI even starts.
        stdin_parts = [prompt]
        if stdin_text:
            stdin_parts.extend(["", "--- stdin context ---", stdin_text])
        stdin_to_send = "\n".join(stdin_parts)

        cmd = self._build_cmd(
            "",
            session_id,
            output_format,
            effective_tools,
            mcp_config_path,
            model,
            prompt_in_stdin=True,
        )

        started = time.monotonic()
        started_at = datetime.now(timezone.utc).isoformat()

        with self._invocation_settings(env_overrides) as settings_path:
            if settings_path:
                cmd[1:1] = ["--settings", settings_path]
            # After the settings insertion, which assumes cmd[0] is the binary.
            cmd = self._sandboxed(cmd)
            if output_format == "stream-json":
                return self._run_stream_json(
                    cmd, stdin_to_send, started_at, started, env_overrides
                )

            proc = subprocess.run(
                cmd,
                input=stdin_to_send,
                text=True,
                capture_output=True,
                check=False,
                timeout=self.timeout,
                env=self._subprocess_env(env_overrides),
            )
        duration_ms = int((time.monotonic() - started) * 1000)
        raw_stdout = proc.stdout.strip()
        if proc.returncode != 0 and not raw_stdout:
            raise RuntimeError(
                "Claude CLI failed with exit code "
                f"{proc.returncode}:\n{proc.stderr.strip()}"
            )

        raw_json = self._coerce_json(raw_stdout)
        return self._build_result(
            raw_stdout=raw_stdout,
            raw_json=raw_json,
            result=str(raw_json.get("result", raw_stdout)),
            stderr=proc.stderr.strip(),
            started_at=started_at,
            duration_ms=duration_ms,
        )

    def _run_stream_json(
        self,
        cmd: list[str],
        stdin_text: Optional[str],
        started_at: str,
        started_monotonic: float,
        env_overrides: Optional[Dict[str, str]] = None,
    ) -> ClaudeResult:
        """Run claude -p with ``--output-format stream-json`` and parse
        the event stream. Each line of stdout is one JSON event; we
        collect ``content_block_delta`` text fragments into ``result``,
        and every ``tool_use`` block into ``tool_calls``.
        """
        env = self._subprocess_env(env_overrides)
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE if stdin_text else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        assert proc.stdout is not None
        text_chunks: list[str] = []
        tool_calls: list[Dict[str, Any]] = []
        raw_lines: list[str] = []
        final: Dict[str, Any] = {}
        try:
            if stdin_text:
                proc.stdin.write(stdin_text)
                proc.stdin.close()
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                raw_lines.append(line)
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    continue
                evt_type = evt.get("type")
                # Real Claude stream-json emits nested
                # message.content[].text / tool_use. Support both the
                # nested shape and a flatter assistant-message variant.
                if evt_type == "content_block_start":
                    block = evt.get("content_block", {})
                    if block.get("type") == "tool_use":
                        tool_calls.append(
                            {
                                "tool": block.get("name", ""),
                                "arguments": block.get("input", {}) or {},
                            }
                        )
                elif evt_type == "content_block_delta":
                    delta = evt.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text_chunks.append(delta.get("text", ""))
                elif evt_type == "assistant":
                    msg = evt.get("message", {}) or {}
                    for blk in msg.get("content", []) or []:
                        btype = blk.get("type")
                        if btype == "text":
                            text_chunks.append(blk.get("text", ""))
                        elif btype == "tool_use":
                            tool_calls.append(
                                {
                                    "tool": blk.get("name", ""),
                                    "arguments": blk.get("input", {}) or {},
                                }
                            )
                elif evt_type == "result":
                    final = evt
        finally:
            try:
                proc.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        duration_ms = int((time.monotonic() - started_monotonic) * 1000)
        if proc.returncode != 0:
            err = proc.stderr.read() if proc.stderr else ""
            # Workaround: with the MiniMax-M3 / unknown-model path the SDK
            # sometimes prints ``[claude-code:unrecognized_model]`` to stderr
            # and exits 1 even though the model successfully produced a
            # ``result`` event (is_error=False, stop_reason=end_turn). If we
            # did receive a valid result, treat the call as a success and
            # just surface the warning in stderr.
            if final and final.get("is_error") is False and final.get("result"):
                pass
            else:
                raise RuntimeError(
                    f"Claude CLI failed with exit code {proc.returncode}:\n{err}"
                )

        # The stream typically ends with a single ``result`` event
        # carrying the same payload the JSON format would. Use it for
        # usage / cost / session_id when present; fall back to joined
        # text. When no ``result`` event arrived, `raw_json` carries the
        # stream placeholder and every counter comes back ``None``.
        result_text = (
            final.get("result")
            if isinstance(final, dict) and final.get("result")
            else "".join(text_chunks)
        )
        return self._build_result(
            raw_stdout="\n".join(raw_lines),
            raw_json=final if final else {"stream": True, "tool_calls": tool_calls},
            result=str(result_text or ""),
            stderr=proc.stderr.read() if proc.stderr else "",
            started_at=started_at,
            duration_ms=duration_ms,
            tool_calls=tool_calls,
        )
