"""Thin Python wrapper around the Claude CLI."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence


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
    started_at: Optional[str] = None
    duration_ms: Optional[int] = None
    # Only populated when stream-json output was requested. Each entry
    # is the short tool name with its parsed input dict (e.g. from an
    # `mcp__<server>__<tool>` use).
    tool_calls: list[Dict[str, Any]] = field(default_factory=list)


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
    ):
        self.binary = binary
        self.model = model
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

    def _build_cmd(
        self,
        prompt: str,
        session_id: Optional[str],
        output_format: str,
        allowed_tools: Optional[Sequence[str]] = None,
    ) -> list[str]:
        # Per-call override falls back to the runner-level default.
        effective_tools = list(allowed_tools) if allowed_tools is not None else list(self.allowed_tools)
        cmd = [self.binary, "-p"]
        if output_format:
            cmd.extend(["--output-format", output_format])
        if session_id:
            cmd.extend(["--resume", session_id])
        if self.model:
            cmd.extend(["--model", self.model])
        if self.mcp_config_path and effective_tools:
            # `--mcp-config` is variadic in recent Claude Code releases.
            # Passing the path as the next argv item can make the final prompt
            # get parsed as another config path. The `--flag=value` form keeps
            # the prompt unambiguous.
            cmd.append(f"--mcp-config={self.mcp_config_path}")
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
        cmd.append(prompt)
        return cmd

    def run(
        self,
        prompt: str,
        stdin_text: Optional[str] = None,
        session_id: Optional[str] = None,
        output_format: str = "json",
        allowed_tools: Optional[Sequence[str]] = None,
    ) -> ClaudeResult:
        self._ensure_available()

        # When an MCP stdio server is attached, `claude -p` needs to own
        # stdin to talk to the server. We therefore cannot pipe the
        # orchestrator's context bundle through stdin. Inline the bundle
        # into the prompt so the model still receives it.
        if self.mcp_config_path and stdin_text:
            effective_prompt = (
                f"{prompt}\n\n--- stdin context (inlined because MCP "
                f"server is using stdio) ---\n{stdin_text}"
            )
            stdin_to_send = None
        else:
            effective_prompt = prompt
            stdin_to_send = stdin_text

        cmd = self._build_cmd(effective_prompt, session_id, output_format, allowed_tools)

        started = time.monotonic()
        started_at = datetime.now(timezone.utc).isoformat()

        if output_format == "stream-json":
            return self._run_stream_json(
                cmd, stdin_to_send, started_at, started
            )

        proc = subprocess.run(
            cmd,
            input=stdin_to_send,
            text=True,
            capture_output=True,
            check=False,
            timeout=self.timeout,
        )
        duration_ms = int((time.monotonic() - started) * 1000)
        raw_stdout = proc.stdout.strip()
        if proc.returncode != 0 and not raw_stdout:
            raise RuntimeError(
                "Claude CLI failed with exit code "
                f"{proc.returncode}:\n{proc.stderr.strip()}"
            )

        raw_json = self._coerce_json(raw_stdout)
        result_text = str(raw_json.get("result", raw_stdout))
        usage = raw_json.get("usage") if isinstance(raw_json.get("usage"), dict) else {}

        session_value = raw_json.get("session_id") or raw_json.get("sessionId")
        total_cost = raw_json.get("total_cost_usd") or raw_json.get("cost_usd")
        total_tokens = raw_json.get("total_tokens")
        input_tokens = raw_json.get("input_tokens") or usage.get("input_tokens")
        output_tokens = raw_json.get("output_tokens") or usage.get("output_tokens")

        return ClaudeResult(
            raw_stdout=raw_stdout,
            raw_json=raw_json,
            result=result_text,
            session_id=str(session_value) if session_value is not None else None,
            total_cost_usd=float(total_cost) if total_cost is not None else None,
            total_tokens=int(total_tokens) if total_tokens is not None else None,
            input_tokens=int(input_tokens) if input_tokens is not None else None,
            output_tokens=int(output_tokens) if output_tokens is not None else None,
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
    ) -> ClaudeResult:
        """Run claude -p with ``--output-format stream-json`` and parse
        the event stream. Each line of stdout is one JSON event; we
        collect ``content_block_delta`` text fragments into ``result``,
        and every ``tool_use`` block into ``tool_calls``.
        """
        # Inject the active conda env's bin/ at the front of PATH so
        # MCP servers spawned by the claude CLI subprocess inherit the
        # right Python (with rdkit, matplotlib, etc.). Without this,
        # ``$PATH`` resolves to base env's python inside the MCP child
        # process even when the runner itself is launched from
        # ``conda activate scimas`` — see #rdkit-not-installed in the
        # post-port batch results.
        env = os.environ.copy()
        conda_prefix = env.get("CONDA_PREFIX", "")
        if conda_prefix and conda_prefix != "/opt/anaconda3":
            scimas_bin = os.path.join(conda_prefix, "bin")
            env["PATH"] = scimas_bin + os.pathsep + env.get("PATH", "")
            env["CONDA_PREFIX"] = conda_prefix
            env["CONDA_DEFAULT_ENV"] = env.get("CONDA_DEFAULT_ENV", "scimas")
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
        # text.
        usage = final.get("usage", {}) if isinstance(final.get("usage"), dict) else {}
        session_value = final.get("session_id") or final.get("sessionId")
        total_cost = final.get("total_cost_usd") or final.get("cost_usd")
        total_tokens = final.get("total_tokens")
        input_tokens = final.get("input_tokens") or usage.get("input_tokens")
        output_tokens = final.get("output_tokens") or usage.get("output_tokens")
        result_text = (
            final.get("result")
            if isinstance(final, dict) and final.get("result")
            else "".join(text_chunks)
        )
        return ClaudeResult(
            raw_stdout="\n".join(raw_lines),
            raw_json=final if final else {"stream": True, "tool_calls": tool_calls},
            result=str(result_text or ""),
            session_id=str(session_value) if session_value is not None else None,
            total_cost_usd=float(total_cost) if total_cost is not None else None,
            total_tokens=int(total_tokens) if total_tokens is not None else None,
            input_tokens=int(input_tokens) if input_tokens is not None else None,
            output_tokens=int(output_tokens) if output_tokens is not None else None,
            stderr=proc.stderr.read() if proc.stderr else "",
            started_at=started_at,
            duration_ms=duration_ms,
            tool_calls=tool_calls,
        )
