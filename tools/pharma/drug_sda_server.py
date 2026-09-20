#!/usr/bin/env python3
"""MCP server wrapper for the remote DrugSDA-Tool SCP Server.

Acts as a proxy between Claude's MCP client and the remote endpoint: it
receives MCP tool calls over stdio and forwards them using the
streamable-http transport.

Every tool the remote exposes is registered here — 81 of them, sourced from
``_tool_schemas.json``, which ``scripts/drug_sda_codegen.py`` writes from the
remote's own ``tools/list``. Registering rather than hand-writing them keeps
the names, descriptions and argument schemas faithful, and lets the remote's
inventory change without editing this file.

Splitting 81 tools across focused ``pharma-drug-sda-*`` skills is what keeps
any single agent step's tool list small; this server having them all is fine,
because Claude only sees the ones the skill layer puts in ``--allowedTools``.

Usage:
    python tools/pharma/drug_sda_server.py
"""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Optional

# Make the sciMAS repo root importable when this file is launched as a
# script via ``python tools/pharma/drug_sda_server.py`` — which is exactly
# how config/mcp.json starts it. Without this the ``from tools.pharma...``
# import below fails with ModuleNotFoundError, because sys.path[0] points
# at tools/pharma/ instead of the repo root. Every other server in this
# repo carries the same bootstrap for the same reason; importing this
# module as a package (tests do) leaves it a no-op.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mcp.server.mcpserver import MCPServer
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from tools.pharma._sda_config import (
    REMOTE_SERVER_URL,
    default_timeout_for,
    resolve_api_key,
)

mcp = MCPServer("pharma-drug-sda")

SCHEMAS_PATH = Path(__file__).resolve().parent / "_tool_schemas.json"

# The runner's per-`claude -p` ceiling is 600s (claude_runner.py). A tool that
# overruns it takes the whole agent step down, so no useful timeout exceeds
# this; heavy tools default below it and fail cleanly instead.
RUNNER_CEILING_SECONDS = 600.0


# --- JSON Schema -> Python annotation -------------------------------------
#
# Measured against the live endpoint: of 395 parameters, 302 are scalars, 75
# are `anyOf: [X, null]`, 17 are arrays of scalars, 1 is an array of objects,
# and there are no `$defs`/`$ref` or nested object types at all. So this
# mapping is faithful rather than best-effort.

_SCALARS: dict[str, Any] = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    "object": dict,
}


def _annotation(schema: dict, name: str) -> Any:
    """Python annotation for one JSON Schema property."""
    if not isinstance(schema, dict):
        return Any

    variants = schema.get("anyOf") or schema.get("oneOf")
    if isinstance(variants, list):
        concrete = [v for v in variants if (v or {}).get("type") != "null"]
        optional = len(concrete) != len(variants)
        inner = _annotation(concrete[0], name) if len(concrete) == 1 else Any
        return Optional[inner] if optional else inner

    kind = schema.get("type")
    if isinstance(kind, list):
        concrete = [k for k in kind if k != "null"]
        inner = _SCALARS.get(concrete[0], Any) if len(concrete) == 1 else Any
        return Optional[inner] if len(concrete) != len(kind) else inner

    if kind == "array":
        items = schema.get("items") or {}
        item_type = _SCALARS.get(items.get("type"), Any)
        if items.get("type") == "object" or "properties" in items:
            item_type = dict
        return list[item_type]

    return _SCALARS.get(kind, Any)


def _parameters(schema: dict) -> list[inspect.Parameter]:
    """Build the keyword parameters for one remote tool's input schema."""
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or ())
    params: list[inspect.Parameter] = []

    for name, prop in properties.items():
        prop = prop if isinstance(prop, dict) else {}
        annotation = _annotation(prop, name)
        if name in required:
            default: Any = inspect.Parameter.empty
        elif "default" in prop:
            default = prop["default"]
        else:
            default = None
            # A non-required property with no default is still optional.
            if not _is_optional(annotation):
                annotation = Optional[annotation]
        params.append(
            inspect.Parameter(
                name,
                inspect.Parameter.KEYWORD_ONLY,
                annotation=annotation,
                default=default,
            )
        )

    params.append(
        inspect.Parameter(
            "timeout_seconds",
            inspect.Parameter.KEYWORD_ONLY,
            annotation=float,
            default=default_timeout_for(schema.get("_tool_name", "")),
        )
    )
    return params


def _is_optional(annotation: Any) -> bool:
    return getattr(annotation, "__origin__", None) is not None and type(None) in getattr(
        annotation, "__args__", ()
    )


def _sub_exceptions(exc: BaseException) -> tuple[BaseException, ...] | None:
    """The members of `exc` if it is an exception group, else None.

    Duck-typed on `.exceptions` rather than `isinstance(exc, BaseExceptionGroup)`:
    that builtin arrived in Python 3.11 and this server is also deployed on
    3.10, where the name does not exist and `isinstance` would raise `NameError`
    from inside the error handler. The `exceptiongroup` backport and the 3.11
    builtin both expose `.exceptions`.
    """
    members = getattr(exc, "exceptions", None)
    if isinstance(members, (list, tuple)) and members:
        return tuple(members)
    return None


def _is_timeout(exc: BaseException) -> bool:
    """True if `exc` is a timeout, unwrapping anyio task-group wrappers.

    Cancelling a call inside the transport's task group makes the context
    managers exit with an `ExceptionGroup` whose leaf is the real
    `TimeoutError`, so a plain `except TimeoutError` never fires.
    """
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return True
    members = _sub_exceptions(exc)
    if members is not None:
        return any(_is_timeout(sub) for sub in members)
    cause = exc.__cause__
    return cause is not None and cause is not exc and _is_timeout(cause)


def _leaf_error(exc: BaseException) -> BaseException:
    """Dig the real failure out of nested task-group wrappers."""
    seen: set[int] = set()
    while True:
        if id(exc) in seen:
            return exc
        seen.add(id(exc))
        members = _sub_exceptions(exc)
        if members is not None:
            exc = members[0]
            continue
        if exc.__cause__ is not None and exc.__cause__ is not exc:
            exc = exc.__cause__
            continue
        return exc


# --- Remote proxy ---------------------------------------------------------


class _DrugSDAProxy:
    """Forwards a tool call to the remote server.

    Uses a new connection per call. That is a round trip more than a pooled
    session, but it keeps failures isolated: a dropped stream costs one call
    rather than poisoning every later one.
    """

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        timeout_seconds: float | None = None,
    ) -> Any:
        """Call a tool on the remote server and return its parsed result."""
        import httpx2

        from mcp.client.streamable_http import create_mcp_http_client

        api_key = resolve_api_key()
        budget = (
            float(timeout_seconds)
            if timeout_seconds is not None
            else default_timeout_for(tool_name)
        )

        # A bare `httpx2.AsyncClient()` defaults to a **5s** timeout, which
        # overrides the transport's recommended 300s read timeout and kills
        # any slow tool almost immediately. Build it through
        # `create_mcp_http_client` and set the read timeout to sit just past
        # our own budget, so `asyncio.wait_for` below is what fires first and
        # the agent gets the actionable timeout message.
        client = create_mcp_http_client(
            headers={
                "SCP-HUB-API-KEY": api_key or "",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            timeout=httpx2.Timeout(30.0, read=budget + 30.0),
        )

        try:
            async with streamable_http_client(
                url=REMOTE_SERVER_URL,
                http_client=client,
            ) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    result = await asyncio.wait_for(
                        session.call_tool(tool_name, arguments),
                        timeout=budget,
                    )
                    if hasattr(result, "content") and result.content:
                        content = result.content[0]
                        if hasattr(content, "text"):
                            try:
                                return json.loads(content.text)
                            except json.JSONDecodeError:
                                return {"result": content.text}
                    return {"result": "Success"}
        except BaseException as e:
            if _is_timeout(e):
                # Not an error in the tool — it is still running remotely. Say
                # so, and give the agent a lever rather than a dead end.
                return {
                    "error": (
                        f"Remote tool '{tool_name}' did not finish within "
                        f"{budget:.0f}s. The job may still be running on the "
                        f"remote server. Retry with a larger timeout_seconds "
                        f"(the runner kills the step at "
                        f"{RUNNER_CEILING_SECONDS:.0f}s, so that is the "
                        f"practical ceiling), or treat the result as "
                        f"unavailable rather than reporting a partial one."
                    ),
                    "timed_out": True,
                    "timeout_seconds": budget,
                }
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            # An ExceptionGroup from the transport's task group is noise around
            # one real failure; report the leaf, not the wrapper.
            leaf = _leaf_error(e)
            import traceback

            traceback.print_exc(file=sys.stderr)
            # httpx2's connection errors stringify to "", which would reach the
            # agent as a blank error; the class name is the actionable part.
            return {
                "error": str(leaf) or type(leaf).__name__,
                "traceback": traceback.format_exc(),
            }
        finally:
            await client.aclose()


_proxy = _DrugSDAProxy()


def _make_tool(tool_name: str, description: str, schema: dict) -> Any:
    """Build a coroutine that forwards to one remote tool."""
    # The MCP layer fills in every declared default before calling us, so an
    # argument the agent never mentioned arrives as `None`. Forwarding that
    # would tell the remote "explicitly null" where it expects "absent", and
    # it would then skip its own default. Drop those keys instead, keeping any
    # `None` the remote itself listed as required (there, null is meaningful).
    required = frozenset(schema.get("required") or ())

    async def _tool(**kwargs: Any) -> str:
        timeout_seconds = kwargs.pop("timeout_seconds", None)
        arguments = {
            key: value
            for key, value in kwargs.items()
            if value is not None or key in required
        }
        result = await _proxy.call_tool(tool_name, arguments, timeout_seconds)
        return json.dumps(result, ensure_ascii=False, indent=2)

    _tool.__name__ = tool_name
    _tool.__qualname__ = tool_name
    _tool.__doc__ = description
    schema_with_name = dict(schema or {})
    schema_with_name["_tool_name"] = tool_name
    _tool.__signature__ = inspect.Signature(_parameters(schema_with_name))
    return _tool


def _register_all() -> tuple[int, str | None]:
    """Register every tool described by the generated schema cache."""
    if not SCHEMAS_PATH.exists():
        return 0, (
            f"{SCHEMAS_PATH.name} is missing, so no DrugSDA tools are available. "
            f"Run: python scripts/drug_sda_codegen.py"
        )
    try:
        schemas = json.loads(SCHEMAS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return 0, f"cannot read {SCHEMAS_PATH.name}: {exc}"

    for tool_name, spec in sorted(schemas.items()):
        fn = _make_tool(tool_name, spec.get("description", ""), spec.get("input_schema"))
        mcp.add_tool(
            fn,
            name=tool_name,
            description=spec.get("description", ""),
            structured_output=False,
        )
        # Expose as a module global so `from tools.pharma import drug_sda_server`
        # then `drug_sda_server.<tool>(...)` keeps working for the standalone
        # scripts and tests that call tools directly.
        globals()[tool_name] = fn
    return len(schemas), None


_TOOL_COUNT, _REGISTRATION_WARNING = _register_all()
if _REGISTRATION_WARNING:
    # stderr only: stdout is the stdio JSON-RPC channel.
    print(f"drug_sda_server: {_REGISTRATION_WARNING}", file=sys.stderr)


if __name__ == "__main__":
    try:
        asyncio.run(mcp.run_stdio_async())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
