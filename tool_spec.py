"""AST-based extractor for `@mcp.tool(...)`-decorated tool signatures.

The orchestrator injects a one-line signature + short description per tool into
both the skill-selection catalog (`_skill_selection_prompt`) and the executing
agent's prompt (`_step_prompt`). Without this, agents have to read server source
files mid-run to learn a tool's calling convention.

Each `*_server.py` in `tools/{chemistry,physics,biology,mathematics,pharma,web,
biomni}/` follows the same shape:

    from mcp.server.mcpserver import MCPServer
    mcp = MCPServer("<server-name>")

    @mcp.tool(description="...")            # or parenthesized multi-line form
    async def tool_name(arg1: float, ...) -> str:
        ...

We extract:
- tool name (function name)
- mcp name (`mcp__<server>__<tool>`)  -- the form Claude CLI accepts via
  `--allowedTools` and emits in `tool_use` blocks.
- signature string `(name: type, ...) -> ret`
- description (the `@mcp.tool(description=...)` value, possibly multi-line)
- ordered parameter list `[(name, type_str), ...]`

Per-file failures (SyntaxError, OSError) print to stderr and continue; the
indexer never raises on a single bad file.
"""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_SERVER_NAME_RE = re.compile(r"^(\w+)\s*=\s*MCPServer\(\s*[\"']([^\"']+)[\"']", re.MULTILINE)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    server: str
    mcp_name: str
    signature: str
    description: str
    parameters: Tuple[Tuple[str, str], ...]

    def one_liner(self) -> str:
        """Compact `name(params) -> ret` form for the skill-selection catalog."""
        return f"{self.name}{self.signature}"


def _server_name_from_instance(server_file: Path, instance: str) -> Optional[str]:
    """Look up the MCPServer("<name>") binding for the given instance name."""
    try:
        source = server_file.read_text(encoding="utf-8")
    except OSError:
        return None
    for match in _SERVER_NAME_RE.finditer(source):
        if match.group(1) == instance:
            return match.group(2)
    return None


def _resolve_description(value: ast.AST) -> str:
    """Best-effort extraction of a tool's description string."""
    try:
        literal = ast.literal_eval(value)
        if isinstance(literal, str):
            return literal.strip()
    except (ValueError, SyntaxError):
        pass
    # Fallback: unparse the AST node. For parenthesized multi-line string
    # concatenation this is often a JoinedStr / BinOp chain that unparse will
    # render back to source. Strip whitespace and trailing semicolons.
    rendered = ast.unparse(value).strip()
    return rendered


def _annotation_str(node: Optional[ast.AST]) -> str:
    """Render an annotation AST node as a readable type string.

    With `from __future__ import annotations` every annotation is stored as a
    plain Name/Subscript; ast.unparse round-trips `list[dict]`, `dict[str, Any]`,
    `int | None` correctly.
    """
    if node is None:
        return "Any"
    try:
        return ast.unparse(node)
    except Exception:
        return "Any"


def _render_signature(func: ast.FunctionDef) -> str:
    """Build `(arg1: type, arg2: type = default, ...) -> ret` for an FunctionDef."""
    args = func.args
    parts: List[str] = []
    posonly = list(args.posonlyargs)
    regular = list(args.args)
    defaults = list(args.defaults)
    # Align defaults: defaults apply to the trailing N positional args.
    pad = len(regular) - len(defaults)
    for idx, arg in enumerate(posonly + regular):
        ann = _annotation_str(arg.annotation)
        default_idx = idx - pad
        if default_idx >= 0 and default_idx < len(defaults):
            try:
                default_repr = ast.unparse(defaults[default_idx])
            except Exception:
                default_repr = "..."
            parts.append(f"{arg.arg}: {ann} = {default_repr}")
        else:
            parts.append(f"{arg.arg}: {ann}")
        if arg in posonly and (posonly + regular).index(arg) == len(posonly) - 1:
            parts.append("/")
    if args.vararg is not None:
        ann = _annotation_str(args.vararg.annotation)
        parts.append(f"*{args.vararg.arg}: {ann}")
    elif args.kwonlyargs:
        parts.append("*")
    for kwarg, default in zip(args.kwonlyargs, args.kw_defaults):
        ann = _annotation_str(kwarg.annotation)
        if default is None:
            parts.append(f"{kwarg.arg}: {ann}")
        else:
            try:
                default_repr = ast.unparse(default)
            except Exception:
                default_repr = "..."
            parts.append(f"{kwarg.arg}: {ann} = {default_repr}")
    if args.kwarg is not None:
        ann = _annotation_str(args.kwarg.annotation)
        parts.append(f"**{args.kwarg.arg}: {ann}")
    ret = _annotation_str(func.returns) if func.returns is not None else "str"
    return f"({', '.join(parts)}) -> {ret}"


def _parameter_list(func: ast.FunctionDef) -> Tuple[Tuple[str, str], ...]:
    """Ordered (name, type_str) pairs for the FunctionDef's parameters."""
    out: List[Tuple[str, str]] = []
    for arg in list(func.args.posonlyargs) + list(func.args.args):
        out.append((arg.arg, _annotation_str(arg.annotation)))
    if func.args.vararg is not None:
        out.append((f"*{func.args.vararg.arg}", _annotation_str(func.args.vararg.annotation)))
    for kwarg in func.args.kwonlyargs:
        out.append((kwarg.arg, _annotation_str(kwarg.annotation)))
    if func.args.kwarg is not None:
        out.append((f"**{func.args.kwarg.arg}", _annotation_str(func.args.kwarg.annotation)))
    return tuple(out)


def _is_mcp_tool_decorator(decorator: ast.AST) -> bool:
    """Match `@<anything>.tool(...)` decorators (covers @mcp.tool, @server.tool)."""
    if not isinstance(decorator, ast.Call):
        return False
    func = decorator.func
    return isinstance(func, ast.Attribute) and func.attr == "tool"


def extract_tool_specs(server_file: Path) -> List[ToolSpec]:
    """Parse a single `*_server.py` file. Returns one ToolSpec per @mcp.tool.

    Per-file failures are caught inside the higher-level `build_tool_spec_index`;
    this function raises whatever the parser raises.
    """
    source = server_file.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(server_file))

    specs: List[ToolSpec] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.decorator_list:
            continue
        tool_decorator: Optional[ast.Call] = None
        for d in node.decorator_list:
            if _is_mcp_tool_decorator(d):
                tool_decorator = d
                break
        if tool_decorator is None:
            continue

        # Resolve the instance name (mcp / server / ...) and look up its MCPServer binding.
        decorator_attr = tool_decorator.func  # ast.Attribute
        if not isinstance(decorator_attr.value, ast.Name):
            continue
        instance = decorator_attr.value.id
        server_name = _server_name_from_instance(server_file, instance)
        if server_name is None:
            # Couldn't find a `mcp = MCPServer("...")` binding; skip the tool.
            continue

        description = ""
        for kw in tool_decorator.keywords:
            if kw.arg == "description" and kw.value is not None:
                description = _resolve_description(kw.value)
                break

        tool_name = node.name
        signature = _render_signature(node)
        params = _parameter_list(node)
        specs.append(
            ToolSpec(
                name=tool_name,
                server=server_name,
                mcp_name=f"mcp__{server_name}__{tool_name}",
                signature=signature,
                description=description,
                parameters=params,
            )
        )
    return specs


_TOOLS_SUBDIRS = (
    "chemistry",
    "physics",
    "biology",
    "mathematics",
    "pharma",
    "web",
    "biomni",
)


def build_tool_spec_index(tools_dir: Path) -> Dict[str, ToolSpec]:
    """Walk every `<sub>/<sub>_*_server.py` under `tools_dir` and return mcp_name -> ToolSpec.

    Failures (missing dir, SyntaxError, OSError) are non-fatal: print to stderr
    and continue. Returns an empty dict if `tools_dir` does not exist.
    """
    if not tools_dir.is_dir():
        return {}
    index: Dict[str, ToolSpec] = {}
    for sub in _TOOLS_SUBDIRS:
        sub_dir = tools_dir / sub
        if not sub_dir.is_dir():
            continue
        for server_file in sorted(sub_dir.glob("*_server.py")):
            try:
                specs = extract_tool_specs(server_file)
            except (SyntaxError, OSError, ValueError) as exc:
                print(
                    f"tool_spec: skipping {server_file}: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                continue
            for spec in specs:
                # On collisions (shouldn't happen — tool names are server-bound),
                # last writer wins. Surface to stderr so we can rename.
                if spec.mcp_name in index:
                    print(
                        f"tool_spec: duplicate mcp_name {spec.mcp_name} "
                        f"in {server_file}, overwriting",
                        file=sys.stderr,
                    )
                index[spec.mcp_name] = spec
    return index


if __name__ == "__main__":  # pragma: no cover -- manual smoke entry
    import json

    base = Path(__file__).resolve().parent / "tools"
    idx = build_tool_spec_index(base)
    print(json.dumps(
        {"n_tools": len(idx),
         "sample": {k: idx[k].one_liner() for k in list(idx)[:5]}},
        indent=2,
    ))