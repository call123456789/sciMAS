"""Restricted Python workflow DSL support for sciMAS planners."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional


MAX_LOOP_RANGE = 10


# A fenced code block, optionally language-tagged. Matches the pattern
# already proven in `scripts/smdd_official_eval.py`. Non-greedy, so a reply
# carrying several blocks yields several matches.
_FENCE_RE = re.compile(r"```([^\n`]*)\n?(.*?)```", re.S)
# Where a workflow function starts, wherever it sits in the reply.
_WORKFLOW_DEF_RE = re.compile(r"^[ \t]*async[ \t]+def[ \t]+workflow[ \t]*\(", re.M)
_PYTHON_FENCE_LANGS = {"", "py", "python", "python3"}
# How far back from the end of a reply we will hunt for the last line of
# code. Bounds the line-by-line trim in `_trim_to_parseable`.
_MAX_TRIM_LINES = 200


class WorkflowDSLValidationError(ValueError):
    """Raised when planner-produced workflow source is outside the DSL."""


@dataclass
class WorkflowDSLProgram:
    source: str
    tree: ast.Module
    static_trace: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class WorkflowValue:
    """Agent result wrapper that preserves workflow data dependencies."""

    node_id: str
    value: Any
    role: str = ""
    # The agent's reply verbatim. ``value`` may be a JSON fragment lifted out of
    # that reply (see ``coerce_agent_result``), which is useful for downstream
    # ``verification["passed"]`` lookups but is a poor final answer: prose that
    # happens to contain "[0, 1]" would be reported as the answer. Keeping the
    # original lets the formatter fall back to what the agent actually said.
    text: str = ""

    def __getitem__(self, key: Any) -> Any:
        return self.value[key]

    def __bool__(self) -> bool:
        return bool(self.value)

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return repr(self.value)

    def get(self, key: Any, default: Any = None) -> Any:
        if hasattr(self.value, "get"):
            return self.value.get(key, default)
        return default


def _fence_bodies(text: str) -> list[str]:
    """Return the body of every fenced block, most workflow-like first.

    Planners routinely wrap the workflow in prose ("I'll design a compact
    workflow: ...") and/or a ```python fence. Ranking blocks by how much they
    look like the workflow means we find the code even when the model also
    fences an example or a JSON snippet.
    """

    ranked: list[tuple[int, int, str]] = []
    for order, match in enumerate(_FENCE_RE.finditer(text)):
        lang = match.group(1).strip().lower().split()[0] if match.group(1).strip() else ""
        body = match.group(2).strip()
        if not body:
            continue
        rank = 0
        if _WORKFLOW_DEF_RE.search(body):
            rank -= 2
        if lang in _PYTHON_FENCE_LANGS:
            rank -= 1
        ranked.append((rank, order, body))
    ranked.sort(key=lambda item: (item[0], item[1]))

    # A reply truncated mid-block has no closing fence, so `_FENCE_RE` never
    # matches it. An odd number of ``` markers means the last one opened a
    # block that was never closed; everything after it is the body.
    markers = [match.start() for match in re.finditer(r"```", text)]
    if len(markers) % 2 == 1:
        tail = text[markers[-1] + 3 :]
        newline = tail.find("\n")
        tail = (tail[newline + 1 :] if newline != -1 else tail).strip()
        if tail:
            ranked.append((0, len(ranked), tail))

    return [body for _rank, _order, body in ranked]


def _trim_to_parseable(candidate: str) -> Optional[str]:
    """Drop trailing prose from `candidate` until it parses as Python.

    Walks from the longest prefix down, returning the first prefix that
    `ast.parse` accepts. Removing trailing lines one at a time steps over
    artefacts models append after the code ("(no final answer)", "Let me
    know if you need anything else!").
    """

    lines = candidate.splitlines()
    lower = max(0, len(lines) - _MAX_TRIM_LINES)
    for end in range(len(lines), lower - 1, -1):
        attempt = "\n".join(lines[:end]).strip()
        if not attempt:
            continue
        try:
            ast.parse(attempt, mode="exec")
        except SyntaxError:
            continue
        return attempt
    return None


def extract_workflow_source(text: str) -> str:
    """Return raw workflow source, tolerating prose and markdown fences.

    Returns the first candidate that parses as Python so downstream callers
    see clean source. When nothing parses, returns the best-effort candidate
    rather than the untouched reply, so the validation error names the real
    DSL problem instead of a stray apostrophe in the surrounding prose.
    """

    stripped = text.strip()
    if not stripped:
        return ""

    candidates: list[str] = [stripped]
    fenced = _fence_bodies(stripped)
    candidates.extend(fenced)

    # Code that follows the prose directly, without any fence at all.
    start = _WORKFLOW_DEF_RE.search(stripped)
    if start:
        candidates.append(stripped[start.start():].strip())

    for candidate in candidates:
        try:
            ast.parse(candidate, mode="exec")
        except SyntaxError:
            continue
        return candidate

    # Nothing parsed cleanly. Try trimming trailing prose off the two
    # likeliest candidates before falling back.
    for candidate in (fenced[0] if fenced else "", candidates[-1] if start else ""):
        if not candidate:
            continue
        trimmed = _trim_to_parseable(candidate)
        if trimmed:
            return trimmed

    if fenced:
        return fenced[0]
    if start:
        return stripped[start.start():].strip()
    return stripped


def _balanced_json_spans(text: str) -> list[str]:
    """Return every top-level ``{...}`` / ``[...]`` span, longest first.

    Unlike the brace scanner in ``tests/llm_judge.py``, this one tracks
    string literals and backslash escapes, so a brace inside a JSON string
    (the usual case, since ``fixed_source`` carries Python source) does not
    throw off the depth count. Sorting longest-first yields the outer object
    rather than one of its nested values.
    """

    spans: list[str] = []
    stack: list[str] = []
    start: Optional[int] = None
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            if not stack:
                start = index
            stack.append(char)
        elif char in "}]":
            if stack:
                opener = stack.pop()
                if (char == "}") != (opener == "{"):
                    # Mismatched closer: the span is not well formed.
                    stack.clear()
                    start = None
                    continue
                if not stack and start is not None:
                    spans.append(text[start : index + 1])
                    start = None
    spans.sort(key=len, reverse=True)
    return spans


def _escape_raw_control_chars(text: str) -> str:
    """Escape control characters that appear *inside* JSON string literals.

    Models habitually write a multi-line ``fixed_source`` as a real newline
    rather than ``\\n``, which is invalid JSON but trivially repairable. Only
    characters inside a string literal are touched, so structural whitespace
    between tokens is left alone.
    """

    out: list[str] = []
    in_string = False
    escaped = False
    replacements = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}
    for char in text:
        if in_string:
            if escaped:
                escaped = False
                out.append(char)
                continue
            if char == "\\":
                escaped = True
                out.append(char)
                continue
            if char == '"':
                in_string = False
                out.append(char)
                continue
            out.append(replacements.get(char, char))
            continue
        if char == '"':
            in_string = True
        out.append(char)
    return "".join(out)


def _first_decodable(candidates: Iterable[str]) -> tuple[bool, Any]:
    """Return the first candidate that decodes, then after control-char repair."""
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        for variant in (candidate, _escape_raw_control_chars(candidate)):
            try:
                return True, json.loads(variant)
            except json.JSONDecodeError:
                continue
    return False, None


def extract_json_value(text: str) -> tuple[bool, Any]:
    """Return ``(found, value)`` for JSON embedded in model output.

    Reviewers, planners and agents are all asked for "strict JSON only" but
    routinely add a sentence before or after it, wrap it in a fence, or emit
    unescaped newlines inside a string value. Each candidate is tried verbatim
    and then after control-character repair; the first one that decodes wins.
    Preserves a top-level array or scalar, so callers that need the raw shape
    (agent results) are not forced into a wrapper object.

    Deliberately *lenient*: it will also take a bare ``[...]``/``{...}`` span
    out of running prose. That is what makes ``verification["passed"]`` work
    when a reply opens with a sentence, but it also means a sentence like
    "normalized to [0, 1]" decodes to the list ``[0, 1]``. Callers that need
    the reply's *payload* rather than a convenient fragment want
    ``extract_explicit_json_value``.
    """
    if not text:
        return False, None

    stripped = text.strip()
    candidates: list[str] = [stripped]
    candidates.extend(_fence_bodies(stripped))
    candidates.extend(_balanced_json_spans(stripped))
    return _first_decodable(candidates)


def extract_explicit_json_value(text: str) -> tuple[bool, Any]:
    """Return ``(found, value)`` for JSON the reply *declares* as its payload.

    Only the whole reply or a fenced block counts — the ``_balanced_json_spans``
    fallback is left out on purpose. A fence is the model saying "this block is
    the data"; an unadorned ``[0, 1]`` inside a sentence is a coincidence of
    punctuation, and treating it as the payload silently replaces a prose
    answer (see ``orchestrator._format_workflow_result``).
    """
    if not text:
        return False, None

    stripped = text.strip()
    candidates: list[str] = [stripped]
    candidates.extend(_fence_bodies(stripped))
    return _first_decodable(candidates)


def extract_json_object(text: str) -> dict[str, Any]:
    """Best-effort extraction of one JSON object from model output.

    The dict-shaped view of `extract_json_value`: a non-object payload is
    wrapped as ``{"result": ...}`` and total failure yields ``{}``, so callers
    can keep a plain falsy check.
    """

    found, value = extract_json_value(text)
    if not found:
        return {}
    return value if isinstance(value, dict) else {"result": value}


def parse_and_validate_workflow(
    source: str,
    role_check: Optional[Callable[[str], bool]] = None,
    allowed_roles: Optional[Iterable[str]] = None,
) -> WorkflowDSLProgram:
    source = extract_workflow_source(source)
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise WorkflowDSLValidationError(f"Invalid workflow Python: {exc}") from exc

    validator = _WorkflowValidator(
        role_check=role_check,
        allowed_roles=allowed_roles,
    )
    validator.validate(tree)
    return WorkflowDSLProgram(
        source=source,
        tree=tree,
        static_trace=validator.static_trace,
    )


def unwrap_workflow_value(value: Any) -> Any:
    if isinstance(value, WorkflowValue):
        return unwrap_workflow_value(value.value)
    if isinstance(value, list):
        return [unwrap_workflow_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(unwrap_workflow_value(item) for item in value)
    if isinstance(value, dict):
        return {
            unwrap_workflow_value(key): unwrap_workflow_value(item)
            for key, item in value.items()
        }
    return value


def collect_workflow_dependencies(value: Any) -> list[str]:
    return [item.node_id for item in collect_workflow_values(value)]


def collect_workflow_values(value: Any) -> list[WorkflowValue]:
    values: list[WorkflowValue] = []
    seen: set[str] = set()

    def walk(item: Any) -> None:
        if isinstance(item, WorkflowValue):
            if item.node_id not in seen:
                values.append(item)
                seen.add(item.node_id)
            walk(item.value)
        elif isinstance(item, dict):
            for key, val in item.items():
                walk(key)
                walk(val)
        elif isinstance(item, (list, tuple, set)):
            for child in item:
                walk(child)

    walk(value)
    return values


def coerce_agent_result(text: str) -> Any:
    """Parse an agent's text reply into JSON when it contains JSON.

    Every agent instruction ends in "Return strict JSON", but replies arrive
    with a preamble or a fence often enough that stripping only a leading
    fence silently handed back raw strings. Shares `extract_json_object` with
    the reviewer/planner paths so all of them tolerate the same shapes.
    """

    found, value = extract_json_value(text)
    return value if found else text


def make_json_safe(value: Any) -> Any:
    value = unwrap_workflow_value(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(make_json_safe(key)): make_json_safe(item) for key, item in value.items()}
    return str(value)


def workflow_guarantees_return(statements: list[ast.stmt]) -> bool:
    """True when every path through the workflow body ends in a ``return``.

    A body that can finish without returning yields ``None``, which is not "no
    answer" anywhere downstream: ``_format_workflow_result`` serializes it and
    the judge is handed the four-character answer ``null``. Measured on
    2026-09-20 — a planner emitted a workflow whose last statement was
    ``if review["passed"]: final = await agent(...)`` with no ``else``, the
    review came back false, and the run burned 30 tool calls to answer nothing.

    Only the last statement can let control escape, so only it is inspected.
    The DSL has no ``while``/``try``/``with``, so the forms to consider are:
    ``return`` (guarantees), ``if`` whose *both* arms guarantee (a missing
    ``else`` falls through), and everything else — including a ``for``, whose
    body may never execute because ``range(0)`` is a legal loop.
    """
    if not statements:
        return False
    last = statements[-1]
    if isinstance(last, ast.Return):
        return True
    if isinstance(last, ast.If):
        return (
            bool(last.orelse)
            and workflow_guarantees_return(last.body)
            and workflow_guarantees_return(last.orelse)
        )
    return False


class _WorkflowValidator:
    _reserved_names = {"agent", "parallel", "range"}

    def __init__(
        self,
        role_check: Optional[Callable[[str], bool]] = None,
        allowed_roles: Optional[Iterable[str]] = None,
    ) -> None:
        self.static_trace: list[dict[str, Any]] = []
        self._assigned_names: set[str] = {"task"}
        # Optional hook supplied by the orchestrator, which owns role
        # normalization (ROLE_ALIASES lives there; importing it here would be
        # circular). When absent, roles are recorded but not checked.
        self._role_check = role_check
        self._allowed_roles = sorted({role for role in (allowed_roles or []) if role})

    def validate(self, tree: ast.Module) -> None:
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.AsyncFunctionDef):
            raise WorkflowDSLValidationError(
                "Workflow DSL must contain exactly one top-level async function."
            )
        func = tree.body[0]
        if func.name != "workflow":
            raise WorkflowDSLValidationError("The workflow function must be named workflow.")
        if func.decorator_list:
            raise WorkflowDSLValidationError("Decorators are not allowed in workflow DSL.")
        if func.returns is not None:
            raise WorkflowDSLValidationError("Return annotations are not allowed.")
        self._validate_arguments(func.args)
        self._collect_assigned_names(func.body)
        self._validate_block(func.body, loop_depth=0)
        # Checked after `_validate_block` so a body with a deeper problem (an
        # unsupported statement, a bad range) reports that first — the planner's
        # reviewer only gets one error string per round.
        if not workflow_guarantees_return(func.body):
            raise WorkflowDSLValidationError(
                "The workflow must end by returning its answer, and every path "
                "through the body must reach a `return <value>`. It does not: "
                "control can fall off the end, which makes the run's answer "
                "None — reported downstream as the literal answer `null`. Make "
                "the last statement a `return`, and if it is an `if` that "
                "guards the answer, add an `else` that returns a value too."
            )

    def _validate_arguments(self, args: ast.arguments) -> None:
        if (
            len(args.args) != 1
            or args.args[0].arg != "task"
            or args.posonlyargs
            or args.kwonlyargs
            or args.vararg
            or args.kwarg
            or args.defaults
            or args.kw_defaults
        ):
            raise WorkflowDSLValidationError(
                "workflow must have exactly one positional argument: task."
            )

    def _collect_assigned_names(self, statements: list[ast.stmt]) -> None:
        for stmt in statements:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    self._collect_target_names(target)
            elif isinstance(stmt, ast.For):
                self._collect_target_names(stmt.target)
                self._collect_assigned_names(stmt.body)
                self._collect_assigned_names(stmt.orelse)
            elif isinstance(stmt, ast.If):
                self._collect_assigned_names(stmt.body)
                self._collect_assigned_names(stmt.orelse)

    def _collect_target_names(self, target: ast.expr) -> None:
        if isinstance(target, ast.Name):
            self._assigned_names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                self._collect_target_names(item)

    def _validate_block(self, statements: list[ast.stmt], loop_depth: int) -> None:
        if not statements:
            raise WorkflowDSLValidationError("Empty workflow blocks are not allowed.")
        for stmt in statements:
            self._validate_statement(stmt, loop_depth)

    def _validate_statement(self, stmt: ast.stmt, loop_depth: int) -> None:
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1:
                raise WorkflowDSLValidationError("Only single-target assignments are allowed.")
            self._validate_target(stmt.targets[0])
            self._validate_expr(stmt.value, allow_await=True)
            return
        if isinstance(stmt, ast.Expr):
            self._validate_expr(stmt.value, allow_await=True)
            return
        if isinstance(stmt, ast.Return):
            if stmt.value is None:
                raise WorkflowDSLValidationError("return must include a value.")
            self.static_trace.append({"event": "return", "line": stmt.lineno})
            self._validate_expr(stmt.value, allow_await=False)
            return
        if isinstance(stmt, ast.If):
            self.static_trace.append({"event": "if", "line": stmt.lineno})
            self._validate_expr(stmt.test, allow_await=False)
            self._validate_block(stmt.body, loop_depth)
            if stmt.orelse:
                self._validate_block(stmt.orelse, loop_depth)
            return
        if isinstance(stmt, ast.For):
            self._validate_for(stmt, loop_depth)
            return
        if isinstance(stmt, ast.Break):
            if loop_depth <= 0:
                raise WorkflowDSLValidationError("break is only allowed inside for loops.")
            self.static_trace.append({"event": "break", "line": stmt.lineno})
            return
        raise WorkflowDSLValidationError(
            f"Unsupported statement: {type(stmt).__name__} at line {getattr(stmt, 'lineno', '?')}."
        )

    def _validate_for(self, stmt: ast.For, loop_depth: int) -> None:
        if stmt.orelse:
            raise WorkflowDSLValidationError("for/else is not allowed.")
        if not isinstance(stmt.target, ast.Name) or stmt.target.id != "_":
            raise WorkflowDSLValidationError("Only `for _ in range(N)` loops are allowed.")
        if not isinstance(stmt.iter, ast.Call):
            raise WorkflowDSLValidationError("Only `for _ in range(N)` loops are allowed.")
        self._validate_range_call(stmt.iter)
        count = stmt.iter.args[0].value
        self.static_trace.append({"event": "for", "line": stmt.lineno, "range": count})
        self._validate_block(stmt.body, loop_depth + 1)

    def _validate_target(self, target: ast.expr) -> None:
        if isinstance(target, ast.Name):
            self._validate_target_name(target.id)
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            if not target.elts:
                raise WorkflowDSLValidationError("Empty assignment unpacking is not allowed.")
            for item in target.elts:
                if not isinstance(item, ast.Name):
                    raise WorkflowDSLValidationError(
                        "Assignment unpacking may only contain names."
                    )
                self._validate_target_name(item.id)
            return
        raise WorkflowDSLValidationError("Assignment targets must be names or tuples of names.")

    def _validate_target_name(self, name: str) -> None:
        if name.startswith("__") or name in self._reserved_names:
            raise WorkflowDSLValidationError(f"Illegal assignment target: {name}.")

    def _validate_expr(self, node: ast.expr, *, allow_await: bool) -> None:
        if isinstance(node, ast.Await):
            if not allow_await:
                raise WorkflowDSLValidationError("await is only allowed in statements.")
            if not isinstance(node.value, ast.Call):
                raise WorkflowDSLValidationError("await may only wrap agent(...) or parallel(...).")
            self._validate_awaited_call(node.value)
            return
        if isinstance(node, ast.Call):
            raise WorkflowDSLValidationError(
                "Function calls are only allowed as await agent(...), "
                "await parallel(...), or range(N) in a for loop."
            )
        if isinstance(node, ast.Name):
            if node.id not in self._assigned_names:
                raise WorkflowDSLValidationError(f"Unknown name in workflow: {node.id}.")
            return
        if isinstance(node, ast.Constant):
            return
        if isinstance(node, (ast.List, ast.Tuple)):
            for item in node.elts:
                self._validate_expr(item, allow_await=False)
            return
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if key is not None:
                    self._validate_expr(key, allow_await=False)
                self._validate_expr(value, allow_await=False)
            return
        if isinstance(node, ast.Subscript):
            self._validate_expr(node.value, allow_await=False)
            self._validate_subscript_slice(node.slice)
            return
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            self._validate_expr(node.operand, allow_await=False)
            return
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            for value in node.values:
                self._validate_expr(value, allow_await=False)
            return
        if isinstance(node, ast.Compare):
            self._validate_expr(node.left, allow_await=False)
            for comparator in node.comparators:
                self._validate_expr(comparator, allow_await=False)
            if not all(
                isinstance(op, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE))
                for op in node.ops
            ):
                raise WorkflowDSLValidationError("Unsupported comparison operator.")
            return
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            self._validate_expr(node.left, allow_await=False)
            self._validate_expr(node.right, allow_await=False)
            return
        if isinstance(node, ast.Attribute):
            raise WorkflowDSLValidationError("Attribute access is not allowed.")
        raise WorkflowDSLValidationError(
            f"Unsupported expression: {type(node).__name__} at line {getattr(node, 'lineno', '?')}."
        )

    def _validate_subscript_slice(self, node: ast.expr) -> None:
        if isinstance(node, ast.Constant):
            return
        if isinstance(node, ast.Name):
            if node.id not in self._assigned_names:
                raise WorkflowDSLValidationError(f"Unknown subscript name: {node.id}.")
            return
        raise WorkflowDSLValidationError("Only simple constant/name subscripts are allowed.")

    def _validate_awaited_call(self, node: ast.Call) -> None:
        name = self._call_name(node)
        if name == "agent":
            self._validate_agent_call(node)
            return
        if name == "parallel":
            self._validate_parallel_call(node)
            return
        raise WorkflowDSLValidationError("Only agent(...) and parallel(...) may be awaited.")

    def _validate_agent_call(self, node: ast.Call) -> None:
        if self._call_name(node) != "agent":
            raise WorkflowDSLValidationError("Expected agent(...) call.")
        if node.args:
            raise WorkflowDSLValidationError("agent(...) only accepts keyword arguments.")
        allowed = {"role", "instruction", "input"}
        seen: set[str] = set()
        role = None
        instruction = None
        for keyword in node.keywords:
            if keyword.arg is None or keyword.arg not in allowed:
                raise WorkflowDSLValidationError("agent(...) only supports role, instruction, input.")
            if keyword.arg in seen:
                raise WorkflowDSLValidationError(f"Duplicate agent(...) keyword: {keyword.arg}.")
            seen.add(keyword.arg)
            if keyword.arg == "role":
                if not isinstance(keyword.value, ast.Constant) or not isinstance(keyword.value.value, str):
                    raise WorkflowDSLValidationError("agent role must be a string literal.")
                role = keyword.value.value
            elif keyword.arg == "instruction":
                if not isinstance(keyword.value, ast.Constant) or not isinstance(keyword.value.value, str):
                    raise WorkflowDSLValidationError("agent instruction must be a string literal.")
                instruction = keyword.value.value
            elif keyword.arg == "input":
                self._validate_expr(keyword.value, allow_await=False)
        if not role or not instruction:
            raise WorkflowDSLValidationError("agent(...) requires role and instruction.")
        if self._role_check is not None and not self._role_check(role):
            # Catching this here beats letting it surface later as a
            # FileNotFoundError from the missing prompt file, mid-execution.
            available = (
                f" Available roles: {', '.join(self._allowed_roles)}."
                if self._allowed_roles
                else ""
            )
            raise WorkflowDSLValidationError(
                f"Unknown agent role {role!r} at line {node.lineno}.{available}"
            )
        self.static_trace.append(
            {
                "event": "agent_call",
                "line": node.lineno,
                "role": role,
                "instruction": instruction,
            }
        )

    def _validate_parallel_call(self, node: ast.Call) -> None:
        if node.keywords:
            raise WorkflowDSLValidationError("parallel(...) does not accept keyword arguments.")
        if not node.args:
            raise WorkflowDSLValidationError("parallel(...) requires at least one agent call.")
        for arg in node.args:
            if not isinstance(arg, ast.Call) or self._call_name(arg) != "agent":
                raise WorkflowDSLValidationError(
                    "parallel(...) arguments must be direct agent(...) calls."
                )
            self._validate_agent_call(arg)
        self.static_trace.append(
            {"event": "parallel_call", "line": node.lineno, "width": len(node.args)}
        )

    def _validate_range_call(self, node: ast.Call) -> None:
        if self._call_name(node) != "range" or node.keywords or len(node.args) != 1:
            raise WorkflowDSLValidationError("Only range(N) is allowed in for loops.")
        arg = node.args[0]
        if not isinstance(arg, ast.Constant) or not isinstance(arg.value, int):
            raise WorkflowDSLValidationError("range(N) must use an integer literal.")
        if arg.value < 0 or arg.value > MAX_LOOP_RANGE:
            raise WorkflowDSLValidationError(
                f"range(N) must be between 0 and {MAX_LOOP_RANGE}."
            )

    def _call_name(self, node: ast.Call) -> str | None:
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            raise WorkflowDSLValidationError("Attribute calls are not allowed.")
        return None
