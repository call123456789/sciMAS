"""Data models for planning and execution."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class PlanStep:
    id: str
    role: str
    objective: str
    depends_on: List[str] = field(default_factory=list)
    expected_output: str = ""
    handoff_to: List[str] = field(default_factory=list)
    confidence: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanStep":
        return cls(
            id=str(data["id"]),
            role=str(data["role"]),
            objective=str(data.get("objective", "")),
            depends_on=[str(item) for item in data.get("depends_on", [])],
            expected_output=str(data.get("expected_output", "")),
            handoff_to=[str(item) for item in data.get("handoff_to", [])],
            confidence=data.get("confidence"),
        )


@dataclass
class ExecutionPlan:
    problem_summary: str
    topology_rationale: str = ""
    assumptions: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    steps: List[PlanStep] = field(default_factory=list)
    execution_order: List[str] = field(default_factory=list)
    final_role: str = "synthesizer"
    termination_criterion: str = ""
    mode: str = "legacy_json"
    workflow_source: str = ""
    workflow_static_trace: List[Dict[str, Any]] = field(default_factory=list)
    workflow_trace: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionPlan":
        return cls(
            problem_summary=str(data.get("problem_summary", "")),
            topology_rationale=str(data.get("topology_rationale", "")),
            assumptions=[str(item) for item in data.get("assumptions", [])],
            risks=[str(item) for item in data.get("risks", [])],
            steps=[PlanStep.from_dict(item) for item in data.get("steps", [])],
            execution_order=[str(item) for item in data.get("execution_order", [])],
            final_role=str(data.get("final_role", "synthesizer")),
            termination_criterion=str(data.get("termination_criterion", "")),
            mode=str(data.get("mode", "legacy_json")),
            workflow_source=str(data.get("workflow_source", "")),
            workflow_static_trace=[
                dict(item) for item in data.get("workflow_static_trace", [])
            ],
            workflow_trace=[dict(item) for item in data.get("workflow_trace", [])],
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def ordered_steps(self) -> List[PlanStep]:
        if self.execution_order:
            lookup = {step.id: step for step in self.steps}
            ordered: List[PlanStep] = []
            for step_id in self.execution_order:
                if step_id in lookup:
                    ordered.append(lookup[step_id])
            if ordered:
                return ordered
        return self._topological_order()

    def _topological_order(self) -> List[PlanStep]:
        lookup = {step.id: step for step in self.steps}
        pending = {step.id: set(step.depends_on) for step in self.steps}
        ready = [step_id for step_id, deps in pending.items() if not deps]
        ordered: List[PlanStep] = []

        while ready:
            current_id = ready.pop(0)
            ordered.append(lookup[current_id])
            for step_id, deps in pending.items():
                if current_id in deps:
                    deps.remove(current_id)
                    if not deps and lookup[step_id] not in ordered and step_id not in ready:
                        ready.append(step_id)

        if len(ordered) != len(self.steps):
            # Fallback to declared order if the plan has gaps or cycles.
            return list(self.steps)
        return ordered


@dataclass
class PlannerRun:
    """Record of the planner's single claude -p invocation.

    Captures everything needed to reproduce or audit the topology that
    was used to dispatch the specialist agents.
    """

    prompt_path: str
    assembled_prompt: str
    problem_text: str
    raw_stdout: str = ""
    raw_json: Dict[str, Any] = field(default_factory=dict)
    parsed_plan: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    started_at: Optional[str] = None
    duration_ms: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    # Keep separate from `input_tokens`, which the CLI reports as uncached
    # input only; `sum_run_usage` folds them together for reporting.
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    stderr: str = ""
    # Populated when the planner is invoked with --output-format stream-json.
    # Each entry is the short tool name with its parsed input dict.
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AgentResult:
    """Public result produced by an agent execution."""

    output: Any
    node_id: str = ""
    role: str = ""
    received_from: List[str] = field(default_factory=list)


@dataclass
class AgentRun:
    role: str
    step_id: str
    prompt_path: str
    assembled_prompt: str = ""
    context_bundle: str = ""
    stdin_text: str = ""
    selected_skills: List[str] = field(default_factory=list)
    skill_selection_result: str = ""
    allowed_tools: List[str] = field(default_factory=list)
    session_id: Optional[str] = None
    result: str = ""
    raw_stdout: str = ""
    raw_json: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[str] = None
    duration_ms: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    # See `PlannerRun`: uncached input and the two cache counters are kept
    # apart here and summed only at reporting time.
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    stderr: str = ""
    # Resumed from a previous session within the same role? If so this
    # step inherits the prior AgentRun's session_id at start of turn.
    resumed_from_session: Optional[str] = None
    # Public information this node is allowed to expose along workflow edges.
    public_output: Any = ""
    # Upstream node ids whose public outputs were explicitly supplied.
    received_from: List[str] = field(default_factory=list)
    # Populated when the agent is invoked with --output-format stream-json.
    # Each entry is the short tool name with its parsed input dict (e.g.
    # an `mcp__<server>__<tool>` invocation).
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RunReport:
    problem: str
    plan: ExecutionPlan
    planner: Optional[PlannerRun] = None
    runs: List[AgentRun] = field(default_factory=list)
    final_answer: str = ""
    session_ids: Dict[str, str] = field(default_factory=dict)
    output_dir: str = ""
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem": self.problem,
            "plan": self.plan.to_dict(),
            "planner": asdict(self.planner) if self.planner else None,
            "runs": [asdict(run) for run in self.runs],
            "final_answer": self.final_answer,
            "session_ids": self.session_ids,
            "output_dir": self.output_dir,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


# The run-metric counters every PlannerRun/AgentRun carries, and the exact
# set `grade_problem` accepts. Kept as one list so the dataclasses, the
# summation below, and the grader cannot drift apart — a counter recorded on
# the dataclasses but missing here is precisely how token usage came to be
# captured per call and never totalled.
RUN_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "cost_usd",
    "duration_ms",
)


def total_tokens(source: Any) -> int:
    """Grand total the model processed, in tokens.

    Accepts either a `sum_run_usage` mapping or any single record carrying
    the same counters (``PlannerRun``, ``AgentRun``, ``GradingResult``), so
    the batch total and a one-problem line share this definition.

    "Input" here means input as the API bills it: fresh input plus cache
    writes plus cache reads. That is the number to read next to ``cost_usd``
    — ``input_tokens`` alone excludes the cache counters, so it understates
    a resumed session by most of its context.
    """
    def counter(name: str) -> Any:
        if isinstance(source, dict):
            return source.get(name)
        return getattr(source, name, None)

    return sum(
        int(counter(name) or 0)
        for name in (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        )
    )


def sum_run_usage(runs: Iterable[Any]) -> Dict[str, Any]:
    """Sum per-invocation usage counters across a run's calls.

    Returns exactly the run-metric keywords ``grade_problem`` takes, so a
    caller splats the result straight in. Entries may be ``PlannerRun`` or
    ``AgentRun``; ``None`` entries are skipped, which lets a caller write
    ``sum_run_usage([report.planner, *report.runs])`` without a guard.

    Note this is called with ``report.runs`` alone to keep aggregate costs
    comparable with runs recorded before the planner was measured. The
    planner is a real, separately-recorded invocation, so folding it in
    would raise every historical total; pass it in explicitly if you want
    that.

    ``None`` counts as 0: the CLI omits counters it cannot report (an
    unknown model, a text-only output format), and a batch total should be
    the sum of what was reported rather than ``None``.
    """
    totals: Dict[str, Any] = {name: 0 for name in RUN_USAGE_FIELDS}
    for run in runs:
        if run is None:
            continue
        for name in RUN_USAGE_FIELDS:
            value = getattr(run, name, None)
            if value is not None:
                totals[name] += value
    return totals
