"""Test the planner-reviewer workflow validation and fix loop."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchestrator import SciMASOrchestrator
from workflow_dsl import WorkflowDSLValidationError, parse_and_validate_workflow


def test_fenced_and_prose_wrapped_code_validates_without_the_reviewer():
    """A fenced workflow is unwrapped locally, so no reviewer call is needed.

    This used to assert the opposite (`pytest.raises` on fenced code), which
    contradicted `extract_workflow_source` and cost a reviewer round-trip for
    a purely cosmetic wrapper.
    """
    wrapped = """I'll design a compact workflow for this problem.

```python
async def workflow(task):
    result = await agent(role="generalist", instruction="Analyze", input=task)
    return result
```
"""

    program = parse_and_validate_workflow(wrapped)
    assert "```" not in program.source
    assert "I'll design" not in program.source
    assert program.source.startswith("async def workflow(task):")
    # The code inside the fence was valid all along.
    assert len(program.static_trace) == 2


@pytest.mark.live
def test_reviewer_fixes_markdown_wrapped_code():
    """Test that reviewer can fix code wrapped in markdown fences."""
    invalid_workflow = """```python
async def workflow(task):
    result = await agent("generalist", "Analyze", task)
    return result
```"""

    orchestrator = SciMASOrchestrator(planner_mode="python_dsl")

    # Positional agent(...) args are a real DSL violation, unlike the fence.
    with pytest.raises(WorkflowDSLValidationError):
        parse_and_validate_workflow(invalid_workflow)

    # Reviewer should fix it
    review_result = orchestrator._review_and_fix_workflow(
        workflow_source=invalid_workflow,
        validation_error="agent(...) only accepts keyword arguments.",
        attempt_number=1,
    )

    assert review_result.get("fixed_source")
    fixed = review_result["fixed_source"]

    # Fixed version should be valid
    program = parse_and_validate_workflow(fixed)
    assert program.source
    assert "```" not in program.source


@pytest.mark.live
def test_reviewer_fixes_missing_return():
    """Test that reviewer adds missing return statement."""
    invalid_workflow = """async def workflow(task):
    result = await agent(role="generalist", instruction="Analyze", input=task)
"""

    orchestrator = SciMASOrchestrator(planner_mode="python_dsl")

    review_result = orchestrator._review_and_fix_workflow(
        workflow_source=invalid_workflow,
        validation_error="Missing return statement",
        attempt_number=1,
    )

    fixed = review_result.get("fixed_source", "")
    assert "return" in fixed.lower()


@pytest.mark.live
def test_reviewer_fixes_invalid_agent_call():
    """Test that reviewer fixes agent calls with positional args."""
    invalid_workflow = """async def workflow(task):
    result = await agent("generalist", "Analyze the task", task)
    return result
"""

    orchestrator = SciMASOrchestrator(planner_mode="python_dsl")

    with pytest.raises(WorkflowDSLValidationError):
        parse_and_validate_workflow(invalid_workflow)

    review_result = orchestrator._review_and_fix_workflow(
        workflow_source=invalid_workflow,
        validation_error="agent() must use keyword arguments only",
        attempt_number=1,
    )

    fixed = review_result.get("fixed_source", "")
    assert "role=" in fixed
    assert "instruction=" in fixed


def test_plan_with_reviewer_loop():
    """Test the full planning flow with reviewer validation."""
    problem = "Calculate the energy of a photon with wavelength 500 nm."

    orchestrator = SciMASOrchestrator(planner_mode="python_dsl")

    # Mock the runner to return invalid then valid code
    class MockRunner:
        def __init__(self):
            self.call_count = 0

        def run(self, prompt, stdin_text, output_format, allowed_tools):
            from claude_runner import ClaudeResult

            self.call_count += 1

            if self.call_count == 1:
                # First call: planner returns a genuine DSL violation
                # (positional args), not just a fence - a fence alone is now
                # unwrapped locally and never reaches the reviewer.
                result_text = """async def workflow(task):
    result = await agent("physicist", "Calculate photon energy", task)
    return result
"""
            else:
                # Second call: reviewer returns fixed code
                result_text = json.dumps({
                    "is_valid": True,
                    "issues": ["agent() called with positional arguments"],
                    "fixed_source": """async def workflow(task):
    result = await agent(role="physicist", instruction="Calculate photon energy", input=task)
    return result
""",
                    "changes_made": ["Converted agent() call to keyword arguments"],
                    "confidence": 0.95,
                })

            return ClaudeResult(
                result=result_text,
                raw_stdout=result_text,
                raw_json={"result": result_text},
                session_id="test-session",
                started_at="2024-01-01T00:00:00Z",
                duration_ms=1000,
                input_tokens=100,
                output_tokens=50,
                total_cost_usd=0.001,
                stderr="",
                tool_calls=[],
            )

    orchestrator.runner = MockRunner()

    # This should succeed after reviewer fixes the code
    plan, planner_run = orchestrator.plan(problem, ["physicist", "generalist"])

    assert plan.mode == "python_dsl"
    assert plan.workflow_source
    assert "```" not in plan.workflow_source
    assert orchestrator.runner.call_count == 2


def test_reviewer_gives_up_after_max_attempts():
    """Test that reviewer stops after max attempts."""
    problem = "Test problem"

    orchestrator = SciMASOrchestrator(planner_mode="python_dsl")

    class MockRunner:
        def run(self, prompt, stdin_text, output_format, allowed_tools):
            from claude_runner import ClaudeResult

            # Always return invalid code
            if "planner-reviewer" in prompt:
                # Reviewer returns same broken code
                result_text = json.dumps({
                    "is_valid": False,
                    "issues": ["Cannot fix this"],
                    "fixed_source": "```python\nasync def workflow(task):\n    pass\n```",
                    "changes_made": [],
                    "confidence": 0.1,
                })
            else:
                # Planner returns broken code
                result_text = "```python\nasync def workflow(task):\n    pass\n```"

            return ClaudeResult(
                result=result_text,
                raw_stdout=result_text,
                raw_json={"result": result_text},
                session_id="test-session",
                started_at="2024-01-01T00:00:00Z",
                duration_ms=1000,
                input_tokens=100,
                output_tokens=50,
                total_cost_usd=0.001,
                stderr="",
                tool_calls=[],
            )

    orchestrator.runner = MockRunner()

    # Should fail after max attempts
    with pytest.raises(RuntimeError, match="validation failed after"):
        orchestrator.plan(problem, ["generalist"])


PLANNER_WRAPPED_BUT_VALID = """I'll design a compact workflow: conformer generation by a
computational chemist, then an independent cross-check.

```python
async def workflow(task):
    a = await agent(role="computational-chemist", instruction="Analyze.", input=task)
    b = await agent(role="physical-chemist", instruction="Verify.", input=[task, a])
    return b
```
"""

PLANNER_BROKEN = """async def workflow(task):
    x = await agent("physicist", "Do it", task)
    return x
"""

PLANNER_FIXED = """async def workflow(task):
    x = await agent(role="physicist", instruction="Do it", input=task)
    return x
"""


class RecordingRunner:
    """Runner stub that records which agent was called and returns canned text."""

    def __init__(self, planner_text, reviewer_text=""):
        self.planner_text = planner_text
        self.reviewer_text = reviewer_text
        self.calls = []

    def run(self, prompt, stdin_text, output_format, allowed_tools):
        from claude_runner import ClaudeResult

        is_reviewer = "workflow code reviewer" in prompt
        self.calls.append("reviewer" if is_reviewer else "planner")
        text = self.reviewer_text if is_reviewer else self.planner_text
        return ClaudeResult(
            result=text,
            raw_stdout=text,
            raw_json={"result": text},
            session_id="test-session",
            started_at="2024-01-01T00:00:00Z",
            duration_ms=1,
            input_tokens=1,
            output_tokens=1,
            total_cost_usd=0.0,
            stderr="",
            tool_calls=[],
        )

    def reviewer_calls(self):
        return self.calls.count("reviewer")


def test_prose_and_fence_wrapped_workflow_never_reaches_the_reviewer():
    """The reported failure: valid code wrapped in prose plus a fence.

    The wrapper is unwrapped locally, so this must plan with zero reviewer
    calls - it previously cost a reviewer round-trip and then killed the run.
    """
    orchestrator = SciMASOrchestrator(planner_mode="python_dsl")
    orchestrator.runner = RecordingRunner(PLANNER_WRAPPED_BUT_VALID)

    plan, _ = orchestrator.plan(
        "conformational analysis", ["computational-chemist", "physical-chemist"]
    )

    assert orchestrator.runner.reviewer_calls() == 0
    assert "I'll design" not in plan.workflow_source
    assert "```" not in plan.workflow_source
    assert len(plan.workflow_static_trace) == 3


def test_reviewer_reply_with_prose_and_trailing_text_is_used():
    """A reviewer reply that is not *exactly* JSON must still be parsed.

    This is the shape that produced "Reviewer did not return valid JSON".
    """
    reply = (
        "Here is my review of the workflow:\n\n"
        "```json\n"
        + json.dumps(
            {
                "is_valid": True,
                "issues": [],
                "fixed_source": PLANNER_FIXED,
                "changes_made": ["keyword arguments"],
                "confidence": 0.9,
            }
        )
        + "\n```\n\nLet me know if you need anything else!"
    )
    orchestrator = SciMASOrchestrator(planner_mode="python_dsl")
    orchestrator.runner = RecordingRunner(PLANNER_BROKEN, reply)

    plan, _ = orchestrator.plan("Test problem", ["physicist"])

    assert orchestrator.runner.reviewer_calls() == 1
    # `workflow_source` is the *extracted* source, which is stripped.
    assert plan.workflow_source == PLANNER_FIXED.strip()


def test_unparseable_reviewer_reply_is_retried_then_reported():
    """A reviewer that never returns JSON gets a retry, then a usable error."""
    orchestrator = SciMASOrchestrator(planner_mode="python_dsl", max_review_attempts=2)
    orchestrator.runner = RecordingRunner(PLANNER_BROKEN, "The code looks fine to me.")

    with pytest.raises(RuntimeError) as ctx:
        orchestrator.plan("Test problem", ["physicist"])

    # Two reviewer rounds, each of which got one corrective parse retry.
    assert orchestrator.runner.reviewer_calls() == 4
    # The reviewer's raw reply is what makes this debuggable.
    assert "The code looks fine to me." in str(ctx.value)


def test_max_review_attempts_controls_the_retry_budget():
    """Every extra attempt buys exactly one more reviewer round.

    Each round costs up to two calls: the round itself plus the single
    corrective retry it makes when the reply will not parse. So the worst
    case is 2 x max_review_attempts - bounded, but worth knowing before
    raising the budget in the dashboard.
    """
    calls = {}
    for budget in (1, 2, 3):
        orchestrator = SciMASOrchestrator(
            planner_mode="python_dsl", max_review_attempts=budget
        )
        orchestrator.runner = RecordingRunner(PLANNER_BROKEN, "not json")
        with pytest.raises(RuntimeError):
            orchestrator.plan("Test problem", ["physicist"])
        calls[budget] = orchestrator.runner.reviewer_calls()

    assert calls == {1: 2, 2: 4, 3: 6}


def test_reviewer_recovers_after_several_bad_rounds():
    """A reviewer that eventually returns good JSON is still honoured."""

    class FlakyRunner(RecordingRunner):
        def run(self, prompt, stdin_text, output_format, allowed_tools):
            if "workflow code reviewer" not in prompt:
                return super().run(prompt, stdin_text, output_format, allowed_tools)
            self.calls.append("reviewer")
            from claude_runner import ClaudeResult

            if self.reviewer_calls() < 3:
                text = "still thinking about it"
            else:
                text = json.dumps(
                    {
                        "is_valid": True,
                        "issues": [],
                        "fixed_source": PLANNER_FIXED,
                        "changes_made": ["keyword arguments"],
                        "confidence": 0.8,
                    }
                )
            return ClaudeResult(
                result=text,
                raw_stdout=text,
                raw_json={"result": text},
                session_id="test-session",
                started_at="2024-01-01T00:00:00Z",
                duration_ms=1,
                input_tokens=1,
                output_tokens=1,
                total_cost_usd=0.0,
                stderr="",
                tool_calls=[],
            )

    orchestrator = SciMASOrchestrator(planner_mode="python_dsl", max_review_attempts=3)
    orchestrator.runner = FlakyRunner(PLANNER_BROKEN)

    plan, _ = orchestrator.plan("Test problem", ["physicist"])

    assert plan.workflow_source == PLANNER_FIXED.strip()
    assert orchestrator.runner.reviewer_calls() == 3


def test_invented_agent_role_fails_at_planning_not_execution():
    """A role the planner made up is caught by the validator, with the list."""
    invented = (
        "async def workflow(task):\n"
        '    x = await agent(role="conformational-analyst", instruction="Do it", input=task)\n'
        "    return x\n"
    )
    orchestrator = SciMASOrchestrator(planner_mode="python_dsl", max_review_attempts=1)
    orchestrator.runner = RecordingRunner(invented, "cannot fix")

    with pytest.raises(RuntimeError) as ctx:
        orchestrator.plan("Test problem", ["physicist", "generalist"])

    assert "conformational-analyst" in str(ctx.value)


if __name__ == "__main__":
    # Run a simple manual test
    print("Testing reviewer with invalid workflow...")

    invalid = """```python
async def workflow(task):
    x = await agent("physicist", "Do physics", task)
    return x
```"""

    orchestrator = SciMASOrchestrator(planner_mode="python_dsl")

    try:
        parse_and_validate_workflow(invalid)
        print("❌ Should have failed validation")
    except WorkflowDSLValidationError as e:
        print(f"✓ Validation failed as expected: {e}")

    print("\nAsking reviewer to fix it...")
    result = orchestrator._review_and_fix_workflow(
        workflow_source=invalid,
        validation_error=str(e),
        attempt_number=1,
    )

    print(f"Reviewer response: {json.dumps(result, indent=2)}")

    if result.get("fixed_source"):
        try:
            program = parse_and_validate_workflow(result["fixed_source"])
            print(f"✓ Fixed workflow is valid!")
            print(f"\nFixed source:\n{program.source}")
        except WorkflowDSLValidationError as e2:
            print(f"❌ Fixed workflow still invalid: {e2}")
