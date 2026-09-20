#!/usr/bin/env python3
"""Demo script showing the planner-reviewer workflow validation and fix loop.

This demonstrates how the two-agent system works:
1. Planner generates workflow code (may have format issues)
2. Reviewer validates and fixes the code
3. If execution fails, reviewer fixes it again
4. The process repeats until the workflow is valid and executable

Usage:
    python demo_reviewer.py
"""

from __future__ import annotations

from __future__ import annotations

from orchestrator import SciMASOrchestrator
from workflow_dsl import WorkflowDSLValidationError, parse_and_validate_workflow


def demo_validation_loop():
    """Demonstrate the validation and fix loop."""
    print("=" * 70)
    print("DEMO: Planner-Reviewer Workflow Validation Loop")
    print("=" * 70)

    # Example 1: Markdown-wrapped code. Planners habitually wrap the workflow
    # in prose and a fence; that wrapper is purely cosmetic and is unwrapped
    # locally, so this costs no reviewer call at all. (It used to be treated
    # as a validation failure, which burned a reviewer round-trip and then
    # killed the run entirely.)
    print("\n1. Testing prose- and markdown-wrapped workflow...")
    print("-" * 70)

    wrapped_workflow = """I'll design a compact workflow: analyze, then solve.

```python
async def workflow(task):
    analysis = await agent(role="generalist", instruction="Analyze task", input=task)
    solution = await agent(role="physicist", instruction="Solve", input=analysis)
    return solution
```
"""

    print("Planner output (prose + fence):")
    print(wrapped_workflow)

    program = parse_and_validate_workflow(wrapped_workflow)
    print("✓ Extracted and validated locally — no reviewer call needed:")
    print(program.source)
    print(f"  static trace nodes: {len(program.static_trace)}")

    orchestrator = SciMASOrchestrator(planner_mode="python_dsl")

    # Example 2: Invalid agent call (positional arguments)
    print("\n\n2. Testing invalid agent call (positional args)...")
    print("-" * 70)

    invalid_workflow2 = """async def workflow(task):
    result = await agent("physicist", "Calculate energy", task)
    return result
"""

    print("Invalid workflow (positional arguments):")
    print(invalid_workflow2)

    try:
        parse_and_validate_workflow(invalid_workflow2)
        print("❌ Should have failed!")
    except WorkflowDSLValidationError as e:
        print(f"\n✓ Validation failed: {e}")

    print("\n→ Asking reviewer to fix it...")
    result2 = orchestrator._review_and_fix_workflow(
        workflow_source=invalid_workflow2,
        validation_error="agent() must use keyword arguments only",
        attempt_number=1,
    )

    fixed2 = result2.get("fixed_source", "")
    if fixed2:
        print(f"\n✓ Fixed workflow:\n{fixed2}")

        try:
            program2 = parse_and_validate_workflow(fixed2)
            print("\n✓ Fixed workflow validates successfully!")
        except WorkflowDSLValidationError as e:
            print(f"\n❌ Fixed workflow still invalid: {e}")

    # Example 3: Missing return statement
    print("\n\n3. Testing missing return statement...")
    print("-" * 70)

    invalid_workflow3 = """async def workflow(task):
    result = await agent(role="generalist", instruction="Analyze", input=task)
"""

    print("Invalid workflow (no return):")
    print(invalid_workflow3)

    print("\n→ Asking reviewer to fix it...")
    result3 = orchestrator._review_and_fix_workflow(
        workflow_source=invalid_workflow3,
        validation_error="Workflow must return a value",
        attempt_number=1,
    )

    fixed3 = result3.get("fixed_source", "")
    if fixed3:
        print(f"\n✓ Fixed workflow:\n{fixed3}")

        try:
            program3 = parse_and_validate_workflow(fixed3)
            print("\n✓ Fixed workflow validates successfully!")
        except WorkflowDSLValidationError as e:
            print(f"\n❌ Fixed workflow still invalid: {e}")

    print("\n" + "=" * 70)
    print("Demo completed!")
    print("=" * 70)


def demo_full_planning_flow():
    """Demonstrate the full planning flow with reviewer integration."""
    print("\n" + "=" * 70)
    print("DEMO: Full Planning Flow with Reviewer")
    print("=" * 70)

    print("\nThis demonstrates how the orchestrator automatically:")
    print("1. Calls planner to generate workflow code")
    print("2. Validates the code")
    print("3. If invalid, calls reviewer to fix it")
    print("4. Retries validation until valid or max attempts reached")
    print("\nNote: This demo would require actual Claude API calls.")
    print("See test_planner_reviewer.py for mock examples.")


if __name__ == "__main__":
    print("\n🔧 sciMAS Planner-Reviewer Demo\n")

    # Run the validation loop demo
    demo_validation_loop()

    # Show info about full flow (doesn't actually run it)
    demo_full_planning_flow()

    print("\n✓ All demos completed!")
    print("\nTo test with actual workflows, run:")
    print("  python web_dashboard.py --port 7860")
    print("  # Then create a run with 'python-dsl' planner mode")
