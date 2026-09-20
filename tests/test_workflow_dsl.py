from pathlib import Path
import json
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from claude_runner import ClaudeResult
from orchestrator import SciMASOrchestrator
from workflow_dsl import WorkflowDSLValidationError, parse_and_validate_workflow


VALID_WORKFLOW = '''
async def workflow(task):
    analysis = await agent(
        role="generalist",
        instruction="Analyze the task.",
        input=task,
    )

    sol1, sol2 = await parallel(
        agent(role="generalist", instruction="Solve with method A.", input=[task, analysis]),
        agent(role="generalist", instruction="Solve with method B.", input=[task, analysis]),
    )

    solution = sol1
    for _ in range(2):
        review = await agent(
            role="generalist",
            instruction="Review the current solution and return strict JSON with passed.",
            input=solution,
        )
        if review["passed"]:
            break
        solution = await agent(
            role="generalist",
            instruction="Improve the solution.",
            input=[task, solution, review],
        )

    return solution
'''


class FakeRunner:
    mcp_config_path = None

    def __init__(self, planner_source: str) -> None:
        self.planner_source = planner_source
        self._counter = 0
        self._review_count = 0

    def run(
        self,
        prompt,
        stdin_text=None,
        session_id=None,
        output_format="json",
        allowed_tools=None,
    ):
        self._counter += 1
        if "async def workflow(task)" in prompt and stdin_text == "demo task":
            result = self.planner_source
        else:
            context = json.loads(stdin_text)
            objective = context["current_step"]["objective"]
            if objective == "Analyze the task.":
                result = '{"analysis": "ok"}'
            elif objective == "Solve with method A.":
                result = "solution A"
            elif objective == "Solve with method B.":
                result = "solution B"
            elif objective.startswith("Review the current solution"):
                self._review_count += 1
                passed = "true" if self._review_count == 2 else "false"
                result = f'{{"passed": {passed}}}'
            elif objective == "Improve the solution.":
                result = "solution refined"
            else:
                result = f"unexpected objective: {objective}"
        return ClaudeResult(
            raw_stdout=json.dumps({"result": result}),
            raw_json={"result": result},
            result=result,
            session_id=f"session-{self._counter}",
            total_cost_usd=0.0,
            total_tokens=0,
            input_tokens=0,
            output_tokens=0,
            stderr="",
            started_at="2026-09-16T00:00:00+00:00",
            duration_ms=1,
        )


class InspectingRunner:
    mcp_config_path = None

    def __init__(self, planner_source: str) -> None:
        self.planner_source = planner_source
        self.contexts_by_objective = {}
        self._counter = 0

    def run(
        self,
        prompt,
        stdin_text=None,
        session_id=None,
        output_format="json",
        allowed_tools=None,
    ):
        self._counter += 1
        if "async def workflow(task)" in prompt and stdin_text == "demo task":
            result = self.planner_source
            raw_json = {"result": result}
            raw_stdout = json.dumps(raw_json)
        else:
            context = json.loads(stdin_text)
            objective = context["current_step"]["objective"]
            self.contexts_by_objective.setdefault(objective, []).append(context)
            result = {
                "A": "PUBLIC_A",
                "B": "PUBLIC_B",
                "C": "PUBLIC_C",
            }.get(objective, f"PUBLIC_{objective}")
            raw_json = {
                "result": result,
                "private_history": f"PRIVATE_{objective}_TOOL_TRACE",
            }
            raw_stdout = json.dumps(raw_json)
        return ClaudeResult(
            raw_stdout=raw_stdout,
            raw_json=raw_json,
            result=result,
            session_id=f"session-{self._counter}",
            total_cost_usd=0.0,
            total_tokens=0,
            input_tokens=0,
            output_tokens=0,
            stderr="",
            started_at="2026-09-16T00:00:00+00:00",
            duration_ms=1,
        )


class LegacyInspectingRunner(InspectingRunner):
    def __init__(self) -> None:
        plan = {
            "problem_summary": "legacy chain",
            "steps": [
                {
                    "id": "a",
                    "role": "generalist",
                    "objective": "A",
                    "depends_on": [],
                    "expected_output": "A",
                },
                {
                    "id": "b",
                    "role": "generalist",
                    "objective": "B",
                    "depends_on": ["a"],
                    "expected_output": "B",
                },
                {
                    "id": "c",
                    "role": "generalist",
                    "objective": "C",
                    "depends_on": ["b"],
                    "expected_output": "C",
                },
            ],
            "execution_order": ["a", "b", "c"],
            "final_role": "generalist",
        }
        super().__init__(json.dumps(plan))

    def run(
        self,
        prompt,
        stdin_text=None,
        session_id=None,
        output_format="json",
        allowed_tools=None,
    ):
        if "Return strict JSON only" in prompt and stdin_text == "demo task":
            self._counter += 1
            return ClaudeResult(
                raw_stdout=json.dumps({"result": self.planner_source}),
                raw_json={"result": self.planner_source},
                result=self.planner_source,
                session_id=f"session-{self._counter}",
                total_cost_usd=0.0,
                total_tokens=0,
                input_tokens=0,
                output_tokens=0,
                stderr="",
                started_at="2026-09-16T00:00:00+00:00",
                duration_ms=1,
            )
        return super().run(
            prompt,
            stdin_text=stdin_text,
            session_id=session_id,
            output_format=output_format,
            allowed_tools=allowed_tools,
        )


def run_workflow_with_inspector(source: str):
    with tempfile.TemporaryDirectory() as tmp:
        empty_skills = Path(tmp) / "empty-skills"
        empty_skills.mkdir()
        orch = SciMASOrchestrator(
            output_dir=Path(tmp),
            skill_dir=empty_skills,
            use_skill_routing=False,
            mcp_config_path="",
            planner_mode="python_dsl",
        )
        runner = InspectingRunner(source)
        orch.runner = runner
        report = orch.run("demo task", roles=["generalist"])
    return report, runner


def run_legacy_with_inspector():
    with tempfile.TemporaryDirectory() as tmp:
        empty_skills = Path(tmp) / "empty-skills"
        empty_skills.mkdir()
        orch = SciMASOrchestrator(
            output_dir=Path(tmp),
            skill_dir=empty_skills,
            use_skill_routing=False,
            mcp_config_path="",
            planner_mode="legacy_json",
        )
        runner = LegacyInspectingRunner()
        orch.runner = runner
        report = orch.run("demo task", roles=["generalist"], auto_synthesize=False)
    return report, runner


class WorkflowDSLValidationTests(unittest.TestCase):
    def test_valid_workflow_supports_parallel_if_and_loop(self) -> None:
        program = parse_and_validate_workflow(VALID_WORKFLOW)

        events = [item["event"] for item in program.static_trace]
        self.assertIn("parallel_call", events)
        self.assertIn("if", events)
        self.assertIn("for", events)

    def test_rejects_disallowed_python(self) -> None:
        bad_sources = [
            "import os\nasync def workflow(task):\n    return task\n",
            "async def workflow(task):\n    return eval('1 + 1')\n",
            "async def workflow(task):\n    return task.__class__\n",
            "async def workflow(task):\n    while True:\n        break\n    return task\n",
            "async def workflow(task):\n    x = await agent(role='generalist', instruction='x', input=open('x'))\n    return x\n",
            "async def workflow(task):\n    for _ in range(999):\n        break\n    return task\n",
        ]
        for source in bad_sources:
            with self.subTest(source=source):
                with self.assertRaises(WorkflowDSLValidationError):
                    parse_and_validate_workflow(source)


class WorkflowDSLExecutionTests(unittest.TestCase):
    def test_orchestrator_executes_dsl_with_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            empty_skills = Path(tmp) / "empty-skills"
            empty_skills.mkdir()
            orch = SciMASOrchestrator(
                output_dir=Path(tmp),
                skill_dir=empty_skills,
                use_skill_routing=False,
                mcp_config_path="",
                planner_mode="python_dsl",
            )
            orch.runner = FakeRunner(VALID_WORKFLOW)

            report = orch.run("demo task", roles=["generalist"])

        self.assertEqual(report.final_answer, "solution refined")
        self.assertEqual(report.plan.mode, "python_dsl")
        self.assertEqual(len(report.runs), 6)

        agent_events = [
            event for event in report.plan.workflow_trace if event["event"] == "agent"
        ]
        by_instruction = {}
        for event in agent_events:
            by_instruction.setdefault(event["instruction"], event)
        analysis_id = by_instruction["Analyze the task."]["id"]
        self.assertIn(
            analysis_id,
            by_instruction["Solve with method A."]["input_dependencies"],
        )
        review_events = [
            event for event in agent_events
            if event["instruction"].startswith("Review the current solution")
        ]
        self.assertIn(by_instruction["Solve with method A."]["id"], review_events[0]["input_dependencies"])
        parallel_events = [
            event for event in report.plan.workflow_trace if event["event"] == "parallel"
        ]
        self.assertEqual(len(parallel_events), 1)
        self.assertEqual(len(parallel_events[0]["children"]), 2)

    def test_edge_passes_public_output_not_private_history(self) -> None:
        source = '''
async def workflow(task):
    a = await agent(role="generalist", instruction="A", input=task)
    b = await agent(role="generalist", instruction="B", input=[task, a])
    return b
'''
        report, runner = run_workflow_with_inspector(source)

        b_context = runner.contexts_by_objective["B"][0]
        serialized = json.dumps(b_context, ensure_ascii=False)
        self.assertIn("PUBLIC_A", serialized)
        self.assertNotIn("PRIVATE_A_TOOL_TRACE", serialized)
        self.assertEqual(b_context["received_upstream_outputs"][0]["public_output"], "PUBLIC_A")
        b_trace = [
            event for event in report.plan.workflow_trace
            if event.get("event") == "agent" and event.get("instruction") == "B"
        ][0]
        self.assertEqual(len(b_trace["received_from"]), 1)
        self.assertEqual(b_trace["public_output"], "PUBLIC_B")

    def test_join_receives_two_explicit_public_outputs(self) -> None:
        source = '''
async def workflow(task):
    a = await agent(role="generalist", instruction="A", input=task)
    b = await agent(role="generalist", instruction="B", input=task)
    c = await agent(role="generalist", instruction="C", input=[a, b])
    return c
'''
        _report, runner = run_workflow_with_inspector(source)

        c_context = runner.contexts_by_objective["C"][0]
        outputs = c_context["received_upstream_outputs"]
        self.assertEqual([item["public_output"] for item in outputs], ["PUBLIC_A", "PUBLIC_B"])
        serialized = json.dumps(c_context, ensure_ascii=False)
        self.assertNotIn("PRIVATE_A_TOOL_TRACE", serialized)
        self.assertNotIn("PRIVATE_B_TOOL_TRACE", serialized)

    def test_chain_does_not_inherit_grandparent_output(self) -> None:
        source = '''
async def workflow(task):
    a = await agent(role="generalist", instruction="A", input=task)
    b = await agent(role="generalist", instruction="B", input=a)
    c = await agent(role="generalist", instruction="C", input=b)
    return c
'''
        report, runner = run_workflow_with_inspector(source)

        c_context = runner.contexts_by_objective["C"][0]
        serialized = json.dumps(c_context, ensure_ascii=False)
        self.assertIn("PUBLIC_B", serialized)
        self.assertNotIn("PUBLIC_A", serialized)
        c_trace = [
            event for event in report.plan.workflow_trace
            if event.get("event") == "agent" and event.get("instruction") == "C"
        ][0]
        b_trace = [
            event for event in report.plan.workflow_trace
            if event.get("event") == "agent" and event.get("instruction") == "B"
        ][0]
        self.assertEqual(c_trace["received_from"], [b_trace["id"]])

    def test_legacy_json_uses_explicit_dependencies_only(self) -> None:
        _report, runner = run_legacy_with_inspector()

        c_context = runner.contexts_by_objective["C"][0]
        serialized = json.dumps(c_context, ensure_ascii=False)
        self.assertIn("PUBLIC_B", serialized)
        self.assertNotIn("PUBLIC_A", serialized)
        self.assertEqual(
            [item["public_output"] for item in c_context["received_upstream_outputs"]],
            ["PUBLIC_B"],
        )


class PlannerOutputShapeTests(unittest.TestCase):
    """Planners wrap workflow code in prose and fences; both parsers must cope.

    Regression tests for the run that died with "Reviewer did not return valid
    JSON" while the planner's code was valid DSL all along.
    """

    CODE = (
        'async def workflow(task):\n'
        '    a = await agent(role="computational-chemist", instruction="Analyze. Return strict JSON.", input=task)\n'
        '    b = await agent(role="physical-chemist", instruction="Verify.", input=[task, a])\n'
        '    return b\n'
    )

    def test_source_shapes_that_must_validate(self) -> None:
        shapes = {
            "bare code": self.CODE,
            "leading fence": f"```python\n{self.CODE}```",
            "prose then fence": (
                "I'll design a compact workflow: conformer generation by a "
                f"computational chemist.\n\n```python\n{self.CODE}```"
            ),
            "prose, fence, trailing prose": (
                f"Here you go.\n\n```python\n{self.CODE}```\n\n(no final answer)"
            ),
            "unterminated fence + trailing prose": (
                f"Here:\n\n```python\n{self.CODE}\nLet me know if you need anything else!"
            ),
            "prose then bare code": f"Sure, here is the workflow:\n\n{self.CODE}",
        }
        for name, text in shapes.items():
            with self.subTest(shape=name):
                program = parse_and_validate_workflow(text)
                self.assertTrue(program.source.startswith("async def workflow(task):"))
                self.assertNotIn("```", program.source)
                self.assertEqual(len(program.static_trace), 3)

    def test_reviewer_reply_shapes_that_must_parse(self) -> None:
        import json as _json

        from workflow_dsl import extract_json_object

        payload = {
            "is_valid": True,
            "issues": [],
            "fixed_source": self.CODE,
            "changes_made": [],
            "confidence": 0.9,
        }
        encoded = _json.dumps(payload)
        shapes = {
            "bare json": encoded,
            "fenced json": f"```json\n{encoded}\n```",
            "prose before fence": f"Here is my review:\n\n```json\n{encoded}\n```",
            "trailing prose after fence": f"```json\n{encoded}\n```\n\nDone.",
            "json then prose": f"{encoded}\n\nLet me know if you need anything else!",
        }
        for name, text in shapes.items():
            with self.subTest(shape=name):
                parsed = extract_json_object(text)
                self.assertEqual(parsed.get("fixed_source"), self.CODE)
                self.assertTrue(parsed.get("is_valid"))

    def test_unescaped_newlines_in_fixed_source_are_repaired(self) -> None:
        from workflow_dsl import extract_json_object

        # A model that escapes its quotes but writes the code as real line
        # breaks rather than \n emits invalid JSON; the repair pass recovers
        # it. (Raw unescaped quotes are genuinely ambiguous - that case is
        # handled by the orchestrator's corrective reviewer retry instead.)
        body = self.CODE.rstrip("\n").replace('"', '\\"')  # quotes escaped, newlines raw
        raw = '```json\n{"is_valid": true, "fixed_source": "' + body + '", "confidence": 0.8}\n```'

        parsed = extract_json_object(raw)
        self.assertEqual(parsed.get("fixed_source"), self.CODE.rstrip("\n"))
        parse_and_validate_workflow(parsed["fixed_source"])

    def test_agent_result_keeps_top_level_array(self) -> None:
        from workflow_dsl import coerce_agent_result

        self.assertEqual(coerce_agent_result("[1, 2, 3]"), [1, 2, 3])
        self.assertEqual(coerce_agent_result("Values:\n[1, 2, 3]"), [1, 2, 3])
        # Not JSON at all stays a string.
        self.assertEqual(coerce_agent_result("7.35 kcal/mol"), "7.35 kcal/mol")

    def test_unknown_agent_role_is_rejected_at_validation(self) -> None:
        source = (
            "async def workflow(task):\n"
            '    x = await agent(role="conformational-analyst", instruction="Do it", input=task)\n'
            "    return x\n"
        )
        allowed = {"physicist", "generalist"}
        with self.assertRaises(WorkflowDSLValidationError) as ctx:
            parse_and_validate_workflow(
                source,
                role_check=lambda role: role in allowed,
                allowed_roles=allowed,
            )
        self.assertIn("conformational-analyst", str(ctx.exception))
        self.assertIn("generalist", str(ctx.exception))
        # Without a role_check the validator records the role but allows it.
        self.assertTrue(parse_and_validate_workflow(source).source)


if __name__ == "__main__":
    unittest.main()
