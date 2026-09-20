"""Regression tests for SciAgentGYM answer grading.

Three defects are covered here, all found from one dashboard run that was
marked ``✓ correct`` while showing ``expected: 2.8%`` next to a model answer
of ≈1.4%:

1. ``Problem.from_raw`` iterated a **dict**-shaped ``metadata.golden_answer``
   as if it were a list, so it kept the dict's *keys* (tool names) and threw
   away every step's output.
2. ``grade_sciagentgym_llm`` read the official ground truth from
   ``problem.task_info["golden_answer"]``, which ``Problem.from_raw`` never
   assigns — dead code, so every answer was judged against the ``answer``
   field alone.
3. The judge never actually ran — see section 5 — so every answer was
   scored by the deterministic fallback, whose "any expected number found
   anywhere in the text" rule marked a 1.4% essay ``✓ correct`` against the
   ``"2.8%"`` gold because the essay happened to quote 2.83% along the way.

The reported dashboard row (``expected: 2.8%`` beside a 1.4% answer) turned
out to have two separate faults, and they were fixed in that order:

1. ``2.8%`` was shown because it *was* the gold the grade came from. That is
   the bug, not the diagnosis: ``2.8%`` is ``horwitz_trumpet``'s own output —
   step 1 of the reference chain — and the question asks for the
   within-laboratory figure the chain computes next. The gold is now resolved
   to ``1.41421`` (section 4c), so the row reads ``expected: 1.41421 %``.

2. The display led with a value the verdict did not come from.

An earlier version of this module asserted the opposite — that the chain's
last step is the target and ``2.8%`` is an intermediate value — and it was
retracted, because in three of the four entries whose chain disagrees with
``answer`` the chain ends on a helper (``calculate_tension`` → -65.5 N after
an acceleration question, ``calculate_tension_force`` → 5.658 N after a
speed question, ``radius_of_gyration`` after a composite-metric one). It was
then re-derived the other way for the one entry where it *is* right, from a
condition narrow enough to exclude those three: the ``answer`` field must
equal an earlier step of the chain **and** the chain must continue to a
later step that carries a value. Only the Horwitz entry meets both halves,
so the resolution is a one-entry rule, not a policy about chains.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import grader
import llm_judge
from dataset import GoldenCall, Problem
from grader import (
    _sciagentgym_reference_workings,
    _sciagentgym_resolved_expected,
    _sciagentgym_solution_steps,
    grade_sciagentgym_llm,
    sciagentgym_expected_display,
)
from llm_judge import judge_correct


def _problem(**overrides) -> Problem:
    base = dict(
        id=1,
        filename="t.json",
        question="What is the minimum expected RSD?",
        answer="2.8%",
        subject="Chemistry",
        topic="Analytical Chemistry",
    )
    base.update(overrides)
    return Problem(**base)


# ---------------------------------------------------------------------------
# 1. The loader must keep every golden step, in both JSON shapes
# ---------------------------------------------------------------------------

# `refine_merged_single_questions.json` keys the chain by tool name.
DICT_SHAPED = {
    "id": 7,
    "question": "q",
    "answer": "2.6*1e5 GeV",
    "metadata": {
        "golden_answer": {
            "analyze_threshold": {
                "call": "analyze_threshold",
                "inputs": {"cmb_energy": 0.001},
                "output": {"threshold_energy_GeV": 261000.0},
            },
            "scan_energy": {
                "call": "scan_energy",
                "inputs": {"slope": -1},
                "output": {"scaling_law_slope": -1.0},
            },
        }
    },
}

# `refine_merged_multi_questions.json` uses an ordered list of steps.
LIST_SHAPED = {
    "id": 12,
    "question": "q",
    "answer": "2.8%",
    "metadata": {
        "golden_answer": [
            {"call": "horwitz_trumpet", "inputs": {"concentration": 0.1}, "output": 2.8284},
            {
                "call": "intra_laboratory_rsd",
                "inputs": {"interlaboratory_rsd": 2.8284, "factor": 0.5},
                "output": 1.4142,
            },
        ]
    },
}


def test_dict_shaped_golden_answer_keeps_its_steps():
    """Iterating a dict yields its keys — that silently dropped every output."""
    problem = Problem.from_raw(DICT_SHAPED, source="single.json")

    assert [c.tool for c in problem.golden_calls] == [
        "analyze_threshold",
        "scan_energy",
    ]
    assert problem.golden_calls[0].output == {"threshold_energy_GeV": 261000.0}


def test_list_shaped_golden_answer_keeps_its_steps():
    problem = Problem.from_raw(LIST_SHAPED, source="multi.json")

    assert [c.tool for c in problem.golden_calls] == [
        "horwitz_trumpet",
        "intra_laboratory_rsd",
    ]
    assert problem.golden_calls[-1].output == 1.4142


def test_step_inputs_may_be_null_or_a_list():
    """16 real steps carry ``inputs: null`` and one carries a list."""
    raw = {
        "id": 1,
        "question": "q",
        "answer": "a",
        "metadata": {
            "golden_answer": [
                {"call": "a", "inputs": None, "output": 1.0},
                {"call": "b", "inputs": [{"x": 1}], "output": 2.0},
            ]
        },
    }

    problem = Problem.from_raw(raw, source="t.json")

    assert [c.inputs for c in problem.golden_calls] == [{}, {}]
    assert [c.tool for c in problem.golden_calls] == ["a", "b"]


def test_non_dict_steps_are_skipped():
    raw = {
        "id": 1,
        "question": "q",
        "answer": "a",
        "metadata": {"golden_answer": [["not", "a", "step"], {"call": "ok", "output": 1}]},
    }

    problem = Problem.from_raw(raw, source="t.json")

    assert [c.tool for c in problem.golden_calls] == ["ok"]


# ---------------------------------------------------------------------------
# 2. The reference chain handed to the judge
# ---------------------------------------------------------------------------


def test_reference_workings_name_every_step_in_order():
    problem = _problem(
        golden_calls=[
            GoldenCall(tool="horwitz_trumpet", output=2.8284, units="%", note="实验室间RSD"),
            GoldenCall(
                tool="intra_laboratory_rsd",
                output=1.4142,
                units="%",
                note="班级内最小期望RSD",
            ),
        ]
    )

    workings, final_value = _sciagentgym_reference_workings(problem)

    assert workings.splitlines() == [
        "1. horwitz_trumpet → 2.8284 %（实验室间RSD）",
        "2. intra_laboratory_rsd → 1.4142 %（班级内最小期望RSD）",
    ]
    # The last step — not the `answer` field — is the question's answer.
    assert final_value == "1.4142 %"


def test_reference_workings_skip_artifact_paths():
    """A step that only draws a file has no value to compare."""
    problem = _problem(
        golden_calls=[
            GoldenCall(tool="count_atoms", output=12),
            GoldenCall(tool="plot_spectrum", output="./tool_visual_images/spectrum.png"),
        ]
    )

    assert _sciagentgym_reference_workings(problem) == ("", "")


def test_reference_workings_skip_nameless_steps():
    """A bare-string ``golden_answer`` loads as a tool name with no tool set."""
    problem = _problem(
        golden_calls=[
            GoldenCall(tool="", output="analyze_threshold"),
            GoldenCall(tool="", output="scan_energy"),
        ]
    )

    assert _sciagentgym_reference_workings(problem) == ("", "")


def test_reference_workings_need_at_least_two_steps():
    """One step adds nothing over ``answer``, so the prompt stays unchanged."""
    problem = _problem(golden_calls=[GoldenCall(tool="horwitz_trumpet", output=2.8284)])

    assert _sciagentgym_reference_workings(problem) == ("", "")


def test_reference_workings_render_a_flat_result_bundle():
    problem = _problem(
        golden_calls=[
            GoldenCall(tool="a", output={"x": 1.0, "y": 2.0}),
            GoldenCall(tool="b", output={"threshold_energy_GeV": 261000.0}),
        ]
    )

    workings, final_value = _sciagentgym_reference_workings(problem)

    assert workings.splitlines()[1] == "2. b → threshold_energy_GeV=261000"
    assert final_value == "261000"


def test_reference_workings_keep_only_the_numbers_in_a_bundle():
    """A bundle's nested lists are structure, not values; the numbers are."""
    problem = _problem(
        golden_calls=[
            GoldenCall(tool="a", output=1.0),
            GoldenCall(
                tool="b",
                output={"max_safe_load": 225882.0, "distribution": [1, 2, 3]},
            ),
        ]
    )

    workings, final_value = _sciagentgym_reference_workings(problem)

    assert workings.splitlines()[1] == "2. b → max_safe_load=225882"
    assert final_value == "225882"


def test_reference_workings_skip_bundles_with_no_numbers():
    problem = _problem(
        golden_calls=[
            GoldenCall(tool="a", output=1.0),
            GoldenCall(tool="b", output={"note": "drew the curve"}),
        ]
    )

    assert _sciagentgym_reference_workings(problem) == ("", "")


# ---------------------------------------------------------------------------
# 3. Wiring: the chain reaches the judge prompt
# ---------------------------------------------------------------------------


def _stub_judge_config(monkeypatch):
    monkeypatch.setattr(
        grader,
        "_llm_resolve_judge_config",
        lambda **_: SimpleNamespace(model="stub", api_base="http://stub"),
    )


def test_grade_sciagentgym_llm_hands_the_chain_to_the_judge(monkeypatch):
    _stub_judge_config(monkeypatch)
    captured: dict = {}

    def fake_judge(**kwargs):
        captured.update(kwargs)
        return True

    monkeypatch.setattr(grader, "_llm_judge_correct", fake_judge)
    problem = _problem(
        golden_calls=[
            GoldenCall(tool="horwitz_trumpet", output=2.8284, units="%"),
            GoldenCall(tool="intra_laboratory_rsd", output=1.4142, units="%"),
        ]
    )

    verdict, score, notes = grade_sciagentgym_llm(problem, "The answer is 1.4%.")

    assert (verdict, score) == (True, 1.0)
    assert captured["expected"] == "2.8%"
    assert "intra_laboratory_rsd" in captured["workings"]
    # The reader can see what the judge saw.
    assert any("metadata.golden_answer" in n for n in notes)


def test_grade_sciagentgym_llm_passes_no_workings_when_there_is_no_chain(monkeypatch):
    _stub_judge_config(monkeypatch)
    captured: dict = {}

    def fake_judge(**kwargs):
        captured.update(kwargs)
        return True

    monkeypatch.setattr(grader, "_llm_judge_correct", fake_judge)

    grade_sciagentgym_llm(_problem(), "2.8%")

    assert captured["workings"] is None


def test_grade_sciagentgym_llm_uses_the_last_step_when_answer_is_empty(monkeypatch):
    """With no ``answer`` field, judging against "" would always fail."""
    _stub_judge_config(monkeypatch)
    captured: dict = {}

    def fake_judge(**kwargs):
        captured.update(kwargs)
        return True

    monkeypatch.setattr(grader, "_llm_judge_correct", fake_judge)
    problem = _problem(
        answer="",
        golden_calls=[
            GoldenCall(tool="horwitz_trumpet", output=2.8284, units="%"),
            GoldenCall(tool="intra_laboratory_rsd", output=1.4142, units="%"),
        ],
    )

    grade_sciagentgym_llm(problem, "1.4%")

    assert captured["expected"] == "1.4142 %"


# ---------------------------------------------------------------------------
# 3b. What the dashboard shows as "expected"
# ---------------------------------------------------------------------------


def test_expected_display_leads_with_the_gold_it_judges_against():
    """The reported row: `expected: 2.8%` beside a 1.4% answer marked ✓.

    The panel explains a verdict, so it must lead with the value the verdict
    came from. On this entry that is 1.4142, not the `answer` field: 2.8% is
    the reference chain's own step 1, and the question asks for the later
    within-laboratory figure. Leading with 2.8% is exactly what made a
    correct 1.4% answer look misgraded.
    """
    problem = _problem(
        answer="2.8%",
        golden_chain_ordered=True,
        golden_calls=[
            GoldenCall(tool="horwitz_trumpet", output=2.8284, units="%"),
            GoldenCall(
                tool="intra_laboratory_rsd",
                output=1.4142,
                units="%",
                note="班级内最小期望RSD(0.5×between-lab)",
            ),
        ],
    )

    display = sciagentgym_expected_display(problem)

    assert display.startswith("1.4142 %")
    assert "判分依据" in display
    # The field that was set aside stays visible, labelled as an intermediate.
    assert "2.8%" in display
    assert "中间量" in display


def test_expected_display_keeps_the_answer_field_when_it_matches_no_step():
    """The pulley entry: `answer` is 13.1 m/s², which appears nowhere in a
    chain running 22.9 m/s² → -65.5 N. It is not an intermediate *of that
    chain*, so it stands and is what the judge accepted — unlike the Horwitz
    entry, where the field is literally the chain's own earlier step."""
    problem = _problem(
        answer="13.1 m/s^2",
        solution_steps=[
            "1. 建立受力方程并确定同绳加速度约束，采用净力/总质量法求a",
            "2. 调用 calculate_pulley_system_acceleration 得到加速度",
        ],
        golden_chain_ordered=True,
        golden_calls=[
            GoldenCall(tool="calculate_pulley_system_acceleration", output=22.9, units="m/s^2"),
            GoldenCall(tool="calculate_tension", output=-65.5, units="N"),
        ],
    )

    display = sciagentgym_expected_display(problem)

    assert display.startswith("13.1 m/s^2")
    assert "判分依据 = dataset answer 字段" in display
    # The steps were handed to the judge, so the panel must not imply the
    # field decided alone.
    assert "解题步骤" in display
    assert "中间量" not in display


def test_expected_display_is_unchanged_when_the_chain_agrees():
    problem = _problem(
        answer=r"\boxed{38.329\Omega}",
        golden_chain_ordered=True,
        golden_calls=[
            GoldenCall(tool="a", output=10.0),
            GoldenCall(tool="b", output=38.3287, units="Ω"),
        ],
    )

    assert sciagentgym_expected_display(problem) == problem.answer


def test_expected_display_ignores_unordered_chains():
    """A dict-shaped chain is a bag of tool calls; its "last step" is noise."""
    problem = _problem(
        answer="2.6*1e5 GeV",
        golden_chain_ordered=False,
        golden_calls=[
            GoldenCall(tool="analyze_threshold", output={"threshold_energy_GeV": 261000.0}),
            GoldenCall(tool="scan_energy", output={"scaling_law_slope": -1.0}),
        ],
    )

    assert sciagentgym_expected_display(problem) == "2.6*1e5 GeV"


def test_expected_display_falls_back_to_the_answer_field():
    assert sciagentgym_expected_display(_problem(answer="2.8%")) == "2.8%"


def test_expected_display_survives_a_non_string_units_field():
    """`units` is a dict in some entries; it must not leak into the report."""
    problem = _problem(
        answer="1.3 T",
        golden_chain_ordered=True,
        golden_calls=[
            GoldenCall(tool="a", output=1.0),
            GoldenCall(tool="b", output=1.43, units={"load": "N"}),
        ],
    )

    display = sciagentgym_expected_display(problem)

    assert display.startswith("1.3 T")
    assert "1.43" in display
    assert "{'load'" not in display


def test_grade_problem_reports_the_expected_value_it_judged_against(monkeypatch):
    """The per-problem record must match the value the verdict came from."""
    _stub_judge_config(monkeypatch)
    monkeypatch.setattr(grader, "_llm_judge_correct", lambda **_: True)
    problem = _problem(
        answer="2.8%",
        golden_chain_ordered=True,
        golden_calls=[
            GoldenCall(tool="horwitz_trumpet", output=2.8284, units="%"),
            GoldenCall(tool="intra_laboratory_rsd", output=1.4142, units="%"),
        ],
    )

    result = grader.grade_problem(problem, predicted_answer="1.4%", tool_calls=[])

    # The judge was asked about 1.4142, so that is what the record shows.
    assert result.expected_answer.startswith("1.4142 %")


# ---------------------------------------------------------------------------
# 4. The prompt itself
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, text):
        self._text = text

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": self._text}}]}


def _capture_prompt(monkeypatch, **judge_kwargs) -> str:
    monkeypatch.setenv("RESEARCH_CLAW_JUDGE_API_KEY", "sk-1")
    monkeypatch.setenv("RESEARCH_CLAW_JUDGE_MODEL", "gpt-4.1")
    captured: dict = {}

    def fake_post(url, headers=None, json=None):
        captured["prompt"] = json["messages"][1]["content"]
        return _FakeResponse("正确")

    fake_client = mock.MagicMock()
    fake_client.__enter__ = mock.MagicMock(return_value=fake_client)
    fake_client.__exit__ = mock.MagicMock(return_value=False)
    fake_client.post = fake_post

    import httpx

    with mock.patch.object(httpx, "Client", return_value=fake_client):
        assert judge_correct("q", "p", "e", **judge_kwargs) is True
    return captured["prompt"]


def test_prompt_is_unchanged_without_workings(monkeypatch):
    """Callers with no chain must still get the official prompt verbatim."""
    prompt = _capture_prompt(monkeypatch)

    assert "参考解题步骤" not in prompt
    assert "5." not in prompt
    assert "4. 若模型答案与标准答案**核心内容一致**，则判断为\"正确\"。" in prompt


def test_workings_are_added_as_context_only(monkeypatch):
    """The chain helps the judge read the question; it never overrides the
    standard answer. An earlier rule said "only the last step counts" and
    was removed — in most entries whose chain disagrees with ``answer`` the
    last step is an unrelated helper (a tension, a radius of gyration)."""
    prompt = _capture_prompt(
        monkeypatch, workings="1. horwitz_trumpet → 2.8284 %\n2. intra_laboratory_rsd → 1.4142 %"
    )

    assert "参考解题步骤" in prompt
    assert "2. intra_laboratory_rsd → 1.4142 %" in prompt
    assert "判断以标准答案为准" in prompt
    # The removed rule would have told the judge to prefer the last step.
    assert "只答出中间步骤" not in prompt
    assert "5." not in prompt


def test_blank_workings_add_nothing(monkeypatch):
    prompt = _capture_prompt(monkeypatch, workings="   \n  ")

    assert "参考解题步骤" not in prompt


# ---------------------------------------------------------------------------
# 4b. ``metadata.solution_steps`` — the dataset's own account of the solution
# ---------------------------------------------------------------------------


def test_solution_steps_add_the_block_and_a_rule(monkeypatch):
    """The steps block carries the rule with it: on entries where ``answer``
    holds an intermediate, the question is what decides. Without this the
    Horwitz-trumpet entry marks the correct 1.4% answer 错误."""
    prompt = _capture_prompt(
        monkeypatch,
        solution_steps=(
            "1. 依据Horwitz喇叭经验关系计算10 wt%的实验室间RSD\n"
            "2. 调用 intra_laboratory_rsd 以系数0.5估算班级内最小RSD\n"
            "3. 核对题意并记录最终数值回答\n"
        ),
    )

    assert "数据集标注的解题步骤" in prompt
    assert "2. 调用 intra_laboratory_rsd 以系数0.5估算班级内最小RSD" in prompt
    assert "请以问题实际所问、且解题步骤最后一步给出的量为准" in prompt
    # The official four rules must survive intact ahead of it.
    assert "4. 若模型答案与标准答案**核心内容一致**，则判断为\"正确\"。" in prompt


def test_no_solution_steps_leaves_the_official_prompt_alone(monkeypatch):
    """30 of the 83 entries carry no steps; those must be judged exactly as
    the official evaluator judges them."""
    prompt = _capture_prompt(monkeypatch)

    assert "数据集标注的解题步骤" not in prompt
    assert "请以问题实际所问" not in prompt
    # The placeholder must not leak through as literal text.
    assert "extra_rules" not in prompt
    assert "{extra_rules}" not in prompt


def test_blank_solution_steps_add_nothing(monkeypatch):
    prompt = _capture_prompt(monkeypatch, solution_steps="  \n ")

    assert "数据集标注的解题步骤" not in prompt
    assert "请以问题实际所问" not in prompt


def test_solution_steps_keep_their_own_numbering():
    """The dataset numbers its own steps; ours must not be added on top."""
    problem = _problem(solution_steps=["1. 先算这个", "  2. 再算那个  ", ""])

    assert _sciagentgym_solution_steps(problem) == "1. 先算这个\n2. 再算那个\n"


def test_no_solution_steps_render_to_an_empty_string():
    assert _sciagentgym_solution_steps(_problem()) == ""


def test_the_grader_passes_solution_steps_to_the_judge(monkeypatch):
    """End to end through the grading entry point, so a caller cannot
    silently drop the field on the way to the prompt."""
    _stub_judge_config(monkeypatch)
    captured: dict = {}

    def fake_judge(**kwargs):
        captured.update(kwargs)
        return True

    monkeypatch.setattr(grader, "_llm_judge_correct", fake_judge)
    problem = _problem(solution_steps=["1. 计算实验室间RSD", "2. 乘以0.5得班级内RSD"])

    verdict, score, notes = grade_sciagentgym_llm(problem, "1.41421 %")

    assert (verdict, score) == (True, 1.0)
    assert captured["solution_steps"] == "1. 计算实验室间RSD\n2. 乘以0.5得班级内RSD\n"
    assert any("metadata.solution_steps" in n for n in notes)


# ---------------------------------------------------------------------------
# 4c. Resolving an `answer` field that holds an intermediate
# ---------------------------------------------------------------------------


def _horwitz_problem(**overrides) -> Problem:
    base = dict(
        answer="2.8%",
        golden_chain_ordered=True,
        golden_calls=[
            GoldenCall(tool="horwitz_trumpet", output=2.8284271247461903, units="%"),
            GoldenCall(
                tool="intra_laboratory_rsd",
                output=1.4142135623730951,
                units="%",
                note="班级内最小期望RSD(0.5×between-lab)",
            ),
        ],
    )
    base.update(overrides)
    return _problem(**base)


def test_an_answer_that_is_an_earlier_step_is_replaced_by_the_final_one():
    """`2.8%` is `horwitz_trumpet`'s own output — step 1 of the reference
    chain, not the answer. The chain continues to 1.4142, which is what the
    question asks for, so that is the gold."""
    gold, reason = _sciagentgym_resolved_expected(_horwitz_problem())

    assert gold == "1.41421 %"
    assert "中间步骤" in reason


def test_an_answer_matching_no_step_is_kept():
    """The pulley entry: 13.1 m/s² is nowhere in the chain, so it is not an
    intermediate of it and must survive untouched."""
    problem = _problem(
        answer="13.1 m/s^2",
        golden_chain_ordered=True,
        golden_calls=[
            GoldenCall(tool="a", output=22.9, units="m/s^2"),
            GoldenCall(tool="b", output=-65.5, units="N"),
        ],
    )

    assert _sciagentgym_resolved_expected(problem) == ("13.1 m/s^2", "")


def test_an_answer_matching_the_last_step_is_kept():
    problem = _problem(
        answer="2.8%",
        golden_chain_ordered=True,
        golden_calls=[
            GoldenCall(tool="a", output=9.9, units="%"),
            GoldenCall(tool="b", output=2.8284271247461903, units="%"),
        ],
    )

    assert _sciagentgym_resolved_expected(problem) == ("2.8%", "")


def test_a_chain_ending_on_a_visualization_does_not_resolve():
    """Twelve entries in the multi-question dump match an earlier step but
    end on a `visualize_*` call with no value. There is nothing to move to,
    so the field stands — this is what keeps the rule to one entry."""
    problem = _problem(
        answer="5.56 m/s",
        golden_chain_ordered=True,
        golden_calls=[
            GoldenCall(tool="solve_final_velocity_conservation", output=5.56, units="m/s"),
            GoldenCall(tool="visualize", output=None),
        ],
    )

    assert _sciagentgym_resolved_expected(problem) == ("5.56 m/s", "")


def test_an_unordered_chain_never_resolves():
    """A dict-shaped chain is a bag of calls; "earlier" and "later" are
    meaningless in it."""
    problem = _problem(
        answer="2.8%",
        golden_chain_ordered=False,
        golden_calls=[
            GoldenCall(tool="a", output=2.8, units="%"),
            GoldenCall(tool="b", output=1.4142, units="%"),
        ],
    )

    assert _sciagentgym_resolved_expected(problem) == ("2.8%", "")


def test_the_resolved_gold_is_what_the_judge_is_asked_about(monkeypatch):
    """The point of the whole change: the judge must be asked about 1.4142,
    and must not additionally be handed the rule that distrusts 标准答案 —
    the gold no longer needs distrusting."""
    _stub_judge_config(monkeypatch)
    captured: dict = {}

    def fake_judge(**kwargs):
        captured.update(kwargs)
        return True

    monkeypatch.setattr(grader, "_llm_judge_correct", fake_judge)

    verdict, score, notes = grade_sciagentgym_llm(_horwitz_problem(), "1.41421 %")

    assert (verdict, score) == (True, 1.0)
    assert captured["expected"] == "1.41421 %"
    assert captured["solution_steps"] is None
    assert any("2.8%" in n and "中间步骤" in n for n in notes)


def test_the_grader_passes_no_solution_steps_when_there_are_none(monkeypatch):
    _stub_judge_config(monkeypatch)
    captured: dict = {}

    def fake_judge(**kwargs):
        captured.update(kwargs)
        return True

    monkeypatch.setattr(grader, "_llm_judge_correct", fake_judge)

    grade_sciagentgym_llm(_problem(), "1.41421 %")

    assert captured["solution_steps"] is None


# ---------------------------------------------------------------------------
# 5. A reasoning model needs room to think before it answers
# ---------------------------------------------------------------------------


class _RecordingClient:
    """An httpx client stub that records the token budget of each request."""

    def __init__(self, replies):
        self.budgets: list[int] = []
        self._replies = list(replies)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def post(self, url, headers=None, json=None):
        self.budgets.append(json["max_tokens"])
        return self._replies.pop(0)


def _stub_transport(monkeypatch, replies) -> _RecordingClient:
    monkeypatch.setenv("RESEARCH_CLAW_JUDGE_API_KEY", "sk-1")
    monkeypatch.setenv("RESEARCH_CLAW_JUDGE_MODEL", "deepseek-flash")
    client = _RecordingClient(replies)

    import httpx

    monkeypatch.setattr(httpx, "Client", lambda **_: client)
    return client


def test_reasoning_model_truncated_to_no_answer_is_retried_with_room(monkeypatch):
    """`max_tokens: 5` buys a reasoning model five tokens of thinking and no
    verdict — `content` comes back empty with `finish_reason: "length"`.
    That is a budget problem, not a wrong answer, so it must be retried."""
    truncated = _FakeResponse("")
    truncated.json = lambda: {
        "choices": [
            {"finish_reason": "length", "message": {"content": "", "reasoning_content": "We need"}}
        ]
    }
    client = _stub_transport(monkeypatch, [truncated, _FakeResponse("错误")])
    configured = llm_judge.resolve_judge_config().max_tokens

    assert judge_correct("q", "1.4%", "2.8%") is False
    assert client.budgets == [configured, llm_judge._REASONING_ESCALATION_TOKENS[0]]


def test_an_empty_answer_that_is_not_a_budget_problem_is_not_retried(monkeypatch):
    """`finish_reason: "stop"` with no text is a model problem; retrying the
    same prompt would only burn a second request."""
    empty = _FakeResponse("")
    empty.json = lambda: {"choices": [{"finish_reason": "stop", "message": {"content": ""}}]}
    client = _stub_transport(monkeypatch, [empty])
    configured = llm_judge.resolve_judge_config().max_tokens

    assert judge_correct("q", "1.4%", "2.8%") is None
    assert client.budgets == [configured]


def test_a_failed_judge_call_reports_why(monkeypatch):
    """The report must distinguish "no judge configured" from "the call
    failed", and from "it failed because the budget was too small"."""
    truncated = _FakeResponse("")
    truncated.json = lambda: {"choices": [{"finish_reason": "length", "message": {"content": ""}}]}
    client = _stub_transport(monkeypatch, [truncated] * 3)

    assert judge_correct("q", "1.4%", "2.8%") is None
    reason = llm_judge.get_last_judge_error()

    assert "no text" in reason
    # The retry escalates rather than giving up after one larger try: a case in
    # the probe exhausted 4096, and losing the verdict there falls back to the
    # deterministic grader, which is where the false positives came from.
    configured = llm_judge.resolve_judge_config().max_tokens
    expected = [configured] + [
        b for b in llm_judge._REASONING_ESCALATION_TOKENS if b > configured
    ]
    assert client.budgets == expected
    assert client.budgets == sorted(client.budgets)
    assert "reasoning model" in reason


def test_grade_problem_keeps_the_judge_failure_reason(monkeypatch):
    """`grade_problem` used to overwrite the judge's notes with a hardcoded
    "unavailable", hiding whether the judge was unconfigured or broken."""
    _stub_judge_config(monkeypatch)

    def failing_judge(**_):
        llm_judge._set_last_error("model 'x' exploded")
        return None

    monkeypatch.setattr(grader, "_llm_judge_correct", failing_judge)

    result = grader.grade_problem(_problem(answer="2.8%"), predicted_answer="2.8%", tool_calls=[])

    joined = " ".join(result.answer_notes)
    assert "exploded" in joined
    assert "fell back to the deterministic grader" in joined


# ---------------------------------------------------------------------------
# 6. The question's own figures reach the judge
# ---------------------------------------------------------------------------
#
# `metadata.image_path` is the *question's* chart, not a ground-truth target:
# unlike ResearchClawBench — whose judge attaches the target figure the report
# is scored against — these figures belong to the problem statement. What the
# judge does with them is a separate question; what these tests pin is that
# they arrive, that they arrive as the copies the solver was told to read, and
# that the record says so when they did not arrive at all.


def _capture_judge(monkeypatch):
    """Stub the judge, capture its kwargs, and answer 正确."""
    _stub_judge_config(monkeypatch)
    captured: dict = {}

    def fake_judge(**kwargs):
        captured.update(kwargs)
        return True

    monkeypatch.setattr(grader, "_llm_judge_correct", fake_judge)
    return captured


def test_grade_sciagentgym_llm_passes_the_question_figures(monkeypatch):
    captured = _capture_judge(monkeypatch)
    problem = _problem(
        image_paths=["/data/chart.png"],
        image_paths_local=["/runs/figure-assets/1/chart.png"],
    )

    _verdict, _score, notes = grade_sciagentgym_llm(problem, "2.8%")

    # The staged copies are what the solver was told to read, so they are what
    # the judge is shown — one set of bytes, one story about what was answered.
    assert captured["images"] == ["/runs/figure-assets/1/chart.png"]
    assert any("judge also saw 1 question figure(s)" in n for n in notes)
    assert any("chart.png" in n for n in notes)


def test_grade_sciagentgym_llm_falls_back_to_the_dataset_paths(monkeypatch):
    """A grader call that skipped staging still shows the judge the figure."""
    captured = _capture_judge(monkeypatch)

    grade_sciagentgym_llm(_problem(image_paths=["/data/chart.png"]), "2.8%")

    assert captured["images"] == ["/data/chart.png"]


def test_grade_sciagentgym_llm_sends_no_images_when_there_are_none(monkeypatch):
    captured = _capture_judge(monkeypatch)

    _verdict, _score, notes = grade_sciagentgym_llm(_problem(), "2.8%")

    assert captured["images"] is None
    assert not any("figure" in n for n in notes)


def test_grade_sciagentgym_llm_judges_the_bare_question(monkeypatch):
    """The figure paths travel with the images, not with the question: a judged
    question carrying local paths would be a different question."""
    captured = _capture_judge(monkeypatch)
    problem = _problem(image_paths_local=["/runs/figure-assets/1/chart.png"])

    grade_sciagentgym_llm(problem, "2.8%")

    assert captured["question"] == problem.question


def test_grade_sciagentgym_llm_reports_figures_that_did_not_reach_the_judge(
    monkeypatch,
):
    """A chart-reading answer can be marked 错误 by a judge that never saw the
    chart — the row has to make that visible."""
    _capture_judge(monkeypatch)
    monkeypatch.setattr(
        grader,
        "_llm_last_image_note",
        lambda: "1 question image(s) not sent — vision call failed (HTTP 400 …)",
    )
    problem = _problem(image_paths=["/data/chart.png"])

    _verdict, _score, notes = grade_sciagentgym_llm(problem, "2.8%")

    assert any("NOT sent to the judge" in n for n in notes)
    assert not any("judge also saw" in n for n in notes)


def test_grade_sciagentgym_llm_reports_figures_named_but_missing(monkeypatch):
    """None of the named figures resolved, so the problem looks figureless;
    the record must say otherwise."""
    _capture_judge(monkeypatch)
    problem = _problem(image_paths_missing=["gym/test_images/gone.png"])

    _verdict, _score, notes = grade_sciagentgym_llm(problem, "2.8%")

    assert any("do not exist on disk" in n for n in notes)
    assert any("gone.png" in n for n in notes)
