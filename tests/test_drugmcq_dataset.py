"""Tests for the DrugMCQ loader and deterministic grader.

DrugMCQ reads ``dataset/MCQ.jsonl`` (50 records, A-J options) and grades by
letter extraction: exact set match → 1.0, partial overlap → recall on the
gold set. There is no LLM judge — the local scorer is the grader.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from dataset import (  # noqa: E402
    DATASET_CHOICES,
    DRUGMCQ,
    default_dataset_root,
    load_dataset,
)
from grader import grade_drugmcq_answer, grade_problem  # noqa: E402


class DrugMCQDatasetTests(unittest.TestCase):
    def _write_fixture(self, root: Path, *, with_kegg: bool = True) -> Path:
        path = root / "MCQ.jsonl"
        ti_row = {
            "id": "target_identification_15",
            "task_type": "target_identification",
            "disease": "Ulcerative Colitis",
            "question": "Which genes represent the most effective therapeutic targets?",
            "options": [
                "A. TNF - cytokine",
                "B. IL10 - cytokine",
                "C. VEGFA - growth factor",
                "D. MMP9 - enzyme",
                "E. PTPN2 - phosphatase",
                "F. SMAD7 - transcription regulator",
                "G. NOD2 - pattern recognition receptor",
                "H. IL6 - cytokine",
            ],
            "answers": ["A", "H"],
            "rationale": "TNF and IL6 are central drivers.",
            "drug_discovery_stage": "target_discovery",
        }
        if with_kegg:
            ti_row["kegg_evidence"] = {
                "pathways": ["Inflammatory bowel disease (IBD)"],
                "target_gene_ids": ["hsa:7124", "hsa:3569"],
            }
        moa_row = {
            "id": "preclinical_research_0",
            "task_type": "moa_pathway_reasoning",
            "question": "Which statements describe the PI3Kalpha inhibitor?",
            "options": [
                "A. It selectively inhibits the PI3Kalpha isoform.",
                "B. Phase I trial for advanced solid tumors.",
                "C. Was developed by a multinational pharma.",
                "D. Orally bioavailable with improved safety.",
                "E. Selectivity over PI3Kbeta was not assessed.",
                "F. Improved PK relative to early-generation inhibitors.",
            ],
            "answers": ["A", "B", "D", "F"],
            "rationale": "Statements A, B, D, and F are confirmed.",
            "drug_discovery_stage": "mechanism_validation",
        }
        path.write_text(
            "\n\n".join(json.dumps(row, ensure_ascii=False, indent=2) for row in [ti_row, moa_row]),
            encoding="utf-8",
        )
        return path

    def test_loader_reads_pretty_printed_concatenated_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._write_fixture(root)

            problems = load_dataset(root, dataset_name=DRUGMCQ)

        self.assertEqual(len(problems), 2)
        first = problems[0]
        self.assertEqual(first.id, "target_identification_15")
        self.assertEqual(Path(first.filename), source.resolve())
        self.assertEqual(first.dataset_name, DRUGMCQ)
        self.assertEqual(first.subject, "DrugMCQ")
        self.assertEqual(first.topic, "Target Identification")
        self.assertIn("Disease/context: Ulcerative Colitis", first.question)
        self.assertIn("A. TNF - cytokine", first.question)
        self.assertIn("Begin your answer with the chosen option letter", first.question)
        self.assertNotIn("TNF and IL6 are central", first.question)
        self.assertEqual(first.task_info["drugmcq_answers"], ["A", "H"])
        self.assertEqual(first.task_info["drugmcq_options"][0], "A. TNF - cytokine")
        self.assertEqual(first.answer, "A, H")

    def test_loader_skips_disease_when_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root, with_kegg=False)
            problems = load_dataset(root, dataset_name=DRUGMCQ)
        moa = next(p for p in problems if p.topic == "Moa Pathway Reasoning")
        self.assertNotIn("Disease/context", moa.question)
        self.assertEqual(moa.subject, "DrugMCQ")
        self.assertEqual(moa.task_info["drugmcq_disease"], "")
        self.assertEqual(moa.task_info["drugmcq_answers"], ["A", "B", "D", "F"])

    def test_loader_accepts_direct_file_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = self._write_fixture(Path(tmp))

            problems = load_dataset(source, dataset_name="drugmcq")

        self.assertEqual(
            [problem.id for problem in problems],
            ["target_identification_15", "preclinical_research_0"],
        )

    def test_loader_normalizes_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._write_fixture(Path(tmp))
            for alias in ("DrugMCQ", "drugmcq", "drug-mcq", "drug_mcq", "mcq"):
                problems = load_dataset(Path(tmp), dataset_name=alias)
                self.assertEqual(len(problems), 2, msg=f"alias {alias!r} did not resolve")

    def test_grader_exact_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            problem = load_dataset(root, dataset_name=DRUGMCQ)[0]

            result = grade_problem(
                problem, predicted_answer="Answer: A, H", tool_calls=[]
            )

        self.assertTrue(result.answer_correct)
        self.assertEqual(result.answer_score, 1.0)
        self.assertEqual(result.answer_verdict_source, "deterministic")
        self.assertTrue(
            any("hit_rate=1.00" in note for note in result.answer_notes),
            msg=f"hit_rate=1.00 not in notes: {result.answer_notes}",
        )

    def test_grader_exact_match_through_prose(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            problem = load_dataset(root, dataset_name=DRUGMCQ)[0]

            # Model never wrote "Answer:" — only the letters in prose.
            # Under the parser-restricted grader this scores 0.0 because
            # there is no structured marker; this test pins that
            # behaviour so prose-letters cannot regress into votes.
            result = grade_problem(
                problem, predicted_answer="The two cytokines are A and H.", tool_calls=[]
            )

        self.assertFalse(result.answer_correct)
        self.assertEqual(result.answer_score, 0.0)
        self.assertTrue(
            any("no structured Answer: marker" in note for note in result.answer_notes),
            msg=result.answer_notes,
        )

    def test_grader_partial_credit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            problem = load_dataset(root, dataset_name=DRUGMCQ)[0]

            result_one = grade_problem(
                problem, predicted_answer="Answer: A", tool_calls=[]
            )
            self.assertFalse(result_one.answer_correct)
            self.assertEqual(result_one.answer_score, 0.5)
            self.assertTrue(
                any("missing gold letters: ['H']" in note for note in result_one.answer_notes),
                msg=result_one.answer_notes,
            )

            result_wrong = grade_problem(
                problem, predicted_answer="Answer: B", tool_calls=[]
            )
            self.assertFalse(result_wrong.answer_correct)
            self.assertEqual(result_wrong.answer_score, 0.0)

    def test_grader_no_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            problem = load_dataset(root, dataset_name=DRUGMCQ)[0]

            result = grade_problem(
                problem,
                predicted_answer="The answer relates to cytokines and inflammation.",
                tool_calls=[],
            )

        self.assertFalse(result.answer_correct)
        self.assertEqual(result.answer_score, 0.0)
        self.assertTrue(
            any("hit_rate=0.00" in note for note in result.answer_notes),
            msg=result.answer_notes,
        )

    def test_grader_handles_letters_beyond_h(self) -> None:
        # The shared ``_MCQ_LETTER_RE`` (A-H) silently drops I and J. The
        # DrugMCQ-private regex must accept them; otherwise the score is
        # wrong even when the model picks every right letter.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "MCQ.jsonl"
            row = {
                "id": "target_identification_jj",
                "task_type": "target_identification",
                "disease": "Test",
                "question": "Pick two late-alphabet options.",
                "options": [f"{chr(ord('A') + i)}. opt-{i}" for i in range(10)],  # A..J
                "answers": ["I", "J"],
                "rationale": "We want the last two.",
                "drug_discovery_stage": "target_discovery",
            }
            path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
            problem = load_dataset(root, dataset_name=DRUGMCQ)[0]

            result = grade_problem(
                problem, predicted_answer="Answer: I, J", tool_calls=[]
            )

        self.assertTrue(result.answer_correct)
        self.assertEqual(result.answer_score, 1.0)
        self.assertEqual(problem.task_info["drugmcq_answers"], ["I", "J"])

    def test_grader_notes_extras(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            problem = load_dataset(root, dataset_name=DRUGMCQ)[0]

            # All gold + an extra letter from the A-J set. Recall is still 1.0
            # but the verdict is False because the sets are not equal — and
            # the grader should surface the extra letter in its notes.
            # (A non-option letter like "Z" would be silently dropped by the
            # A-J regex; using a valid one proves the extras path runs.)
            result = grade_problem(
                problem, predicted_answer="Answer: A, H, B", tool_calls=[]
            )

        self.assertFalse(result.answer_correct)
        self.assertEqual(result.answer_score, 1.0)
        self.assertTrue(
            any("extra letters not in gold" in note for note in result.answer_notes),
            msg=result.answer_notes,
        )

    def test_grader_empty_predicted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            problem = load_dataset(root, dataset_name=DRUGMCQ)[0]

            correct, score, notes = grade_drugmcq_answer(problem, "")

        self.assertFalse(correct)
        self.assertEqual(score, 0.0)
        self.assertTrue(any("no predicted answer" in n for n in notes), msg=notes)

    def test_dashboard_dataset_choice_is_available(self) -> None:
        self.assertIn(DRUGMCQ, DATASET_CHOICES)
        root = default_dataset_root(DRUGMCQ)
        self.assertTrue(
            str(root).endswith("/dataset") or root.is_file(),
            msg=f"unexpected default root {root!r}",
        )

    def test_dataset_documents_drugmcq_in_module(self) -> None:
        # Loader sanity: dataset.DATASET_CHOICES should include DrugMCQ
        # (dashboard surfaces it automatically) and aliases should include
        # the lowercase / hyphenated forms.
        from dataset import _DATASET_ALIASES

        self.assertEqual(_DATASET_ALIASES["drugmcq"], DRUGMCQ)
        self.assertEqual(_DATASET_ALIASES["mcq"], DRUGMCQ)

    # ------------------------------------------------------------------
    # Regression tests for the parser-restricted grader. The DrugMCQ
    # grader used to scan the whole predicted_answer text — including
    # prose like "**C** — Contradicted", "H1047R", "Phase I", "rat F" —
    # and reported those prose letters as "extra letters not in gold"
    # on every correct response. The fix is to only count letters that
    # appear inside a structured answer marker (JSON ``"answer":`` field,
    # ``Answer: …`` line, ``## Final answer`` + next line). These cases
    # pin the new behaviour so the old behaviour cannot regress.
    # ------------------------------------------------------------------

    def _make_problem(self, tmp_dir):
        root = Path(tmp_dir)
        self._write_fixture(root)
        return load_dataset(root, dataset_name=DRUGMCQ)[0]

    def test_grader_ignores_prose_letters_when_answer_marker_present(self) -> None:
        # This is the exact failure mode observed in the
        # 20260923-124534-513fd5e5 run on DrugMCQ: the model wrote
        # ``Answer: A, H`` correctly but the prose then mentioned
        # ``**B** — same Q1 feature``, ``H1047R``, ``Phase I``, ``rat F ≈``
        # and the previous grader reported B/C/E/F/G/I as extras.
        with tempfile.TemporaryDirectory() as tmp:
            problem = self._make_problem(tmp)
            predicted = (
                "Answer: A, H\n\n"
                "Some prose mentioning ``**B** — same Q1 feature`` and "
                "the compound name H1047R, plus Phase I, plus ``rat F ≈ "
                "37%``, plus variables like PIK3CA and EGFRvIII. "
                "Even more capital letters: G, I, C, D, E, F. All prose.\n"
            )
            result = grade_problem(problem, predicted_answer=predicted, tool_calls=[])
        self.assertTrue(
            result.answer_correct,
            msg=f"expected exact match A,H; notes={result.answer_notes}",
        )
        self.assertEqual(result.answer_score, 1.0)

    def test_grader_accepts_bold_markdown_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            problem = self._make_problem(tmp)
            predicted = (
                "Here is some reasoning first.\n\n"
                "**Answer: A, H**\n\n"
                "- **A** — TNF is a central driver (corroborated)\n"
                "- **H** — IL6 is the cytokine partner\n"
            )
            result = grade_problem(problem, predicted_answer=predicted, tool_calls=[])
        self.assertTrue(result.answer_correct)
        self.assertEqual(result.answer_score, 1.0)

    def test_grader_accepts_markdown_header_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            problem = self._make_problem(tmp)
            predicted = (
                "## Final answer\n\n"
                "Answer: A, H\n\n"
                "Prose mentioning **B** and **G** is unrelated.\n"
            )
            result = grade_problem(problem, predicted_answer=predicted, tool_calls=[])
        self.assertTrue(result.answer_correct)
        self.assertEqual(result.answer_score, 1.0)

    def test_grader_accepts_json_answer_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            problem = self._make_problem(tmp)
            predicted = (
                '{\n  "passed": true,\n  "answer": "A, H",\n  "reason": "..."\n}'
            )
            result = grade_problem(problem, predicted_answer=predicted, tool_calls=[])
        self.assertTrue(result.answer_correct)
        self.assertEqual(result.answer_score, 1.0)

    def test_grader_picks_first_answer_marker_when_multiple(self) -> None:
        # Two markers — model first wrote a draft, then corrected itself.
        # Use the FIRST one.
        with tempfile.TemporaryDirectory() as tmp:
            problem = self._make_problem(tmp)
            predicted = (
                "Answer: A, B\n\n"  # draft — wrong (B is an extra)
                "Actually, after review:\n"
                "Answer: A, H\n"
            )
            result = grade_problem(problem, predicted_answer=predicted, tool_calls=[])
        self.assertFalse(
            result.answer_correct,
            msg="first marker (A, B) wins; not exact match against gold A, H",
        )
        # the first marker had A right and B as an extra — so recall=0.5, score=0.5
        self.assertEqual(result.answer_score, 0.5)
        self.assertTrue(
            any("extra letters not in gold: ['B']" in n for n in result.answer_notes),
            msg=result.answer_notes,
        )

    def test_grader_returns_zero_when_no_marker(self) -> None:
        # Model never wrote ``Answer: …`` and did not return JSON. Prose
        # mentions many capital letters — under the OLD grader these would
        # be counted; under the NEW grader they are all ignored.
        with tempfile.TemporaryDirectory() as tmp:
            problem = self._make_problem(tmp)
            predicted = (
                "Some prose full of capital letters: A, B, C, D, E, F, G, H, "
                "I, J, K, TNF, IL6, IL10, PIK3CA, EGFRvIII, NCT03544905, "
                "Phase I, rat F ≈ 37%, H1047R, E545K, R26-Pik3ca."
            )
            result = grade_problem(problem, predicted_answer=predicted, tool_calls=[])
        self.assertFalse(result.answer_correct)
        self.assertEqual(result.answer_score, 0.0)
        self.assertTrue(
            any("no structured Answer: marker" in n for n in result.answer_notes),
            msg=result.answer_notes,
        )

    def test_grader_trailing_answer_marker_is_picked(self) -> None:
        # Some models only summarise it at the very end after a long
        # explanation. The grader must scan the full text, not just the
        # first line.
        with tempfile.TemporaryDirectory() as tmp:
            problem = self._make_problem(tmp)
            predicted = (
                "Long explanation… **B**, **C**, **G**, **I** appear in prose.\n"
                "More text mentioning H1047R, Phase I, rat F.\n"
                "Answer: A, H\n"
            )
            result = grade_problem(problem, predicted_answer=predicted, tool_calls=[])
        self.assertTrue(result.answer_correct)
        self.assertEqual(result.answer_score, 1.0)

    def test_grader_extracts_marker_helper_directly(self) -> None:
        # Direct unit test for the marker extractor, so future refactors
        # of ``grade_drugmcq_answer`` cannot accidentally widen its scope.
        from grader import _extract_drugmcq_answer_text

        cases = [
            ("Answer: A, H", "Answer: A, H"),
            ("**Answer: A, H**", "**Answer: A, H**"),
            ("## Answer: A, H\nbody", "## Answer: A, H"),
            ("## Final answer\n\nAnswer: A, H\nmore", "Answer: A, H"),
            (
                '{\n  "answer": "A, H",\n  "reason": "..."\n}',
                "A, H",
            ),
            (
                '{\n  "passed": true,\n  "answer": "A, H"\n}',
                "A, H",
            ),
            ("Prose only, no Answer: anywhere.", ""),
            ("", ""),
        ]
        for input_text, expected_marker in cases:
            with self.subTest(input_text=input_text):
                self.assertEqual(
                    _extract_drugmcq_answer_text(input_text),
                    expected_marker,
                )


if __name__ == "__main__":
    unittest.main()