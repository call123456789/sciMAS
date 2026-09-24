from pathlib import Path
import json
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from dataset import DATASET_CHOICES, DRUGQA, default_dataset_root, load_dataset
from grader import grade_problem


class DrugQADatasetTests(unittest.TestCase):
    def _write_fixture(self, root: Path) -> Path:
        path = root / "target_identification_and_moa_openended.jsonl"
        rows = [
            {
                "id": "target_identification_15",
                "task_type": "target_identification",
                "disease": "Ulcerative Colitis",
                "question": "Select two genes for modulating inflammation in UC.",
                "answers": ["TNF - cytokine", "IL6 - cytokine"],
                "kegg_evidence": {
                    "pathways": ["Inflammatory bowel disease (IBD)"],
                    "target_gene_ids": ["hsa:7124", "hsa:3569"],
                },
                "rationale": "TNF and IL6 are central inflammatory drivers.",
            },
            {
                "id": "preclinical_research_0",
                "task_type": "preclinical_research",
                "question": "Which statements describe the PI3K alpha inhibitor?",
                "answers": [
                    "It selectively inhibits the PI3Kalpha isoform with high potency.",
                    "It is orally bioavailable and optimized for improved safety.",
                ],
                "rationale": "Open-ended preclinical QA rationale.",
            },
        ]
        path.write_text(
            "\n\n".join(json.dumps(row, ensure_ascii=False, indent=2) for row in rows),
            encoding="utf-8",
        )
        return path

    def test_loader_reads_pretty_printed_concatenated_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._write_fixture(root)

            problems = load_dataset(root, dataset_name=DRUGQA)

        self.assertEqual(len(problems), 2)
        first = problems[0]
        self.assertEqual(first.id, "target_identification_15")
        self.assertEqual(Path(first.filename), source.resolve())
        self.assertEqual(first.dataset_name, DRUGQA)
        self.assertEqual(first.subject, "DrugQA")
        self.assertEqual(first.topic, "Target Identification")
        self.assertIn("Disease/context: Ulcerative Colitis", first.question)
        self.assertNotIn("TNF and IL6 are central", first.question)
        self.assertEqual(first.task_info["drugqa_answers"], ["TNF - cytokine", "IL6 - cytokine"])

    def test_loader_accepts_direct_file_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = self._write_fixture(Path(tmp))

            problems = load_dataset(source, dataset_name="target_identification_and_moa_openended")

        self.assertEqual([problem.id for problem in problems], ["target_identification_15", "preclinical_research_0"])

    def test_grader_scores_answer_list_items(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            problem = load_dataset(root, dataset_name=DRUGQA)[0]

        result = grade_problem(
            problem,
            predicted_answer="The two strongest targets are TNF and IL6.",
            tool_calls=[],
        )

        self.assertTrue(result.answer_correct)
        self.assertEqual(result.answer_score, 1.0)
        self.assertTrue(any("matched 2/2" in note for note in result.answer_notes))

    def test_grader_reports_partial_credit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            problem = load_dataset(root, dataset_name=DRUGQA)[0]

        result = grade_problem(problem, predicted_answer="TNF is the key cytokine.", tool_calls=[])

        self.assertFalse(result.answer_correct)
        self.assertEqual(result.answer_score, 0.5)

    def test_dashboard_dataset_choice_is_available(self) -> None:
        self.assertIn(DRUGQA, DATASET_CHOICES)
        self.assertTrue(
            str(default_dataset_root(DRUGQA)).endswith("/dataset")
            or default_dataset_root(DRUGQA).is_file()
        )


if __name__ == "__main__":
    unittest.main()
