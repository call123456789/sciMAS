from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from dataset import MADD, load_dataset
from grader import ToolCall, grade_problem


class MaddDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.problems = load_dataset(REPO_ROOT / "dataset" / MADD, dataset_name=MADD)

    def test_loader_reads_migrated_benchmark(self) -> None:
        self.assertEqual(len(self.problems), 100)
        first = self.problems[0]

        self.assertEqual(first.id, "MADD_0001")
        self.assertEqual(first.dataset_name, MADD)
        self.assertEqual(first.subject, "Pharmaceutical Sciences")
        self.assertIn("KRAS G12C", first.question)
        self.assertNotIn("gen_mols_lung_cancer", first.question)
        self.assertEqual(
            first.expected_tools[:4],
            [
                "gen_mols_lung_cancer",
                "gen_mols_multiple_sclerosis",
                "gen_mols_dyslipidemia",
                "gen_mols_parkinson",
            ],
        )
        self.assertTrue(first.task_info["madd_tool_answers"])

    def test_grader_scores_golden_madd_answer(self) -> None:
        first = self.problems[0]
        calls = [
            ToolCall("generate_molecules_by_case", "pharma-drug-discovery", {"case": "lung cancer"}),
            ToolCall("generate_molecules_by_case", "pharma-drug-discovery", {"case": "multiple sclerosis"}),
            ToolCall("generate_molecules_by_case", "pharma-drug-discovery", {"case": "dyslipidemia"}),
            ToolCall("generate_molecules_by_case", "pharma-drug-discovery", {"case": "parkinson"}),
        ]

        result = grade_problem(first, predicted_answer=first.answer, tool_calls=calls)

        self.assertTrue(result.answer_correct)
        self.assertEqual(result.answer_score, 1.0)
        self.assertEqual(result.tool_coverage, 1.0)

    def test_madd_tool_mapping_respects_case_arguments(self) -> None:
        first = self.problems[0]
        calls = [
            ToolCall("generate_molecules_by_case", "pharma-drug-discovery", {"case": "lung cancer"}),
        ]

        result = grade_problem(first, predicted_answer=first.answer, tool_calls=calls)

        self.assertEqual(result.tool_coverage, 0.25)
        self.assertIn("gen_mols_multiple_sclerosis", result.missing_tools)


if __name__ == "__main__":
    unittest.main()
