from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from dataset import DRUG_DISCOVERY_BENCH, Problem, load_dataset
from grader import grade_problem


class DrugDiscoveryBenchDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.problems = load_dataset(
            REPO_ROOT / "dataset" / DRUG_DISCOVERY_BENCH,
            dataset_name=DRUG_DISCOVERY_BENCH,
        )

    def test_loader_reads_migrated_harbor_tasks(self) -> None:
        self.assertEqual(len(self.problems), 82)
        first = self.problems[0]

        self.assertEqual(first.id, "69b025e20c10fe76b7aaf812")
        self.assertEqual(first.dataset_name, DRUG_DISCOVERY_BENCH)
        self.assertEqual(first.subject, "Biomedical Sciences")
        self.assertIn("colon cancer", first.question)
        self.assertNotIn("/workspace/answer.md. Keep it terse", first.question)
        self.assertIn("ddb_outcome_rubrics", first.task_info)
        self.assertEqual(first.task_info["ddb_outcome_rubrics"], [])

    def test_loader_exposes_environment_inputs(self) -> None:
        by_id = {str(problem.id): problem for problem in self.problems}
        task = by_id["6a232796afa0e46251ad3d57"]

        self.assertEqual(len(task.data_files), 2)
        self.assertTrue(all(Path(path).is_absolute() for path in task.data_files))
        self.assertIn("fda-substances.csv", task.question)

    def test_empty_public_rubrics_are_reported_as_ungradable(self) -> None:
        first = self.problems[0]

        result = grade_problem(first, predicted_answer="YES, 30.0% and 25.0%", tool_calls=[])

        self.assertFalse(result.answer_correct)
        self.assertEqual(result.answer_score, 0.0)
        self.assertTrue(any("populate_rubrics.py" in note for note in result.answer_notes))

    def test_populated_ground_truth_scores_with_existing_answer_matcher(self) -> None:
        problem = Problem(
            id="ddb-synthetic",
            filename="synthetic",
            question="Synthetic DrugDiscoveryBench problem",
            answer="NO, 30.0% vs 0.0%",
            subject="Biomedical Sciences",
            topic="biomni",
            dataset_name=DRUG_DISCOVERY_BENCH,
            task_info={
                "ddb_ground_truth": "NO, 30.0% vs 0.0%",
                "ddb_outcome_rubrics": [],
                "ddb_process_rubrics": [],
            },
        )

        result = grade_problem(problem, predicted_answer="NO, 30.0% vs 0.0%", tool_calls=[])

        self.assertTrue(result.answer_correct)
        self.assertEqual(result.answer_score, 1.0)


if __name__ == "__main__":
    unittest.main()
