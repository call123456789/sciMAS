from pathlib import Path
import csv
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from dataset import SCIPREDICT, load_dataset
from grader import grade_problem


class SciPredictDatasetTests(unittest.TestCase):
    def _write_fixture(self, root: Path) -> None:
        data_dir = root / "data"
        data_dir.mkdir(parents=True)
        rows = [
            {
                "TASK": "sci-mcq",
                "DOMAIN": "Physics",
                "FIELD": "Experimental Condensed Matter Physics",
                "PQ_FORMAT": "MCQ",
                "TITLE": "Synthetic MCQ",
                "URL": "https://example.test/mcq",
                "PUBLISHING_DATE": "September 16, 2025",
                "EXPERIMENTAL_SETUP": "A copper target is compressed and probed.",
                "MEASUREMENT_TAKEN": "X-ray absorption spectra.",
                "OUTCOME_PREDICTION_QUESTION": "Which option is closest? A. low B. high C. none",
                "GTA": "B. high",
                "CLEAN_GTA": "B",
                "REQUIRED_BACKGROUND_KNOWLEDGE": "Warm dense matter raises ionization.",
            },
            {
                "TASK": "sci-num",
                "DOMAIN": "Chemistry",
                "FIELD": "Physical Chemistry",
                "PQ_FORMAT": "Numerical",
                "TITLE": "Synthetic numerical",
                "URL": "https://example.test/num",
                "PUBLISHING_DATE": "April 2, 2026",
                "EXPERIMENTAL_SETUP": "A reaction is monitored over time.",
                "MEASUREMENT_TAKEN": "Yield percentage.",
                "OUTCOME_PREDICTION_QUESTION": "Predict the final yield.",
                "GTA": "7.2-8.8%",
                "CLEAN_GTA": "7.2-8.8%",
                "REQUIRED_BACKGROUND_KNOWLEDGE": "The catalyst improves yield.",
            },
            {
                "TASK": "sci-free",
                "DOMAIN": "Biology",
                "FIELD": "Cell Biology",
                "PQ_FORMAT": "Free-form",
                "TITLE": "Synthetic free form",
                "URL": "https://example.test/free",
                "PUBLISHING_DATE": "May 1, 2026",
                "EXPERIMENTAL_SETUP": "Cells receive a stress perturbation.",
                "MEASUREMENT_TAKEN": "Transcriptomic response.",
                "OUTCOME_PREDICTION_QUESTION": "Describe the response.",
                "GTA": "Stress response genes increase.",
                "CLEAN_GTA": "Stress response genes increase.",
                "REQUIRED_BACKGROUND_KNOWLEDGE": "Heat shock induces stress genes.",
            },
        ]
        with (data_dir / "main_ds.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        rubric_rows = [
            {
                "TASK": "sci-free",
                "CRITERION_1": "mentions stress response genes increase",
            }
        ]
        with (data_dir / "rubrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rubric_rows[0].keys()))
            writer.writeheader()
            writer.writerows(rubric_rows)

    def test_loader_reads_scipredict_csv_and_background_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)

            problems = load_dataset(root, dataset_name=SCIPREDICT)
            no_background = load_dataset(root, dataset_name=SCIPREDICT, split="nbk")

        self.assertEqual(len(problems), 3)
        first = problems[0]
        self.assertEqual(first.id, "sci-mcq")
        self.assertEqual(first.dataset_name, SCIPREDICT)
        self.assertEqual(first.subject, "Physics")
        self.assertIn("Background knowledge", first.question)
        self.assertEqual(first.answer, "B")
        self.assertEqual(no_background[0].id, "sci-mcq")
        self.assertNotIn("Background knowledge", no_background[0].question)
        self.assertIn("scipredict_raw_row", first.task_info)

    def test_grader_scores_mcq_numeric_and_free_form(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            problems = {str(problem.id): problem for problem in load_dataset(root, dataset_name=SCIPREDICT)}

        mcq = grade_problem(problems["sci-mcq"], predicted_answer="Answer: B", tool_calls=[])
        numeric = grade_problem(problems["sci-num"], predicted_answer="The final yield is about 8.1%.", tool_calls=[])
        free = grade_problem(
            problems["sci-free"],
            predicted_answer="Stress response genes increase after the perturbation.",
            tool_calls=[],
        )

        self.assertTrue(mcq.answer_correct)
        self.assertEqual(mcq.answer_score, 1.0)
        self.assertTrue(numeric.answer_correct)
        self.assertEqual(numeric.answer_score, 1.0)
        self.assertTrue(free.answer_correct)
        self.assertGreaterEqual(free.answer_score, 0.5)


if __name__ == "__main__":
    unittest.main()
