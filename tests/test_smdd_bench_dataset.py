from pathlib import Path
import json
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from dataset import SMDD_BENCH, load_dataset
from grader import grade_problem


class SMDDBenchDatasetTests(unittest.TestCase):
    def _write_fixture(self, root: Path) -> Path:
        tasks_root = root / "tasks"
        task_dir = tasks_root / "004_lead_optimization_smoke"
        task_dir.mkdir(parents=True)
        (task_dir / "ligand.smi").write_text("CCO ethanol\n", encoding="utf-8")
        (task_dir / "task.yaml").write_text(
            "\n".join(
                [
                    "id: 004_lead_optimization_smoke",
                    "type: '004'",
                    "description: Optimize the seed ligand while preserving the alcohol motif.",
                    "output:",
                    "  file_path: answer.smi",
                    "evaluation:",
                    "  metric: toy",
                ]
            ),
            encoding="utf-8",
        )
        return task_dir

    def test_loader_reads_task_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)

            problems = load_dataset(root, dataset_name=SMDD_BENCH)

        self.assertEqual(len(problems), 1)
        first = problems[0]
        self.assertEqual(first.id, "004_lead_optimization_smoke")
        self.assertEqual(first.dataset_name, SMDD_BENCH)
        self.assertEqual(first.subject, "Drug Discovery")
        self.assertEqual(first.topic, "Lead Optimization")
        self.assertIn("Optimize the seed ligand", first.question)
        self.assertIn("answer.smi", first.question)
        self.assertEqual(len(first.data_files), 1)
        self.assertEqual(first.task_info["smdd_output_file"], "answer.smi")
        self.assertEqual(first.task_info["smdd_type_code"], "004")

    def test_grader_reads_official_result_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = self._write_fixture(root)
            (task_dir / "result.json").write_text(
                json.dumps({"score": 0.75}),
                encoding="utf-8",
            )
            problem = load_dataset(root, dataset_name=SMDD_BENCH)[0]
            result = grade_problem(problem, predicted_answer="CCCO", tool_calls=[])

        self.assertFalse(result.answer_correct)
        self.assertEqual(result.answer_score, 0.75)
        self.assertTrue(any("official evaluator" in note for note in result.answer_notes))

    def test_grader_marks_missing_official_score_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            problem = load_dataset(root, dataset_name=SMDD_BENCH)[0]

        result = grade_problem(problem, predicted_answer="CCCO", tool_calls=[])

        self.assertFalse(result.answer_correct)
        self.assertEqual(result.answer_score, 0.0)
        self.assertTrue(any("official score unavailable" in note for note in result.answer_notes))


if __name__ == "__main__":
    unittest.main()
