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
from scripts.smdd_official_eval import (
    evaluate_task,
    extract_artifact_content,
    write_agent_output,
)


class SMDDBenchOfficialEvalTests(unittest.TestCase):
    def _write_task(self, root: Path) -> Path:
        task_dir = root / "tasks" / "smdd_001_fixture_0"
        task_dir.mkdir(parents=True)
        (task_dir / "task.yaml").write_text(
            "\n".join(
                [
                    "id: smdd_001_fixture_0",
                    "description: Write a Python pharmacophore checker.",
                    "output_config:",
                    "  file_path: solution.py",
                    "evaluation: []",
                ]
            ),
            encoding="utf-8",
        )
        return task_dir

    def test_extracts_matching_fenced_artifact(self) -> None:
        answer = "Use this file:\n```python\ndef check_pharmacophore(smiles):\n    return True\n```"

        artifact = extract_artifact_content(answer, "solution.py")

        self.assertEqual(
            artifact,
            "def check_pharmacophore(smiles):\n    return True",
        )

    def test_writes_official_agent_output_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = self._write_task(root)
            output_path = write_agent_output(
                task_dir,
                "```python\ndef check_pharmacophore(smiles):\n    return True\n```",
                root / "agent_outputs",
            )

            self.assertEqual(output_path.name, "solution.py")
            self.assertEqual(output_path.parent.name, "smdd_001_fixture_0")
            self.assertIn("check_pharmacophore", output_path.read_text(encoding="utf-8"))

    def test_evaluate_task_writes_error_when_artifact_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = self._write_task(root)

            result = evaluate_task(
                task_dir,
                root / "agent_outputs",
                root / "results",
                docker_image="definitely-missing-smdd-image",
            )
            result_path = root / "results" / "smdd_001_fixture_0" / "result.json"

            self.assertEqual(result["status"], "errored")
            self.assertIn("Missing agent output artifact", result["error"])
            self.assertTrue(result_path.exists())

    def test_grader_scores_official_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = self._write_task(root)
            (task_dir / "result.json").write_text(
                json.dumps(
                    {
                        "task_id": "smdd_001_fixture_0",
                        "status": "failed",
                        "steps": [
                            {"name": "one", "status": "passed"},
                            {"name": "two", "status": "failed"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            problem = load_dataset(root, dataset_name=SMDD_BENCH)[0]

            result = grade_problem(problem, predicted_answer="", tool_calls=[])

        self.assertFalse(result.answer_correct)
        self.assertEqual(result.answer_score, 0.5)
        self.assertTrue(any("passed_steps=1/2" in note for note in result.answer_notes))


if __name__ == "__main__":
    unittest.main()
