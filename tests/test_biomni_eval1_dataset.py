from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from dataset import BIOMNI_EVAL1, load_dataset
from grader import grade_problem


class BiomniEval1DatasetTests(unittest.TestCase):
    def _write_fixture(self, root: Path) -> None:
        data_dir = root / "data"
        data_dir.mkdir(parents=True)
        frame = pd.DataFrame(
            [
                {
                    "instance_id": "lab_001",
                    "task_instance_id": "001",
                    "prompt": "Which lab result is most abnormal? A. ALT B. sodium",
                    "task_name": "lab_test_analyzing",
                    "split": "test",
                    "answer": "A",
                },
                {
                    "instance_id": "gene_001",
                    "task_instance_id": "002",
                    "prompt": "Convert the alias p53 to the approved symbol.",
                    "task_name": "gene_name_conversion",
                    "split": "test",
                    "answer": "TP53",
                },
                {
                    "instance_id": "rare_001",
                    "task_instance_id": "003",
                    "prompt": "Diagnose the rare disease.",
                    "task_name": "rare_disease_diagnosis",
                    "split": "test",
                    "answer": '{"disease_name": "Example disease", "OMIM_ID": "123456"}',
                },
                {
                    "instance_id": "patient_001",
                    "task_instance_id": "004",
                    "prompt": "Find the causal gene.",
                    "task_name": "patient_gene_detection",
                    "split": "test",
                    "answer": '{"causal_gene": ["BRCA1"]}',
                },
            ]
        )
        frame.to_parquet(data_dir / "test-00000-of-00001.parquet", index=False)

    def test_loader_reads_local_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)

            problems = load_dataset(root, dataset_name=BIOMNI_EVAL1)

        self.assertEqual(len(problems), 4)
        first = problems[0]
        self.assertEqual(first.id, "lab_001")
        self.assertEqual(first.dataset_name, BIOMNI_EVAL1)
        self.assertEqual(first.subject, "Biomedical Sciences")
        self.assertEqual(first.topic, "lab_test_analyzing")
        self.assertIn("Task name: lab_test_analyzing", first.question)
        self.assertEqual(first.answer, "A")

    def test_grader_scores_public_answer_formats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            problems = {str(problem.id): problem for problem in load_dataset(root, dataset_name=BIOMNI_EVAL1)}

        lab = grade_problem(problems["lab_001"], predicted_answer="Answer: A", tool_calls=[])
        gene = grade_problem(problems["gene_001"], predicted_answer="tp53", tool_calls=[])
        rare = grade_problem(
            problems["rare_001"],
            predicted_answer='{"OMIM_ID": "123456", "disease_name": "Example disease"}',
            tool_calls=[],
        )
        patient = grade_problem(
            problems["patient_001"],
            predicted_answer='{"causal_gene": ["BRCA1"]}',
            tool_calls=[],
        )

        self.assertEqual(lab.answer_score, 1.0)
        self.assertEqual(gene.answer_score, 1.0)
        self.assertEqual(rare.answer_score, 1.0)
        self.assertEqual(patient.answer_score, 1.0)


if __name__ == "__main__":
    unittest.main()
