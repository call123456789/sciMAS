"""Bridge sciMAS SMDDBench runs to the official SMDD-Bench evaluator.

The official benchmark evaluates an artifact file inside a Docker image. This
module adapts sciMAS final answers into that file layout and launches the
upstream evaluator container when requested.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


DEFAULT_DOCKER_IMAGE = "smdd-evals"
DEFAULT_TIMEOUT_S = 3600.0


_FENCE_RE = re.compile(r"```([^\n`]*)\n?(.*?)```", re.S)


def load_task_config(task_dir: Path) -> dict[str, Any]:
    """Load a task.yaml file as a dictionary."""
    task_yaml = Path(task_dir) / "task.yaml"
    with task_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {task_yaml}")
    return data


def task_id_from_config(task_dir: Path, config: dict[str, Any] | None = None) -> str:
    """Return the official task id from task.yaml, falling back to dir name."""
    cfg = config if config is not None else load_task_config(task_dir)
    task_id = str(
        cfg.get("id")
        or cfg.get("task_id")
        or cfg.get("name")
        or Path(task_dir).name
    ).strip()
    if not task_id:
        raise ValueError(f"Could not infer task id for {task_dir}")
    return task_id


def output_file_from_config(config: dict[str, Any]) -> str:
    """Return the artifact path expected by the official evaluator."""
    candidate_keys = (
        "file_path",
        "output_file",
        "output_path",
        "submission_file",
        "answer_file",
        "target_file",
    )
    for key in candidate_keys:
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("output_config", "output", "submission", "deliverable"):
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            for subkey in (*candidate_keys, "file", "path", "filename"):
                nested = value.get(subkey)
                if isinstance(nested, str) and nested.strip():
                    return nested.strip()
    raise ValueError("task.yaml does not define an output file path")


def extract_artifact_content(answer_text: str, output_file: str = "") -> str:
    """Extract the artifact body from a final answer.

    If the model returned fenced code, prefer a fence matching the expected file
    extension. Otherwise use the first fenced block. With no fences, preserve the
    answer text after trimming outer whitespace.
    """
    text = answer_text or ""
    fences = [
        (lang.strip().lower(), body.strip())
        for lang, body in _FENCE_RE.findall(text)
        if body.strip()
    ]
    if not fences:
        return text.strip()

    suffix = Path(output_file).suffix.lower().lstrip(".")
    preferred_langs = {
        "py": {"py", "python"},
        "json": {"json"},
        "smi": {"smi", "smiles"},
        "smiles": {"smi", "smiles"},
        "sdf": {"sdf"},
        "csv": {"csv"},
        "tsv": {"tsv"},
        "txt": {"text", "txt"},
    }.get(suffix, set())
    if preferred_langs:
        for lang, body in fences:
            normalized = re.split(r"\s+", lang)[0] if lang else ""
            if normalized in preferred_langs:
                return body
    return max((body for _lang, body in fences), key=len)


def write_agent_output(
    task_dir: Path,
    answer_text: str,
    agent_outputs_root: Path,
) -> Path:
    """Write the sciMAS answer to the official agent-output folder layout."""
    task_dir = Path(task_dir).resolve()
    agent_outputs_root = Path(agent_outputs_root).resolve()
    config = load_task_config(task_dir)
    task_id = task_id_from_config(task_dir, config)
    output_file = output_file_from_config(config)
    rel_output = Path(output_file)
    if rel_output.is_absolute():
        raise ValueError(f"SMDDBench output path must be relative: {output_file}")

    output_dir = agent_outputs_root / task_id
    output_path = (output_dir / rel_output).resolve()
    output_dir_resolved = output_dir.resolve()
    if output_path != output_dir_resolved and output_dir_resolved not in output_path.parents:
        raise ValueError(f"SMDDBench output path escapes task output dir: {output_file}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = extract_artifact_content(answer_text, output_file)
    output_path.write_text(artifact, encoding="utf-8")
    return output_path


def result_json_path(results_root: Path, task_id: str) -> Path:
    return Path(results_root).resolve() / task_id / "result.json"


def _write_result(results_root: Path, task_id: str, result: dict[str, Any]) -> Path:
    result_path = result_json_path(results_root, task_id)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result_path


def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except Exception:
        return False
    return True


def docker_image_exists(image: str) -> bool:
    try:
        subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except Exception:
        return False
    return True


def build_docker_image(
    upstream_root: Path,
    *,
    image: str = DEFAULT_DOCKER_IMAGE,
) -> None:
    """Build the official evaluator image from the vendored upstream checkout."""
    upstream_root = Path(upstream_root).resolve()
    dockerfile = upstream_root / "evaluator" / "Dockerfile"
    if not dockerfile.exists():
        raise FileNotFoundError(f"Missing SMDDBench Dockerfile: {dockerfile}")
    if not docker_available():
        raise RuntimeError("Docker is not installed or not reachable")
    subprocess.run(
        ["docker", "build", "-t", image, "-f", str(dockerfile), str(upstream_root)],
        check=True,
    )


def evaluate_task(
    task_dir: Path,
    agent_outputs_root: Path,
    results_root: Path,
    *,
    docker_image: str = DEFAULT_DOCKER_IMAGE,
    timeout: float = DEFAULT_TIMEOUT_S,
    gpu_id: str | int | None = None,
) -> dict[str, Any]:
    """Run the official evaluator container for one task.

    A structured errored result is written when Docker or the image is missing,
    so the normal sciMAS grader can still report a precise failure.
    """
    task_dir = Path(task_dir).resolve()
    agent_outputs_root = Path(agent_outputs_root).resolve()
    results_root = Path(results_root).resolve()
    config = load_task_config(task_dir)
    task_id = task_id_from_config(task_dir, config)
    output_file = output_file_from_config(config)
    output_dir = agent_outputs_root / task_id
    output_path = output_dir / output_file
    result_dir = results_root / task_id
    result_dir.mkdir(parents=True, exist_ok=True)

    if not output_path.exists():
        result = {
            "task_id": task_id,
            "status": "errored",
            "steps": [],
            "error": f"Missing agent output artifact: {output_path}",
            "hint": "Write the model answer to agent_outputs/<task_id>/<output_config.file_path> first.",
        }
        _write_result(results_root, task_id, result)
        return result

    if not docker_available():
        result = {
            "task_id": task_id,
            "status": "errored",
            "steps": [],
            "error": "Docker is not installed or not reachable on this host.",
            "hint": (
                "Install/start Docker, then build the evaluator image with "
                "`docker build -t smdd-evals -f dataset/SMDDBench/upstream/evaluator/Dockerfile "
                "dataset/SMDDBench/upstream`."
            ),
        }
        _write_result(results_root, task_id, result)
        return result

    if not docker_image_exists(docker_image):
        result = {
            "task_id": task_id,
            "status": "errored",
            "steps": [],
            "error": f"Docker image not found: {docker_image}",
            "hint": (
                "Build the official evaluator image with "
                f"`docker build -t {docker_image} -f dataset/SMDDBench/upstream/evaluator/Dockerfile "
                "dataset/SMDDBench/upstream`."
            ),
        }
        _write_result(results_root, task_id, result)
        return result

    cmd = ["docker", "run", "--rm", "--shm-size=32g"]
    if gpu_id is not None and str(gpu_id).strip():
        cmd.extend(["--gpus", f"device={gpu_id}"])
    cmd.extend(
        [
            "-v",
            f"{task_dir}:/eval/task:ro",
            "-v",
            f"{output_dir.resolve()}:/eval/output:ro",
            "-v",
            f"{result_dir.resolve()}:/eval/results:rw",
            docker_image,
            "python",
            "/eval/eval_entrypoint.py",
        ]
    )

    result_path = result_dir / "result.json"
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result_path.exists():
            parsed = json.loads(result_path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                return parsed
        result = {
            "task_id": task_id,
            "status": "errored",
            "steps": [],
            "error": f"Container exited with code {proc.returncode} but produced no result.json",
            "stdout": (proc.stdout or "")[-2000:],
            "stderr": (proc.stderr or "")[-2000:],
        }
    except subprocess.TimeoutExpired:
        result = {
            "task_id": task_id,
            "status": "errored",
            "steps": [],
            "error": f"Container timed out after {timeout:g} seconds",
        }
    except Exception as exc:
        result = {
            "task_id": task_id,
            "status": "errored",
            "steps": [],
            "error": str(exc),
        }
    _write_result(results_root, task_id, result)
    return result


def materialize_and_evaluate_answer(
    task_dir: Path,
    answer_text: str,
    agent_outputs_root: Path,
    results_root: Path,
    *,
    docker_image: str = DEFAULT_DOCKER_IMAGE,
    timeout: float = DEFAULT_TIMEOUT_S,
    gpu_id: str | int | None = None,
) -> dict[str, Any]:
    """Write the answer artifact, then run the official evaluator."""
    write_agent_output(task_dir, answer_text, agent_outputs_root)
    return evaluate_task(
        task_dir,
        agent_outputs_root,
        results_root,
        docker_image=docker_image,
        timeout=timeout,
        gpu_id=gpu_id,
    )


def _read_answer(args: argparse.Namespace) -> str:
    if args.answer_file:
        return Path(args.answer_file).read_text(encoding="utf-8")
    return args.answer_text or ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run official SMDDBench evaluation for one sciMAS answer")
    parser.add_argument("--task-dir", required=True, help="Path to an SMDDBench task directory")
    answer = parser.add_mutually_exclusive_group(required=True)
    answer.add_argument("--answer-file", help="File containing the sciMAS final answer")
    answer.add_argument("--answer-text", help="sciMAS final answer text")
    parser.add_argument("--agent-outputs", required=True, help="Root for official agent output folders")
    parser.add_argument("--results-dir", required=True, help="Root for official evaluator result folders")
    parser.add_argument("--docker-image", default=DEFAULT_DOCKER_IMAGE)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--gpu-id", default=None)
    parser.add_argument("--build-image", action="store_true", help="Build the official evaluator image first")
    parser.add_argument(
        "--upstream-root",
        default=str(Path(__file__).resolve().parent.parent / "dataset" / "SMDDBench" / "upstream"),
        help="Vendored SMDD-Bench checkout used for --build-image",
    )
    args = parser.parse_args(argv)

    if args.build_image:
        build_docker_image(Path(args.upstream_root), image=args.docker_image)

    result = materialize_and_evaluate_answer(
        Path(args.task_dir),
        _read_answer(args),
        Path(args.agent_outputs),
        Path(args.results_dir),
        docker_image=args.docker_image,
        timeout=args.timeout,
        gpu_id=args.gpu_id,
    )
    task_id = task_id_from_config(Path(args.task_dir), load_task_config(Path(args.task_dir)))
    path = result_json_path(Path(args.results_dir), task_id)
    print(json.dumps({"result_path": str(path), "status": result.get("status")}, ensure_ascii=False))
    return 0 if result.get("status") != "errored" else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
