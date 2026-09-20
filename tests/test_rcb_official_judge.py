"""Tests for the ResearchClawBench official LLM judge and its grader wiring.

The LLM transport (raw httpx to an OpenAI-compatible API) is mocked so
no real API key is needed. Image handling, cache behaviour, parse-failure
recovery, and grader dispatch (env set / unset / disabled) are covered.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

import rcb_official_judge as rcb  # noqa: E402
import grader  # noqa: E402
from dataset import Problem  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CHECKLIST = [
    {
        "type": "text",
        "content": "Criterion A: posterior mass should be ~9 M_sun",
        "keywords": ["posterior", "M_sun"],
        "weight": 0.2,
    },
    {
        "type": "image",
        "path": "images/figure.png",
        "content": "Criterion B: exclusion curve falls below 0.05",
        "keywords": ["exclusion", "0.05"],
        "weight": 0.3,
    },
    {
        "type": "text",
        "content": "Criterion C: derive upper limit on g",
        "keywords": ["g", "upper"],
        "weight": 0.5,
    },
]


def _mock_judge_response(score: int, reasoning: str = "ok") -> dict:
    return {
        "choices": [
            {"message": {"content": json.dumps({"reasoning": reasoning, "score": score})}}
        ]
    }


def _problem(checklist=None, target_paper=None, target_images=None) -> Problem:
    return Problem(
        id="Astronomy_000",
        filename="Astronomy_000.json",
        question="q",
        answer="a",
        subject="Astronomy",
        topic="Diagnostic Analysis",
        checklist=checklist or SAMPLE_CHECKLIST,
        target_paper=target_paper,
        target_images=target_images or [],
        task_info={"task": "Constrain ULB properties from M33 X-7"},
        dataset_name="ResearchClawBench",
        assets_root=str(REPO_ROOT / "dataset" / "ResearchClawBench" / "repo"),
    )


# ---------------------------------------------------------------------------
# Module-level tests
# ---------------------------------------------------------------------------


class RCBJudgeModuleTests(unittest.TestCase):
    def test_constants_match_upstream(self):
        # Spot-check that the RUBRIC preserves the official Mode A / Mode B
        # anchors and the "50 = matches paper" bar.
        self.assertIn("Mode A", rcb.RUBRIC)
        self.assertIn("Mode B", rcb.RUBRIC)
        self.assertIn("41-50", rcb.RUBRIC)  # upper edge of "matches paper"
        self.assertEqual(rcb.PROMPT_VERSION, "rcb-judge-v1")
        self.assertEqual(
            rcb.IMAGE_EXTENSIONS,
            {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"},
        )
        # SVG must NOT be in the whitelist (XSS / vision-model quirks).
        self.assertNotIn(".svg", rcb.IMAGE_EXTENSIONS)

    def test_short_circuit_on_empty_predicted(self):
        with mock.patch.object(rcb, "_call_judge") as mock_call:
            payload = rcb.score_checklist(
                task_id="X", predicted="", checklist=SAMPLE_CHECKLIST,
                target_paper_path=None, target_image_paths=[],
                judge_api_key="k", judge_model="m",
            )
        self.assertEqual(payload["total_score"], 0.0)
        self.assertEqual(payload["judge_calls"], 0)
        mock_call.assert_not_called()

    def test_short_circuit_on_empty_checklist(self):
        with mock.patch.object(rcb, "_call_judge") as mock_call:
            payload = rcb.score_checklist(
                task_id="X", predicted="hi", checklist=[],
                target_paper_path=None, target_image_paths=[],
                judge_api_key="k", judge_model="m",
            )
        self.assertEqual(payload["total_score"], 0.0)
        self.assertIn("empty checklist", payload.get("note", ""))
        mock_call.assert_not_called()


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


class ResponseParsingTests(unittest.TestCase):
    def test_normal_json(self):
        out = rcb._parse_judge_response('{"reasoning": "good", "score": 42}')
        self.assertEqual(out["score"], 42)
        self.assertEqual(out["reasoning"], "good")

    def test_json_embedded_in_prose(self):
        out = rcb._parse_judge_response(
            "Sure, here: {\"reasoning\": \"x\", \"score\": 73}"
        )
        self.assertEqual(out["score"], 73)

    def test_clamp_out_of_range(self):
        out = rcb._parse_judge_response('{"score": 150}')
        self.assertEqual(out["score"], 100)
        out = rcb._parse_judge_response('{"score": -10}')
        self.assertEqual(out["score"], 0)

    def test_parse_failure_returns_zero(self):
        out = rcb._parse_judge_response("totally not json")
        self.assertEqual(out["score"], 0)
        self.assertIn("Failed to parse", out["reasoning"])


# ---------------------------------------------------------------------------
# Image handling
# ---------------------------------------------------------------------------


class ImageHandlingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        # Real PNG fixtures: a 1x1 red PNG (tiny) and a big synthetic PNG.
        self.tiny_png = self.tmp_path / "tiny.png"
        self._write_tiny_png(self.tiny_png)
        self.big_png = self.tmp_path / "big.png"
        self._write_synthetic_large_png(self.big_png, target_bytes=2_500_000)

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def _write_tiny_png(path: Path) -> None:
        # 1x1 transparent PNG, ~67 bytes.
        import base64
        png_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4"
            "2mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        )
        path.write_bytes(base64.b64decode(png_b64))

    @staticmethod
    def _write_synthetic_large_png(path: Path, target_bytes: int) -> None:
        import random
        from PIL import Image
        # Make a noisy image big enough that the JPEG downscale is exercised.
        # Random bytes don't compress well in PNG so we get a chunky file
        # without expensive per-pixel Python iteration.
        rng = random.Random(0xC0FFEE)
        side = 1200
        data = bytes(rng.randrange(256) for _ in range(side * side * 3))
        im = Image.frombytes("RGB", (side, side), data)
        im.save(path, format="PNG", optimize=False)
        # If the file happened to compress well and is too small, bump
        # resolution until we exceed the target.
        while path.stat().st_size < target_bytes and side < 2200:
            side += 200
            data = bytes(rng.randrange(256) for _ in range(side * side * 3))
            im = Image.frombytes("RGB", (side, side), data)
            im.save(path, format="PNG", optimize=False)

    def test_resolve_image_returns_path_for_existing_file(self):
        task_root = self.tmp_path / "task"
        (task_root / "target_study" / "images").mkdir(parents=True)
        (task_root / "target_study" / "images" / "f.png").write_bytes(
            self.tiny_png.read_bytes()
        )
        path, reason = rcb._resolve_image("images/f.png", task_root, [])
        self.assertIsNotNone(path)
        self.assertIsNone(reason)

    def test_resolve_image_missing_returns_none_with_reason(self):
        task_root = self.tmp_path / "task"
        (task_root / "target_study").mkdir(parents=True)
        path, reason = rcb._resolve_image("images/missing.png", task_root, [])
        self.assertIsNone(path)
        self.assertIn("not found", reason)

    def test_resolve_image_rejects_svg(self):
        task_root = self.tmp_path / "task"
        (task_root / "target_study" / "images").mkdir(parents=True)
        (task_root / "target_study" / "images" / "x.svg").write_text("<svg/>")
        path, reason = rcb._resolve_image("images/x.svg", task_root, [])
        self.assertIsNone(path)
        self.assertIn("not whitelisted", reason)

    def test_downscale_image_returns_path_within_size(self):
        out = rcb._downscale_image(self.big_png, max_bytes=300_000)
        try:
            self.assertTrue(out.exists())
            self.assertLessEqual(out.stat().st_size, 300_000)
        finally:
            if out != self.big_png:
                out.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Cache behaviour
# ---------------------------------------------------------------------------


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_cache_hit_skips_judge_call(self):
        # First call should hit the API; second call should hit cache.
        with mock.patch.object(
            rcb, "_call_judge", return_value={"reasoning": "x", "score": 77}
        ) as mock_call:
            first = rcb.score_checklist(
                task_id="X", predicted="some report text", checklist=[
                    {"type": "text", "content": "c", "weight": 1.0}
                ],
                target_paper_path=None, target_image_paths=[],
                judge_api_key="k", judge_model="m",
                cache_dir=str(self.cache_dir),
            )
            second = rcb.score_checklist(
                task_id="X", predicted="some report text", checklist=[
                    {"type": "text", "content": "c", "weight": 1.0}
                ],
                target_paper_path=None, target_image_paths=[],
                judge_api_key="k", judge_model="m",
                cache_dir=str(self.cache_dir),
            )
        self.assertEqual(first["judge_calls"], 1)
        self.assertEqual(second["judge_calls"], 0)
        self.assertTrue(second["cached"])
        self.assertEqual(first["total_score"], second["total_score"])
        mock_call.assert_called_once()

    def test_cache_key_differs_per_model(self):
        # Same predicted, different model → two cache files, two API calls.
        with mock.patch.object(
            rcb, "_call_judge", return_value={"reasoning": "x", "score": 50}
        ) as mock_call:
            rcb.score_checklist(
                task_id="X", predicted="hi", checklist=[
                    {"type": "text", "content": "c", "weight": 1.0}
                ],
                target_paper_path=None, target_image_paths=[],
                judge_api_key="k", judge_model="model-A",
                cache_dir=str(self.cache_dir),
            )
            rcb.score_checklist(
                task_id="X", predicted="hi", checklist=[
                    {"type": "text", "content": "c", "weight": 1.0}
                ],
                target_paper_path=None, target_image_paths=[],
                judge_api_key="k", judge_model="model-B",
                cache_dir=str(self.cache_dir),
            )
        self.assertEqual(mock_call.call_count, 2)
        cache_files = list(self.cache_dir.glob("X__*.json"))
        self.assertEqual(len(cache_files), 2)
        model_names = {f.stem.split("__")[2] for f in cache_files}
        self.assertEqual(model_names, {"model-A", "model-B"})


# ---------------------------------------------------------------------------
# End-to-end scoring (mocked httpx)
# ---------------------------------------------------------------------------


class ScoringTests(unittest.TestCase):
    def test_text_only_weighted_aggregation(self):
        # Three text items; mock returns 100 for all → total_score = 100.
        with mock.patch.object(
            rcb, "_call_judge",
            return_value={"reasoning": "ok", "score": 100},
        ):
            payload = rcb.score_checklist(
                task_id="X", predicted="good report", checklist=[
                    {"type": "text", "content": "a", "weight": 0.2},
                    {"type": "text", "content": "b", "weight": 0.3},
                    {"type": "text", "content": "c", "weight": 0.5},
                ],
                target_paper_path=None, target_image_paths=[],
                judge_api_key="k", judge_model="m",
            )
        self.assertAlmostEqual(payload["total_score"], 100.0, places=2)
        self.assertEqual(len(payload["items"]), 3)

    def test_image_item_sends_image_url(self):
        tmp = tempfile.TemporaryDirectory()
        try:
            tmp_path = Path(tmp.name)
            task_root = tmp_path / "task"
            (task_root / "target_study" / "images").mkdir(parents=True)
            # tiny PNG (re-use helper inline)
            import base64
            png = base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4"
                "2mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
            )
            (task_root / "target_study" / "images" / "f.png").write_bytes(png)
            target_paper = task_root / "target_study" / "paper.pdf"
            target_paper.write_bytes(b"%PDF-1.4 fake")

            captured = {}

            def fake_call(prompt, image_paths, **kw):
                captured["image_paths"] = image_paths
                captured["prompt_has_image_branch"] = "Compare the AI-generated images" in prompt
                return {"reasoning": "ok", "score": 60}

            with mock.patch.object(rcb, "_call_judge", side_effect=fake_call):
                payload = rcb.score_checklist(
                    task_id="X",
                    predicted="see figure 1",
                    checklist=[{
                        "type": "image", "path": "images/f.png",
                        "content": "exclusion curve", "weight": 1.0,
                    }],
                    target_paper_path=str(target_paper),
                    target_image_paths=[],
                    judge_api_key="k", judge_model="m",
                )
            self.assertEqual(payload["items"][0]["type"], "image")
            self.assertFalse(payload["items"][0]["degraded"])
            self.assertEqual(len(captured.get("image_paths", [])), 1)
            self.assertTrue(captured["prompt_has_image_branch"])
        finally:
            tmp.cleanup()

    def test_image_degrades_when_path_missing(self):
        # Path hint present but file gone → text-only prompt, degraded=True.
        captured = {}

        def fake_call(prompt, image_paths, **kw):
            captured["image_paths"] = image_paths
            captured["prompt"] = prompt
            return {"reasoning": "ok", "score": 55}

        with mock.patch.object(rcb, "_call_judge", side_effect=fake_call):
            payload = rcb.score_checklist(
                task_id="X", predicted="report", checklist=[{
                    "type": "image", "path": "images/missing.png",
                    "content": "exclusion curve", "weight": 1.0,
                }],
                target_paper_path=None,
                target_image_paths=[],
                judge_api_key="k", judge_model="m",
            )
        self.assertTrue(payload["items"][0]["degraded"])
        self.assertIn("not found", payload["items"][0]["reasoning"])
        # No image content attached, prompt should be the text variant.
        self.assertIsNone(captured.get("image_paths"))
        self.assertIn("## Key Technical Aspects", captured["prompt"])


# ---------------------------------------------------------------------------
# Grader dispatch
# ---------------------------------------------------------------------------


class GraderDispatchTests(unittest.TestCase):
    def setUp(self):
        # Save and clear all judge-related env vars.
        self._saved = {
            k: os.environ.get(k)
            for k in (
                "RESEARCH_CLAW_JUDGE_API_KEY",
                "RESEARCH_CLAW_JUDGE_API_BASE",
                "RESEARCH_CLAW_JUDGE_MODEL",
                "RESEARCH_CLAW_JUDGE_DISABLED",
                "JUDGE_API_KEY",
                "JUDGE_API_BASE",
                "JUDGE_MODEL_NAME",
            )
        }
        for k in self._saved:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_no_env_falls_back_to_proxy(self):
        problem = _problem()
        result = grader.grade_problem(
            problem,
            predicted_answer="some text",
            tool_calls=[],
            available_tools=set(),
        )
        self.assertEqual(problem.dataset_name, "ResearchClawBench")
        # Proxy was used: notes should NOT mention "official LLM judge".
        joined = " ".join(result.answer_notes)
        self.assertNotIn("official LLM judge", joined)
        self.assertIn("weighted checklist", joined)

    def test_env_set_dispatches_to_official(self):
        os.environ["RESEARCH_CLAW_JUDGE_API_KEY"] = "sk-test"
        os.environ["RESEARCH_CLAW_JUDGE_MODEL"] = "fake-judge"
        # Mock the underlying score_checklist *as imported by grader*, so no
        # real API call is made. (grader.py imports it under its own alias.)
        with mock.patch.object(
            grader, "_rcb_score_checklist",
            return_value={
                "items": [
                    {
                        "index": 0, "type": "text", "content": "c",
                        "weight": 1.0, "score": 75, "reasoning": "good",
                        "degraded": False, "judge_calls": 1,
                    }
                ],
                "total_score": 75.0,
                "total_weight": 1.0,
                "model": "fake-judge",
                "judge_calls": 1,
                "cached": False,
                "prompt_version": "rcb-judge-v1",
            },
        ):
            result = grader.grade_problem(
                _problem(checklist=[{
                    "type": "text", "content": "c", "weight": 1.0,
                }]),
                predicted_answer="some text",
                tool_calls=[],
                available_tools=set(),
            )
        self.assertAlmostEqual(result.answer_score, 0.75, places=3)
        self.assertTrue(result.answer_correct)  # 0.75 >= 0.5
        joined = " ".join(result.answer_notes)
        self.assertIn("official LLM judge", joined)
        self.assertIn("fake-judge", joined)
        self.assertIn("score=75/100", joined)

    def test_disabled_flag_uses_proxy(self):
        os.environ["RESEARCH_CLAW_JUDGE_API_KEY"] = "sk-test"
        os.environ["RESEARCH_CLAW_JUDGE_MODEL"] = "fake-judge"
        os.environ["RESEARCH_CLAW_JUDGE_DISABLED"] = "1"
        result = grader.grade_problem(
            _problem(),
            predicted_answer="some text",
            tool_calls=[],
            available_tools=set(),
        )
        joined = " ".join(result.answer_notes)
        self.assertNotIn("official LLM judge", joined)
        self.assertIn("weighted checklist", joined)

    def test_judge_failure_falls_back_to_proxy(self):
        os.environ["RESEARCH_CLAW_JUDGE_API_KEY"] = "sk-test"
        os.environ["RESEARCH_CLAW_JUDGE_MODEL"] = "fake-judge"

        def boom(*a, **kw):
            raise RuntimeError("simulated judge outage")

        # Patch the name grader imported (alias), not the source module.
        with mock.patch.object(grader, "_rcb_score_checklist", side_effect=boom):
            result = grader.grade_problem(
                _problem(),
                predicted_answer="some text",
                tool_calls=[],
                available_tools=set(),
            )
        joined = " ".join(result.answer_notes)
        self.assertIn("Official judge failed", joined)
        self.assertIn("simulated judge outage", joined)
        # Fell back to proxy, so notes now include the proxy summary.
        self.assertIn("weighted checklist", joined)

    def test_resolve_judge_config_upstream_fallback(self):
        os.environ["JUDGE_API_KEY"] = "sk-upstream"
        os.environ["JUDGE_MODEL_NAME"] = "gpt-5.1"
        cfg = grader._resolve_judge_config()
        self.assertIsNotNone(cfg)
        # grader._resolve_judge_config delegates to tests.llm_judge, which
        # returns an LLMJudgeConfig dataclass (not a plain dict).
        from llm_judge import LLMJudgeConfig
        self.assertIsInstance(cfg, LLMJudgeConfig)
        self.assertEqual(cfg.api_key, "sk-upstream")
        self.assertEqual(cfg.model, "gpt-5.1")
        # New prefix wins when both are set.
        os.environ["RESEARCH_CLAW_JUDGE_API_KEY"] = "sk-new"
        os.environ["RESEARCH_CLAW_JUDGE_MODEL"] = "fake-judge"
        cfg = grader._resolve_judge_config()
        self.assertEqual(cfg.api_key, "sk-new")
        self.assertEqual(cfg.model, "fake-judge")

    def test_judge_cache_dir_handle(self):
        grader.set_judge_cache_dir("/tmp/some-cache")
        self.assertEqual(grader.get_judge_cache_dir(), "/tmp/some-cache")
        grader.set_judge_cache_dir(None)
        self.assertIsNone(grader.get_judge_cache_dir())


if __name__ == "__main__":
    unittest.main()