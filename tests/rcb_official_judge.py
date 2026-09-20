"""ResearchClawBench official multimodal LLM judge.

This module re-implements the scoring loop from upstream
``evaluation/score.py`` (InternScience/ResearchClawBench) so sciMAS can
score its agent reports against the bundled per-task checklists without
vendoring the upstream Flask app, ``structai.LLMAgent``, or
``researchharness``.

The rubric prompt, per-item prompt templates, score scale (0–100 where
50 = "matches paper"), and weighted aggregation all mirror upstream
behavior. The only thing replaced is the LLM transport: we use raw
``httpx`` against any OpenAI-compatible ``/chat/completions`` endpoint,
which keeps sciMAS free of the upstream ``structai``/``openai`` pip
dependencies.

Per-item behavior:

* ``type == "text"`` — single text-only chat-completions call.
* ``type == "image"`` — chat-completions call with the target image
  attached as an OpenAI vision content block. sciMAS agents do not
  produce figures, so only the ground-truth target image is attached
  (the upstream prompt assumes target image first, then agent
  images — we just send the target).

If the target image is missing, has a non-whitelisted extension, or
exceeds :data:`MAX_IMAGE_BYTES` after Pillow downscale, the item is
scored in text-only mode and the per-item record is flagged with
``degraded=True``. The aggregate score is unaffected.

Results are cached under ``<cache_dir>/<task>__<hash16>__<model>__v1.json``
so re-grading an existing run never re-bills the API.
"""

from __future__ import annotations

import base64
import concurrent.futures
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable

import httpx

try:
    from PIL import Image
    _HAS_PILLOW = True
except ImportError:  # pragma: no cover - Pillow is in environment.yml
    _HAS_PILLOW = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROMPT_VERSION = "rcb-judge-v1"

# 4 MB raw → ~5.4 MB base64 → fits inside OpenAI's 20 MB request budget with
# room for the report text + other image items.
MAX_IMAGE_BYTES = 4_000_000

# Whitelist matches the upstream IMAGE_EXTENSIONS minus SVG (which we drop to
# sidestep SVG-XSS edge cases in vision models).
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}

DEFAULT_TIMEOUT = 180.0
DEFAULT_MAX_WORKERS = 8
DEFAULT_API_BASE = "https://api.openai.com/v1"

# ---------------------------------------------------------------------------
# RUBRIC — copied verbatim from upstream InternScience/ResearchClawBench
# evaluation/score.py (mode A "Objective" / mode B "Subjective", 0-100 scale
# where 50 = "matches paper"). Keep this in sync if upstream tightens the
# rubric; bump PROMPT_VERSION to invalidate any cached scores.
# ---------------------------------------------------------------------------

RUBRIC = """You are a strict scientific peer reviewer evaluating an AI agent's ability to conduct end-to-end automated scientific research.

You are given:
1. The INSTRUCTIONS.md that was provided to the AI agent (the research task it was asked to solve).
2. The AI-generated research report (the agent's output).
3. A specific evaluation criterion derived from the original published paper.

IMPORTANT: Your role is ONLY to score the AI report against the criterion. Do NOT attempt to answer or solve the research task yourself. Focus solely on evaluating what the AI agent produced.

## Evaluation Modes

Each checklist item falls into one of two categories. Determine which applies based on the criterion's nature:

### Mode A: Objective Evaluation (Metric Optimization / Quantitative Results)
Use this when the criterion involves specific numerical results, metrics, benchmarks, or quantitative outcomes.

- **0**: The criterion is completely absent from the report.
- **1-10**: Mentioned but no quantitative results provided.
- **11-20**: Quantitative results given but the methodology has fundamental errors.
- **21-30**: Methodology has significant flaws; metrics deviate severely from the paper.
- **31-40**: Methodology is mostly correct but metrics are notably worse than the paper.
- **41-50**: Metrics are roughly comparable to the original paper.
- **51-60**: Metrics are slightly better than the paper.
- **61-70**: Metrics are clearly better than the paper.
- **71-80**: Both methodology and metrics show substantial improvements over the paper.
- **81-90**: Metrics dramatically surpass the paper.
- **91-100**: Breakthrough results far exceeding the paper.

### Mode B: Subjective Evaluation (Mechanism Analysis / Qualitative Reasoning)
Use this when the criterion involves theoretical explanations, mechanistic insights, logical arguments, or interpretive analysis.

- **0**: The criterion is completely absent from the report.
- **1-10**: Mentioned only with vague, generic statements.
- **11-20**: Some description present but no substantive analysis.
- **21-30**: Some analysis attempted but evidence is insufficient or reasoning has logical gaps.
- **31-40**: Analysis direction is correct but lacks depth; key arguments are missing.
- **41-50**: Analysis depth and logical rigor are roughly comparable to the original paper.
- **51-60**: More supporting evidence provided than the paper.
- **61-70**: More complete logical chain and more rigorous argumentation than the paper.
- **71-80**: Significantly deeper analysis; raises valuable insights not covered in the paper.
- **81-90**: Analysis depth far exceeds the paper.
- **91-100**: Original contributions with breakthrough insights beyond the paper.

## CRITICAL RULES
- 50 means "as good as the actual published paper" — this is a high bar.
- First determine if the criterion is Objective (Mode A) or Subjective (Mode B), then apply the corresponding rubric.
- No credit for vague or generic statements. Must demonstrate specific, concrete analysis.
- No inflation for well-written but shallow content. Substance over style. Longer does not mean better.
- Be highly skeptical of AI-generated content: it may sound plausible but contain factual errors, fabricated numbers, or unsupported conclusions. Verify claims against the criterion carefully.
- Be strict but fair.
"""


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def _resolve_image(
    path_hint: str | None,
    task_root: Path | None,
    target_image_paths: Iterable[str] | None,
) -> tuple[Path | None, str | None]:
    """Resolve a checklist image path to a local file.

    Returns ``(path, reason)`` where ``reason`` is non-None when the image
    is unusable and the caller should degrade to text-only scoring.
    """
    if not path_hint:
        return None, "no path hint"

    if task_root is not None:
        candidate = (task_root / "target_study" / path_hint).resolve()
        if candidate.is_file():
            ext = candidate.suffix.lower()
            if ext not in IMAGE_EXTENSIONS:
                return None, f"extension {ext!r} not whitelisted"
            return candidate, None

    # Fall back to scanning the pre-resolved target_images list. HF rows
    # already pre-resolve these; the explicit task-root resolve above is
    # the upstream-faithful path.
    if target_image_paths:
        wanted_name = Path(path_hint).name
        for raw in target_image_paths:
            p = Path(raw)
            if p.is_file() and p.name == wanted_name:
                ext = p.suffix.lower()
                if ext not in IMAGE_EXTENSIONS:
                    return None, f"extension {ext!r} not whitelisted"
                return p, None

    return None, f"target image not found ({path_hint})"


def _downscale_image(path: Path, max_bytes: int = MAX_IMAGE_BYTES) -> Path:
    """Return a JPEG copy of ``path`` no larger than ``max_bytes`` bytes.

    Falls back to the original path if Pillow is unavailable or downscale
    fails. Caller should still respect ``max_bytes`` via the returned size.
    """
    if not _HAS_PILLOW:
        return path
    try:
        if path.stat().st_size <= max_bytes and path.suffix.lower() in {".jpg", ".jpeg"}:
            return path
        with Image.open(path) as im:
            im = im.convert("RGB")
            w0, h0 = im.size
            # Iterate (scale, quality) combinations in order until the JPEG
            # fits under max_bytes. Cap the number of attempts to guarantee
            # termination on adversarial images.
            candidates: list[tuple[float, int]] = []
            for scale in (1.0, 0.75, 0.5, 0.35, 0.25, 0.18):
                for quality in (85, 75, 65, 55):
                    candidates.append((scale, quality))
            last_path: Path | None = None
            for scale, quality in candidates:
                w = max(1, int(w0 * scale))
                h = max(1, int(h0 * scale))
                resized = im if scale == 1.0 else im.resize((w, h), Image.LANCZOS)
                buf = tempfile.NamedTemporaryFile(
                    prefix="rcb-judge-", suffix=".jpg", delete=False
                )
                tmp_path = Path(buf.name)
                buf.close()
                resized.save(tmp_path, format="JPEG", quality=quality, optimize=True)
                if tmp_path.stat().st_size <= max_bytes:
                    return tmp_path
                last_path = tmp_path
            # Couldn't fit under max_bytes; return the smallest we produced.
            # Caller checks size and degrades to text-only if still too big.
            return last_path or path
    except Exception:
        return path


def _data_uri(path: Path, mime: str | None = None) -> str:
    mime = mime or _mime_for(path)
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _mime_for(path: Path) -> str:
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
    }.get(path.suffix.lower(), "application/octet-stream")


# ---------------------------------------------------------------------------
# Prompt builders (verbatim from upstream)
# ---------------------------------------------------------------------------


def _build_text_prompt(report_text: str, item: dict, instructions: str) -> str:
    criteria = str(item.get("content", "") or "")
    keywords = item.get("keywords", []) or []
    keywords_str = ", ".join(str(k) for k in keywords) if keywords else "None specified"
    return (
        f"{RUBRIC}\n\n"
        f"## Research Task Background (INSTRUCTIONS.md given to the AI agent)\n"
        f"{instructions}\n\n"
        f"## Evaluation Criterion (from the original paper)\n"
        f"{criteria}\n\n"
        f"## Key Technical Aspects to Verify\n"
        f"{keywords_str}\n\n"
        f"## AI-Generated Research Report\n"
        f"{report_text}\n\n"
        f"## Task\n"
        f"Rate how well this report addresses the criterion compared to the original paper.\n"
        f"First determine if this criterion is Objective (Mode A) or Subjective (Mode B), then apply the corresponding rubric strictly.\n\n"
        f'Return your answer as a JSON object: {{"reasoning": "<2-3 sentences>", "score": <0-100>}}'
    )


def _build_image_prompt(
    report_text: str, item: dict, instructions: str, image_count: int
) -> str:
    criteria = str(item.get("content", "") or "")
    keywords = item.get("keywords", []) or []
    keywords_str = ", ".join(str(k) for k in keywords) if keywords else "None specified"
    excerpt = (report_text or "No report text available.")[:10000]
    return (
        f"{RUBRIC}\n\n"
        f"## Research Task Background (INSTRUCTIONS.md given to the AI agent)\n"
        f"{instructions}\n\n"
        f"## Evaluation Criterion (from the original paper)\n"
        f"{criteria}\n\n"
        f"## Key Visual/Technical Aspects to Verify\n"
        f"{keywords_str}\n\n"
        f"## AI-Generated Report Text (excerpt)\n"
        f"{excerpt}\n\n"
        f"## Task\n"
        f"Compare the AI-generated images against the target image from the original paper.\n"
        f"When images are attached, the first image is always the ground-truth target image from the original paper. "
        f"All subsequent images are from the AI agent's workspace/report.\n"
        f"First determine if this criterion is Objective (Mode A) or Subjective (Mode B), then apply the corresponding rubric strictly.\n"
        f"Superficially similar plots with wrong scales, missing data, or incorrect trends should score low.\n\n"
        f'Return your answer as a JSON object: {{"reasoning": "<2-3 sentences>", "score": <0-100>}}'
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

# Greedy but anchored: grab the first {...} JSON object, handling nested braces
# for the reasoning string. JSON spec allows newlines inside strings, so use a
# non-greedy match with a fallback to ast.literal_eval on the raw text.
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_judge_response(raw: str, max_try: int = 2) -> dict:
    """Best-effort JSON parse of an LLM judge response.

    Returns ``{"reasoning": str, "score": int 0..100}``. Mirrors upstream's
    behavior: clamp into [0, 100]; if both attempts fail, return
    ``{"reasoning": "Failed to parse: <truncated>", "score": 0}``.
    """
    last_err = ""
    text = raw or ""
    for attempt in range(max_try):
        # Try strict JSON first.
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return _normalize_judge_obj(obj)
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
        # Then try extracting the first {...} block.
        match = _JSON_OBJ_RE.search(text)
        if match:
            try:
                obj = json.loads(match.group(0))
                if isinstance(obj, dict):
                    return _normalize_judge_obj(obj)
            except Exception as exc:
                last_err = f"{type(exc).__name__}: {exc}"
        # Last-ditch: look for a bare integer score.
        nums = re.findall(r"\b(\d{1,3})\b", text)
        for n in reversed(nums):
            val = int(n)
            if 0 <= val <= 100:
                return {"reasoning": text[:400], "score": val}
        text = ""
    truncated = (raw or "")[:200]
    return {"reasoning": f"Failed to parse: {truncated}", "score": 0}


def _normalize_judge_obj(obj: dict) -> dict:
    score_raw = obj.get("score", 0)
    try:
        score = int(round(float(score_raw)))
    except (TypeError, ValueError):
        score = 0
    score = max(0, min(100, score))
    reasoning = str(obj.get("reasoning", "") or "")
    return {"reasoning": reasoning, "score": score}


# ---------------------------------------------------------------------------
# LLM call (raw httpx, OpenAI-compatible)
# ---------------------------------------------------------------------------


def _call_judge(
    prompt: str,
    image_paths: list[Path] | None,
    *,
    api_base: str,
    api_key: str,
    model: str,
    timeout: float,
) -> dict:
    """Issue a single chat-completions request and parse the response.

    Raises ``RuntimeError`` on transport errors so callers can decide
    whether to retry, cache, or fall back.
    """
    base = api_base.rstrip("/")
    url = f"{base}/chat/completions"

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for img in image_paths or []:
        try:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _data_uri(img)},
                }
            )
        except Exception as exc:
            # Skip unreadable images but keep going.
            content.append(
                {
                    "type": "text",
                    "text": f"[image {img.name} unreadable: {exc}]",
                }
            )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict scientific peer reviewer evaluating "
                    "AI-generated research. Score the report against the "
                    "criterion only — do not attempt to solve the research "
                    "task yourself."
                ),
            },
            {"role": "user", "content": content},
        ],
        "temperature": 0,
        "max_tokens": 500,
    }

    with httpx.Client(timeout=timeout) as client:
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                resp = client.post(url, headers=headers, json=body)
                resp.raise_for_status()
                payload = resp.json()
                break
            except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                last_exc = exc
                continue
            except Exception as exc:
                last_exc = exc
                continue
        else:
            raise RuntimeError(f"judge API failed: {last_exc!r}")

    try:
        text = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"judge API returned unexpected payload: {exc}") from exc

    # Some OpenAI-compatible servers wrap content as a list of parts.
    if isinstance(text, list):
        text = "".join(
            part.get("text", "") for part in text if isinstance(part, dict)
        )
    text = str(text or "")
    return _parse_judge_response(text)


# ---------------------------------------------------------------------------
# Per-item scoring (text vs image, with degradation)
# ---------------------------------------------------------------------------


def _score_item(args: dict) -> dict:
    """Score a single checklist item. Designed for executor.map()."""
    item = args["item"]
    predicted = args["predicted"]
    instructions = args["instructions"]
    task_root = args["task_root"]
    target_image_paths = args["target_image_paths"]
    api_base = args["api_base"]
    api_key = args["api_key"]
    model = args["model"]
    timeout = args["timeout"]

    item_type = str(item.get("type", "text")).lower()
    weight = float(item.get("weight", 1.0) or 0.0)
    content = str(item.get("content", "") or "")
    keywords = item.get("keywords", []) or []

    record: dict[str, Any] = {
        "index": args["index"],
        "type": item_type,
        "content": content[:200],
        "weight": weight,
        "score": 0,
        "reasoning": "",
        "degraded": False,
        "judge_calls": 0,
    }

    if not predicted.strip():
        record["reasoning"] = "Skipped: empty predicted answer"
        return record

    try:
        if item_type == "image":
            image_path, degradation_reason = _resolve_image(
                item.get("path"), task_root, target_image_paths
            )
            if image_path is None:
                # Graceful degradation: score as text-only.
                prompt = _build_text_prompt(predicted, item, instructions)
                record["degraded"] = True
                record["reasoning"] = f"degraded: {degradation_reason}; "
                result = _call_judge(
                    prompt, None,
                    api_base=api_base, api_key=api_key, model=model, timeout=timeout,
                )
            else:
                prepared = _downscale_image(image_path)
                if prepared.stat().st_size > MAX_IMAGE_BYTES:
                    # Still too big after downscale — degrade.
                    prompt = _build_text_prompt(predicted, item, instructions)
                    record["degraded"] = True
                    record["reasoning"] = (
                        f"degraded: image too large "
                        f"({prepared.stat().st_size} bytes); "
                    )
                    result = _call_judge(
                        prompt, None,
                        api_base=api_base, api_key=api_key, model=model, timeout=timeout,
                    )
                else:
                    prompt = _build_image_prompt(
                        predicted, item, instructions, image_count=1
                    )
                    result = _call_judge(
                        prompt, [prepared],
                        api_base=api_base, api_key=api_key, model=model, timeout=timeout,
                    )
        else:
            prompt = _build_text_prompt(predicted, item, instructions)
            result = _call_judge(
                prompt, None,
                api_base=api_base, api_key=api_key, model=model, timeout=timeout,
            )
        record["score"] = int(result.get("score", 0))
        record["reasoning"] = str(record["reasoning"]) + str(result.get("reasoning", ""))
        record["judge_calls"] = 1
    except Exception as exc:
        record["score"] = 0
        record["reasoning"] = (
            str(record["reasoning"]) + f"judge call failed: {exc!r}"
        )
    return record


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _safe_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)[:64] or "model"


def _cache_key(task_id: str, predicted: str, model: str) -> str:
    digest = hashlib.sha256(predicted.encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"{task_id}__{digest}__{_safe_model_name(model)}__{PROMPT_VERSION}"


def _read_cache(cache_dir: Path | None, key: str) -> dict | None:
    if cache_dir is None:
        return None
    path = cache_dir / f"{key}.json"
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _write_cache_atomic(cache_dir: Path | None, key: str, payload: dict) -> None:
    if cache_dir is None:
        return
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    final = cache_dir / f"{key}.json"
    tmp = cache_dir / f"{key}.json.tmp"
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        tmp.replace(final)
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_checklist(
    task_id: str,
    predicted: str,
    checklist: list[dict[str, Any]],
    target_paper_path: str | os.PathLike[str] | None,
    target_image_paths: list[str] | None,
    *,
    judge_api_base: str = DEFAULT_API_BASE,
    judge_api_key: str,
    judge_model: str,
    max_workers: int = DEFAULT_MAX_WORKERS,
    timeout: float = DEFAULT_TIMEOUT,
    cache_dir: str | os.PathLike[str] | None = None,
    instructions: str | None = None,
) -> dict[str, Any]:
    """Score ``predicted`` against ``checklist`` using the upstream judge.

    Returns ``{"items": [...], "total_score": float (0-100),
    "model": str, "judge_calls": int, "cached": bool}``.

    ``total_score`` is the weighted mean over non-zero-weight items, in
    the upstream 0-100 scale (caller normalises to 0-1.0 if needed).
    """
    checklist = list(checklist or [])
    target_image_paths = list(target_image_paths or [])

    if not predicted or not predicted.strip():
        return {
            "items": [],
            "total_score": 0.0,
            "model": judge_model,
            "judge_calls": 0,
            "cached": False,
            "note": "empty predicted answer; short-circuit",
        }
    if not checklist:
        return {
            "items": [],
            "total_score": 0.0,
            "model": judge_model,
            "judge_calls": 0,
            "cached": False,
            "note": "empty checklist; short-circuit",
        }

    cache_dir_path = Path(cache_dir) if cache_dir else None
    key = _cache_key(task_id, predicted, judge_model)
    cached = _read_cache(cache_dir_path, key)
    if cached is not None:
        cached = dict(cached)
        cached["cached"] = True
        cached["judge_calls"] = 0
        return cached

    # target_paper lives at <root>/tasks/<task>/target_study/paper.pdf so the
    # parent.parent is the task directory itself. That is what
    # ``_resolve_image`` joins ``target_study/<path_hint>`` onto.
    task_root = Path(target_paper_path).parent.parent if target_paper_path else None

    # Default instructions to the task's own description if not provided.
    if not instructions:
        instructions = ""

    work_items: list[dict[str, Any]] = []
    for idx, item in enumerate(checklist):
        work_items.append(
            {
                "index": idx,
                "item": item,
                "predicted": predicted,
                "instructions": instructions,
                "task_root": task_root,
                "target_image_paths": target_image_paths,
                "api_base": judge_api_base,
                "api_key": judge_api_key,
                "model": judge_model,
                "timeout": timeout,
            }
        )

    results: list[dict[str, Any]] = []
    if len(work_items) == 1:
        results = [_score_item(work_items[0])]
    else:
        worker_count = max(1, min(len(work_items), max_workers))
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as pool:
            for record in pool.map(_score_item, work_items):
                results.append(record)

    # Weighted aggregation, ignoring items with weight <= 0.
    total_weighted = 0.0
    total_weight = 0.0
    for record in results:
        if record["weight"] > 0:
            total_weighted += record["score"] * record["weight"]
            total_weight += record["weight"]
    final_score = (total_weighted / total_weight) if total_weight > 0 else 0.0

    payload = {
        "items": results,
        "total_score": round(final_score, 2),
        "total_weight": round(total_weight, 4),
        "model": judge_model,
        "judge_calls": sum(int(r.get("judge_calls", 0)) for r in results),
        "cached": False,
        "prompt_version": PROMPT_VERSION,
    }

    _write_cache_atomic(cache_dir_path, key, payload)
    return payload


__all__ = [
    "DEFAULT_API_BASE",
    "DEFAULT_MAX_WORKERS",
    "DEFAULT_TIMEOUT",
    "IMAGE_EXTENSIONS",
    "MAX_IMAGE_BYTES",
    "PROMPT_VERSION",
    "RUBRIC",
    "score_checklist",
]