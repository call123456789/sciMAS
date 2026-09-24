"""Centralized LLM-judge configuration and helpers for sciMAS grading.

This module is the single source of truth for "should we call an LLM to
grade this?" and "if so, where do we send the request?". Both
``tests.grader.grade_research_claw_official`` and the new SciAgentGYM
LLM-judge path consume :func:`resolve_judge_config` so the same env-var
prefix chain works for both datasets.

Configuration precedence
------------------------

1. **Disabled flag** (``<disabled_env_flag>`` env var == ``"1"``) short-circuits
   to :data:`NO_CONFIG` and the caller falls back to the deterministic
   grader. Default flag: ``RESEARCH_CLAW_JUDGE_DISABLED``.
2. **Env-var prefixes** are tried in order; the first prefix that supplies
   a full ``(KEY, BASE, MODEL)`` triple wins. Default prefixes (from
   ``config/llm_judge.json``):

   - ``RESEARCH_CLAW_JUDGE`` — original ResearchClawBench env vars
   - ``JUDGE`` — upstream-compatible naming
   - ``SCIMAS_LLM_JUDGE`` — sciMAS-native prefix
3. **JSON defaults** (``config/llm_judge.json``) supply ``api_base``,
   ``model``, ``timeout``, ``max_tokens`` when env vars don't override.

Each prefix expects three env vars: ``<PREFIX>_KEY``, ``<PREFIX>_BASE``,
``<PREFIX>_MODEL``.

Public helpers
--------------

- :class:`LLMJudgeConfig` — resolved configuration dataclass.
- :func:`resolve_judge_config` — read env + JSON, return config or
  :data:`NO_CONFIG`.
- :func:`is_judge_enabled` — boolean shortcut used by callers that just
  want to know whether to call the LLM at all.
- :func:`judge_correct` — official SciAgentGYM ``is_answer_correct``
  prompt; returns True/False/None on transport failure.
- :func:`secondary_verify` — equivalent-match rescue used when the
  deterministic match fails (mirrors SciAgentGYM's
  ``secondary_verification_with_llm``).
- :func:`judge_drugqa_items` — drug-discovery QA over a *list* of acceptable
  answer items; returns both a full-match verdict and the fraction of items
  the answer hit. See :class:`DrugQAJudgement`.
- :func:`extract_boxed` — pull the last ``\\boxed{...}`` value out of a
  model response. Falls back to None when not present.

Question figures
----------------

:func:`judge_correct` accepts ``images`` — the question's own figures, sent as
OpenAI vision content blocks alongside the prompt. With no images the request
body is byte-identical to the text-only one. When the configured model cannot
accept image content the vision call fails, the judge is retried text-only so
the verdict still arrives, and :func:`get_last_image_note` reports that the
figures were not seen. This module is also where the image caps live
(``MAX_IMAGE_BYTES``, ``MAX_TOTAL_IMAGE_BYTES``, ``MAX_IMAGES``); they are
deliberately not config keys, because a key that is never read looks
configurable and silently is not.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "llm_judge.json"

# Per-machine credentials and overrides, merged over ``CONFIG_PATH``.
# ``CONFIG_PATH`` is tracked by git, so a key written there would be
# committed; this file is git-ignored and is where a key belongs. Shape is
# the same as ``CONFIG_PATH``: ``{"api_key": ..., "api_base": ..., "model": ...}``.
LOCAL_CONFIG_PATH = CONFIG_PATH.with_suffix(".local.json")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMJudgeConfig:
    api_base: str
    api_key: str
    model: str
    timeout: float = 30.0
    max_tokens: int = 5
    temperature: float = 0.0
    prefix: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "api_base": self.api_base,
            "api_key": self.api_key,
            "model": self.model,
            "timeout": self.timeout,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "prefix": self.prefix,
        }


NO_CONFIG: Optional[LLMJudgeConfig] = None


# ---------------------------------------------------------------------------
# Config file loader
# ---------------------------------------------------------------------------


_DEFAULT_CONFIG: dict[str, Any] = {
    "env_var_prefixes": ["RESEARCH_CLAW_JUDGE", "JUDGE", "SCIMAS_LLM_JUDGE"],
    "disabled_env_flag": "RESEARCH_CLAW_JUDGE_DISABLED",
    "default_api_base": "https://api.openai.com/v1",
    "default_model": "gpt-4.1",
    "default_timeout_seconds": 30.0,
    "default_max_tokens": 5,
}


def _merge_config_file(merged: dict[str, Any], path: Path) -> dict[str, Any]:
    """Merge one JSON config file over ``merged``; ignore a missing/bad file."""
    if not path.is_file():
        return merged
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return merged
    if not isinstance(data, dict):
        return merged
    for key, value in data.items():
        if key.startswith("$"):
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


def _load_json_config() -> dict[str, Any]:
    """Read the judge config, falling back to in-process defaults.

    ``config/llm_judge.json`` (tracked) supplies the shared settings and is
    merged first; ``config/llm_judge.local.json`` (git-ignored) is merged
    over it so a per-machine credential never has to be committed.
    """
    merged = dict(_DEFAULT_CONFIG)
    merged = _merge_config_file(merged, CONFIG_PATH)
    return _merge_config_file(merged, LOCAL_CONFIG_PATH)


def dataset_setting(dataset_name: str) -> dict[str, Any]:
    """Return the per-dataset sub-config for ``dataset_name`` (or ``{}``)."""
    cfg = _load_json_config()
    per = cfg.get("per_dataset") or {}
    if not isinstance(per, dict):
        return {}
    value = per.get(dataset_name)
    return dict(value) if isinstance(value, dict) else {}


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


def _is_disabled(cfg: dict[str, Any]) -> bool:
    flag = str(cfg.get("disabled_env_flag") or "").strip()
    if not flag:
        return False
    return os.environ.get(flag, "").strip() == "1"


def _try_prefix(
    prefix: str, cfg: dict[str, Any], dataset: Optional[str] = None
) -> Optional[LLMJudgeConfig]:
    """Try to assemble an :class:`LLMJudgeConfig` from one env-var prefix.

    Accepts both ``<PREFIX>_API_KEY`` (historical RCB convention) and the
    shorter ``<PREFIX>_KEY`` suffix. The base URL is read from
    ``<PREFIX>_API_BASE`` (or ``<PREFIX>_BASE`` as a fallback) and the
    model from ``<PREFIX>_MODEL`` (or ``<PREFIX>_MODEL_NAME``). The
    longer suffix wins when both are set so historical deployments keep
    working without renaming env vars.
    """
    api_key = (
        os.environ.get(f"{prefix}_API_KEY", "").strip()
        or os.environ.get(f"{prefix}_KEY", "").strip()
    )
    if not api_key:
        return None
    api_base = (
        os.environ.get(f"{prefix}_API_BASE", "").strip()
        or os.environ.get(f"{prefix}_BASE", "").strip()
        or str(cfg.get("default_api_base") or _DEFAULT_CONFIG["default_api_base"])
    )
    model = (
        os.environ.get(f"{prefix}_MODEL", "").strip()
        or os.environ.get(f"{prefix}_MODEL_NAME", "").strip()
    )
    if not model:
        # A prefix that has a key but no model is incomplete; skip it so the
        # next prefix can try to supply one.
        return None
    return _build_config(
        cfg,
        api_key=api_key,
        api_base=api_base,
        model=model,
        prefix=prefix,
        dataset=dataset,
    )


def _build_config(
    cfg: dict[str, Any],
    *,
    api_key: str,
    api_base: str,
    model: str,
    prefix: str,
    dataset: Optional[str] = None,
) -> LLMJudgeConfig:
    """Assemble a config from a credential source plus the shared settings.

    ``dataset``'s own ``per_dataset`` block may override ``max_tokens`` and
    ``temperature`` — the two knobs that describe *how much thinking this
    dataset's prompt provokes*, which is a property of the prompt rather than
    of the deployment. DrugQA hands the judge a list of gold items to rule on
    one by one and a JSON object to write; SciAgentGYM hands it two short
    values and expects 正确/错误. Sizing both at one number means either
    overpaying for the short one or truncating the long one.

    Deliberately not overridable per dataset: ``model`` and the credentials.
    A model named in the environment is the deployment's declared judge, and a
    file on the machine outvoting an exported variable is the trap
    :func:`_try_config_credentials` already avoids.
    """
    try:
        timeout = float(cfg.get("default_timeout_seconds") or _DEFAULT_CONFIG["default_timeout_seconds"])
    except (TypeError, ValueError):
        timeout = _DEFAULT_CONFIG["default_timeout_seconds"]
    try:
        max_tokens = int(cfg.get("default_max_tokens") or _DEFAULT_CONFIG["default_max_tokens"])
    except (TypeError, ValueError):
        max_tokens = _DEFAULT_CONFIG["default_max_tokens"]
    temperature = 0.0
    if dataset is not None:
        per = dataset_setting(dataset)
        try:
            # ``or max_tokens`` would make a deliberate 0 unsettable; these are
            # "absent means inherit", so test for absence.
            if per.get("max_tokens") is not None:
                max_tokens = int(per["max_tokens"])
        except (TypeError, ValueError):
            pass
        try:
            if per.get("temperature") is not None:
                temperature = float(per["temperature"])
        except (TypeError, ValueError):
            pass
    return LLMJudgeConfig(
        temperature=temperature,
        api_base=api_base,
        api_key=api_key,
        model=model,
        timeout=timeout,
        max_tokens=max_tokens,
        prefix=prefix,
    )


def _try_config_credentials(
    cfg: dict[str, Any], dataset: Optional[str] = None
) -> Optional[LLMJudgeConfig]:
    """Assemble a config from credentials stored in the JSON config itself.

    Consulted only after every env-var prefix has failed, so an exported key
    still wins — env vars are the documented override, and a deployment that
    sets them must not be silently overridden by a file on the machine.

    The credentials live in ``api_key`` (plus optional ``api_base`` and
    ``model``), normally in the git-ignored ``llm_judge.local.json``. The
    ``env_var_layout`` block is *documentation* — spelling the key there was
    a long-standing trap, because nothing ever read it.
    """
    api_key = str(cfg.get("api_key") or "").strip()
    if not api_key:
        return None
    model = str(cfg.get("model") or cfg.get("default_model") or "").strip()
    if not model:
        return None
    api_base = str(
        cfg.get("api_base")
        or cfg.get("default_api_base")
        or _DEFAULT_CONFIG["default_api_base"]
    ).strip()
    return _build_config(
        cfg,
        api_key=api_key,
        api_base=api_base,
        model=model,
        prefix="<config>",
        dataset=dataset,
    )


def resolve_judge_config(*, dataset: Optional[str] = None) -> Optional[LLMJudgeConfig]:
    """Resolve the LLM-judge config, honoring ``dataset``'s enable flag.

    Credentials come from the first env-var prefix that supplies a full
    ``(KEY, MODEL)`` pair, else from the JSON config's own ``api_key`` /
    ``model`` (see :func:`_try_config_credentials`).

    Returns :data:`NO_CONFIG` (= ``None``) when disabled, when neither
    source supplies a credential, or when the per-dataset flag ``enabled``
    is set to ``False`` in ``config/llm_judge.json``.
    """
    cfg = _load_json_config()
    if _is_disabled(cfg):
        return None
    if dataset is not None:
        per = dataset_setting(dataset)
        if per.get("enabled") is False:
            return None
    prefixes = cfg.get("env_var_prefixes") or _DEFAULT_CONFIG["env_var_prefixes"]
    if not isinstance(prefixes, list):
        prefixes = list(_DEFAULT_CONFIG["env_var_prefixes"])
    for prefix in prefixes:
        prefix = str(prefix).strip()
        if not prefix:
            continue
        resolved = _try_prefix(prefix, cfg, dataset)
        if resolved is not None:
            return resolved
    # No env var carried a credential — fall back to the config file's own.
    return _try_config_credentials(cfg, dataset)


def is_judge_enabled(dataset: Optional[str] = None) -> bool:
    """Convenience boolean used by callers that just want a yes/no answer."""
    return resolve_judge_config(dataset=dataset) is not None


# ---------------------------------------------------------------------------
# Judge-cache directory (shared by both RCB and SciAgentGYM paths)
# ---------------------------------------------------------------------------


_judge_cache_dir: Optional[str] = None


def set_judge_cache_dir(path: Optional[str]) -> None:
    """Pin the directory LLM judges may use for response caching."""
    global _judge_cache_dir
    _judge_cache_dir = str(path) if path else None


def get_judge_cache_dir() -> Optional[str]:
    return _judge_cache_dir


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


_BOXED_PATTERNS = (
    r"\\boxed\{",
    r"\$\\boxed\{",
    r"boxed\{",
)


def _extract_balanced_braces(text: str, start_pos: int) -> Optional[str]:
    """Return the contents of the balanced ``{...}`` group starting at
    ``start_pos``, supporting arbitrary nesting. Returns None on mismatch.
    """
    if not text or start_pos < 0 or start_pos >= len(text):
        return None
    if text[start_pos] != "{":
        return None
    stack = 0
    out: list[str] = []
    for idx in range(start_pos, len(text)):
        ch = text[idx]
        if ch == "{":
            stack += 1
            if stack > 1:
                out.append(ch)
        elif ch == "}":
            stack -= 1
            if stack == 0:
                return "".join(out)
            if stack < 0:
                return None
            out.append(ch)
        elif stack >= 1:
            out.append(ch)
    return None


def extract_boxed(text: Optional[str]) -> Optional[str]:
    """Return the last ``\\boxed{...}`` value found in ``text``, or None."""
    if not text or not isinstance(text, str):
        return None
    matches: list[str] = []
    for pattern in _BOXED_PATTERNS:
        for match in re.finditer(pattern, text):
            start = match.end() - 1  # position of '{'
            inner = _extract_balanced_braces(text, start)
            if inner is not None:
                matches.append(inner.strip())
    return matches[-1] if matches else None


# ---------------------------------------------------------------------------
# LLM transport
# ---------------------------------------------------------------------------


# ``max_tokens`` was sized for a judge that answers with a bare 正确/错误
# (the official prompt's value is 5). A *reasoning* model bills its hidden
# thinking against the same budget, so at 5 tokens it can spend everything on
# reasoning and return an empty ``content`` with ``finish_reason == "length"``
# — the verdict never arrives and the caller silently falls back. When that
# happens the call is retried at each of these budgets in turn, so a model that
# exhausts 4096 still gets a larger try rather than losing the verdict. (One
# case in the probe below did exhaust 4096, which is why this escalates instead
# of retrying once.)
#
# 8192 was not the top: the two datasets that reached the shared judge both had
# problems that burned every rung and still returned nothing (SciAgentGYM
# problem 11 in five separate runs, DrugQA preclinical_research_2), so the
# ladder continues past it. Every rung is a real request, so the ceiling is a
# cost decision as much as a quality one — see ``_STARTING_BUDGET`` for how the
# discovery is paid for once rather than per problem.
_REASONING_ESCALATION_TOKENS = (4096, 8192, 16384, 32768)

#: What each model needs to be *started* at, learned in-process. Keyed by
#: ``(api_base, model)`` because the same model name behind two endpoints is
#: two different deployments.
#:
#: Without this, a model that needs 16384 pays for the discovery on every single
#: call: a failure at each rung bills 2048+4096+8192+16384+32768 output tokens,
#: nearly all of them reasoning nobody reads, and a batch of 21 problems that
#: all trip the same wall pays it 21 times. The memo holds the high-water mark
#: of what has been *attempted* (not just what succeeded), so a model that
#: cannot finish at any rung is tried once at the top and then skips straight to
#: the top on the next problem — same verdict, one request instead of five.
#: It is a floor for the next call, never a ceiling: ``cfg.max_tokens`` is
#: still tried if it is larger.
_STARTING_BUDGET: dict[tuple[str, str], int] = {}


def reset_starting_budget() -> None:
    """Forget the per-model budget memo (tests, and after a config reload)."""
    _STARTING_BUDGET.clear()

# Why the last ``_call_judge`` produced no verdict. Callers collapse
# "no judge configured" and "the judge call failed" into the same fallback,
# so the reason is kept here for the report notes — see
# :func:`get_last_judge_error`.
_LAST_ERROR: str = ""


def get_last_judge_error() -> str:
    """Return why the last judge call returned no text, or ``""``."""
    return _LAST_ERROR


def _set_last_error(reason: str) -> None:
    global _LAST_ERROR
    _LAST_ERROR = reason


# Question figures ride along as OpenAI vision content blocks. The caps mirror
# ``rcb_official_judge``: 4 MB of raw bytes becomes ~5.4 MB of base64 inside
# the request body, comfortably inside the 20 MB request budget. The
# SciAgentGYM gallery tops out at 2.36 MB, so the downscale path is a guard
# rail rather than the common case. 12 is above the 8 figures the largest dump
# entry carries — a cap that actually bit would silently drop a figure.
MAX_IMAGE_BYTES = 4_000_000
MAX_TOTAL_IMAGE_BYTES = 12_000_000
MAX_IMAGES = 12

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}

# SVG is deliberately absent from IMAGE_EXTENSIONS: a vision model needs a
# raster, and the chat-completions API rejects image/svg+xml.
_IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
}

try:  # Pillow is in environment.yml; the import guard keeps judge-only
    from PIL import Image as _PILImage  # installs working without it.

    _HAS_PILLOW = True
except ImportError:  # pragma: no cover - depends on the environment
    _HAS_PILLOW = False


def _image_mime(path: Path) -> str:
    return _IMAGE_MIME_TYPES.get(path.suffix.lower(), "application/octet-stream")


def _image_data_uri(path: Path) -> str:
    """Inline one image as a ``data:`` URI the chat API can fetch."""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{_image_mime(path)};base64,{encoded}"


def _prepare_image(path: Path, max_bytes: int) -> tuple[Optional[Path], str]:
    """Return ``(readable path, "")`` or ``(None, reason to skip it)``.

    Unlike ``rcb_official_judge._downscale_image``, the downscaled JPEG is
    returned from a temp directory the caller is expected to leave behind for
    the process (it is read once, immediately) — no file is left in the
    dataset tree and nothing is written next to the source.
    """
    if not path.is_file():
        return None, f"{path}: not found"
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        return None, f"{path}: {path.suffix!r} is not an accepted image type"
    size = path.stat().st_size
    if size <= max_bytes:
        return path, ""
    if not _HAS_PILLOW:
        return None, f"{path}: {size} bytes > {max_bytes} and Pillow is unavailable"
    try:
        with _PILImage.open(path) as raw:
            image = raw.convert("RGB")
            for scale, quality in (
                (1.0, 85),
                (0.75, 85),
                (0.5, 75),
                (0.35, 65),
                (0.25, 55),
                (0.18, 55),
            ):
                sized = (
                    image
                    if scale == 1.0
                    else image.resize(
                        (
                            max(1, int(image.width * scale)),
                            max(1, int(image.height * scale)),
                        ),
                        _PILImage.LANCZOS,
                    )
                )
                buffer = io.BytesIO()
                sized.save(buffer, format="JPEG", quality=quality, optimize=True)
                if buffer.tell() <= max_bytes:
                    tmp_dir = Path(tempfile.mkdtemp(prefix="scimas-judge-"))
                    target = tmp_dir / "figure.jpg"
                    target.write_bytes(buffer.getvalue())
                    return target, ""
    except Exception as exc:
        return None, f"{path}: downscale failed ({type(exc).__name__}: {exc})"
    return None, f"{path}: still over {max_bytes} bytes after downscaling"


def _attachable_images(images: Iterable[str]) -> tuple[list[str], list[str]]:
    """Split ``images`` into the attachable ones and why the rest are not.

    The returned paths are *prepared*: an over-cap image comes back as the
    downscaled JPEG, which is what has to be encoded. Never raises.
    """
    entries = [str(image).strip() for image in images if str(image).strip()]
    attachable: list[str] = []
    dropped: list[str] = []
    for entry in entries[MAX_IMAGES:]:
        dropped.append(f"{entry}: skipped, over {MAX_IMAGES} images")
    for entry in entries[:MAX_IMAGES]:
        prepared, reason = _prepare_image(Path(entry), MAX_IMAGE_BYTES)
        if prepared is None:
            dropped.append(reason)
        else:
            attachable.append(str(prepared))
    return attachable, dropped


def _image_content_blocks(
    paths: Iterable[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Encode prepared images as vision content blocks. Never raises.

    Returns ``(blocks, dropped)`` where ``dropped`` names every image that did
    not make it and why, so the grader can say so in the run notes instead of
    letting a judge that never saw the chart mark a chart-reading answer wrong
    in silence.
    """
    blocks: list[dict[str, Any]] = []
    dropped: list[str] = []
    total = 0
    for path in paths:
        try:
            uri = _image_data_uri(Path(path))
        except Exception as exc:
            dropped.append(f"{path}: unreadable ({type(exc).__name__}: {exc})")
            continue
        # Checked on the encoded length, before the block is built.
        if total + len(uri) > MAX_TOTAL_IMAGE_BYTES:
            dropped.append(
                f"{path}: skipped, total image payload would exceed "
                f"{MAX_TOTAL_IMAGE_BYTES} bytes"
            )
            continue
        total += len(uri)
        blocks.append({"type": "image_url", "image_url": {"url": uri}})
    return blocks, dropped


# Why the last judge call did not carry the question's figures. Kept apart from
# ``_LAST_ERROR``, which is the *no verdict* reason: a call that succeeded
# text-only after a vision failure has no error at all, yet the verdict must
# not pretend the figures were seen.
_LAST_IMAGE_NOTE: str = ""


def get_last_image_note() -> str:
    """Return a note about the last call's question images, or ``""``."""
    return _LAST_IMAGE_NOTE


def _set_image_note(reason: str) -> None:
    global _LAST_IMAGE_NOTE
    _LAST_IMAGE_NOTE = reason


def _judge_choice(payload: Any) -> tuple[str, str]:
    """Return ``(content, finish_reason)`` from a chat-completions payload."""
    try:
        choice = payload["choices"][0]
        return str(choice["message"]["content"] or "").strip(), str(
            choice.get("finish_reason") or ""
        )
    except (KeyError, IndexError, TypeError):
        return "", ""


def _call_judge(
    prompt: str,
    cfg: LLMJudgeConfig,
    system: str,
    images: Optional[Iterable[str]] = None,
) -> Optional[str]:
    """Issue a single chat-completions request. Returns the assistant text
    or ``None`` on any transport / parse failure (never raises).

    ``images`` optionally attaches figures to the user turn as vision content
    blocks. With no images the user content stays a plain string — the shape
    every caller had before figures existed, and the shape the prompt-level
    tests assert on.
    """
    try:
        import httpx
    except ImportError:
        _set_last_error("httpx is not installed")
        return None
    base = cfg.api_base.rstrip("/")
    url = f"{base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    # Encode once, outside ``send``: the reasoning-escalation retry re-sends the
    # same request, and re-base64-encoding megabytes per attempt is waste.
    attachable, dropped = _attachable_images(images or [])
    blocks, unencodable = _image_content_blocks(attachable)
    _set_image_note("; ".join([*dropped, *unencodable]))
    user_content: Any = (
        [{"type": "text", "text": prompt}, *blocks] if blocks else prompt
    )

    def send(max_tokens: int) -> Optional[Any]:
        body = {
            "model": cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            "temperature": cfg.temperature,
            "max_tokens": max_tokens,
        }
        try:
            with httpx.Client(timeout=cfg.timeout) as client:
                resp = client.post(url, headers=headers, json=body)
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            _set_last_error(
                f"HTTP {status} from {url}" if status
                else f"request to {url} failed: {type(exc).__name__}: {exc}"
            )
            return None

    _set_last_error("")
    # Start where this model last needed to start, then climb. ``start`` is a
    # floor, so a config that raises ``max_tokens`` above the memo still wins.
    memo = _STARTING_BUDGET.get((cfg.api_base, cfg.model), 0)
    start = max(cfg.max_tokens, memo)
    budgets = [start, *(b for b in _REASONING_ESCALATION_TOKENS if b > start)]
    #: The best reply so far. Kept because escalating can come back *worse*:
    #: a larger budget that returns nothing must not discard a shorter reply
    #: that was cut off mid-sentence, which a caller able to salvage a partial
    #: answer (``_parse_drugqa_judgement``) still gets value from.
    best: tuple[str, str] = ("", "")
    #: Tracked apart from ``best`` because ``best`` only moves for a non-empty
    #: reply, and the reason an *empty* one was empty is exactly what the
    #: error message has to report ("length" — a reasoning model — versus
    #: "stop", which is a model problem no larger budget will fix).
    last_finish = ""
    transport_failed = False
    for budget in budgets:
        payload = send(budget)
        if payload is None:
            # Transport failure, not a budget one. Stop climbing — a bigger
            # ``max_tokens`` does not fix a dead endpoint — but keep whatever
            # an earlier rung returned instead of discarding it.
            transport_failed = True
            break
        content, finish = _judge_choice(payload)
        last_finish = finish
        _STARTING_BUDGET[(cfg.api_base, cfg.model)] = max(memo, budget)
        if content and finish != "length":
            return content
        if len(content) > len(best[0]):
            best = (content, finish)
        if finish != "length":
            # A model that stopped on its own will not stop differently for
            # more room; only a truncated reply is worth a bigger budget.
            break
    content, finish = best
    if not content:
        finish = last_finish
        # ``send`` already recorded *why* the last request died, and "HTTP 400
        # from <url>" and "the model thought until its budget ran out" need
        # different fixes — so a transport failure keeps its own reason rather
        # than being relabelled here as an empty completion.
        if not transport_failed:
            _set_last_error(
                f"model {cfg.model!r} returned no text "
                f"(finish_reason={finish or 'unknown'})"
                + (
                    "; it appears to be a reasoning model — raise default_max_tokens"
                    " in config/llm_judge.json"
                    if finish == "length" else ""
                )
            )
        return None
    # Truncated at every rung, but non-empty: return it rather than nothing —
    # the caller decides whether the fragment is usable.
    return content


# ---------------------------------------------------------------------------
# Public judge helpers
# ---------------------------------------------------------------------------


_ANSWER_CORRECT_PROMPT = (
    "这是一个问题、一个标准答案，以及一个由AI模型生成的答案。请你判断AI模型的答案是否正确。\n"
    "评判要求：\n"
    "1. 忽略答案中的格式差异。\n"
    "2. 忽略无关的前缀或文字修饰。\n"
    "3. 如果标准答案是一个表达式或数值，只要模型答案在逻辑、计算或数值上等价，也视为正确。\n"
    "4. 若模型答案与标准答案**核心内容一致**，则判断为\"正确\"。\n"
    "{extra_rules}"
    "\n"
    "请只输出：`正确` 或 `错误`。\n"
    "\n"
    "---\n"
    "问题：\n{question}\n"
    "---\n"
    "{workings}"
    "标准答案：\n{expected}\n"
    "---\n"
    "AI模型的答案：\n{predicted}\n"
    "---\n"
)

# An earlier version of this block also carried a rule telling the judge that
# "only the last step counts". That rule was removed: across the SciAgentGYM
# dumps the chain's last *valued* step is frequently a helper result rather
# than the quantity the question asks for (a tension, a radius of gyration, a
# raw property), so the rule named the wrong target for most entries whose
# chain disagrees with the `answer` field. Measured against the API it never
# changed a verdict anyway — the judge reads the block below as context and
# rules on 标准答案 — so the block is now labelled as context only.
_ANSWER_WORKINGS_BLOCK = (
    "参考解题步骤（供理解题意，判断以标准答案为准）：\n{steps}\n---\n"
)

# ``metadata.solution_steps`` — the dataset's own prose account of how the
# entry is meant to be solved. Unlike the tool chain above, its last step
# names the quantity the question actually asks for.
#
# It is needed because on a substantial minority of entries the ``answer``
# field holds an *intermediate* value. The worked example that forced this:
# id 12 asks for "the minimum expected relative standard deviation for the
# class"; its ``answer`` field is 2.8% — the between-laboratory RSD from
# ``horwitz_trumpet`` — while its ``solution_steps`` read "1. 依据Horwitz喇叭
# 经验关系计算…实验室间RSD / 2. 调用 intra_laboratory_rsd 以系数0.5估算班级内
# 最小RSD / 3. 核对题意并记录最终数值回答". The question asks for the second
# step, so the correct response is 1.41421, and a judge ruling on ``answer``
# alone marks it 错误. Measured against the API, adding this block plus the
# rule below flips id 12 to 正确 while leaving ids 21 and 37 — where the
# ``answer`` field *is* the asked-for quantity — correct.
#
# This is the inverse of the mistake the removed "only the last step counts"
# rule made: there the *tool chain* was named as the target, and its last
# valued step is frequently a helper. Here the target is the question, and
# solution_steps is evidence about what the question asks for.
_ANSWER_SOLUTION_STEPS_BLOCK = (
    "数据集标注的解题步骤（说明本题实际所问的最终量）：\n{steps}\n---\n"
)

_ANSWER_SOLUTION_STEPS_RULE = (
    "5. 上面的「数据集标注的解题步骤」说明了本题实际所问的最终量。"
    "标准答案字段有时只记录了中间步骤的数值（例如只写实验室间RSD，"
    "而题目问的是再乘以系数之后的班级内RSD）。遇到这种情况，"
    "请以问题实际所问、且解题步骤最后一步给出的量为准。\n"
)

# The question's own figures, attached to the same message as vision content
# blocks (see :func:`_image_content_blocks`). The wording carries the whole
# weight of a distinction that is easy to get backwards: here the image belongs
# to the QUESTION. ResearchClawBench's judge attaches images too, but there
# image 0 is the ground-truth *target* the report is scored against — a judge
# given this block without the sentence below would read the figures as part of
# the model's answer and grade the chart instead of the text.
_ANSWER_IMAGES_BLOCK = (
    "题目附图（本题原题附有 {count} 张图表，见本条消息末尾的图像内容；"
    "它们是**题目**的一部分，不是AI模型的答案）：\n"
    "---\n"
)

_ANSWER_IMAGES_RULE = (
    "{index}. 上面附的图像属于**原题**（题目给出的图表），不是AI模型的答案。"
    "若题目或参考答案依赖图中的数值、曲线或坐标轴，请先读图再判断模型答案"
    "是否与之一致。\n"
)

_ANSWER_CORRECT_SYSTEM = "You are a strict scientific answer judge."


def judge_correct(
    question: str,
    predicted: str,
    expected: str,
    *,
    dataset: Optional[str] = None,
    workings: Optional[str] = None,
    solution_steps: Optional[str] = None,
    images: Optional[Iterable[str]] = None,
) -> Optional[bool]:
    """Ask the LLM whether ``predicted`` correctly answers ``question``.

    Mirrors the official SciAgentGYM ``is_answer_correct`` prompt.
    Returns ``True``/``False`` on a clean response; returns ``None`` if
    the API call fails, the model is misconfigured, or the response is
    not parseable — callers should fall back to the deterministic
    grader on ``None``.

    ``workings`` optionally supplies the reference solution's tool chain
    (one step per line) as *context*: it lets the judge see the method the
    reference used, which helps when ``expected`` is a symbolic expression
    and the model answered with its numeric value. It never overrides
    ``expected`` — see the note on :data:`_ANSWER_WORKINGS_BLOCK`.

    ``solution_steps`` optionally supplies ``metadata.solution_steps``, the
    dataset's own account of the intended solution. When present it adds a
    rule telling the judge that ``expected`` may be an intermediate step —
    see the note on :data:`_ANSWER_SOLUTION_STEPS_BLOCK`. Omitted, the
    prompt is byte-identical to the official one.

    ``images`` optionally supplies the question's own figures. They are sent as
    vision content blocks *and* announced in the prompt (see the note on
    :data:`_ANSWER_IMAGES_BLOCK`). A judge model that cannot take images fails
    the vision call, in which case the call is retried text-only so the verdict
    still arrives and :func:`get_last_image_note` explains what was lost.
    """
    cfg = resolve_judge_config(dataset=dataset)
    if cfg is None:
        return None
    steps = str(workings or "").strip()
    solution = str(solution_steps or "").strip()
    figures = [str(image).strip() for image in (images or []) if str(image).strip()]
    # What will actually ride along. The same filter runs again inside
    # ``_call_judge``, which is where the drop reasons are recorded — this call
    # exists so the prompt's "N figures" line matches the blocks, instead of
    # pointing the judge at an image that was never attached. (Re-preparing is
    # not free for an over-cap image, but SciAgentGYM has none: its largest
    # figure is 2.36 MB against a 4 MB cap.)
    attachable, _dropped = _attachable_images(figures)
    extra_rules = _ANSWER_SOLUTION_STEPS_RULE if solution else ""
    if attachable:
        # 5 is taken by the solution-steps rule; numbering this 5 as well would
        # read like a truncated list.
        extra_rules += _ANSWER_IMAGES_RULE.format(index=6 if solution else 5)
    prompt = _ANSWER_CORRECT_PROMPT.format(
        question=str(question or ""),
        expected=str(expected or ""),
        predicted=str(predicted or ""),
        workings=(
            _ANSWER_WORKINGS_BLOCK.format(steps=steps) if steps else ""
        ) + (
            _ANSWER_SOLUTION_STEPS_BLOCK.format(steps=solution) if solution else ""
        ) + (
            _ANSWER_IMAGES_BLOCK.format(count=len(attachable)) if attachable else ""
        ),
        extra_rules=extra_rules,
    )
    response = _call_judge(
        prompt, cfg, system=_ANSWER_CORRECT_SYSTEM, images=figures or None
    )
    if response is None and figures:
        # Snapshot before the retry: the retry's own ``_call_judge`` resets
        # ``_LAST_ERROR`` and would erase why the vision attempt died (an HTTP
        # 400 from a model without image support, almost always).
        vision_failure = get_last_judge_error() or "no verdict"
        response = _call_judge(prompt, cfg, system=_ANSWER_CORRECT_SYSTEM)
        if response is None:
            # Keep both reasons: ``_LAST_ERROR`` now holds the text-only
            # failure, the image note holds the vision one.
            _set_image_note(
                f"{len(figures)} question image(s) not sent — vision call failed "
                f"({vision_failure}); the text-only retry failed too"
            )
            return None
        _set_image_note(
            f"{len(figures)} question image(s) not sent — vision call failed "
            f"({vision_failure}); judged text-only"
        )
    if response is None:
        return None
    normalized = response.strip()
    if normalized.startswith("正确"):
        return True
    if normalized.startswith("错误"):
        return False
    if normalized.lower() in {"correct", "yes", "true", "match", "匹配"}:
        return True
    if normalized.lower() in {"incorrect", "no", "false", "mismatch", "不匹配"}:
        return False
    return None


_MATCH_PROMPT = (
    "请判断以下两个答案是否在含义上等价或匹配。\n"
    "即使表达方式不同，只要核心含义、数值、逻辑关系一致就应该判断为匹配。\n"
    "请只回答 '匹配' 或 '不匹配'。\n"
    "\n"
    "---\n"
    "标准答案：{expected}\n"
    "---\n"
    "模型答案：{actual}\n"
    "---\n"
)


_MATCH_SYSTEM = "You are a strict scientific answer judge."


def secondary_verify(
    actual: Any,
    expected: Any,
    *,
    dataset: Optional[str] = None,
) -> Optional[bool]:
    """LLM rescue for the deterministic grader.

    Mirrors SciAgentGYM's ``secondary_verification_with_llm`` (used when
    a leaf-value match fails). Returns ``True``/``False`` for a clean
    response, ``None`` if the API is unavailable.
    """
    cfg = resolve_judge_config(dataset=dataset)
    if cfg is None:
        return None
    prompt = _MATCH_PROMPT.format(
        expected=str(expected),
        actual=str(actual),
    )
    response = _call_judge(prompt, cfg, system=_MATCH_SYSTEM)
    if response is None:
        return None
    if response.startswith("匹配"):
        return True
    if response.startswith("不匹配"):
        return False
    if response.lower() in {"match", "correct", "yes", "true", "匹配"}:
        return True
    if response.lower() in {"mismatch", "incorrect", "no", "false", "不匹配"}:
        return False
    return None


# ---------------------------------------------------------------------------
# DrugQA item-level judge
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DrugQAJudgement:
    """A DrugQA judge verdict: did the answer match *all* the items?

    DrugQA entries carry a list of acceptable answer items
    (``task_info["drugqa_answers"]``), and an answer only counts as correct
    when it names every one of them. That gives two separate signals, and
    the report already aggregates both — ``correct`` drives
    ``answer_accuracy``, ``hit_rate`` drives ``answer_score_avg``:

    - ``correct``   — every item was hit (the dataset's own pass/fail).
    - ``hit_rate``  — ``matched / total``, the partial-credit signal.
    """

    correct: bool
    hit_rate: float
    matched: int
    total: int
    reason: str = ""


# A list-shaped gold does not fit ``_ANSWER_CORRECT_PROMPT``: that prompt has a
# single ``expected`` slot and answers with a bare 正确/错误, so it can express
# neither "which of the N items were hit" nor a hit rate. Rather than renumber
# its rules (5 is the solution-steps rule, 6 the image rule) and put the
# SciAgentGYM wording at risk, DrugQA gets its own prompt and a structured
# reply. Keep every literal brace below doubled — the prompt goes through
# ``str.format``.
_DRUGQA_PROMPT = (
    "这是一个问题、一组标准答案条目，以及一个由AI模型生成的答案。\n"
    "标准答案可能包含多个条目，请你**逐条**判断模型答案是否命中。\n"
    "评判要求：\n"
    "1. 忽略答案中的格式差异、无关的前缀或文字修饰。\n"
    "2. 基因、蛋白、靶点、生物标志物的名称，忽略大小写、连字符与希腊字母写法的"
    "差异（例如 TNF-alpha、TNFα、TNFA 视为同一个条目）。\n"
    "3. 「命中」指模型答案明确给出了该条目所指的实体、结论或机制；"
    "仅仅提到该词但结论相反、或把它排除在外，不算命中。\n"
    "4. 模型答案与某条目在含义上等价（换用同义表述、给出该基因的别名）即算命中。\n"
    "5. correct 表示是否**全部**条目都命中。\n"
    "\n"
    "请只输出一个 JSON 对象，不要输出任何其他内容：\n"
    '{{"correct": true或false, "matched": 命中条目数, "total": {total}, '
    '"hit_rate": 命中率(0到1之间的小数), "reason": "一句话说明"}}\n'
    "\n"
    "---\n"
    "问题：\n{question}\n"
    "---\n"
    "标准答案条目（共 {total} 条，编号仅供参考，请按内容匹配）：\n{expected}\n"
    "---\n"
    "AI模型的答案：\n{predicted}\n"
    "---\n"
)

_DRUGQA_SYSTEM = "You are a strict scientific answer judge."


def _json_object_from_text(text: Optional[str]) -> Optional[dict[str, Any]]:
    """Return the first parseable JSON object embedded in ``text``.

    Judge replies come back wrapped in prose or a ```json fence often enough
    that slicing at the first ``{`` and scanning forward is worth doing: a
    strict ``json.loads`` on the whole reply would throw the verdict away.
    """
    if not text:
        return None
    start = text.find("{")
    while start != -1:
        inner = _extract_balanced_braces(text, start)
        if inner is not None:
            try:
                parsed = json.loads("{" + inner + "}")
            except (ValueError, TypeError):
                parsed = None
            if isinstance(parsed, dict):
                return parsed
        start = text.find("{", start + 1)
    return None


# A whole ``"key": value`` pair, where the value is a JSON scalar. Anchored on
# the opening quote so a key that appears inside a *string* value cannot be
# mistaken for a field.
_TRUNCATED_FIELD = re.compile(
    r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:\s*'
    r'("(?:[^"\\]|\\.)*"|true|false|null|-?\d+(?:\.\d+)?)'
)


def _salvage_truncated_object(text: Optional[str]) -> Optional[dict[str, Any]]:
    """Recover the completed fields of an object that was cut off mid-write.

    A reasoning model whose budget runs out mid-JSON returns something like
    ``{"correct": false,`` — the brace never closes, so it is not JSON and
    :func:`_json_object_from_text` (correctly) refuses it. But ``correct`` is
    the *first* field of :data:`_DRUGQA_PROMPT`'s object by design, so the
    verdict is very often exactly the part that survived. Reading back the
    fields that were written whole recovers a real judge ruling instead of
    discarding it for a token-overlap guess.

    Only ever called for a reply that is *unterminated*. A reply that closes
    its brace and still fails to parse is malformed in some other way, and
    guessing at it is not this function's job.
    """
    if not text:
        return None
    start = text.find("{")
    if start == -1 or _extract_balanced_braces(text, start) is not None:
        return None
    payload: dict[str, Any] = {}
    for match in _TRUNCATED_FIELD.finditer(text[start:]):
        key, raw = match.group(1), match.group(2)
        if key in payload:
            continue
        try:
            payload[key] = json.loads(raw)
        except (ValueError, TypeError):
            continue
    return payload or None


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "正确"}:
            return True
        if lowered in {"false", "no", "错误"}:
            return False
    return None


def _coerce_int(value: Any) -> Optional[int]:
    # ``bool`` is an ``int`` in Python; ``true`` must not read as 1 match.
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        match = re.search(r"\d+", value)
        if match:
            return int(match.group())
    return None


def _coerce_rate(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        rate = float(value)
    elif isinstance(value, str):
        match = re.search(r"\d+(?:\.\d+)?", value)
        if not match:
            return None
        rate = float(match.group())
    else:
        return None
    # A judge that answers 50 for "half" means percent, not 50x.
    if 1.0 < rate <= 100.0:
        rate = rate / 100.0
    return max(0.0, min(rate, 1.0))


def _parse_drugqa_judgement(
    text: Optional[str],
    *,
    total: int,
) -> Optional[DrugQAJudgement]:
    """Parse a DrugQA judge reply into a :class:`DrugQAJudgement`.

    ``total`` is supplied by the caller and never read from the model: a judge
    that under-counts the gold items would otherwise raise its own hit rate.

    ``hit_rate`` is derived from ``matched`` rather than trusted, so the two
    fields can never contradict each other. When the judge's own ``correct``
    flag disagrees with that count, neither value is rewritten — the
    disagreement is recorded in ``reason`` so a reader can audit it.

    Returns ``None`` only when the reply carries no usable count *and* no
    verdict, which sends the caller to the deterministic scorer.
    """
    if total <= 0:
        return None
    payload = _json_object_from_text(text)
    salvaged = False
    if payload is None:
        payload = _salvage_truncated_object(text)
        salvaged = payload is not None
        # A cut-off reply that never wrote its verdict has nothing to salvage;
        # ``correct`` is the one field worth recovering.
        if payload is not None and "correct" not in payload:
            return None
    if payload is None:
        return None
    notes: list[str] = []
    if salvaged:
        notes.append(
            "reply was cut off mid-object; read the fields it did complete"
        )
    matched = _coerce_int(payload.get("matched"))
    rate = _coerce_rate(payload.get("hit_rate"))
    if matched is None:
        if rate is not None:
            matched = int(round(rate * total))
    elif rate is not None and abs(rate - matched / total) > 0.01:
        # ``matched`` wins, but a judge whose own rate disagrees with its own
        # count is worth flagging: it is the signal that the reply is sloppy.
        notes.append(f"judge reported hit_rate={rate:.2f} with matched={matched}/{total}")
    correct = _coerce_bool(payload.get("correct"))
    if matched is None:
        if correct is None:
            return None
        # No count at all, but the judge did rule. Dropping that verdict to the
        # fallback would contradict "the judge is authoritative": True means
        # every item was hit, False leaves the count unknown so it reads as 0.
        matched = total if correct else 0
        notes.append(f"judge gave no item count; read as matched={matched}/{total}")
    matched = max(0, min(matched, total))
    hit_rate = matched / total
    if correct is None:
        correct = matched == total
    elif correct != (hit_rate >= 1.0):
        notes.append(f"judge reported correct={correct} with matched={matched}/{total}")
    reason = str(payload.get("reason") or "").strip()
    if notes:
        suffix = "；".join(notes)
        reason = f"{reason}（{suffix}）" if reason else suffix
    return DrugQAJudgement(
        correct=correct,
        hit_rate=hit_rate,
        matched=matched,
        total=total,
        reason=reason,
    )


def judge_drugqa_items(
    question: str,
    predicted: str,
    expected_items: Iterable[str],
    *,
    dataset: Optional[str] = None,
) -> Optional[DrugQAJudgement]:
    """Judge a DrugQA answer against its list of acceptable items.

    Returns ``None`` when there are no items to judge, no judge is configured,
    the call fails, or the reply carries no usable match count — callers then
    fall back to the deterministic scorer. A reply that parses to a count
    always yields a :class:`DrugQAJudgement`; see
    :func:`_parse_drugqa_judgement` for how the two fields are derived.
    """
    items = [str(item).strip() for item in (expected_items or []) if str(item).strip()]
    if not items:
        return None
    cfg = resolve_judge_config(dataset=dataset)
    if cfg is None:
        return None
    prompt = _DRUGQA_PROMPT.format(
        question=str(question or ""),
        expected="\n".join(f"{idx}. {item}" for idx, item in enumerate(items, start=1)),
        predicted=str(predicted or ""),
        total=len(items),
    )
    response = _call_judge(prompt, cfg, system=_DRUGQA_SYSTEM)
    if response is None:
        return None
    judgement = _parse_drugqa_judgement(response, total=len(items))
    if judgement is None:
        # ``_call_judge`` succeeded, so ``_LAST_ERROR`` is empty; without this
        # the report would show a bare fallback with no reason at all.
        _set_last_error(
            f"model {cfg.model!r} returned an unparseable judgement: "
            f"{response.strip()[:200]!r}"
        )
    return judgement


# ---------------------------------------------------------------------------
# MADD requirement-completion judge
# ---------------------------------------------------------------------------


#: Ceiling on the molecules the judge is asked to enumerate. A pathological
#: answer is not worth a 32k-token reply, and the cap is applied here rather
#: than by truncating ``predicted`` so the note can say what was dropped.
_MADD_MAX_MOLECULES = 200


@dataclass(frozen=True)
class MADDCompletionJudgement:
    """A MADD judge verdict on *requirement completion*, not correctness.

    ``dataset/MADD/dataset_L.xlsx`` carries no ground truth, so there is
    no answer to grade against. What the judge reads is whether the work
    was delivered, in the two halves the grader then scores:

    - ``answered`` — one flag per molecule-generation requirement, in the
      order the table lists them: did the answer give a molecule for it?
    - ``molecule_metric_coverage`` — per molecule the answer reported,
      the fraction of the required properties it carried (0..1).

    The judge is deliberately not asked whether the molecules are
    chemically sound, and its arithmetic is not trusted: it returns
    counts, and :func:`grader._madd_completion_from_parts` turns those
    into the 0..1 ``answer_score``.
    """

    answered: list[bool]
    molecule_metric_coverage: list[float]
    reason: str = ""


# MADD's answer metric used to be a hit rate against a gold molecule list
# and was asked for one. There is no list any more, so the model is asked
# for the two *counts* the halves need — the same "structured reply, score
# computed in Python" shape the DrugQA judge uses.
_MADD_COMPLETION_PROMPT = (
    "这是一个药物设计任务、该任务的子要求列表，以及一个由AI模型生成的答案。\n"
    "请只判定**完成度**，**不要**评判分子本身：不判断结构是否合理、不判断性质数值"
    "是否可信、不判断设计方案好不好。只回答下面两个问题。\n"
    "\n"
    "A. 对每一个子要求，答案是否给出了至少一个**具体的分子**"
    "（SMILES 或明确的结构标识）？\n"
    "   - 只谈类别（如「已生成若干候选分子」「前 10 名」）而没有列出任何具体分子 = false。\n"
    "   - 同一个分子可以同时算作多个子要求的回答。\n"
    "   - 必须**恰好 {total} 项**、按下面给出的子要求顺序返回，"
    "即使答案完全忽略了某个子要求，也要为它返回一项 false。\n"
    "B. 答案里报告的**每一个分子**，给出了几项必需性质？\n"
    "   - 数一数该分子那一行（或那一段）里，下列 {metric_count} 项必需指标中"
    "**有值**的有几项。\n"
    "   - 必需指标（顺序固定）：{metrics}\n"
    "   - 写成「n/a」「N/A」「-」「未知」这类表示没算出来的占位符，**不算**有值。\n"
    "   - 不要自己算分数或百分比，只给出这个 0 到 {metric_count} 的整数个数。\n"
    "\n"
    "请只输出一个 JSON 对象，不要输出任何其他内容：\n"
    '{{"subtask_answered": [true或false，共 {total} 项，按子要求顺序],\n'
    '  "molecules": [{{"id": "该分子的 SMILES 或表内标识", '
    '"metrics_reported": 0到{metric_count}的整数}}],\n'
    '  "reason": "一句话说明"}}\n'
    "\n"
    "---\n"
    "任务：\n{question}\n"
    "---\n"
    "子要求（共 {total} 个，请按此顺序返回 {total} 项）：\n{subtasks}\n"
    "---\n"
    "AI模型的答案：\n{predicted}\n"
    "---\n"
)

_MADD_COMPLETION_SYSTEM = "You are a strict scientific requirement-completeness judge."


def _coerce_bool_list(value: Any) -> Optional[list[bool]]:
    """Coerce the reply's ``subtask_answered`` into a list of flags.

    A bare bool is *not* accepted — the whole point of the field is that it
    is per-requirement, and reading a scalar as "all true" would hand out
    half a point for a reply that never answered the question. ``None`` for
    anything that is not a list of booleans.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        return None
    flags: list[bool] = []
    for item in value:
        if isinstance(item, dict):
            item = item.get("has_molecules", item.get("answered"))
        flag = _coerce_bool(item)
        if flag is None:
            return None
        flags.append(flag)
    return flags


def _parse_madd_completion(
    text: Optional[str],
    *,
    sub_tasks: int,
    metric_count: int,
) -> Optional[MADDCompletionJudgement]:
    """Parse a completion reply into a :class:`MADDCompletionJudgement`.

    ``sub_tasks`` and ``metric_count`` come from the caller, never from the
    reply — a judge that under-counts the requirements would otherwise
    raise its own Half A, exactly the hole :func:`_parse_madd_judgement`
    closed for DrugQA with its caller-supplied ``total``.

    Returns ``None`` when nothing usable is left to rule on: no JSON
    object, a ``subtask_answered`` that is not exactly ``sub_tasks`` long,
    or a ``molecules`` list whose every entry carried an out-of-range count.
    A genuinely empty ``molecules`` array is *not* that case — it is a
    judgement that the answer reported no molecules, and it scores zero
    rather than sending the grader down its "give up" path.
    """
    if sub_tasks <= 0 or metric_count <= 0:
        return None
    payload = _json_object_from_text(text)
    salvaged = False
    if payload is None:
        payload = _salvage_truncated_object(text)
        salvaged = payload is not None
        # A truncated reply that never reached the two arrays has nothing to
        # salvage: the fields it did emit are the ones we do not want.
        if payload is not None and not (
            isinstance(payload.get("subtask_answered"), list)
            and isinstance(payload.get("molecules"), list)
        ):
            return None
    if payload is None:
        return None

    notes: list[str] = []
    if salvaged:
        notes.append("reply was cut off mid-object; read the fields it did complete")

    answered = _coerce_bool_list(payload.get("subtask_answered"))
    if answered is None:
        return None
    if len(answered) != sub_tasks:
        # Wrong length means the model did not follow the one instruction the
        # whole half depends on; guessing which entries align would be worse
        # than declaring the call unusable.
        return None

    raw_molecules = payload.get("molecules")
    if not isinstance(raw_molecules, list):
        return None
    if len(raw_molecules) > _MADD_MAX_MOLECULES:
        notes.append(
            f"judge enumerated {len(raw_molecules)} molecules; "
            f"kept the first {_MADD_MAX_MOLECULES}"
        )
        raw_molecules = raw_molecules[:_MADD_MAX_MOLECULES]

    coverage: list[float] = []
    dropped = 0
    for entry in raw_molecules:
        if not isinstance(entry, dict):
            dropped += 1
            continue
        count = _coerce_int(entry.get("metrics_reported"))
        if count is None or not 0 <= count <= metric_count:
            dropped += 1
            continue
        coverage.append(count / metric_count)
    if raw_molecules and not coverage:
        # Every row the judge returned was unusable, so there is no honest
        # reading of the second half — including "zero", which is an
        # assertion about the answer rather than about the reply.
        return None
    if dropped:
        notes.append(f"ignored {dropped} molecule row(s) with an out-of-range count")

    reason = str(payload.get("reason") or "").strip()
    if notes:
        suffix = "；".join(notes)
        reason = f"{reason}（{suffix}）" if reason else suffix
    return MADDCompletionJudgement(
        answered=answered,
        molecule_metric_coverage=coverage,
        reason=reason,
    )


def judge_madd_completion(
    question: str,
    subtask_cases: Iterable[str],
    required_metrics: Iterable[str],
    predicted: str,
    *,
    dataset: Optional[str] = None,
) -> Optional[MADDCompletionJudgement]:
    """Judge how much of a MADD task's stated work the answer delivered.

    ``subtask_cases`` is the requirement list, in the order ``case`` gives
    it, one label per requirement; the reply must carry exactly that many
    flags. Returns ``None`` when there are no requirements, no judge is
    configured, the call fails, or the reply cannot be read — the caller
    treats the first as unscoreable and the rest per its own policy.
    """
    subtasks = [str(item).strip() for item in (subtask_cases or []) if str(item).strip()]
    metrics = [str(item).strip() for item in (required_metrics or []) if str(item).strip()]
    if not subtasks or not metrics:
        return None
    cfg = resolve_judge_config(dataset=dataset)
    if cfg is None:
        return None
    prompt = _MADD_COMPLETION_PROMPT.format(
        question=str(question or ""),
        subtasks="\n".join(
            f"{idx}. {name}" for idx, name in enumerate(subtasks, start=1)
        ),
        predicted=str(predicted or ""),
        total=len(subtasks),
        metrics=", ".join(metrics),
        metric_count=len(metrics),
    )
    response = _call_judge(prompt, cfg, system=_MADD_COMPLETION_SYSTEM)
    if response is None:
        return None
    judgement = _parse_madd_completion(
        response, sub_tasks=len(subtasks), metric_count=len(metrics)
    )
    if judgement is None:
        _set_last_error(
            f"model {cfg.model!r} returned an unparseable judgement: "
            f"{response.strip()[:200]!r}"
        )
    return judgement


__all__ = [
    "CONFIG_PATH",
    "DrugQAJudgement",
    "LLMJudgeConfig",
    "MADDCompletionJudgement",
    "NO_CONFIG",
    "dataset_setting",
    "extract_boxed",
    "get_judge_cache_dir",
    "get_last_image_note",
    "get_last_judge_error",
    "is_judge_enabled",
    "judge_correct",
    "judge_drugqa_items",
    "judge_madd_completion",
    "reset_starting_budget",
    "resolve_judge_config",
    "secondary_verify",
    "set_judge_cache_dir",
]