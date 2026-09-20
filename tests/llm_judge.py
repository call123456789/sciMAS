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


def _try_prefix(prefix: str, cfg: dict[str, Any]) -> Optional[LLMJudgeConfig]:
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
    return _build_config(cfg, api_key=api_key, api_base=api_base, model=model, prefix=prefix)


def _build_config(
    cfg: dict[str, Any],
    *,
    api_key: str,
    api_base: str,
    model: str,
    prefix: str,
) -> LLMJudgeConfig:
    """Assemble a config from a credential source plus the shared settings."""
    try:
        timeout = float(cfg.get("default_timeout_seconds") or _DEFAULT_CONFIG["default_timeout_seconds"])
    except (TypeError, ValueError):
        timeout = _DEFAULT_CONFIG["default_timeout_seconds"]
    try:
        max_tokens = int(cfg.get("default_max_tokens") or _DEFAULT_CONFIG["default_max_tokens"])
    except (TypeError, ValueError):
        max_tokens = _DEFAULT_CONFIG["default_max_tokens"]
    return LLMJudgeConfig(
        api_base=api_base,
        api_key=api_key,
        model=model,
        timeout=timeout,
        max_tokens=max_tokens,
        prefix=prefix,
    )


def _try_config_credentials(cfg: dict[str, Any]) -> Optional[LLMJudgeConfig]:
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
        cfg, api_key=api_key, api_base=api_base, model=model, prefix="<config>"
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
        resolved = _try_prefix(prefix, cfg)
        if resolved is not None:
            return resolved
    # No env var carried a credential — fall back to the config file's own.
    return _try_config_credentials(cfg)


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
_REASONING_ESCALATION_TOKENS = (4096, 8192)

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
    payload = send(cfg.max_tokens)
    if payload is None:
        return None
    content, finish = _judge_choice(payload)
    for budget in _REASONING_ESCALATION_TOKENS:
        if content or finish != "length" or cfg.max_tokens >= budget:
            break
        # The budget ran out before any answer was written — a reasoning
        # model thinking out loud. Give it more room and try again.
        payload = send(budget)
        if payload is None:
            return None
        content, finish = _judge_choice(payload)
    if not content:
        _set_last_error(
            f"model {cfg.model!r} returned no text (finish_reason={finish or 'unknown'})"
            + (
                "; it appears to be a reasoning model — raise default_max_tokens"
                " in config/llm_judge.json"
                if finish == "length" else ""
            )
        )
        return None
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


__all__ = [
    "CONFIG_PATH",
    "LLMJudgeConfig",
    "NO_CONFIG",
    "dataset_setting",
    "extract_boxed",
    "get_judge_cache_dir",
    "get_last_image_note",
    "get_last_judge_error",
    "is_judge_enabled",
    "judge_correct",
    "resolve_judge_config",
    "secondary_verify",
    "set_judge_cache_dir",
]