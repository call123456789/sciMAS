"""Shared helpers for BiOMNI-backed sciMAS MCP servers.

BiOMNI source is not installed as a package; we load its
``biomni/tool/<category>.py`` files on demand via importlib so the
wrappers do not require any extra install step. Heavy imports inside
the BiOMNI functions (matplotlib, Bio.*, scipy, ...) happen at call
time, which is why every wrapper routes execution through
``call_with_error_boundary``: a missing dependency surfaces as a JSON
error instead of crashing the MCP stdio server.

The MCP layer only knows JSON-RPC types (str / int / float / bool /
list / dict), so numpy arrays and pandas DataFrames are coerced here.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence


# BiOMNI is a sibling checkout, not a dependency, so its location differs per
# machine. Resolve it lazily and portably: the env var wins, then a couple of
# conventional layouts, and if none exist we raise on first use with the list
# of paths we tried. Hard-coding one developer's absolute path (which this
# module used to do) made all 34 biomni-* MCP servers look installed while
# every call failed with a confusing ImportError on any other host.
_SCIMAS_ROOT = Path(__file__).resolve().parent.parent.parent


def _biomni_tool_dir_candidates() -> List[Path]:
    env = os.environ.get("BIOMNI_TOOL_DIR")
    if env:
        return [Path(env).expanduser()]
    return [
        # <parent>/Biomni/biomni/tool  — sibling checkout (dev layout)
        _SCIMAS_ROOT.parent / "Biomni" / "biomni" / "tool",
        # <sciMAS>/Biomni/biomni/tool  — vendored inside the repo
        _SCIMAS_ROOT / "Biomni" / "biomni" / "tool",
    ]


def biomni_tool_dir() -> Path:
    """Return the first existing BiOMNI ``tool`` directory.

    Raises ``ImportError`` naming every candidate when none is present, so a
    misconfigured host fails loudly at the point of use rather than at import
    time (importing this module must stay side-effect free — the MCP servers
    import it just to register their tool list).
    """
    candidates = _biomni_tool_dir_candidates()
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    tried = "\n  ".join(str(c) for c in candidates)
    raise ImportError(
        "BiOMNI source not found. Set BIOMNI_TOOL_DIR to the directory "
        "containing biomni/tool/<category>.py, or clone BiOMNI next to "
        f"sciMAS. Tried:\n  {tried}"
    )


_DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get("SCIMAS_BIOMNI_OUTPUT_DIR", "/tmp/scimas_biomni")
)


_MODULE_CACHE: Dict[str, Any] = {}


def load_biomni_module(category: str) -> Any:
    """Import ``biomni/tool/<category>.py`` as a Python module.

    Cached so we pay the cost once per process. Raises ``ImportError``
    with a clear message if the file is missing.
    """
    if category in _MODULE_CACHE:
        return _MODULE_CACHE[category]
    path = biomni_tool_dir() / f"{category}.py"
    if not path.exists():
        siblings = sorted(p.stem for p in path.parent.glob("*.py"))
        raise ImportError(
            f"BiOMNI has no category {category!r} at {path}. "
            f"Available: {', '.join(siblings) or '(none)'}"
        )
    spec = importlib.util.spec_from_file_location(
        f"biomni.tool.{category}", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 — surface as JSON later
        raise ImportError(
            f"Failed to load {path}: {type(exc).__name__}: {exc}"
        ) from exc
    _MODULE_CACHE[category] = module
    return module


def reset_module_cache() -> None:
    """Drop cached BiOMNI modules. Used by tests and after edits."""
    _MODULE_CACHE.clear()


def ok(tool: str, **fields: Any) -> str:
    return json.dumps(
        {"status": "ok", "tool": tool, **fields},
        ensure_ascii=False,
    )


def err(message: str, **fields: Any) -> str:
    return json.dumps(
        {"status": "error", "message": message, **fields},
        ensure_ascii=False,
    )


def normalize_output_dir(output_dir: Any, tool_name: str) -> str:
    """Default ``output_dir`` to /tmp/scimas_biomni/<tool>."""
    text = str(output_dir or "").strip()
    if not text or text in {".", "./"}:
        text = str(_DEFAULT_OUTPUT_ROOT / tool_name)
    path = Path(text).expanduser()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return str(exc)
    return str(path)


def coerce_array(
    value: Any,
    *,
    ndim: Optional[int] = None,
    name: str = "array",
) -> Any:
    """Best-effort conversion of JSON-shaped data to a numpy array.

    ``ndim`` is informational only; we still return whatever shape the
    input encodes. Returns the original value if it cannot be coerced
    so the downstream tool sees the same data shape the caller sent.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        import numpy as np  # local import — only needed when used
    except ImportError:
        return value
    try:
        arr = np.array(value)
    except (TypeError, ValueError):
        return value
    if ndim is not None and arr.ndim != ndim:
        return value
    return arr


def coerce_dataframe(
    value: Any,
    *,
    columns: Optional[Sequence[str]] = None,
    name: str = "dataframe",
) -> Any:
    """Best-effort conversion to ``pandas.DataFrame``.

    Accepts either a list-of-dicts (records) or a CSV/TSV file path.
    Returns the original value if ``pandas`` is unavailable.
    """
    if value is None:
        return None
    if isinstance(value, str) and value:
        if os.path.exists(value):
            try:
                import pandas as pd
                sep = "\t" if value.lower().endswith((".tsv", ".tab")) else ","
                return pd.read_csv(value, sep=sep)
            except ImportError:
                return value
        return value
    if isinstance(value, list):
        try:
            import pandas as pd
            return pd.DataFrame(value, columns=list(columns) if columns else None)
        except ImportError:
            return value
    return value


def call_with_error_boundary(
    tool_name: str,
    func: Callable[..., Any],
    *,
    return_keys: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> str:
    """Invoke ``func(**kwargs)`` and return a JSON envelope.

    Any exception (ImportError for missing BiOMNI deps, RuntimeError
    for failed network calls, etc.) is captured and surfaced as a
    JSON error. The MCP stdio loop stays alive even when a single
    tool call fails.
    """
    try:
        result = func(**kwargs)
    except Exception as exc:  # noqa: BLE001
        return err(
            f"{type(exc).__name__}: {exc}",
            tool=tool_name,
            traceback=traceback.format_exc(limit=4).splitlines(),
        )

    if isinstance(result, str):
        # Most BiOMNI tools already return a research log string; wrap
        # it but preserve the log verbatim under "log".
        if return_keys:
            return ok(tool_name, result_text=result, **{k: kwargs.get(k) for k in return_keys})
        return ok(tool_name, log=result, **{k: v for k, v in kwargs.items() if _safe_field(v)})

    payload: Dict[str, Any] = {"tool": tool_name}
    payload.update(_safe_kwargs(kwargs))
    if isinstance(result, dict):
        payload["result"] = result
    else:
        payload["result_repr"] = repr(result)
    return json.dumps({"status": "ok", **payload}, ensure_ascii=False)


def _safe_field(value: Any) -> bool:
    if isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, (list, dict)) and len(repr(value)) < 240:
        return True
    return False


def _safe_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in kwargs.items() if _safe_field(v)}


def require_positive(name: str, value: float) -> None:
    if value is None:
        raise ValueError(f"{name} must be a finite number")
    try:
        fvalue = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(fvalue) or fvalue <= 0:
        raise ValueError(f"{name} must be > 0 (got {value})")


def truncate(text: str, limit: int = 8000) -> str:
    if not text or len(text) <= limit:
        return text or ""
    return text[:limit] + f"\n...[truncated, {len(text) - limit} more chars]"


def import_optional(name: str) -> Optional[Any]:
    try:
        return importlib.import_module(name)
    except ImportError:
        return None