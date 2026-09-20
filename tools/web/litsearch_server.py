#!/usr/bin/env python3
"""OpenAlex-backed MCP server exposing `search_literature` for sciMAS.

Registered via `config/mcp.json` and made available to `claude -p` through
`--mcp-config`. The server speaks stdio JSON-RPC and exposes a single tool
that returns normalized academic-paper records from the OpenAlex `/works`
endpoint.

Tested against mcp >= 2.0 (uses MCPServer, the FastMCP successor).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

import requests
from mcp.server.mcpserver import MCPServer

mcp = MCPServer("litsearch")

OPENALEX_API = "https://api.openalex.org"
DEFAULT_TIMEOUT_SECONDS = 20.0
MAX_RESULTS = 100
# This file lives at <repo>/tools/web/litsearch_server.py, so two parents up
# is the repo root. Deriving it from __file__ rather than the cwd keeps the
# lookup working when the Claude CLI spawns us from another directory.
LOCAL_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "openalex.local.json"
SELECT_FIELDS = ",".join(
    [
        "id",
        "ids",
        "doi",
        "title",
        "display_name",
        "publication_year",
        "publication_date",
        "type",
        "authorships",
        "primary_location",
        "open_access",
        "abstract_inverted_index",
        "referenced_works_count",
        "cited_by_count",
        "language",
    ]
)


def _dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _local_config_key() -> str | None:
    """Read the API key from the git-ignored ``config/openalex.local.json``.

    Consulted only after every environment variable has failed, so an
    exported key still wins — env vars are the documented override, and a
    deployment that sets them must not be silently overridden by a file on
    the machine. Mirrors ``tests/llm_judge.py:_try_config_credentials``.

    A missing or empty file is not an error: the server falls back to
    anonymous OpenAlex access, which still works at a lower daily quota.
    """
    try:
        data = json.loads(LOCAL_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    key = data.get("api_key") or data.get("OPENALEX_API_KEY")
    if not key:
        return None
    return str(key).strip() or None


def _openalex_api_key() -> str | None:
    key = os.environ.get("OPENALEX_API_KEY") or os.environ.get("OPENALEX_KEY")
    if key:
        return key.strip()

    key_file = os.environ.get("OPENALEX_API_KEY_FILE")
    if key_file:
        try:
            from_file = open(key_file, encoding="utf-8").read().strip()
        except OSError:
            from_file = ""
        if from_file:
            return from_file

    return _local_config_key()


def _coerce_year(value: int | str | None, field: str) -> int | None:
    if value in (None, ""):
        return None
    try:
        year = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a four-digit year") from exc
    if year < 1600 or year > 3000:
        raise ValueError(f"{field} must be between 1600 and 3000")
    return year


def _coerce_bool(value: bool | str | None, field: str) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"{field} must be true or false")


def _clean_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _clean_doi(value: str | None) -> str | None:
    text = _clean_optional_text(value)
    if not text:
        return None
    return re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text, flags=re.I)


def _reconstruct_abstract(inverted_index: Any) -> str | None:
    if not isinstance(inverted_index, dict) or not inverted_index:
        return None

    positions: list[tuple[int, str]] = []
    for word, indexes in inverted_index.items():
        if not isinstance(indexes, list):
            continue
        for index in indexes:
            if isinstance(index, int) and index >= 0:
                positions.append((index, str(word)))

    if not positions:
        return None
    positions.sort(key=lambda item: item[0])
    return " ".join(word for _, word in positions)


def _truncate(text: str | None, max_chars: int) -> tuple[str | None, bool]:
    if not text:
        return None, False
    max_chars = max(0, int(max_chars))
    if max_chars == 0 or len(text) <= max_chars:
        return text, False
    if max_chars <= 3:
        return text[:max_chars], True
    return text[: max_chars - 3].rstrip() + "...", True


def _author_names(authorships: Any, limit: int = 12) -> list[str]:
    if not isinstance(authorships, list):
        return []
    authors: list[str] = []
    for authorship in authorships[:limit]:
        if not isinstance(authorship, dict):
            continue
        author = authorship.get("author")
        if isinstance(author, dict):
            name = _clean_optional_text(author.get("display_name"))
            if name:
                authors.append(name)
    if len(authorships) > limit:
        authors.append(f"et al. ({len(authorships)} authors)")
    return authors


def _source_name(primary_location: Any) -> str | None:
    if not isinstance(primary_location, dict):
        return None
    source = primary_location.get("source")
    if isinstance(source, dict):
        return _clean_optional_text(source.get("display_name"))
    return None


def _normalize_work(work: dict[str, Any], include_abstract: bool, abstract_chars: int) -> dict[str, Any]:
    primary_location = work.get("primary_location") or {}
    open_access = work.get("open_access") or {}
    ids = work.get("ids") or {}
    abstract, abstract_truncated = (None, False)
    if include_abstract:
        abstract, abstract_truncated = _truncate(
            _reconstruct_abstract(work.get("abstract_inverted_index")),
            abstract_chars,
        )

    landing_page_url = None
    pdf_url = None
    if isinstance(primary_location, dict):
        landing_page_url = _clean_optional_text(primary_location.get("landing_page_url"))
        pdf_url = _clean_optional_text(primary_location.get("pdf_url"))
    if not pdf_url and isinstance(open_access, dict):
        pdf_url = _clean_optional_text(open_access.get("oa_url"))

    doi = _clean_doi(work.get("doi") or ids.get("doi"))
    return {
        "title": work.get("title") or work.get("display_name"),
        "authors": _author_names(work.get("authorships")),
        "publication_year": work.get("publication_year"),
        "publication_date": work.get("publication_date"),
        "type": work.get("type"),
        "source": _source_name(primary_location),
        "doi": doi,
        "doi_url": f"https://doi.org/{doi}" if doi else None,
        "openalex_id": work.get("id"),
        "openalex_url": work.get("id"),
        "landing_page_url": landing_page_url,
        "pdf_url": pdf_url,
        "is_open_access": open_access.get("is_oa") if isinstance(open_access, dict) else None,
        "open_access_status": open_access.get("oa_status") if isinstance(open_access, dict) else None,
        "cited_by_count": work.get("cited_by_count"),
        "referenced_works_count": work.get("referenced_works_count"),
        "language": work.get("language"),
        "abstract": abstract,
        "abstract_truncated": abstract_truncated,
    }


def _filter_param(
    filters: str | None,
    from_year: int | str | None,
    to_year: int | str | None,
    is_oa: bool | str | None,
    work_type: str | None,
) -> str | None:
    pieces: list[str] = []
    base_filter = _clean_optional_text(filters)
    if base_filter:
        pieces.extend(part.strip() for part in base_filter.split(",") if part.strip())

    start_year = _coerce_year(from_year, "from_year")
    end_year = _coerce_year(to_year, "to_year")
    if start_year and end_year and start_year > end_year:
        raise ValueError("from_year must be <= to_year")
    if start_year:
        pieces.append(f"from_publication_date:{start_year}-01-01")
    if end_year:
        pieces.append(f"to_publication_date:{end_year}-12-31")

    oa = _coerce_bool(is_oa, "is_oa")
    if oa is not None:
        pieces.append(f"is_oa:{str(oa).lower()}")

    kind = _clean_optional_text(work_type)
    if kind:
        pieces.append(f"type:{kind}")

    return ",".join(pieces) if pieces else None


def _sort_param(sort: str | None) -> str | None:
    text = (sort or "relevance").strip().lower()
    if text in {"", "relevance", "relevance_score"}:
        return None
    aliases = {
        "cited": "cited_by_count:desc",
        "cited_by_count": "cited_by_count:desc",
        "most_cited": "cited_by_count:desc",
        "newest": "publication_date:desc",
        "publication_date": "publication_date:desc",
        "year": "publication_year:desc",
        "publication_year": "publication_year:desc",
        "oldest": "publication_date:asc",
    }
    if text in aliases:
        return aliases[text]
    if ":" in text:
        return text
    raise ValueError(
        "sort must be relevance, cited_by_count, most_cited, newest, oldest, "
        "publication_date, publication_year, or an OpenAlex sort expression"
    )


def _request_openalex_works(params: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
    try:
        response = requests.get(f"{OPENALEX_API}/works", params=params, timeout=timeout_seconds)
        response.raise_for_status()
        return response.json()
    except requests.Timeout as exc:
        raise RuntimeError(f"OpenAlex request timed out after {timeout_seconds:g}s") from exc
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        body = exc.response.text[:500] if exc.response is not None else ""
        raise RuntimeError(f"OpenAlex HTTP {status}: {body}") from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"OpenAlex request failed: {exc}") from exc
    except ValueError as exc:
        raise RuntimeError(f"OpenAlex returned non-JSON: {exc}") from exc


@mcp.tool(
    description=(
        "Search OpenAlex academic literature and return up to k works with "
        "title, authors, source, year/date, DOI, OpenAlex ID, citation count, "
        "open-access links, and abstract snippets. Use when the user asks "
        "about papers, citations, references, prior work, or literature. "
        "Optional filters accept OpenAlex filter syntax; API authentication "
        "uses OPENALEX_API_KEY or OPENALEX_API_KEY_FILE from the environment, "
        "falling back to the git-ignored config/openalex.local.json."
    )
)
async def search_literature(
    query: str,
    k: int = 5,
    filters: str | None = None,
    from_year: int | str | None = None,
    to_year: int | str | None = None,
    is_oa: bool | str | None = None,
    work_type: str | None = None,
    sort: str = "relevance",
    corpus: str | None = None,
    include_abstract: bool = True,
    abstract_chars: int = 1200,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """Search OpenAlex works and return normalized paper metadata."""
    query = query.strip()
    if not query:
        return _dumps({"ok": False, "error": "query must be non-empty"})

    try:
        limit = max(1, min(MAX_RESULTS, int(k)))
        timeout = max(1.0, float(timeout_seconds))
        filter_param = _filter_param(filters, from_year, to_year, is_oa, work_type)
        sort_param = _sort_param(sort)
    except (TypeError, ValueError) as exc:
        return _dumps({"ok": False, "error": str(exc)})

    api_key = _openalex_api_key()
    params: dict[str, Any] = {
        "search": query,
        "per_page": limit,
        "select": SELECT_FIELDS,
    }
    if filter_param:
        params["filter"] = filter_param
    if sort_param:
        params["sort"] = sort_param
    corpus_param = _clean_optional_text(corpus)
    if corpus_param:
        params["corpus"] = corpus_param
    if api_key:
        params["api_key"] = api_key

    try:
        data = _request_openalex_works(params, timeout)
    except RuntimeError as exc:
        return _dumps(
            {
                "ok": False,
                "source": "OpenAlex",
                "query": query,
                "error": str(exc),
                "authenticated": bool(api_key),
            }
        )

    works = data.get("results") or []
    normalized = [
        _normalize_work(work, include_abstract=include_abstract, abstract_chars=int(abstract_chars))
        for work in works
        if isinstance(work, dict)
    ]
    return _dumps(
        {
            "ok": True,
            "source": "OpenAlex",
            "query": query,
            "authenticated": bool(api_key),
            "meta": data.get("meta") or {},
            "filters": filter_param,
            "sort": sort_param or "relevance",
            "corpus": corpus_param or "core-default",
            "results": normalized,
        }
    )


if __name__ == "__main__":
    asyncio.run(mcp.run_stdio_async())
