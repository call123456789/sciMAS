---
name: biomni-literature-arxiv-pubmed
description: Use for arXiv preprint or PubMed bibliographic lookups when the existing OpenAlex ``search_literature`` is insufficient (e.g. user wants a specific preprint or MeSH-constrained PubMed query).
x-scimas-role: literature-searcher
x-scimas-server: biomni-literature
x-scimas-tools:
  - query_arxiv
  - query_pubmed
---

# arXiv & PubMed lookups

## When to use
- OpenAlex lacks coverage of a recent arXiv preprint or a PubMed-only record.
- User wants raw title / author / abstract / DOI strings in arXiv or PubMed format rather than OpenAlex normalized records.

## Limitations
- Uses BiOMNI's HTTP wrappers; rate-limit and availability reflect the BiOMNI service, not sciMAS.
- The arXiv / PubMed result is a research-log string, not a structured JSON list of papers — parse carefully before comparing to OpenAlex output.
