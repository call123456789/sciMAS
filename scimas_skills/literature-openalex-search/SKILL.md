---
name: literature-openalex-search
description: Use for searching scholarly literature, papers, citations, prior work, reviews, and recent evidence through OpenAlex.
x-scimas-role: literature-searcher
x-scimas-server: litsearch
x-scimas-tools:
  - search_literature
---
# Literature OpenAlex Search

Use this skill when the current step needs real paper discovery or citation
metadata. The available tool queries OpenAlex works and returns normalized
records with title, authors, publication date, source, DOI, OpenAlex ID,
citation count, open-access status, links, and optional abstract snippets.

Search guidance:
- Start with a concise technical query that contains the core concept and
  domain terms.
- Use `from_year` / `to_year` for recency-constrained searches.
- Use `is_oa=true` only when the task specifically needs accessible full text.
- Use `sort="cited_by_count"` or `sort="most_cited"` for foundational work,
  and `sort="newest"` for current literature.
- Use `work_type="article"` or raw OpenAlex `filters` when the task needs a
  specific publication type or fielded restriction.

Report:
- Include DOI or OpenAlex URL for traceability.
- Highlight why each selected work is relevant to the user's scientific
  question.
- Do not treat abstracts as verified full-paper evidence; mark conclusions
  based only on metadata/abstract search results as preliminary.
