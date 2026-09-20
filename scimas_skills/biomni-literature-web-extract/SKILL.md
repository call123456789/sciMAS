---
name: biomni-literature-web-extract
description: Use to fetch and extract plain text from a URL or PDF (e.g. extract figure caption text from an open-access PDF).
x-scimas-role: literature-searcher
x-scimas-server: biomni-literature
x-scimas-tools:
  - extract_url_content
  - extract_pdf_content
---

# URL & PDF content extraction

## When to use
- User has a specific URL / PDF and wants the raw text content.

## Limitations
- Heavy lifting by BiOMNI helpers; needs ``requests`` and a PDF extractor (``pymupdf`` / ``PyPDF2``). Returns JSON error if missing.
- No citation parsing — combine with the OpenAlex or arXiv skill when the URL is a known paper.
