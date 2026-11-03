"""pdf → page-keyed raw text. M1.7, and nothing more.

Parsing only. No chunking, no embeddings, no table extraction, no retrieval — those are M3.2
onward and they extend this file rather than replacing it. Keeping the page number as the key
from the very first line is what makes citation possible later without a migration.
"""

import pdfplumber


def extract_pages(file, name: str | None = None) -> dict[int, str]:
    """{1: "text of page one", 2: ...} — 1-indexed, because that is how a citation reads."""
    with pdfplumber.open(file) as pdf:
        return {i: (page.extract_text() or "") for i, page in enumerate(pdf.pages, start=1)}


def preview(pages: dict[int, str], page: int = 1, chars: int = 1500) -> str:
    text = pages.get(page, "")
    return text[:chars] + ("…" if len(text) > chars else "")
