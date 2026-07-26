"""Zero-shot document classification via a vision-capable LLM.

A single high-level entry point — `classify` — converts the pages of a
PDF or image file into base64 PNG inputs and asks the LLM to label the
document with a snake_case identifier, either drawn from a caller-supplied
class list or invented freely when none is given.

Designed as a building block for the data-generation pipeline (DESIGN.md
step 1 — cataloging the real corpus). Higher-level orchestration (walking
directories, aggregating into `catalog.json`) lives elsewhere.
"""

from __future__ import annotations

import asyncio
import re
from typing import Optional

from paperwerk.llm import LLM

from datagen.utils import document_to_data_urls

_MAX_PAGES = 3


_IDENT_RE = re.compile(r"[a-z][a-z0-9_]*")


def _normalize(raw: str) -> str:
    """Extract a snake_case identifier from the model's free-form reply."""
    cleaned = raw.strip().lower().replace("-", "_").replace(" ", "_")
    m = _IDENT_RE.search(cleaned)
    if not m:
        raise RuntimeError(f"classifier returned no recognizable identifier: {raw!r}")
    return m.group(0)


async def classify(
    llm: LLM,
    path: str,
    classes: Optional[list[str]] = None,
) -> str:
    """Zero-shot classify the document at `path` into a snake_case label.

    If `classes` is provided, the model is constrained to pick one of those
    labels. If `classes` is None, the model is asked to invent a reasonable
    snake_case identifier (e.g. `cv`, `passport`, `bank_statement`).
    """
    urls = await asyncio.to_thread(document_to_data_urls, path, end_page=_MAX_PAGES)
    if not urls:
        raise RuntimeError(f"no pages extracted from {path}")

    if classes:
        joined = "\n".join(f"  - {c}" for c in classes)
        instruction = (
            "Classify the attached document into exactly one of the "
            f"following snake_case classes:\n{joined}\n\n"
            "Output ONLY the chosen class name, on a single line, with no "
            "punctuation or explanation."
        )
    else:
        instruction = (
            "What type of document is the attached file? Output exactly one "
            "snake_case identifier (e.g. cv, passport, bank_statement, "
            "marriage_certificate, invoice). Output ONLY the identifier, on "
            "a single line, with no punctuation or explanation."
        )

    user_content: list[dict] = [{"type": "text", "text": instruction}]
    for url in urls:
        user_content.append({"type": "image_url", "image_url": {"url": url}})

    resp = await llm.ainvoke(
        messages=[{"role": "user", "content": user_content}],
        max_tokens=50,
        temperature=0.0,
    )
    return _normalize(resp.choices[0].message.content)
