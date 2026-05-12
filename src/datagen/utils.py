"""Shared low-level utilities for the datagen pipeline."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

from pdf2image import convert_from_path
from PIL import Image


_DEFAULT_DPI = 150


def document_to_data_urls(
    path: str | Path,
    *,
    max_pages: int | None = None,
    dpi: int = _DEFAULT_DPI,
) -> list[str]:
    """Convert a PDF (one image per page) or single image file into base64 PNG data URLs.

    Used by the classifier and template builder to feed real documents to a
    vision-capable LLM. `max_pages` caps the number of PDF pages returned
    (None means all pages).
    """
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        images = convert_from_path(str(p), dpi=dpi)
        if max_pages is not None:
            images = images[:max_pages]
    else:
        images = [Image.open(p).convert("RGB")]
    urls: list[str] = []
    for img in images:
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        urls.append(f"data:image/png;base64,{b64}")
    return urls
