"""Shared low-level utilities for the datagen pipeline."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

from pdf2image import convert_from_path
from PIL import Image


_DEFAULT_DPI = 150
# Anthropic's recommended max long edge for vision inputs; staying at or under
# this keeps PNGs comfortably below the 5 MB per-image API limit and matches
# what Anthropic resizes to server-side anyway.
_DEFAULT_MAX_SIDE = 1568
# Hard ceiling on the encoded image bytes; Anthropic rejects > 5 MB.
_MAX_PNG_BYTES = 5 * 1024 * 1024
# Lower bound when adaptively shrinking — below this, layout becomes unreadable.
_MIN_SIDE = 512


def _downscale(img: Image.Image, max_side: int) -> Image.Image:
    """Downscale `img` so its longest side is at most `max_side` pixels."""
    w, h = img.size
    longest = max(w, h)
    if longest <= max_side:
        return img
    scale = max_side / longest
    return img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)


def _encode_png_under_limit(
    img: Image.Image, max_side: int, max_bytes: int
) -> bytes:
    """Encode `img` as PNG, shrinking until the encoded bytes fit `max_bytes`.

    Starts at `max_side` and halves the longest side on each retry until the
    encoded size fits, or `_MIN_SIDE` is reached.
    """
    side = max_side
    while True:
        candidate = _downscale(img, side)
        buf = BytesIO()
        candidate.save(buf, format="PNG", optimize=True)
        data = buf.getvalue()
        if len(data) <= max_bytes or side <= _MIN_SIDE:
            return data
        side = max(_MIN_SIDE, side // 2)


def document_to_data_urls(
    path: str | Path,
    *,
    start_page: int = 0,
    end_page: int | None = None,
    dpi: int = _DEFAULT_DPI,
    max_side: int | None = _DEFAULT_MAX_SIDE,
) -> list[str]:
    """Convert a PDF (one image per page) or single image file into base64 PNG data URLs.

    Page selection uses Python-slice semantics: `start_page` is 0-indexed
    inclusive, `end_page` is 0-indexed exclusive (None means through the
    last page). Default is "all pages". For PDFs the range is pushed into
    pdf2image (via its 1-indexed inclusive first_page/last_page args) so we
    don't rasterize pages we're about to discard - important when the
    caller only wants a small window of a long document.

    Each page is downscaled so its longest side is at most `max_side`
    pixels (pass `None` to disable resizing), then adaptively shrunk
    further if the encoded PNG would exceed the 5 MB vision-API limit.
    """
    p = Path(path)
    if end_page is not None and start_page >= end_page:
        return []
    if p.suffix.lower() == ".pdf":
        # pdf2image: first_page/last_page are 1-indexed inclusive; None is
        # unbounded on that side. 0-indexed-exclusive `end_page` and
        # 1-indexed-inclusive last_page happen to be numerically equal.
        images = convert_from_path(
            str(p),
            dpi=dpi,
            first_page=start_page + 1,
            last_page=end_page,
        )
    else:
        if start_page > 0:
            return []
        images = [Image.open(p).convert("RGB")]
    urls: list[str] = []
    for img in images:
        if max_side is None:
            buf = BytesIO()
            img.save(buf, format="PNG")
            data = buf.getvalue()
        else:
            data = _encode_png_under_limit(img, max_side, _MAX_PNG_BYTES)
        b64 = base64.b64encode(data).decode("ascii")
        urls.append(f"data:image/png;base64,{b64}")
    return urls
