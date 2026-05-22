"""Render Jinja2 templates to PDF and capture per-field bounding boxes.

`render` is the single entry point: it expands a Jinja2 HTML template with
a data dict, drives Chromium's print pipeline at A4 (96 DPI, 794×1123 CSS
px), and returns the PDF bytes alongside a `Field` per `[data-field]`
element. Bboxes are stored as per-page normalized `(x0, y0, x1, y1)` in
`[0, 1]` with top-left origin, so they're decoupled from any raster DPI a
downstream consumer might choose.

`annotate` is a separate helper that rasterizes the PDF and overlays the
bboxes on each page — useful for visual sanity-checking, not part of the
hot path.

Template contract:
- Each page is a `<div class="page">` block sized 794 × 1123 CSS px with
  `page-break-after: always` and `overflow: hidden`.
- `@page { size: A4; margin: 0; }` so Chromium emits one PDF page per div.
- Variable fields are wrapped in `<span data-field="NAME">{{ ... }}</span>`.
  Names may include Jinja loop expansions (e.g. `items[0].description`);
  the renderer treats each post-expansion occurrence as a distinct field.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from io import BytesIO

from jinja2 import Environment
from pdf2image import convert_from_bytes
from PIL import ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

from datagen.handwriting import apply_handwriting


A4_WIDTH_PX = 794
A4_HEIGHT_PX = 1123


@dataclass
class Field:
    name: str
    value: str
    page: int
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1) in [0, 1], top-left origin


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def render(
    template_str: str,
    data: dict,
    *,
    handwritten_fields: list[str] | tuple[str, ...] = (),
    seed: int | None = None,
) -> tuple[bytes, list[Field]]:
    """Expand a Jinja2 template with `data` and render it to PDF.

    `handwritten_fields` names (schema-style — including `foo[]` /
    `foo[].bar`) get styled with a randomly-chosen bundled handwriting
    font, larger size, blue "ink" color, and slight tilt. The choice is
    deterministic given `seed`.

    Returns (pdf_bytes, fields). Each `Field` carries the post-expansion
    `data-field` name (e.g. `line_items[0].description`), its rendered text,
    a 0-indexed `page`, and a `bbox` normalized to per-page `[0, 1]` with
    top-left origin.
    """
    rng = random.Random(seed)
    html = Environment(autoescape=True).from_string(template_str).render(**data)
    if handwritten_fields:
        html = apply_handwriting(html, handwritten_fields, rng)

    fields: list[Field] = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page(
                viewport={"width": A4_WIDTH_PX, "height": A4_HEIGHT_PX},
                device_scale_factor=1,
            )
            page.emulate_media(media="print")
            page.set_content(html, wait_until="networkidle")

            for loc in page.locator("[data-field]").all():
                box = loc.bounding_box()
                if box is None:
                    continue
                name = loc.get_attribute("data-field") or ""
                value = loc.inner_text()
                x = box["x"]
                y = box["y"]
                w = box["width"]
                h = box["height"]
                page_idx = int(y // A4_HEIGHT_PX)
                local_y = y - page_idx * A4_HEIGHT_PX
                fields.append(
                    Field(
                        name=name,
                        value=value,
                        page=page_idx,
                        bbox=(
                            _clamp01(x / A4_WIDTH_PX),
                            _clamp01(local_y / A4_HEIGHT_PX),
                            _clamp01((x + w) / A4_WIDTH_PX),
                            _clamp01((local_y + h) / A4_HEIGHT_PX),
                        ),
                    )
                )

            pdf_bytes = page.pdf(
                width=f"{A4_WIDTH_PX}px",
                height=f"{A4_HEIGHT_PX}px",
                margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
                print_background=True,
            )
        finally:
            browser.close()
    return pdf_bytes, fields


def annotate(
    pdf_bytes: bytes,
    fields: list[Field],
    *,
    dpi: int = 150,
) -> list[bytes]:
    """Rasterize the PDF at `dpi` and overlay each field's bbox on its page.

    Returns one PNG per PDF page in document order. Used for visual
    inspection; downstream consumers that need raster pixels at a specific
    DPI should rasterize the PDF themselves and scale bboxes by the actual
    image dimensions.
    """
    images = convert_from_bytes(pdf_bytes, dpi=dpi)
    by_page: dict[int, list[Field]] = {}
    for f in fields:
        by_page.setdefault(f.page, []).append(f)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11
        )
    except OSError:
        font = ImageFont.load_default()

    out: list[bytes] = []
    for page_idx, img in enumerate(images):
        canvas = img.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        w, h = canvas.size
        for f in by_page.get(page_idx, []):
            x0 = int(f.bbox[0] * w)
            y0 = int(f.bbox[1] * h)
            x1 = int(f.bbox[2] * w)
            y1 = int(f.bbox[3] * h)
            draw.rectangle((x0, y0, x1, y1), outline=(220, 30, 30), width=2)
            draw.text(
                (x0 + 2, max(0, y0 - 13)), f.name, fill=(220, 30, 30), font=font
            )
        buf = BytesIO()
        canvas.save(buf, format="PNG")
        out.append(buf.getvalue())
    return out
