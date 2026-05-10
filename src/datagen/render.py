"""Shared Playwright rendering for synthetic document templates.

All `datagen` templates honor the same contract:
- Each page is a `<div class="page">` block sized exactly 794 × 1123 CSS px
  (A4 at 96 DPI), with `page-break-after: always` and `overflow: hidden`.
- `@page { size: A4; margin: 0; }` so Chromium's print pipeline emits one
  PDF page per div and the full-page screenshot tiles cleanly into
  N × 1123 px slices.
- Variable fields are wrapped in `<span data-field="NAME">{{NAME}}</span>`.
  Placeholders use double curly braces — CSS `{...}` rules are left alone.

`render_template` fills the placeholders, drives Playwright, and returns
both the PDF (for downstream consumers) and the full-page screenshot
together with one `Field` per `data-field` occurrence (page-indexed,
within-page CSS-px bbox). `annotate` overlays the bboxes onto per-page
slices of that screenshot for visual inspection.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from io import BytesIO

from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright


A4_WIDTH_PX = 794
A4_HEIGHT_PX = 1123

_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")


@dataclass
class Field:
    name: str
    value: str
    page: int
    bbox: tuple[int, int, int, int]  # (x0, y0, x1, y1) in within-page CSS px


def fill_template(template: str, data: dict[str, str]) -> str:
    """Substitute `{{field}}` placeholders with values; CSS `{...}` is untouched."""
    return _PLACEHOLDER_RE.sub(lambda m: data.get(m.group(1), ""), template)


def render_template(
    template: str, data: dict[str, str]
) -> tuple[bytes, bytes, list[Field]]:
    """Render a `{{field}}` template with `data`; return (pdf, full_page_png, fields).

    The PNG is a single full-page screenshot covering all pages stacked
    vertically. Pass it (with `fields`) to `annotate` to get per-page PNGs.
    """
    html = fill_template(template, data)
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

            names = page.evaluate(
                "() => Array.from(new Set("
                "Array.from(document.querySelectorAll('[data-field]'))"
                ".map(e => e.getAttribute('data-field'))"
                "))"
            )
            for name in names:
                for loc in page.locator(f'[data-field="{name}"]').all():
                    box = loc.bounding_box()
                    if box is None:
                        continue
                    x = round(box["x"])
                    y = round(box["y"])
                    w = round(box["width"])
                    h = round(box["height"])
                    page_idx = y // A4_HEIGHT_PX
                    local_y = y - page_idx * A4_HEIGHT_PX
                    fields.append(
                        Field(
                            name=name,
                            value=data.get(name, ""),
                            page=page_idx,
                            bbox=(x, local_y, x + w, local_y + h),
                        )
                    )

            png_bytes = page.screenshot(type="png", full_page=True)
            pdf_bytes = page.pdf(
                width=f"{A4_WIDTH_PX}px",
                height=f"{A4_HEIGHT_PX}px",
                margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
                print_background=True,
            )
        finally:
            browser.close()
    return pdf_bytes, png_bytes, fields


def annotate(png_bytes: bytes, fields: list[Field]) -> list[bytes]:
    """Slice the full-page screenshot into per-page PNGs with bboxes overlaid."""
    img = Image.open(BytesIO(png_bytes)).convert("RGB")
    n_pages = max(1, math.ceil(img.height / A4_HEIGHT_PX))
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
    for p in range(n_pages):
        top = p * A4_HEIGHT_PX
        bottom = min(img.height, top + A4_HEIGHT_PX)
        slice_img = img.crop((0, top, img.width, bottom)).copy()
        draw = ImageDraw.Draw(slice_img)
        for f in by_page.get(p, []):
            x0, y0, x1, y1 = f.bbox
            draw.rectangle((x0, y0, x1, y1), outline=(220, 30, 30), width=2)
            draw.text(
                (x0 + 2, max(0, y0 - 13)), f.name, fill=(220, 30, 30), font=font
            )
        buf = BytesIO()
        slice_img.save(buf, format="PNG")
        out.append(buf.getvalue())
    return out
