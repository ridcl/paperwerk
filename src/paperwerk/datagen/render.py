"""Render Jinja2 templates to PDF and capture per-field bounding boxes.

`render` (async) and `render_sync` (blocking) are the entry points: they
expand a Jinja2 HTML template with a data dict, drive Chromium's print
pipeline, and return the PDF bytes alongside a `Field` per `[data-field]`
element. Each `.page` div becomes one PDF sheet at that div's own size
(A4 portrait or landscape); bboxes are stored as per-page normalized
`(x0, y0, x1, y1)` in `[0, 1]` with top-left origin, so they're decoupled
from page size and from any raster DPI a downstream consumer might choose.

`annotate` is a separate helper that rasterizes the PDF and overlays the
bboxes on each page — useful for visual sanity-checking, not part of the
hot path.

Template contract:
- Each page is a `<div class="page">` block — A4 portrait (794 × 1123 CSS
  px) or landscape (1123 × 794) — with `page-break-after: always` and
  `overflow: hidden`.
- `@page { size: A4[ landscape]; margin: 0; }`; the sheet is sized to the
  `.page` div so Chromium emits exactly one PDF page per div.
- Variable fields are wrapped in `<span data-field="NAME">{{ ... }}</span>`.
  Names may include Jinja loop expansions (e.g. `items[0].description`);
  the renderer treats each post-expansion occurrence as a distinct field.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from jinja2 import ChainableUndefined, Environment
from pdf2image import convert_from_bytes
from PIL import ImageDraw, ImageFont
from playwright.async_api import async_playwright
from playwright.sync_api import sync_playwright

from paperwerk.datagen.handwriting import apply_handwriting

A4_WIDTH_PX = 794
A4_HEIGHT_PX = 1123

# Measurement viewport — wide/tall enough to contain a landscape A4 page
# (1123px). Field bboxes are normalized relative to each `.page` element's own
# measured box, so the viewport size does not affect the results.
_VIEWPORT_WIDTH = 1123
_VIEWPORT_HEIGHT = 1123


@dataclass
class Field:
    name: str
    value: str
    page: int
    bbox: tuple[
        float, float, float, float
    ]  # (x0, y0, x1, y1) in [0, 1], top-left origin


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


_FONTS_DIR = Path(__file__).parent / "assets" / "fonts"


def _annotation_font(size: int):
    """Load a bundled font for annotation labels.

    Uses the fonts shipped under ``assets/fonts`` rather than assuming any
    particular system font is installed; falls back to PIL's built-in default
    if none load.
    """
    for ttf in sorted(_FONTS_DIR.glob("*.ttf")):
        try:
            return ImageFont.truetype(str(ttf), size)
        except OSError:
            continue
    return ImageFont.load_default()


def _finalize(value):
    """Stringify containers rendered directly as `{{ x }}`.

    Value synthesis (especially smaller local models) occasionally returns a
    dict or list for a field the template outputs directly, which would
    otherwise render as a raw Python repr (``{'street': '123 Main St', ...}``).
    Flatten those to a readable comma-joined string. Nested access
    (``{{ x.y }}``) and normal scalars pass through untouched, since finalize
    only sees the final expression value.
    """
    if isinstance(value, dict):
        return ", ".join(
            str(_finalize(v)) for v in value.values() if v not in (None, "")
        )
    if isinstance(value, (list, tuple)):
        return ", ".join(str(_finalize(v)) for v in value if v not in (None, ""))
    return value


def _field_value(loc) -> str:
    """Read a field element's rendered value.

    Form controls (input/textarea/select) carry their value in the `value`
    property, not as inner text (inner_text() returns "" for them). For
    non-HTML nodes (e.g. a `data-field` on an SVG element) `inner_text()`
    raises "Node is not an HTMLElement", so fall back to `text_content()`.
    """
    tag = loc.evaluate("el => el.tagName.toLowerCase()")
    if tag in ("input", "textarea", "select"):
        return loc.input_value()
    try:
        return loc.inner_text()
    except Exception:
        return loc.text_content() or ""


def _fields_in_page(page_el, page_idx: int) -> list[Field]:
    """Normalize every `[data-field]` inside one `.page` to that page's box.

    Coordinates are taken *relative to the page element's own bounding box*, so
    the result is independent of the page's absolute position, its size, and
    the orientation (portrait or landscape) — the `.page` need not be A4.
    """
    pbox = page_el.bounding_box()
    if pbox is None or pbox["width"] <= 0 or pbox["height"] <= 0:
        return []
    px, py, pw, ph = pbox["x"], pbox["y"], pbox["width"], pbox["height"]
    out: list[Field] = []
    for loc in page_el.locator("[data-field]").all():
        box = loc.bounding_box()
        if box is None:
            continue
        out.append(
            Field(
                name=loc.get_attribute("data-field") or "",
                value=_field_value(loc),
                page=page_idx,
                bbox=(
                    _clamp01((box["x"] - px) / pw),
                    _clamp01((box["y"] - py) / ph),
                    _clamp01((box["x"] + box["width"] - px) / pw),
                    _clamp01((box["y"] + box["height"] - py) / ph),
                ),
            )
        )
    return out


def render_sync(
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
    html = (
        Environment(autoescape=True, finalize=_finalize, undefined=ChainableUndefined)
        .from_string(template_str)
        .render(**data)
    )
    if handwritten_fields:
        html = apply_handwriting(html, handwritten_fields, rng)

    fields: list[Field] = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page(
                viewport={"width": _VIEWPORT_WIDTH, "height": _VIEWPORT_HEIGHT},
                device_scale_factor=1,
            )
            page.emulate_media(media="print")
            page.set_content(html, wait_until="networkidle")
            # Neutralize the UA default body margin: an 8px margin can push a
            # full-height page (esp. landscape) onto a blank second sheet. Field
            # bboxes are page-relative, so this does not move them.
            page.add_style_tag(content="html, body { margin: 0; padding: 0; }")

            # Each `.page` div is one PDF page. Measure fields relative to their
            # own page box so bboxes are correct for any page size/orientation.
            page_els = page.locator(".page").all()
            first_box = None
            for page_idx, page_el in enumerate(page_els):
                box = page_el.bounding_box()
                if page_idx == 0:
                    first_box = box
                fields.extend(_fields_in_page(page_el, page_idx))

            # Size the PDF sheet to the actual `.page` box (portrait or
            # landscape) so each div maps to exactly one sheet. Matching the
            # div's own pixel size avoids the sub-pixel overflow blank pages you
            # get from `prefer_css_page_size` (whose mm-derived A4 size differs
            # slightly from the div's px size). All pages share one orientation.
            pw = round(first_box["width"]) if first_box else A4_WIDTH_PX
            ph = round(first_box["height"]) if first_box else A4_HEIGHT_PX
            pdf_bytes = page.pdf(
                width=f"{pw}px",
                height=f"{ph}px",
                margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
                print_background=True,
            )
        finally:
            browser.close()
    return pdf_bytes, fields


async def _field_value_async(loc) -> str:
    """Async counterpart of `_field_value`."""
    tag = await loc.evaluate("el => el.tagName.toLowerCase()")
    if tag in ("input", "textarea", "select"):
        return await loc.input_value()
    try:
        return await loc.inner_text()
    except Exception:
        return (await loc.text_content()) or ""


async def _fields_in_page_async(
    page_el, page_idx: int, pbox: dict | None
) -> list[Field]:
    """Async counterpart of `_fields_in_page` (page box passed in, not remeasured)."""
    if pbox is None or pbox["width"] <= 0 or pbox["height"] <= 0:
        return []
    px, py, pw, ph = pbox["x"], pbox["y"], pbox["width"], pbox["height"]
    out: list[Field] = []
    for loc in await page_el.locator("[data-field]").all():
        box = await loc.bounding_box()
        if box is None:
            continue
        name = await loc.get_attribute("data-field")
        out.append(
            Field(
                name=name or "",
                value=await _field_value_async(loc),
                page=page_idx,
                bbox=(
                    _clamp01((box["x"] - px) / pw),
                    _clamp01((box["y"] - py) / ph),
                    _clamp01((box["x"] + box["width"] - px) / pw),
                    _clamp01((box["y"] + box["height"] - py) / ph),
                ),
            )
        )
    return out


async def render(
    template_str: str,
    data: dict,
    *,
    handwritten_fields: list[str] | tuple[str, ...] = (),
    seed: int | None = None,
) -> tuple[bytes, list[Field]]:
    """Async version of `render_sync` — identical output via Playwright's async API.

    Preferred for async callers (e.g. dataset builders): `await render(...)`
    directly instead of pushing `render_sync` onto a thread. See `render_sync`
    for the full contract (handwriting, per-page bboxes, portrait/landscape).
    """
    rng = random.Random(seed)
    html = (
        Environment(autoescape=True, finalize=_finalize, undefined=ChainableUndefined)
        .from_string(template_str)
        .render(**data)
    )
    if handwritten_fields:
        html = apply_handwriting(html, handwritten_fields, rng)

    fields: list[Field] = []
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        try:
            page = await browser.new_page(
                viewport={"width": _VIEWPORT_WIDTH, "height": _VIEWPORT_HEIGHT},
                device_scale_factor=1,
            )
            await page.emulate_media(media="print")
            await page.set_content(html, wait_until="networkidle")
            # Neutralize the UA default body margin (see render_sync).
            await page.add_style_tag(content="html, body { margin: 0; padding: 0; }")

            page_els = await page.locator(".page").all()
            first_box = None
            for page_idx, page_el in enumerate(page_els):
                pbox = await page_el.bounding_box()
                if page_idx == 0:
                    first_box = pbox
                fields.extend(await _fields_in_page_async(page_el, page_idx, pbox))

            pw = round(first_box["width"]) if first_box else A4_WIDTH_PX
            ph = round(first_box["height"]) if first_box else A4_HEIGHT_PX
            pdf_bytes = await page.pdf(
                width=f"{pw}px",
                height=f"{ph}px",
                margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
                print_background=True,
            )
        finally:
            await browser.close()
    return pdf_bytes, fields


def annotate(
    pdf_bytes: bytes,
    fields: list[Field],
    *,
    dpi: int = 150,
) -> bytes:
    """Rasterize the PDF at `dpi`, overlay each field's bbox + name, and repack.

    Takes PDF bytes and returns PDF bytes (an image-only PDF, one annotated
    page per input page), so input and output types match. For visual
    inspection only; downstream consumers that need raster pixels at a specific
    DPI should rasterize the PDF themselves and scale bboxes by the actual
    image dimensions.
    """
    images = convert_from_bytes(pdf_bytes, dpi=dpi)
    by_page: dict[int, list[Field]] = {}
    for f in fields:
        by_page.setdefault(f.page, []).append(f)

    font = _annotation_font(11)

    annotated: list = []
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
            draw.text((x0 + 2, max(0, y0 - 13)), f.name, fill=(220, 30, 30), font=font)
        annotated.append(canvas)

    if not annotated:
        return pdf_bytes
    buf = BytesIO()
    annotated[0].save(
        buf,
        format="PDF",
        save_all=True,
        append_images=annotated[1:],
        resolution=dpi,
    )
    return buf.getvalue()
