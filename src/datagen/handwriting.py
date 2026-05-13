"""Tier-1 handwriting emulation: style picked `data-field` elements with a
hand-writing font.

`apply_handwriting` takes a fully-rendered HTML string, a list of schema
field names (which may include the `foo[]` / `foo[].bar` array notation
used by `templates.py`), and a `random.Random` for picking a font + tilt
angle. It injects a single `<style>` block into the document's `<head>`
that:

- Embeds one randomly-chosen bundled font via an `@font-face` data URL.
- Uses CSS attribute selectors to target every matching `data-field`
  element, including the post-Jinja-expansion concrete names produced
  by loop iterations (e.g. `line_items[3].description`).
- Applies a tilt, larger size, and dark-blue "ink" color so the rendered
  fields look distinct from the printed labels around them.

Bundled fonts live in `src/datagen/assets/fonts/` and are SIL Open Font
License (see `OFL.txt` next to them). The HTML manipulation is
regex-free: we just append a style block, so existing classes and
inline styles on the spans are preserved.
"""

from __future__ import annotations

import base64
import random
from pathlib import Path


_FONTS_DIR = Path(__file__).parent / "assets" / "fonts"
_FONT_PATHS = sorted(_FONTS_DIR.glob("*.ttf"))


def _field_to_selector(name: str) -> str:
    """Convert a schema field name (possibly with `[]`) to a CSS attribute selector.

    - `"full_name"`               -> `[data-field="full_name"]`
    - `"employer.name"`           -> `[data-field="employer.name"]`
    - `"skills[]"`                -> `[data-field^="skills["][data-field$="]"]`
    - `"line_items[].description"`-> `[data-field^="line_items["][data-field$="].description"]`
    """
    if "[]" not in name:
        return f'[data-field="{name}"]'
    prefix, _, suffix = name.partition("[]")
    if suffix:
        return f'[data-field^="{prefix}["][data-field$="{suffix}"]'
    return f'[data-field^="{prefix}["][data-field$="]"]'


def _font_data_url(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:font/ttf;base64,{b64}"


def apply_handwriting(
    html: str,
    handwritten_fields: list[str] | tuple[str, ...],
    rng: random.Random,
) -> str:
    """Inject a `<style>` block that renders listed fields in a handwriting font.

    Returns the HTML unchanged if no fields are listed or no fonts are bundled.
    """
    if not handwritten_fields or not _FONT_PATHS:
        return html

    font_path = rng.choice(_FONT_PATHS)
    font_url = _font_data_url(font_path)
    tilt = rng.uniform(-1.2, 1.2)
    selectors = ",\n".join(_field_to_selector(n) for n in handwritten_fields)

    style = f"""
<style>
@font-face {{
  font-family: 'HandwrittenSynth';
  src: url({font_url}) format('truetype');
}}
{selectors} {{
  font-family: 'HandwrittenSynth', cursive !important;
  font-size: 1.5em !important;
  color: #1a3a8a !important;
  font-weight: 400 !important;
  letter-spacing: 0.5px !important;
  display: inline-block;
  transform: rotate({tilt:.2f}deg);
}}
</style>"""

    if "</head>" in html:
        return html.replace("</head>", style + "</head>", 1)
    if "<body" in html:
        return html.replace("<body", style + "<body", 1)
    return style + html
