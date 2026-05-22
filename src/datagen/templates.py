"""Jinja2 template generation from a real document image or PDF.

`make_template` is the single entry point: given a path to a real document,
it asks Claude (vision) for a Jinja2 HTML template that mirrors the source
layout with `<span data-field="...">{{ ... }}</span>` markers, and returns
the template together with a flat schema of field names.

Conventions for the field list:
- Scalar:           "full_name"
- Nested object:    "employer.name"
- Array of dicts:   "line_items[].description"   (note the empty [])
- Array of strings: "skills[]"

The `[]` marker tells downstream data synthesis that this field expands to
a JSON array. Inside the template, loop iterations carry concrete indices:
`<span data-field="line_items[{{ loop.index0 }}].description">…`, so after
rendering each occurrence has a unique data-field name and bbox capture
treats each row as its own field instance.
"""

from __future__ import annotations

import asyncio
import re

from pile.llm import LLM

from datagen.utils import document_to_data_urls

_MAX_INPUT_PAGES = 6


_HTML_FENCE_RE = re.compile(r"```html\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_ANY_FENCE_RE = re.compile(r"```\w*\s*(.*?)\s*```", re.DOTALL)


def _extract_html(text: str) -> str:
    """Pull the HTML out of a fenced block, falling back to the whole reply."""
    m = _HTML_FENCE_RE.search(text)
    if m:
        return m.group(1)
    m = _ANY_FENCE_RE.search(text)
    return m.group(1) if m else text.strip()


_DATA_FIELD_ATTR_RE = re.compile(r"""data-field=(["'])(.*?)\1""", re.DOTALL)
_LOOP_INDEX_RE = re.compile(r"\[\s*\{\{[^}]*\}\}\s*\]")


def _normalize_field_name(name: str) -> str:
    """Collapse `[{{ loop.index0 }}]` inside a data-field name to `[]`."""
    return _LOOP_INDEX_RE.sub("[]", name)


def discover_fields(template: str) -> list[str]:
    """Return all `data-field` names in document order, deduped, with array
    notation normalized.

    Concrete loop indices (`[{{ loop.index0 }}]`) collapse to `[]` so the
    returned list represents the schema, not the post-render enumeration.
    """
    seen: set[str] = set()
    out: list[str] = []
    for m in _DATA_FIELD_ATTR_RE.finditer(template):
        norm = _normalize_field_name(m.group(2))
        if norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


_SYSTEM_PROMPT = """You are converting a real document into a Jinja2 HTML template for synthetic data generation. The source is a layout reference ONLY. You must not reproduce any specific personal data from it into the template.

Output a single, self-contained Jinja2 HTML document inside one ```html``` code fence and nothing else.

LAYOUT
1. Mirror the visual layout of the input: headings, columns, dividers, font weights, alignment, relative spacing, section ordering. Use only system fonts: "DejaVu Serif", "Liberation Serif", "DejaVu Sans", "Liberation Sans", Georgia, Helvetica, Arial. DO NOT @import Google Fonts or load any external resource.

2. Use exactly this page-wrapper structure — one <div class="page">…</div> per page in the source — with these exact .page rules in <style>:
     .page {
       width: 794px; height: 1123px;
       padding: 50px 60px; box-sizing: border-box;
       overflow: hidden; page-break-after: always;
       position: relative;
     }
     .page:last-child { page-break-after: auto; }
   And include at the top of <style>:
     @page { size: A4; margin: 0; }

FIELDS
3. Wrap every variable text in a <span data-field="..."> element. The span's inner content MUST be a Jinja2 expression of the form {{ ... }} — never source text from the document.

4. Naming conventions (use these exactly):
   - Scalar:            <span data-field="full_name">{{ full_name }}</span>
   - Nested object:     <span data-field="employer.name">{{ employer.name }}</span>
   - Array of dicts:
       {% for item in line_items %}
         <tr>
           <td><span data-field="line_items[{{ loop.index0 }}].description">{{ item.description }}</span></td>
           <td><span data-field="line_items[{{ loop.index0 }}].amount">{{ item.amount }}</span></td>
         </tr>
       {% endfor %}
   - Array of strings:
       {% for s in skills %}
         <span data-field="skills[{{ loop.index0 }}]">{{ s }}</span>
       {% endfor %}

5. Use Jinja loops ONLY for sections whose count truly varies in real-world instances (invoice line items, bank-statement transactions, CV work-experience or education entries). For singular/header/footer/summary content, use plain scalar fields.

6. Static labels ("Email:", "Description", "Subtotal", "Date") stay as plain text. Only the variable values are marked as fields.

PRIVACY — NON-NEGOTIABLE
7. Every <span data-field="X"> element's inner content must be a Jinja expression of the form {{ ... }}. Do NOT copy any specific names, dates, phone numbers, email addresses, postal addresses, employer names, school names, city or country names, or any other concrete value from the source. Static section/UI labels are not personal data and may be copied verbatim.

8. Output only the fenced HTML block. No prose before or after."""


_USER_TEXT = (
    "Convert the attached document image(s) into the Jinja2 HTML template "
    "described in the system prompt. Match layout faithfully, use loops "
    "wherever a section's row count varies in real-world instances of this "
    "document type, and mark every variable value with a data-field span."
)


async def make_template(llm: LLM, path: str) -> tuple[str, list[str]]:
    """Generate a Jinja2 HTML template that mirrors the document at `path`.

    Returns (template, fields):
    - `template`: a Jinja2 HTML string with `<span data-field="...">{{ ... }}</span>`
      markers and (where appropriate) `{% for … %}` loops for variable-length
      sections.
    - `fields`: flat snake_case names in document order. Array fields use the
      `[]` convention, e.g. `line_items[].description`, `skills[]`.
    """
    urls = await asyncio.to_thread(
        document_to_data_urls, path, max_pages=_MAX_INPUT_PAGES
    )
    if not urls:
        raise RuntimeError(f"no pages extracted from {path}")

    user_content: list[dict] = [{"type": "text", "text": _USER_TEXT}]
    for url in urls:
        user_content.append({"type": "image_url", "image_url": {"url": url}})

    resp = await llm.ainvoke(
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_tokens=16000,
        temperature=0.2,
    )
    template = _extract_html(resp.choices[0].message.content)
    fields = discover_fields(template)
    return template, fields
