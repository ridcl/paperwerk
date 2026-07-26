"""Jinja2 template generation from a real document image or PDF.

`make_template` is the single entry point: given a path to a real document,
it asks an LLM for a Jinja2 HTML template that mirrors the source
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
from pathlib import Path

import imagehash
from pdf2image import pdfinfo_from_path
from PIL import Image

from paperwerk.llm import LLM

from paperwerk.datagen.utils import document_to_data_urls

# Output-token budget for a single LLM call. Auto-chunking (below) keeps each
# chunk small enough that this ceiling is never the bottleneck.
_DEFAULT_MAX_OUTPUT_TOKENS = 32000

# Pages per chunk when `make_template` is called without an explicit end_page.
# 8 is empirically safe: even dense forms (~16 fields/page) fit comfortably
# under 32k output tokens per chunk after Jinja+HTML expansion.
_DEFAULT_CHUNK_PAGES = 8

_BODY_RE = re.compile(r"<body\b[^>]*>(.*?)</body>", re.DOTALL | re.IGNORECASE)
_STYLE_RE = re.compile(r"<style\b[^>]*>.*?</style>", re.DOTALL | re.IGNORECASE)
_CLOSE_HEAD_RE = re.compile(r"</head>", re.IGNORECASE)


_HTML_FENCE_RE = re.compile(r"```html\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_ANY_FENCE_RE = re.compile(r"```\w*\s*(.*?)\s*```", re.DOTALL)


_HTML_TAG_RE = re.compile(r"<html\b", re.IGNORECASE)
_BODY_TAG_RE = re.compile(r"<body\b", re.IGNORECASE)


def _ensure_skeleton(html: str) -> str:
    """Wrap a bare fragment in a complete `<html><head><body>` document.

    The model is asked for a full document, but (especially smaller models)
    it often returns just a `<style>` block plus `.page` divs. Normalising
    here guarantees every template has the same structure, which keeps
    `_merge_templates`, `apply_handwriting`, and any DOM-shaped consumer on
    their well-defined code path instead of the fragment fallbacks.
    """
    if _HTML_TAG_RE.search(html):
        return html  # already a full document
    if _BODY_TAG_RE.search(html):
        # Has <head>/<body> but no <html> wrapper — just wrap it.
        return f"<!DOCTYPE html>\n<html>\n{html.strip()}\n</html>"
    # Bare fragment: hoist <style> blocks into <head>, the rest into <body>.
    styles = "\n".join(_STYLE_RE.findall(html))
    body = _STYLE_RE.sub("", html).strip()
    return (
        "<!DOCTYPE html>\n<html>\n<head>\n"
        f"{styles}\n</head>\n<body>\n{body}\n</body>\n</html>"
    )


def _extract_and_normalize_html(text: str) -> str:
    """Pull the HTML out of a fenced block and normalize it to a full document.

    Extraction falls back to any fenced block, then to the whole reply; the
    result is then passed through `_ensure_skeleton` so a bare
    `<style>`+`.page` fragment becomes a complete `<html><head><body>` document.
    """
    m = _HTML_FENCE_RE.search(text)
    if m:
        html = m.group(1)
    else:
        m = _ANY_FENCE_RE.search(text)
        html = m.group(1) if m else text.strip()
    return _ensure_skeleton(html)


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


_SYSTEM_PROMPT = """You are converting a real document into a Jinja2 HTML template for synthetic data generation. The source is a layout reference ONLY. You must not reproduce any specific sensitive data from it into the template.

Output a single, self-contained Jinja2 HTML document inside one ```html``` code fence and nothing else.

It MUST be a complete HTML document, not a fragment: wrap everything in `<!DOCTYPE html><html><head>...</head><body>...</body></html>`, with the `<style>` block inside `<head>` and every `<div class="page">` inside `<body>`.

LAYOUT
1. Mirror the visual layout of the input: headings, columns, dividers, font weights, alignment, relative spacing, section ordering. Use only system fonts: "DejaVu Serif", "Liberation Serif", "DejaVu Sans", "Liberation Sans", Georgia, Helvetica, Arial. DO NOT @import Google Fonts or load any external resource. Where the source shows a company logo or brand mark, do NOT reproduce it (no real wordmark, brand icon, or copied image/SVG); replace it with a simple neutral placeholder built from a small inline SVG — e.g. a plain rectangle or circle with a generic monogram or shape — of roughly the same size and position as the original.

2. Use exactly this page-wrapper structure — one <div class="page">...</div> per page in the source — with these exact .page rules in <style>:
     .page {
       width: 794px; height: 1123px;
       padding: 50px 60px; box-sizing: border-box;
       overflow: hidden; page-break-after: always;
       position: relative;
     }
     .page:last-child { page-break-after: auto; }
   And include at the top of <style>:
     @page { size: A4; margin: 0; }

   ORIENTATION: the above is A4 PORTRAIT. If the source page is LANDSCAPE
   (clearly wider than it is tall), swap the dimensions and page size instead:
   set `.page { width: 1123px; height: 794px; ... }` (keep the other .page
   rules identical) and `@page { size: A4 landscape; margin: 0; }`. Match the
   source's orientation. All pages within one document use the same orientation.

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

6. Generic printed labels ("Email:", "Description", "Subtotal", "Date") stay as plain text. Only the variable values are marked as fields — and a concrete name, company, place, or number is a value, never a label (see rule 7).

PRIVACY — NON-NEGOTIABLE
7. Do NOT copy ANY specific value from the source into the template — not inside a field, not in a heading, not in a header/letterhead, not in a footer, not anywhere. This includes names, dates, phone numbers, email addresses, postal addresses, employer/company/organization/bank names, logo or letterhead text, account/routing/reference numbers, city or country names, and every other concrete value. Every such value must instead be a <span data-field="X"> element whose inner content is a Jinja expression of the form {{ ... }} — never the source text.

   The document's OWN issuer — the company, organization, or bank whose name, logo, or address appears in the header/letterhead or the payment/footer block — is VARIABLE, not fixed branding: wrap it in data-field spans (e.g. issuer_name, issuer_address, bank_name), never copy it verbatim (e.g. do NOT leave "Microsoft" or "Bank of America" in the template).

   The ONLY text you may copy verbatim is generic printed labels that carry no concrete value — e.g. "Invoice", "Sold To", "Date:", "Subtotal", "FEIN:". If in doubt, make it a field.

8. Output only the fenced HTML block. No prose before or after."""


_USER_TEXT = (
    "Convert the attached document image(s) into the Jinja2 HTML template "
    "described in the system prompt. Match layout faithfully, use loops "
    "wherever a section's row count varies in real-world instances of this "
    "document type, and mark every variable value with a data-field span."
)


_DEFAULT_PHASH_SIZE = 8  # 8 -> 64-bit hash
_DEFAULT_HAMMING_THRESHOLD = 8  # bits of tolerance for "near-duplicate"


class TemplateDeduper:
    """Near-duplicate detector for template *source* images via perceptual hashing.

    kvp10k (and any other scanned-forms corpus) contains many rows drawn from
    the same underlying form family; running `make_template` on all of them
    burns LLM calls and produces layout-redundant templates. This class
    perceptual-hashes each candidate's page image and rejects candidates
    whose phash lands within `threshold` Hamming bits of any already-kept
    hash.

    We hash the *source* image, not a rendered version of the generated
    template - two visually similar sources will almost always yield
    layout-similar templates, and hashing the source avoids the render/LLM
    round-trip needed to hash the output.

    Small O(N) linear scan per query; adequate for corpora of a few thousand
    templates. Not thread-safe.
    """

    def __init__(
        self,
        hash_size: int = _DEFAULT_PHASH_SIZE,
        threshold: int = _DEFAULT_HAMMING_THRESHOLD,
    ) -> None:
        self._hash_size = hash_size
        self._threshold = threshold
        self._hashes: list[imagehash.ImageHash] = []

    def __len__(self) -> int:
        return len(self._hashes)

    def _phash(self, image: Image.Image) -> imagehash.ImageHash:
        return imagehash.phash(image, hash_size=self._hash_size)

    def is_duplicate(self, image: Image.Image) -> bool:
        """Return True if `image`'s phash is within threshold of any kept hash."""
        h = self._phash(image)
        return any((h - kept) <= self._threshold for kept in self._hashes)

    def add(self, image: Image.Image) -> imagehash.ImageHash:
        """Store `image`'s phash and return it. Does not check for duplicates."""
        h = self._phash(image)
        self._hashes.append(h)
        return h

    def check_and_add(self, image: Image.Image) -> bool:
        """One-shot: True if `image` is a near-duplicate; otherwise record it and return False."""
        h = self._phash(image)
        for kept in self._hashes:
            if (h - kept) <= self._threshold:
                return True
        self._hashes.append(h)
        return False


def _total_pdf_pages(path: str) -> int:
    """Total page count for a PDF path; 1 for a standalone image file."""
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        return pdfinfo_from_path(str(p))["Pages"]
    return 1


def _merge_templates(
    chunks: list[tuple[str, list[str]]],
) -> tuple[str, list[str]]:
    """Stitch chunked-template outputs into one valid HTML document.

    - Uses the first chunk's ``<html>/<head>/<body>`` wrapper as the frame.
    - Appends ``<style>`` blocks from every subsequent chunk inside the
      first chunk's ``<head>`` (before ``</head>``) so class definitions
      unique to later chunks survive.
    - Concatenates ``<body>`` inner content across chunks in order.
    - Merges field lists preserving first-appearance order.
    """
    if len(chunks) == 1:
        return chunks[0]

    first_html, _ = chunks[0]
    m = _BODY_RE.search(first_html)
    if not m:
        # Fallback: raw concatenation. Not strictly valid HTML but preserves
        # the emitted content so downstream tooling can still extract fields.
        merged_html = "\n\n".join(html for html, _ in chunks)
    else:
        body_inner_start, body_inner_end = m.span(1)
        head_and_body_open = first_html[:body_inner_start]
        body_close_and_after = first_html[body_inner_end:]

        extra_styles: list[str] = []
        for html, _ in chunks[1:]:
            extra_styles.extend(_STYLE_RE.findall(html))
        if extra_styles:
            insertion = "\n" + "\n".join(extra_styles) + "\n</head>"
            head_and_body_open = _CLOSE_HEAD_RE.sub(
                insertion, head_and_body_open, count=1
            )

        body_parts: list[str] = [m.group(1)]
        for html, _ in chunks[1:]:
            m2 = _BODY_RE.search(html)
            body_parts.append(m2.group(1) if m2 else html)
        merged_html = head_and_body_open + "\n".join(body_parts) + body_close_and_after

    seen: set[str] = set()
    merged_fields: list[str] = []
    for _, fields in chunks:
        for f in fields:
            if f not in seen:
                seen.add(f)
                merged_fields.append(f)
    return merged_html, merged_fields


async def make_template(
    llm: LLM,
    path: str,
    *,
    start_page: int = 0,
    end_page: int | None = None,
) -> tuple[str, list[str]]:
    """Generate a Jinja2 HTML template that mirrors the document at `path`.

    When called without an explicit `end_page`, `make_template` auto-chunks:
    it reads the source's total page count, splits into consecutive
    ``_DEFAULT_CHUNK_PAGES``-page windows, calls itself recursively on
    each window (each recursive call receives an explicit ``end_page`` and
    hits the single-LLM-call base case below), and stitches the returned
    templates into one document via `_merge_templates`. Chunking keeps
    each LLM output well under `_DEFAULT_MAX_OUTPUT_TOKENS`, so long
    documents no longer get truncated mid-emission.

    Pass an explicit `end_page` (0-indexed, half-open — Python slice
    semantics) to disable auto-chunking and force a single LLM call over
    the exact range. Useful when you're already driving the chunk
    boundary yourself or when you only want a small slice.

    Returns (template, fields):
    - `template`: a Jinja2 HTML string with `<span data-field="...">{{ ... }}</span>`
      markers and (where appropriate) `{% for … %}` loops for variable-length
      sections.
    - `fields`: flat snake_case names in document order. Array fields use the
      `[]` convention, e.g. `line_items[].description`, `skills[]`.
    """
    if end_page is None:
        total = await asyncio.to_thread(_total_pdf_pages, path)
        remaining = max(0, total - start_page)
        if remaining > _DEFAULT_CHUNK_PAGES:
            chunks = await asyncio.gather(
                *(
                    make_template(
                        llm,
                        path,
                        start_page=s,
                        end_page=min(s + _DEFAULT_CHUNK_PAGES, total),
                    )
                    for s in range(start_page, total, _DEFAULT_CHUNK_PAGES)
                )
            )
            return _merge_templates(list(chunks))
        # Small enough for one call — fall through with an explicit range so
        # we take the base-case path below.
        end_page = total

    urls = await asyncio.to_thread(
        document_to_data_urls,
        path,
        start_page=start_page,
        end_page=end_page,
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
        max_tokens=_DEFAULT_MAX_OUTPUT_TOKENS,
        temperature=0.2,
    )
    template = _extract_and_normalize_html(resp.choices[0].message.content)
    fields = discover_fields(template)
    return template, fields
