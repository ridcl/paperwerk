"""Template generation: turn cataloged real documents into PII-free HTML templates.

Consumes `catalog.json` from `datagen.catalog`, groups documents by
(doc_type, layout_variant), picks one representative source per group, and
asks Claude to produce an HTML template that mirrors the layout with
`<span data-field="X">{{X}}</span>` placeholders.

Templates are written to `<out_dir>/<doc_type>/<layout_variant>.html` and
become the durable, reviewed inputs for downstream rendering (step 5 of
DESIGN.md). Each template is run through `_force_placeholder_content`
before being written, which guarantees that every data-field element
contains only its placeholder. Any leaks (the LLM ignoring the privacy
rule) are reported in the run output for manual inspection.

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    python -m datagen.templates catalog.json -o src/datagen/templates
    python -m datagen.templates catalog.json -o templates --types cv passport
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import sys
from io import BytesIO
from pathlib import Path

from pdf2image import convert_from_path
from PIL import Image

from pile.async_utils import gather_limited
from pile.llm import LLM


_ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1/"
_DEFAULT_MODEL = "claude-sonnet-4-6"
_MAX_INPUT_PAGES = 6
_INPUT_DPI = 150
_DEFAULT_CONCURRENCY = 4


_TEMPLATE_SYSTEM_PROMPT = """You are converting a real document into an HTML/CSS template for synthetic data generation. The source is a layout reference ONLY. You must not reproduce any specific personal data from it.

Output a single, self-contained HTML document inside one ```html``` code fence and nothing else. The HTML must satisfy ALL of the following:

1. Mirror the visual layout of the input as closely as you can: headings, columns, dividers, font weights, alignment, relative spacing, section ordering. Use only system fonts: "DejaVu Serif", "Liberation Serif", "DejaVu Sans", "Liberation Sans", Georgia, Helvetica, Arial. DO NOT @import Google Fonts or load any external resource (no remote images, no CDN CSS).

2. Use exactly this page-wrapper structure — one <div class="page">…</div> per page in the source document — with these exact .page rules in <style>:
     .page {
       width: 794px; height: 1123px;
       padding: 50px 60px; box-sizing: border-box;
       overflow: hidden; page-break-after: always;
       position: relative;
     }
     .page:last-child { page-break-after: auto; }
   And include at the top of <style>:
     @page { size: A4; margin: 0; }

3. Wrap every variable text field in:
     <span data-field="FIELD_NAME">{{FIELD_NAME}}</span>
   Rules for FIELD_NAME:
   - snake_case, ASCII letters/digits/underscore, unique within the document.
   - Static labels (e.g. "Email:", "Experience", "Education") stay as plain text — only mark the values as fields.
   - For repeated sections (jobs, education entries, projects), use indexed names: experience_1_company, experience_1_title, experience_1_dates, experience_1_summary, etc.
   - Free-text descriptions can be a single field; multi-line content is fine inside the span.
   - The placeholder uses DOUBLE curly braces: {{FIELD_NAME}}. CSS keeps single braces.

4. PRIVACY — NON-NEGOTIABLE: the text inside every <span data-field="X"> element must be EXACTLY the literal string "{{X}}" — nothing else. Do NOT copy any specific names, dates, phone numbers, email addresses, postal addresses, employer names, school names, project titles, country/city names, or any other concrete text from the source into the template. If you see "John Smith, Senior Engineer at Acme, jan 2020 – present", you emit `<span data-field="full_name">{{full_name}}</span>, <span data-field="role">{{role}}</span> at <span data-field="employer">{{employer}}</span>, <span data-field="dates">{{dates}}</span>` — never the original strings. Static UI/section labels ("Experience", "Skills", "Phone:") are not personal data and may be copied verbatim.

5. Output only the fenced HTML block. No prose before or after."""


_TEMPLATE_USER_TEXT_BASE = (
    "Convert the attached document image(s) into the HTML template described "
    "in the system prompt. Match layout faithfully and mark every variable "
    "field with a data-field span."
)


def _document_to_data_urls(path: Path, max_pages: int = _MAX_INPUT_PAGES) -> list[str]:
    """Convert a PDF (one image per page) or image file into base64 PNG data URLs."""
    if path.suffix.lower() == ".pdf":
        images = convert_from_path(str(path), dpi=_INPUT_DPI)[:max_pages]
    else:
        images = [Image.open(path).convert("RGB")]
    urls: list[str] = []
    for img in images:
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        urls.append(f"data:image/png;base64,{b64}")
    return urls


def extract_block(text: str, lang: str) -> str:
    """Pull content out of the first ```<lang> … ``` fence, falling back to whole text."""
    m = re.search(rf"```{lang}\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"```\w*\s*(.*?)\s*```", text, re.DOTALL)
    return m.group(1) if m else text.strip()


_DATA_FIELD_ELEMENT_RE = re.compile(
    r'(<(\w+)\b[^>]*\sdata-field=["\'](\w+)["\'][^>]*>)(.*?)(</\2>)',
    re.DOTALL | re.IGNORECASE,
)


def force_placeholder_content(html: str) -> tuple[str, list[tuple[str, str]]]:
    """Force every `data-field` element's inner content to be exactly `{{field}}`.

    Returns (fixed_html, leaks) where each leak is (field_name, original_snippet).
    Static labels outside `data-field` markers are NOT touched — they may still
    leak source text, but those are typically section headings ("Experience",
    "Skills") rather than personal data.
    """
    leaks: list[tuple[str, str]] = []

    def _replace(m: re.Match[str]) -> str:
        open_tag, _elem, name, inner, close_tag = m.groups()
        expected = f"{{{{{name}}}}}"
        if inner.strip() != expected:
            snippet = re.sub(r"\s+", " ", inner.strip())[:80]
            if snippet:
                leaks.append((name, snippet))
        return f"{open_tag}{expected}{close_tag}"

    fixed = _DATA_FIELD_ELEMENT_RE.sub(_replace, html)
    return fixed, leaks


def discover_fields(html: str) -> list[str]:
    """Return all `data-field=\"...\"` names in document order, deduped."""
    seen: set[str] = set()
    names: list[str] = []
    for m in re.finditer(r'data-field=["\'](\w+)["\']', html):
        n = m.group(1)
        if n not in seen:
            seen.add(n)
            names.append(n)
    return names


def generate_template(
    source: Path,
    llm: LLM,
    *,
    variant: str | None = None,
    notes: str | None = None,
) -> tuple[str, list[tuple[str, str]]]:
    """Ask the LLM to produce a PII-free HTML template that mirrors `source`.

    Optional `variant` and `notes` describe the target layout variant; they
    are appended to the user message as a hint so different variants don't
    collapse into near-duplicate output. Returns (html, leaks); leaks lists
    any data-field elements whose content was rewritten by the scrubber.
    """
    urls = _document_to_data_urls(source)
    if not urls:
        raise RuntimeError(f"no pages extracted from {source}")

    user_text = _TEMPLATE_USER_TEXT_BASE
    if variant:
        user_text += (
            f"\n\nThis template represents the `{variant}` variant of its "
            "document type. Make sure the produced layout reflects that "
            "variant's column structure, section order, and overall feel."
        )
    if notes:
        user_text += f"\n\nCatalog notes about this layout: {notes}"

    user_content: list[dict] = [{"type": "text", "text": user_text}]
    for url in urls:
        user_content.append({"type": "image_url", "image_url": {"url": url}})

    resp = llm.invoke(
        messages=[
            {"role": "system", "content": _TEMPLATE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_tokens=16000,
        temperature=0.2,
    )
    raw = extract_block(resp.choices[0].message.content, "html")
    return force_placeholder_content(raw)


# CLI ----------------------------------------------------------------------


def _select_groups(
    catalog: dict,
    types_filter: set[str] | None,
    variants_per_type: int | None,
) -> list[tuple[str, str, dict]]:
    """Group catalog entries by (doc_type, variant); pick one representative each.

    `variants_per_type` caps how many variants are emitted per type (most
    common first, ties broken alphabetically). None means no cap.
    """
    docs = catalog.get("documents", [])
    by_key: dict[tuple[str, str], list[dict]] = {}
    for d in docs:
        if "doc_type" not in d or "layout_variant" not in d:
            continue
        if types_filter and d["doc_type"] not in types_filter:
            continue
        by_key.setdefault((d["doc_type"], d["layout_variant"]), []).append(d)

    by_type: dict[str, list[tuple[str, list[dict]]]] = {}
    for (t, v), entries in by_key.items():
        by_type.setdefault(t, []).append((v, entries))

    selected: list[tuple[str, str, dict]] = []
    for t in sorted(by_type):
        vs = by_type[t]
        vs.sort(key=lambda kv: (-len(kv[1]), kv[0]))
        if variants_per_type is not None:
            vs = vs[:variants_per_type]
        for v, entries in vs:
            entries.sort(key=lambda d: d["source_path"])
            selected.append((t, v, entries[0]))
    return selected


async def _build_one(
    doc_type: str,
    variant: str,
    entry: dict,
    llm: LLM,
    out_dir: Path,
) -> dict:
    source = Path(entry["source_path"])
    notes = entry.get("notes", "")
    try:
        html, leaks = await asyncio.to_thread(
            generate_template, source, llm, variant=variant, notes=notes
        )
        target = out_dir / doc_type / f"{variant}.html"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(html)
        n_fields = len(discover_fields(html))
        rel = target.relative_to(out_dir.parent) if out_dir.parent in target.parents else target
        print(
            f"  {doc_type}/{variant}: wrote {rel} "
            f"({n_fields} field(s), {len(leaks)} leak(s) scrubbed)"
        )
        return {
            "doc_type": doc_type,
            "variant": variant,
            "source_path": entry["source_path"],
            "template_path": str(target),
            "n_fields": n_fields,
            "leaks": leaks,
        }
    except Exception as e:
        print(f"  {doc_type}/{variant}: ERROR {e!r}", file=sys.stderr)
        return {
            "doc_type": doc_type,
            "variant": variant,
            "source_path": entry["source_path"],
            "error": repr(e),
        }


async def _run(
    catalog_path: str,
    out_dir: str,
    model: str,
    types_filter: set[str] | None,
    variants_per_type: int | None,
    concurrency: int,
) -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("error: ANTHROPIC_API_KEY environment variable is not set")

    catalog = json.loads(Path(catalog_path).read_text())
    selected = _select_groups(catalog, types_filter, variants_per_type)
    if not selected:
        sys.exit(
            "error: no (doc_type, layout_variant) groups selected — "
            "check --types filter and the catalog content"
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"generating {len(selected)} template(s) into {out} via {model} "
        f"(concurrency={concurrency})"
    )

    llm = LLM(
        base_url=_ANTHROPIC_BASE_URL,
        api_key=api_key,
        model=model,
        max_concurrency=concurrency,
    )
    coros = [_build_one(t, v, entry, llm, out) for t, v, entry in selected]
    results = await gather_limited(coros, concurrency)

    summary = {
        "model": model,
        "catalog": catalog_path,
        "out_dir": str(out),
        "templates": results,
    }
    index_path = out / "templates_index.json"
    index_path.write_text(json.dumps(summary, indent=2))

    n_ok = sum(1 for r in results if "error" not in r)
    n_err = len(results) - n_ok
    print()
    print(
        f"wrote {n_ok} template(s); {n_err} error(s); "
        f"index -> {index_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("catalog", help="Path to catalog.json from `datagen.catalog`")
    parser.add_argument(
        "-o", "--out-dir", default="src/datagen/templates",
        help="Where to write generated templates (default: src/datagen/templates)",
    )
    parser.add_argument("--model", default=_DEFAULT_MODEL)
    parser.add_argument(
        "--types", nargs="+", default=None,
        help="Only generate templates for these doc_types",
    )
    parser.add_argument(
        "--variants-per-type", type=int, default=None,
        help="Cap variants per type (most common first). Default: all",
    )
    parser.add_argument(
        "--concurrency", type=int, default=_DEFAULT_CONCURRENCY,
        help=f"Max concurrent LLM calls (default: {_DEFAULT_CONCURRENCY})",
    )
    args = parser.parse_args()
    types_filter = set(args.types) if args.types else None
    asyncio.run(
        _run(
            catalog_path=args.catalog,
            out_dir=args.out_dir,
            model=args.model,
            types_filter=types_filter,
            variants_per_type=args.variants_per_type,
            concurrency=args.concurrency,
        )
    )


if __name__ == "__main__":
    main()
