"""Catalog a corpus of real reference documents for the synthetic data pipeline.

For each document under a directory tree, ask Claude (vision) to identify:
- the document type (snake_case, drawn from a starter vocabulary or proposed
  fresh by the model),
- a coarse layout variant (e.g. single_column_chronological, table_heavy),
- every variable text region that would change between two real instances of
  the same document type, named in snake_case with a coarse kind tag.

Output is a single `catalog.json` keyed by source path. It feeds template
generation (step 2) and per-type axes-config authoring (step 3) of the
DESIGN.md plan.

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    python -m datagen.catalog /data/Documents -o catalog.json
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import sys
from collections import Counter
from io import BytesIO
from pathlib import Path

from pdf2image import convert_from_path
from PIL import Image

from pile.async_utils import gather_limited
from pile.llm import LLM


_ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1/"
_DEFAULT_MODEL = "claude-sonnet-4-6"
_MAX_PAGES_PER_DOC = 3
_INPUT_DPI = 150
_DEFAULT_CONCURRENCY = 8
_SUPPORTED_EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}


# Starter vocabulary; the model may propose new types and they are reported
# back to the user. Keep this list short and broad — it is a hint, not a
# constraint.
_INITIAL_DOC_TYPES = [
    "cv",
    "employment_contract",
    "marriage_certificate",
    "birth_certificate",
    "death_certificate",
    "passport",
    "national_id",
    "drivers_license",
    "bank_statement",
    "invoice",
    "receipt",
    "purchase_order",
    "tax_return",
    "lease_agreement",
    "utility_bill",
    "medical_record",
    "academic_transcript",
    "diploma",
    "shipping_label",
    "letter",
]


_SYSTEM_PROMPT = """You catalog real documents for a synthetic data generation pipeline.

For each document image set you receive, produce a structured analysis as a single JSON object inside one ```json``` code fence and nothing else.

Output schema:
{{
  "doc_type": "<snake_case identifier of the document type>",
  "layout_variant": "<short snake_case identifier of the visual layout>",
  "n_pages": <integer page count, matching the number of attached images>,
  "fields": [
    {{"name": "<snake_case>", "kind": "<one of: text|name|date|amount|address|phone|email|id_number|paragraph|other>"}}
  ],
  "notes": "<1-2 sentence summary of distinguishing features>"
}}

Rules:
- doc_type: prefer one of the known types if applicable: {known_types}. If none fits, invent a new snake_case name (e.g. utility_bill, lease_agreement, w2_form). ASCII letters, digits, underscore only.
- layout_variant: short snake_case slug. Examples: single_column_chronological, two_column_sidebar, table_heavy, letter_with_signature, multi_page_form, two_page_certificate, mrz_photo_id. Be concise.
- fields: include EVERY variable text region — values that would change between two real documents of this type (names, dates, amounts, addresses, IDs, descriptions, line items). DO NOT include static labels (e.g. "Email:", "Date of Birth", section headings). Use snake_case names that describe the field semantically. For repeated rows (line items, transactions, jobs, education entries), include indexed names with as many rows as appear: line_item_1_description, line_item_1_amount, line_item_2_description, …
- kind classifies the value type loosely. Use "paragraph" for multi-sentence free text, "name" for personal/company names, "amount" for currency or numeric values, "id_number" for account numbers / IDs / serials, "address" for postal addresses, "text" if nothing else fits, "other" only if truly ambiguous.
- notes: highlight anything noteworthy — multiple parties, signatures, watermarks, photographs, machine-readable zones, tables, official seals.

Output ONLY the JSON code fence. No prose before or after.

PRIVACY: do not transcribe specific personal values from the source into your output. Field names are general (e.g. "full_name"), not the actual person's name."""


_USER_TEXT = (
    "Catalog the attached document by producing the JSON described in the system prompt."
)


def _document_to_data_urls(path: Path, max_pages: int) -> list[str]:
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


def _extract_block(text: str, lang: str) -> str:
    m = re.search(rf"```{lang}\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"```\w*\s*(.*?)\s*```", text, re.DOTALL)
    return m.group(1) if m else text.strip()


async def _catalog_one(
    path: Path, llm: LLM, known_types: list[str], max_pages: int
) -> dict:
    """Catalog a single document. Image conversion runs in a worker thread
    so it doesn't block the event loop while other docs make LLM calls."""
    urls = await asyncio.to_thread(_document_to_data_urls, path, max_pages)
    if not urls:
        raise RuntimeError("no pages extracted")

    user_content: list[dict] = [{"type": "text", "text": _USER_TEXT}]
    for url in urls:
        user_content.append({"type": "image_url", "image_url": {"url": url}})

    system = _SYSTEM_PROMPT.format(known_types=", ".join(known_types))
    resp = await llm.ainvoke(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        max_tokens=2000,
        temperature=0.2,
    )
    blob = _extract_block(resp.choices[0].message.content, "json")
    parsed = json.loads(blob)
    parsed["source_path"] = str(path)
    parsed.setdefault("n_pages", len(urls))
    return parsed


def _walk(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return [
        p for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTS
    ]


async def _run(
    root: str,
    out_path: str,
    model: str,
    max_pages: int,
    concurrency: int,
) -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("error: ANTHROPIC_API_KEY environment variable is not set")

    root_path = Path(root).expanduser()
    if not root_path.exists():
        sys.exit(f"error: {root_path} does not exist")

    docs = _walk(root_path)
    if not docs:
        sys.exit(f"error: no supported documents found under {root_path}")
    print(f"cataloging {len(docs)} document(s) under {root_path} via {model}")

    llm = LLM(
        base_url=_ANTHROPIC_BASE_URL,
        api_key=api_key,
        model=model,
        max_concurrency=concurrency,
    )

    async def _wrapped(p: Path) -> tuple[str, dict]:
        try:
            entry = await _catalog_one(p, llm, _INITIAL_DOC_TYPES, max_pages)
            print(
                f"  {p.name}: {entry.get('doc_type')!r} / "
                f"{entry.get('layout_variant')!r} "
                f"({len(entry.get('fields', []))} fields)"
            )
            return ("ok", entry)
        except Exception as e:
            print(f"  {p.name}: ERROR {e!r}", file=sys.stderr)
            return ("err", {"source_path": str(p), "error": repr(e)})

    results = await gather_limited([_wrapped(p) for p in docs], concurrency)
    documents = [r[1] for r in results if r[0] == "ok"]
    errors = [r[1] for r in results if r[0] == "err"]

    type_counts: Counter[str] = Counter(
        d["doc_type"] for d in documents if "doc_type" in d
    )
    novel = sorted(t for t in type_counts if t not in _INITIAL_DOC_TYPES)

    catalog = {
        "model": model,
        "root": str(root_path),
        "doc_type_counts": dict(sorted(type_counts.items())),
        "novel_types": novel,
        "documents": documents,
        "errors": errors,
    }

    out = Path(out_path)
    out.write_text(json.dumps(catalog, indent=2))
    print()
    print(
        f"wrote {out}: {len(documents)} document(s) cataloged, "
        f"{len(errors)} error(s)"
    )
    if type_counts:
        print(f"types: {dict(type_counts)}")
    if novel:
        print(f"novel types (not in starter vocab): {novel}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "root", help="Directory (walked recursively) or a single file to catalog"
    )
    parser.add_argument("-o", "--output", default="catalog.json")
    parser.add_argument("--model", default=_DEFAULT_MODEL)
    parser.add_argument(
        "--max-pages",
        type=int,
        default=_MAX_PAGES_PER_DOC,
        help="Max pages per document sent to the LLM (default: 3)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=_DEFAULT_CONCURRENCY,
        help="Max concurrent LLM calls (default: 8)",
    )
    args = parser.parse_args()
    asyncio.run(
        _run(
            root=args.root,
            out_path=args.output,
            model=args.model,
            max_pages=args.max_pages,
            concurrency=args.concurrency,
        )
    )


if __name__ == "__main__":
    main()
