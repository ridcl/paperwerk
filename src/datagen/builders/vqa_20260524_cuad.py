"""Build a VQA dataset from CUAD-like synthetic documents.

Reads PDFs and per-field JSON sidecars from `/data/paperwerk/documents_20260517/`
(produced by `documents_20260517_cuad.py`). For each logical document, tiles
the page list with non-overlapping windows of `--images-per-datapoint` pages
and asks the LLM to convert the fields visible in each window into 5-10
natural-language queries plus 1-2 unanswerable queries. A single query may
produce multiple answers (e.g. "amount due" across every row of a payments
table) - in that case the query string appears once in `queries` and once per
matched field in `answers`.

The clear and phone_photo variants of the same logical document share an
identical field schema (augmentation is pixel-only), so the LLM is called
once per `(logical_doc, window)` and the result is reused across variants.

Each output row is a single VQA datapoint:

  - images:  list[binary]                 -- PNG bytes, one per page in window
  - queries: list[string]                 -- answerable + unanswerable, shuffled
  - answers: list[struct{                 -- one entry per answerable query
        query:        string,
        value:        string,             -- literal or derived (e.g. "2021"
                                          --   from "August 28, 2021")
        bounding_box: list[double, 4],    -- [x0,y0,x1,y1] on the source page,
                                          --   normalized; the field is the
                                          --   evidence for the answer
        index:        int32,              -- which `images` entry holds the
                                          --   evidence
    }]
  - source:     string                    -- "<class>_<variant>_<idx>"
  - variant:    string                    -- "clear" or "phone_photo"
  - page_start: int32
  - page_end:   int32                     -- inclusive

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    python -m datagen.builders.vqa_20260524_cuad
    python -m datagen.builders.vqa_20260524_cuad --limit 20 --images-per-datapoint 1
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import random
import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pdf2image import convert_from_bytes
from PIL import Image, ImageDraw, ImageFont

from paperwerk.llm import LLM

_DOCS_DIR = Path("/data/paperwerk/documents_20260517")
_OUT_PATH = Path("/data/paperwerk/vqa_20260524_cuad.parquet")
_ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1/"
_MODEL = "claude-sonnet-4-6"
_CONCURRENCY = 10
_RENDER_CONCURRENCY = 4
_DEFAULT_SEED = 0
_DEFAULT_IMAGES_PER_DATAPOINT = 1
_DEFAULT_DPI = 150
_DEFAULT_VARIANTS = ("clear", "phone_photo")
_MIN_FIELDS_PER_WINDOW = 3
_MAX_FIELD_VALUE_CHARS_IN_PROMPT = 500
_MAX_UNANSWERABLE = 2

_NAME_RE = re.compile(r"^(?P<cls>.+?)_(?P<variant>clear|phone_photo)_(?P<idx>\d{4})$")
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


_PROMPT_TEMPLATE = """\
You are building a Visual Question Answering training set from a contract \
document.

Document class: {cls}

User profile: individual, looking for infomation about personal or business matters \
including tax and payment information, employment or project conditions, names, \
dates, places, etc.

Below are the fields extracted from page(s) {pages_label} of this document. \
Each field has a name (a stable identifier from the template) and a value \
(the realized text on the page).

Fields:
{field_lines}

Task:
  1) Pick the 5-10 MOST IMPORTANT pieces of information - the ones a real \
user reading or searching this document would actually care about. Skip \
boilerplate (page numbers, running headers, repeated short tokens). When \
several fields carry the same information under different names, choose only \
one.

  2) For each chosen piece of information, write a query and one or more \
answers:
     - The query is what a real user would type to find this information. \
Phrase it as a search-style phrase. Do NOT quote the \
raw field name.
     - A query MAY map to multiple fields. If several fields in the input \
list are equally valid evidence for the same query (e.g. every row of a \
"amount due" column in a payments table, every bullet of a "restricted \
content" list, each party's address), emit ONE query with MULTIPLE entries \
in its `answers` array - one per matching field. Otherwise emit a single \
answer.
     - Each answer's `value` may be LITERAL or DERIVED:
         * literal - the same string as the field value (e.g. query \
"agreement date" -> "August 28, 2021");
         * derived - a short transformation that actually answers the query \
(e.g. query "what year was the document signed?" -> "2021" even though the \
field value is "August 28, 2021"; query "does this contract allow unlimited \
sick days?" -> "Yes" even though the field value is a full paragraph of \
policy text).
     - Keep values short and direct. Yes/no questions get "Yes" or "No". Do \
not restate the question in the value.
     - `source_field_name` on each answer must be the EXACT field name from \
the input list above; we use it to recover the bounding box. Every answer \
under one query must point to a DIFFERENT field.

  3) Also produce 1-{max_unanswerable} "unanswerable" queries - plausible \
questions a user might ask about a contract of this type that are NOT \
answered by the fields above. They should look natural, not adversarial. Do \
NOT invent answers for them.

Return one JSON object inside a single ```json``` fence and nothing else:

```json
{{
  "answerable": [
    {{
      "query": "...",
      "answers": [
        {{"value": "...", "source_field_name": "..."}}
      ]
    }}
  ],
  "unanswerable": ["..."]
}}
```
"""


def _parse_name(stem: str) -> tuple[str, str, int] | None:
    m = _NAME_RE.match(stem)
    if not m:
        return None
    return m.group("cls"), m.group("variant"), int(m.group("idx"))


def _windows(page_count: int, window_size: int) -> list[tuple[int, int]]:
    """Non-overlapping page windows over [0, page_count). Last may be shorter."""
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    return [
        (s, min(s + window_size - 1, page_count - 1))
        for s in range(0, page_count, window_size)
    ]


def _truncate(s: str, n: int) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "..."


def _png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


async def _ask_llm(
    llm: LLM,
    cls: str,
    page_start: int,
    page_end: int,
    fields: list[dict],
) -> dict:
    pages_label = (
        f"{page_start + 1}"
        if page_start == page_end
        else f"{page_start + 1}-{page_end + 1}"
    )
    field_lines = "\n".join(
        f"  - {f['name']}: "
        f"{json.dumps(_truncate(f['value'], _MAX_FIELD_VALUE_CHARS_IN_PROMPT), ensure_ascii=False)}"
        for f in fields
    )
    prompt = _PROMPT_TEMPLATE.format(
        cls=cls,
        pages_label=pages_label,
        field_lines=field_lines,
        max_unanswerable=_MAX_UNANSWERABLE,
    )
    resp = await llm.ainvoke(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4000,
        temperature=0.7,
    )
    text = resp.choices[0].message.content
    m = _JSON_FENCE_RE.search(text)
    blob = (m.group(1) if m else text).strip()
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"LLM did not return valid JSON ({e}); raw response:\n{text}"
        ) from e
    if not isinstance(data, dict):
        raise RuntimeError(f"LLM returned non-object: {type(data).__name__}")
    if not isinstance(data.get("answerable"), list) or not isinstance(
        data.get("unanswerable"), list
    ):
        raise RuntimeError("'answerable' and 'unanswerable' must be lists")
    return data


def _assemble_datapoint(
    llm_out: dict,
    fields_by_name: dict[str, dict],
    window_start: int,
    rng: random.Random,
) -> tuple[list[str], list[dict]] | None:
    """Filter LLM output against the field schema, shuffle, return (queries, answers).

    Returns None if no answerable items survive validation - we don't emit
    datapoints with no grounded evidence.
    """
    answers: list[dict] = []
    answerable_queries: list[str] = []
    for item in llm_out.get("answerable", []):
        if not isinstance(item, dict):
            continue
        query = item.get("query")
        raw_answers = item.get("answers")
        if not (isinstance(query, str) and isinstance(raw_answers, list)):
            continue
        per_query: list[dict] = []
        seen_sources: set[str] = set()
        for raw in raw_answers:
            if not isinstance(raw, dict):
                continue
            value = raw.get("value")
            source = raw.get("source_field_name")
            if not (isinstance(value, str) and isinstance(source, str)):
                continue
            if source in seen_sources:
                continue
            field = fields_by_name.get(source)
            if field is None:
                continue
            bbox = field.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            seen_sources.add(source)
            per_query.append(
                {
                    "query": query,
                    "value": value,
                    "bounding_box": [float(x) for x in bbox],
                    "index": int(field["page"]) - window_start,
                }
            )
        if not per_query:
            continue
        answers.extend(per_query)
        answerable_queries.append(query)

    if not answers:
        return None

    unanswerable = [q for q in llm_out.get("unanswerable", []) if isinstance(q, str)][
        :_MAX_UNANSWERABLE
    ]
    queries = answerable_queries + unanswerable
    rng.shuffle(queries)
    return queries, answers


_VIS_PALETTE = (
    "#e6194B",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#469990",
    "#9A6324",
    "#800000",
    "#808000",
    "#000075",
    "#a9a9a9",
)


def visualize_row(
    row,
    output_path: str | Path | None = None,
    font_size: int = 14,
) -> list[Image.Image]:
    """Render answer bounding boxes onto each page image of a VQA row.

    `row` is one record from the parquet (a dict or a pandas Series with
    `images`, `queries`, `answers` keys). All answers sharing the same
    query are drawn in the same color, so multi-answer queries (e.g.
    "amount due" across every row of a payments table) read as a group.
    Unanswerable queries don't appear visually - find them in `row["queries"]`
    that aren't present as `answer["query"]`.

    Returns one annotated `PIL.Image` per `images` entry, in order. If
    `output_path` is given, writes the annotated pages to disk: a single
    file when the row has one image, otherwise `<stem>_p{i}<ext>` per page.
    """
    annotated = [Image.open(io.BytesIO(b)).convert("RGB") for b in row["images"]]
    draws = [ImageDraw.Draw(im) for im in annotated]
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    query_order: list[str] = []
    for a in row["answers"]:
        q = a["query"]
        if q not in query_order:
            query_order.append(q)

    for a in row["answers"]:
        idx = int(a["index"])
        if not 0 <= idx < len(annotated):
            continue
        x0, y0, x1, y1 = (float(v) for v in a["bounding_box"])
        W, H = annotated[idx].size
        px0, py0, px1, py1 = x0 * W, y0 * H, x1 * W, y1 * H
        color = _VIS_PALETTE[query_order.index(a["query"]) % len(_VIS_PALETTE)]
        draws[idx].rectangle([px0, py0, px1, py1], outline=color, width=3)
        label = f"{a['query']} -> {a['value']}"
        if len(label) > 80:
            label = label[:77] + "..."
        draws[idx].text(
            (px0, max(0.0, py0 - font_size - 2)), label, fill=color, font=font
        )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if len(annotated) == 1:
            annotated[0].save(output_path)
        else:
            stem, suffix = output_path.stem, output_path.suffix
            for i, im in enumerate(annotated):
                im.save(output_path.with_name(f"{stem}_p{i}{suffix}"))

    return annotated


def _discover_docs(
    docs_dir: Path,
    variants: tuple[str, ...],
) -> dict[tuple[str, int], dict]:
    """Group `*.pdf` files by `(class, index)` -> {variants: {variant: path}, json: path}.

    Drops logical docs whose JSON sidecar is missing or whose `variants`
    subdict is empty after filtering by the requested variant set.
    """
    docs: dict[tuple[str, int], dict] = {}
    for pdf in sorted(docs_dir.glob("*.pdf")):
        parsed = _parse_name(pdf.stem)
        if parsed is None:
            continue
        cls, variant, idx = parsed
        if variant not in variants:
            continue
        json_path = pdf.with_suffix(".json")
        if not json_path.exists():
            continue
        entry = docs.setdefault((cls, idx), {"variants": {}, "json": json_path})
        entry["variants"][variant] = pdf
    return {k: v for k, v in docs.items() if v["variants"]}


def _plan_windows(
    docs: dict[tuple[str, int], dict],
    images_per_datapoint: int,
) -> list[dict]:
    """Flatten docs to one task per (cls, idx, window) for the LLM phase."""
    tasks: list[dict] = []
    for (cls, idx), info in docs.items():
        try:
            fields_raw = json.loads(info["json"].read_text())
        except Exception as e:
            print(f"  [{cls}#{idx:04}] JSON ERROR {e!r}", file=sys.stderr)
            continue
        if not fields_raw:
            continue
        page_count = max(int(f["page"]) for f in fields_raw) + 1
        for win_start, win_end in _windows(page_count, images_per_datapoint):
            window_fields = [
                f for f in fields_raw if win_start <= int(f["page"]) <= win_end
            ]
            if len(window_fields) < _MIN_FIELDS_PER_WINDOW:
                continue
            tasks.append(
                {
                    "cls": cls,
                    "idx": idx,
                    "win_start": win_start,
                    "win_end": win_end,
                    "fields": window_fields,
                }
            )
    return tasks


async def _llm_task(llm: LLM, sem: asyncio.Semaphore, task: dict) -> dict | None:
    async with sem:
        try:
            out = await _ask_llm(
                llm,
                task["cls"],
                task["win_start"],
                task["win_end"],
                task["fields"],
            )
        except Exception as e:
            print(
                f"  [{task['cls']}#{task['idx']:04} "
                f"p{task['win_start']}-{task['win_end']}] LLM ERROR {e!r}",
                file=sys.stderr,
            )
            return None
    return out


async def _emit_rows_for_variant(
    cls: str,
    idx: int,
    variant: str,
    pdf_path: Path,
    dpi: int,
    seed: int,
    windows_for_doc: list[tuple[tuple[int, int], dict, list[dict]]],
    render_sem: asyncio.Semaphore,
) -> list[dict]:
    """Rasterize one PDF and emit one row per matched window.

    The render semaphore is held for the full lifetime of the local `pages`
    list (raw PIL images dominate peak memory). PNG-encoded rows are small
    and stay in `rows` only until the caller writes them out.
    """
    async with render_sem:
        try:
            pages = await asyncio.to_thread(
                convert_from_bytes, pdf_path.read_bytes(), dpi=dpi
            )
        except Exception as e:
            print(f"  [{pdf_path.name}] RENDER ERROR {e!r}", file=sys.stderr)
            return []

        rng = random.Random(f"{seed}|{cls}|{idx}|{variant}")
        rows: list[dict] = []
        for (win_start, win_end), llm_out, window_fields in windows_for_doc:
            if win_end >= len(pages):
                continue
            fields_by_name = {f["name"]: f for f in window_fields}
            assembled = _assemble_datapoint(llm_out, fields_by_name, win_start, rng)
            if assembled is None:
                continue
            queries, answers = assembled
            images = [_png_bytes(pages[p]) for p in range(win_start, win_end + 1)]
            rows.append(
                {
                    "images": images,
                    "queries": queries,
                    "answers": answers,
                    "source": pdf_path.stem,
                    "variant": variant,
                    "page_start": win_start,
                    "page_end": win_end,
                }
            )
        return rows


_PARQUET_SCHEMA = pa.schema(
    [
        ("images", pa.list_(pa.binary())),
        ("queries", pa.list_(pa.string())),
        (
            "answers",
            pa.list_(
                pa.struct(
                    [
                        ("query", pa.string()),
                        ("value", pa.string()),
                        ("bounding_box", pa.list_(pa.float64())),
                        ("index", pa.int32()),
                    ]
                )
            ),
        ),
        ("source", pa.string()),
        ("variant", pa.string()),
        ("page_start", pa.int32()),
        ("page_end", pa.int32()),
    ]
)


async def _run(
    limit: int,
    seed: int,
    images_per_datapoint: int,
    variants: tuple[str, ...],
    dpi: int,
    output_path: Path,
) -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("error: ANTHROPIC_API_KEY environment variable is not set")
    if not _DOCS_DIR.exists():
        sys.exit(f"error: documents dir not found: {_DOCS_DIR}")

    docs = _discover_docs(_DOCS_DIR, variants)
    if limit > 0:
        keys = sorted(docs.keys())[:limit]
        docs = {k: docs[k] for k in keys}
    if not docs:
        sys.exit(f"error: no matching documents in {_DOCS_DIR}")

    tasks = _plan_windows(docs, images_per_datapoint)
    if not tasks:
        sys.exit("error: no windows planned (every doc filtered out)")

    print(
        f"processing {len(docs)} logical doc(s), {len(tasks)} window(s), "
        f"variants={variants}, images_per_datapoint={images_per_datapoint}, "
        f"dpi={dpi}, seed={seed}"
    )

    llm = LLM(
        base_url=_ANTHROPIC_BASE_URL,
        api_key=api_key,
        model=_MODEL,
        max_concurrency=_CONCURRENCY,
    )
    sem = asyncio.Semaphore(_CONCURRENCY)

    llm_outputs = await asyncio.gather(*(_llm_task(llm, sem, t) for t in tasks))

    # Bucket successful LLM outputs by (cls, idx) for the rendering phase.
    by_doc: dict[tuple[str, int], list[tuple[tuple[int, int], dict, list[dict]]]] = {}
    for task, out in zip(tasks, llm_outputs):
        if out is None:
            continue
        by_doc.setdefault((task["cls"], task["idx"]), []).append(
            ((task["win_start"], task["win_end"]), out, task["fields"])
        )

    render_sem = asyncio.Semaphore(_RENDER_CONCURRENCY)
    emit_tasks = []
    for (cls, idx), info in docs.items():
        windows_for_doc = by_doc.get((cls, idx))
        if not windows_for_doc:
            continue
        for variant, pdf_path in info["variants"].items():
            emit_tasks.append(
                _emit_rows_for_variant(
                    cls=cls,
                    idx=idx,
                    variant=variant,
                    pdf_path=pdf_path,
                    dpi=dpi,
                    seed=seed,
                    windows_for_doc=windows_for_doc,
                    render_sem=render_sem,
                )
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_total = len(emit_tasks)
    n_done = 0
    n_written = 0
    writer = pq.ParquetWriter(output_path, _PARQUET_SCHEMA)
    try:
        for fut in asyncio.as_completed(emit_tasks):
            batch = await fut
            n_done += 1
            if batch:
                table = pa.Table.from_pylist(batch, schema=_PARQUET_SCHEMA)
                writer.write_table(table)
                n_written += len(batch)
            if n_done % 50 == 0 or n_done == n_total:
                print(
                    f"  {n_done}/{n_total} doc-variants done, "
                    f"{n_written} datapoint(s) written"
                )
    finally:
        writer.close()

    if n_written == 0:
        output_path.unlink(missing_ok=True)
        sys.exit("error: no datapoints produced; nothing to write")

    print(
        f"\nwrote {output_path}: {n_written} datapoint(s) "
        f"from {len(docs)} logical doc(s)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=0,
        help="Process only the first N logical documents (0 = all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_DEFAULT_SEED,
        help=f"Seed for query shuffling (default: {_DEFAULT_SEED})",
    )
    parser.add_argument(
        "--images-per-datapoint",
        type=int,
        default=_DEFAULT_IMAGES_PER_DATAPOINT,
        help=(
            f"Page-window size: pages per datapoint "
            f"(default: {_DEFAULT_IMAGES_PER_DATAPOINT})"
        ),
    )
    parser.add_argument(
        "--variants",
        default=",".join(_DEFAULT_VARIANTS),
        help=(
            f"Comma-separated PDF variants to include "
            f"(default: {','.join(_DEFAULT_VARIANTS)})"
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=_DEFAULT_DPI,
        help=f"DPI for page rasterization (default: {_DEFAULT_DPI})",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=str(_OUT_PATH),
        help=f"Output parquet path (default: {_OUT_PATH})",
    )
    args = parser.parse_args()
    variants = tuple(v.strip() for v in args.variants.split(",") if v.strip())
    if not variants:
        sys.exit("error: --variants must list at least one variant")
    asyncio.run(
        _run(
            limit=args.limit,
            seed=args.seed,
            images_per_datapoint=args.images_per_datapoint,
            variants=variants,
            dpi=args.dpi,
            output_path=Path(args.output),
        )
    )


if __name__ == "__main__" and "__file__" in globals():
    main()
