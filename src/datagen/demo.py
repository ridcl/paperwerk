"""End-to-end demo: real document -> Jinja2 template -> fake data -> PDF + bboxes.

Pipeline:
1. `templates.make_template(llm, source)` builds a Jinja2 HTML template that
   mirrors the layout of `source` and discovers its field schema.
2. `_synthesize_data(llm, fields)` asks the LLM for one realistic-looking
   but completely synthetic record matching that schema (handles nested
   objects and arrays via the `[]` / `.` conventions).
3. `render.render(template, data)` produces a PDF and per-field bboxes
   normalized to [0, 1] with top-left origin.
4. `render.annotate(pdf, fields)` rasterizes the PDF and overlays the
   bboxes for visual inspection.

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    python -m datagen.demo /path/to/real.pdf --out-dir output/demo
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path

from pile.llm import LLM

from datagen.render import annotate, render
from datagen.templates import make_template

_ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1/"
_DEFAULT_MODEL = "claude-sonnet-4-6"


async def _synthesize_data(llm: LLM, fields: list[str]) -> dict:
    """Ask the LLM to invent realistic but synthetic values for `fields`.

    Field-name conventions match the schema returned by `make_template`:
    - `"name"`           — scalar string
    - `"obj.attr"`       — nested {"obj": {"attr": ...}}
    - `"arr[]"`          — JSON array of strings
    - `"arr[].attr"`     — JSON array of objects with `attr`

    Returns a dict ready to feed into `render()`.
    """
    field_list = "\n".join(f"  - {f}" for f in fields)
    prompt = (
        "Generate one synthetic record for a document template. Do not use "
        "real PII; invent plausible names, companies, dates, addresses, etc.\n\n"
        "Field schema (snake_case; `[]` marks array fields, `.` marks nested "
        "object access):\n"
        f"{field_list}\n\n"
        "For array fields, choose a count that fits the document type "
        "(2-5 invoice line items, 4-10 bank transactions, 3-8 CV skills, "
        "etc.). Return one JSON object with the appropriate nesting and "
        "array contents. Wrap the JSON in one ```json``` code fence and "
        "output nothing else."
    )
    resp = await llm.ainvoke(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=16000,
        temperature=1.0,
    )
    text = resp.choices[0].message.content
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    blob = (m.group(1) if m else text).strip()
    try:
        return json.loads(blob)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"data synthesis returned invalid JSON ({e}); raw response:\n{text}"
        ) from e


async def _run(source: str, out_dir: str, model: str) -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("error: ANTHROPIC_API_KEY environment variable is not set")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    llm = LLM(base_url=_ANTHROPIC_BASE_URL, api_key=api_key, model=model)

    print(f"[1/4] making template from {source} (model={model})")
    template, schema = await make_template(llm, source)
    (out / "template.html").write_text(template)
    print(f"      -> {out / 'template.html'} ({len(schema)} field(s))")
    for f in schema:
        print(f"         - {f}")

    print("[2/4] synthesizing fake data")
    data = await _synthesize_data(llm, schema)
    (out / "data.json").write_text(json.dumps(data, indent=2))
    print(f"      -> {out / 'data.json'}")

    # `render` and `annotate` use sync_playwright / PIL, so run them in a
    # worker thread; sync_playwright refuses to start under a running
    # asyncio event loop.
    print("[3/4] rendering")
    pdf_bytes, fields = await asyncio.to_thread(render, template, data)
    (out / "sample.pdf").write_bytes(pdf_bytes)
    (out / "fields.json").write_text(json.dumps([asdict(f) for f in fields], indent=2))
    print(f"      -> {out / 'sample.pdf'} " f"({len(fields)} field occurrence(s))")

    print("[4/4] annotating preview")
    pages = await asyncio.to_thread(annotate, pdf_bytes, fields)
    for i, png in enumerate(pages):
        (out / f"sample_p{i}_annotated.png").write_bytes(png)
    print(f"      -> {len(pages)} annotated page(s) in {out}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source", help="Path to a real PDF or image")
    parser.add_argument("--out-dir", default="output/demo")
    parser.add_argument("--model", default=_DEFAULT_MODEL)
    args = parser.parse_args()
    asyncio.run(_run(args.source, args.out_dir, args.model))


if __name__ == "__main__":
    main()
