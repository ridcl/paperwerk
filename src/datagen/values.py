"""LLM-driven synthesis of one set of field values for a document template.

Given a flat field schema produced by `datagen.templates.make_template`, ask
the LLM for a single realistic-looking but fully synthetic record. Output is
a `dict` ready to feed into `datagen.render.render` alongside the template.

Field-name conventions (mirrors the schema returned by `make_template`):
- `"name"`           — scalar string
- `"obj.attr"`       — nested {"obj": {"attr": ...}}
- `"arr[]"`          — JSON array of strings
- `"arr[].attr"`     — JSON array of objects with `attr`
"""

from __future__ import annotations

import json
import re

from pile.llm import LLM


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


async def synthesize_values(llm: LLM, fields: list[str]) -> dict:
    """Ask the LLM to invent realistic but synthetic values for `fields`."""
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
    m = _JSON_FENCE_RE.search(text)
    blob = (m.group(1) if m else text).strip()
    try:
        return json.loads(blob)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"value synthesis returned invalid JSON ({e}); raw response:\n{text}"
        ) from e
