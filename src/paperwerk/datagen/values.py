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
import random
import re

from paperwerk.llm import LLM

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


# Axes used to inject variety across synthetic records. Each call draws one
# value per axis from `hint_rng`, which the prompt receives as a short profile
# string so successive samples don't collapse to generic defaults.
_INDUSTRIES = [
    "technology / software",
    "finance / banking",
    "healthcare / hospital",
    "education / academia",
    "retail / e-commerce",
    "manufacturing / industrial",
    "media / entertainment",
    "non-profit / NGO",
    "management consulting",
    "government / public sector",
    "biotech / pharma",
    "renewable energy",
    "transportation / logistics",
    "real estate",
    "agriculture / food",
    "marketing / advertising",
    "legal services",
    "architecture / construction",
]
_SENIORITY = [
    "early-career professional (1-3 years of experience)",
    "mid-career professional (4-7 years of experience)",
    "senior professional (8-15 years of experience)",
    "executive / director level (15+ years of experience)",
    "recent graduate just entering the workforce",
]
_REGIONS = [
    "based in the United States or Canada",
    "based in the United Kingdom or Ireland",
    "based in continental Western Europe (Germany, France, Netherlands, Spain, Italy)",
    "based in the Nordic countries (Sweden, Norway, Denmark, Finland)",
    "based in Central or Eastern Europe (Poland, Czech Republic, Romania, Ukraine)",
    "based in Latin America (Mexico, Brazil, Argentina, Chile, Colombia)",
    "based in South Asia (India, Pakistan, Bangladesh, Sri Lanka)",
    "based in East Asia (Japan, South Korea, China, Taiwan)",
    "based in Southeast Asia (Singapore, Indonesia, Philippines, Vietnam)",
    "based in Australia or New Zealand",
    "based in the Middle East (UAE, Saudi Arabia, Israel, Turkey)",
    "based in Sub-Saharan Africa (Nigeria, Kenya, South Africa)",
]
_NAME_STYLES = [
    "Anglo-American",
    "Hispanic / Latino",
    "Slavic / Eastern European",
    "South Asian (Indian, Pakistani, Bangladeshi)",
    "East Asian (Chinese, Korean, Japanese)",
    "Arabic / Middle Eastern",
    "West African",
    "East African",
    "Scandinavian / Nordic",
    "French / Italian / Iberian",
    "German / Dutch",
    "Greek / Balkan",
]


def _diversity_hint(rng: random.Random) -> str:
    return (
        "Subject profile for this sample: "
        f"{rng.choice(_SENIORITY)}, "
        f"working in {rng.choice(_INDUSTRIES)}, "
        f"{rng.choice(_REGIONS)}. "
        f"Use {rng.choice(_NAME_STYLES)}-style personal and place names where applicable, "
        "and choose company names, schools, and addresses consistent with that region. "
        "Lean into the specifics — avoid generic 'Acme Corp' / 'John Smith' / '123 Main St' defaults."
    )


async def random_values(
    llm: LLM, field_names: list[str], hint_rng: random.Random | None = None
) -> dict:
    """Ask the LLM to invent realistic but synthetic values for `fields`.

    A random diversity profile (seniority × industry × region × name style),
    drawn from `hint_rng`, is folded into the prompt so successive calls vary
    across samples instead of collapsing to 'Acme Corp / John Smith / 123 Main
    St'. Pass a seeded `hint_rng` for reproducibility; when omitted, an
    unseeded one is used.
    """
    rng = hint_rng or random.Random()
    field_list = "\n".join(f"  - {f}" for f in field_names)
    prompt = (
        "Generate one synthetic record for a document template. Do not use "
        "real PII; invent plausible names, companies, dates, addresses, etc.\n\n"
        f"{_diversity_hint(rng)}\n\n"
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
