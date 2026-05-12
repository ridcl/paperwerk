"""LLM-driven synthetic document generator.

Pipeline:
  1. Read a real document (PDF or image) and convert each page to a base64 PNG.
  2. Send the page images to Claude (via the OpenAI-compatible endpoint at
     https://api.anthropic.com/v1/) and ask it to produce an HTML+CSS template
     that mirrors the original layout, with every variable text wrapped in
     <span data-field="FIELD_NAME">{{FIELD_NAME}}</span> placeholders.
  3. For each requested sample, ask Claude to generate one synthetic data
     record matching the discovered fields.
  4. Substitute the data into the template and render it via Playwright,
     producing a multi-page PDF, a per-occurrence bbox JSON, and per-page
     annotated PNGs.

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    python -m datagen.generator /path/to/real.pdf --out-dir output/datagen --n 3
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path

from pile.llm import LLM

from datagen.render import annotate, render
from datagen._templates import (
    discover_fields,
    extract_block,
    generate_template,
)

_ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1/"
_DEFAULT_MODEL = "claude-sonnet-4-6"


_DATA_PROMPT = """Generate one set of synthetic, realistic-looking values for the template fields below. Do not use real PII; invent plausible names, companies, dates, addresses, etc. Match the typical length and style of each field (a job title is short; a job description is a sentence or two). For repeated, indexed sections (e.g. experience_1_*, experience_2_*), keep each entry internally consistent.

{hint}

Fields:
{field_list}

Return a single JSON object that maps each field name to its synthetic value. Wrap it in one ```json``` code fence and output nothing else."""


# Axes used to inject variety across synthetic samples. The per-sample rng
# picks one value from each axis, which the data-synthesis prompt receives
# as a short profile string.
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


def synthesize_data(field_names: list[str], llm: LLM, hint: str = "") -> dict[str, str]:
    """Ask the LLM for one synthetic record matching the given field schema."""
    field_list = "\n".join(f"- {n}" for n in field_names)
    prompt = _DATA_PROMPT.format(hint=hint, field_list=field_list)
    resp = llm.invoke(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4000,
        temperature=1.0,
    )
    blob = extract_block(resp.choices[0].message.content, "json")
    parsed = json.loads(blob)
    # Coerce missing entries to empty strings; ensure every value is a string.
    return {n: str(parsed.get(n, "")) for n in field_names}


def main(
    source: str,
    out_dir: str = "output/datagen",
    n: int = 3,
    model: str = _DEFAULT_MODEL,
    template_path: str | None = None,
    seed: int = 0,
) -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("error: ANTHROPIC_API_KEY environment variable is not set")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    llm = LLM(base_url=_ANTHROPIC_BASE_URL, api_key=api_key, model=model)

    template_file = out / "template.html"
    if template_path:
        template = Path(template_path).read_text()
        print(f"reusing template from {template_path}")
    else:
        print(f"generating template from {source} via {model} ...")
        template, leaks = generate_template(Path(source), llm)
        template_file.write_text(template)
        print(f"saved template -> {template_file}")
        if leaks:
            print(
                f"  warning: rewrote {len(leaks)} data-field element(s) whose "
                "content was not the expected placeholder (likely PII leak from source):"
            )
            for name, snippet in leaks[:10]:
                print(f"    - {name}: {snippet!r}")
            if len(leaks) > 10:
                print(f"    ... and {len(leaks) - 10} more")

    field_names = discover_fields(template)
    if not field_names:
        sys.exit("error: generated template has no [data-field] markers")
    print(f"template defines {len(field_names)} fields: {', '.join(field_names)}")

    for i in range(n):
        sample_rng = random.Random(f"{seed}:{i}")
        hint = _diversity_hint(sample_rng)
        print(f"[{i + 1}/{n}] {hint}")
        data = synthesize_data(field_names, llm, hint=hint)
        pdf, fields = render(template, data)
        pages = annotate(pdf, fields)
        base = out / f"sample_{i:02d}"
        base.with_suffix(".pdf").write_bytes(pdf)
        base.with_suffix(".json").write_text(
            json.dumps(
                {"hint": hint, "data": data, "fields": [asdict(f) for f in fields]},
                indent=2,
            )
        )
        for p, png_bytes in enumerate(pages):
            out.joinpath(f"sample_{i:02d}_p{p}_annotated.png").write_bytes(png_bytes)
        print(
            f"  wrote {base}.pdf "
            f"({len(pages)} page(s), {len(fields)} field occurrence(s))"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "source",
        help="Path to a real PDF (or image) to use as the layout reference",
    )
    parser.add_argument("--out-dir", default="output/datagen")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--model", default=_DEFAULT_MODEL)
    parser.add_argument(
        "--template",
        default=None,
        help="Reuse an existing template HTML instead of regenerating from source",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the per-sample diversity-axis selection",
    )
    args = parser.parse_args()
    main(
        source=args.source,
        out_dir=args.out_dir,
        n=args.n,
        model=args.model,
        template_path=args.template,
        seed=args.seed,
    )
