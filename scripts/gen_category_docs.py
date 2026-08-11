"""Generate sample documents from random templates, grouped by category.

For every category directory under ``/data/paperwerk/assets/templates`` this
picks N random templates (3 by default, 1 for ``financial_statement``),
synthesizes a realistic-but-synthetic record per template via
``paperwerk.datagen.values.random_values`` (Anthropic Sonnet), renders each to
PDF via ``render_sync``, and writes them to::

    output/docs/<category>/doc_{i:03}.pdf   (i is 1-based within the category)

Rendering can fail on the occasional bad template (e.g. an LLM-mis-nested value
that trips Jinja's strict undefined), so each slot retries with a freshly-drawn
template + values a few times before giving up.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import sys
from pathlib import Path

from paperwerk.llm import LLM
from paperwerk.datagen.render import render_sync
from paperwerk.datagen.values import random_values

_ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1/"
_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

_TEMPLATES_ROOT = Path("/data/paperwerk/assets/templates")
_OUT_DIR = Path("output") / "docs"

_DEFAULT_COUNT = 3
_COUNTS = {"financial_statement": 1}  # per-category overrides

_SEED = 20260727
_MAX_ATTEMPTS = 4          # per-slot render retries
_LLM_CONCURRENCY = 8       # in-flight value-gen calls
_RENDER_CONCURRENCY = 4    # concurrent headless-chromium renders

_render_sem = asyncio.Semaphore(_RENDER_CONCURRENCY)


def _templates_with_schema(cat_dir: Path) -> list[Path]:
    """Template files under `cat_dir` that have a sidecar schema JSON."""
    return [
        p
        for p in sorted(cat_dir.glob("*.html.j2"))
        if p.with_suffix("").with_suffix(".json").exists()
    ]


def _draw(rng: random.Random, pool: list[Path], n: int) -> list[Path]:
    """n templates from `pool`; without replacement when possible, else with."""
    if len(pool) >= n:
        return rng.sample(pool, n)
    return rng.choices(pool, k=n)  # tiny category (e.g. invoice) -> reuse


def _handwritten_subset(schema: list[str], rng: random.Random) -> list[str]:
    """A few scalar fields to render as handwriting (skip tabular arrays)."""
    scalars = [f for f in schema if "[]" not in f]
    if not scalars:
        return []
    k = min(len(scalars), rng.randint(2, 4))
    return rng.sample(scalars, k)


async def _build_slot(
    llm: LLM,
    category: str,
    idx: int,
    pool: list[Path],
    rng: random.Random,
) -> tuple[str, int, bool]:
    """Fill one doc slot, retrying with fresh templates/values on failure."""
    out_path = _OUT_DIR / category / f"doc_{idx:03}.pdf"
    last_err: Exception | None = None

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        template_path = _draw(rng, pool, 1)[0]
        try:
            meta = json.loads(
                template_path.with_suffix("").with_suffix(".json").read_text()
            )
            schema: list[str] = meta["schema"]
            template = template_path.read_text()

            data = await random_values(llm, schema, hint_rng=rng)
            handwritten = _handwritten_subset(schema, rng)

            async with _render_sem:
                pdf, fields = await asyncio.to_thread(
                    render_sync,
                    template,
                    data,
                    handwritten_fields=handwritten,
                    seed=_SEED + idx,
                )

            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(pdf)
            n_pages = (max((f.page for f in fields), default=0) + 1) if fields else 1
            print(
                f"[{category}/doc_{idx:03}] OK  {template_path.name} "
                f"({n_pages}p, {len(fields)} field occ, {len(handwritten)} hw)"
            )
            return category, idx, True
        except Exception as e:  # noqa: BLE001 - retry on any render/value failure
            last_err = e
            print(
                f"[{category}/doc_{idx:03}] attempt {attempt}/{_MAX_ATTEMPTS} "
                f"failed on {template_path.name}: {type(e).__name__}: {e}"
            )

    print(f"[{category}/doc_{idx:03}] GAVE UP after {_MAX_ATTEMPTS}: {last_err}")
    return category, idx, False


async def _main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("error: ANTHROPIC_API_KEY is not set")
    if not _TEMPLATES_ROOT.is_dir():
        sys.exit(f"error: templates root not found: {_TEMPLATES_ROOT}")

    categories = sorted(p for p in _TEMPLATES_ROOT.iterdir() if p.is_dir())
    llm = LLM(
        base_url=_ANTHROPIC_BASE_URL,
        api_key=api_key,
        model=_MODEL,
        max_concurrency=_LLM_CONCURRENCY,
    )

    tasks = []
    for cat_dir in categories:
        category = cat_dir.name
        pool = _templates_with_schema(cat_dir)
        if not pool:
            print(f"[{category}] no templates with schema, skipping")
            continue
        count = _COUNTS.get(category, _DEFAULT_COUNT)
        for i in range(1, count + 1):
            rng = random.Random(f"{_SEED}:{category}:{i}")
            tasks.append(_build_slot(llm, category, i, pool, rng))

    results = await asyncio.gather(*tasks)

    ok = sum(1 for _, _, s in results if s)
    print(f"\n=== done: {ok}/{len(results)} document(s) written under {_OUT_DIR} ===")
    for category, idx, success in sorted(results):
        if not success:
            print(f"  FAILED: {category}/doc_{idx:03}.pdf")


if __name__ == "__main__":
    asyncio.run(_main())
