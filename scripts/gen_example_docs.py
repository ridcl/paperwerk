"""Generate 5 example documents from random KVP10k templates.

For each of 5 randomly-chosen templates under
``src/datagen/assets/templates`` this:

  1. loads the Jinja2 template + its field schema (sidecar ``.json``),
  2. synthesizes one realistic-but-synthetic record via
     ``datagen.values.synthesize_values`` (LLM, Anthropic endpoint),
  3. picks a few form-like fields to render in a handwriting font
     (``datagen.handwriting`` via ``render``'s ``handwritten_fields``),
  4. renders the filled template to PDF via ``datagen.render.render``,
  5. writes ``output/example_{n}.pdf``.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import sys
from pathlib import Path

from paperwerk.llm import LLM

import datagen
from datagen.render import render
from datagen.values import synthesize_values

_ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1/"
_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

_TEMPLATES_ROOT = Path(datagen.__file__).resolve().parent / "assets" / "templates"
_OUT_DIR = Path("output")
_N = 5
_SEED = 20260719


def _pick_templates(rng: random.Random, n: int) -> list[Path]:
    """Random sample of n template files that have a sidecar schema JSON."""
    candidates = [
        p
        for p in _TEMPLATES_ROOT.rglob("*.html.j2")
        if p.with_suffix("").with_suffix(".json").exists()
    ]
    if len(candidates) < n:
        raise RuntimeError(f"only {len(candidates)} templates available")
    return rng.sample(candidates, n)


def _handwritten_subset(schema: list[str], rng: random.Random) -> list[str]:
    """Pick a small subset of scalar fields to render as handwriting.

    Skips array fields (`foo[]` / `foo[].bar`): those are usually tabular
    body content where handwriting reads oddly. On documents with no scalar
    fields we simply return nothing (fully typeset).
    """
    scalars = [f for f in schema if "[]" not in f]
    if not scalars:
        return []
    k = min(len(scalars), rng.randint(2, 5))
    return rng.sample(scalars, k)


async def _build_one(
    llm: LLM, idx: int, template_path: Path, rng: random.Random
) -> None:
    schema_path = template_path.with_suffix("").with_suffix(".json")
    meta = json.loads(schema_path.read_text())
    schema: list[str] = meta["schema"]
    template = template_path.read_text()

    cls = meta.get("class", template_path.parent.name)
    print(f"[{idx}] {cls}: {len(schema)} field(s) <- {template_path.name}")

    data = await synthesize_values(llm, schema)
    handwritten = _handwritten_subset(schema, rng)

    pdf, fields = await asyncio.to_thread(
        render,
        template,
        data,
        handwritten_fields=handwritten,
        seed=_SEED + idx,
    )

    out_path = _OUT_DIR / f"example_{idx}.pdf"
    out_path.write_bytes(pdf)
    n_pages = (max((f.page for f in fields), default=0) + 1) if fields else 1
    print(
        f"[{idx}] wrote {out_path} "
        f"({n_pages} page(s), {len(fields)} field occurrence(s), "
        f"{len(handwritten)} handwritten)"
    )


async def _main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("error: ANTHROPIC_API_KEY is not set")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(_SEED)
    templates = _pick_templates(rng, _N)

    llm = LLM(
        base_url=_ANTHROPIC_BASE_URL,
        api_key=api_key,
        model=_MODEL,
        max_concurrency=_N,
    )

    # Give each document its own rng stream for reproducible handwriting picks.
    await asyncio.gather(
        *(
            _build_one(llm, i + 1, tpl, random.Random(f"{_SEED}:{i}"))
            for i, tpl in enumerate(templates)
        )
    )


if __name__ == "__main__":
    asyncio.run(_main())
