"""Render synthetic documents from the CUAD template/schema parquet.

Reads `/data/pile/templates/cuad_20260517.parquet` (produced by
`templates_20260517_cuad.py`). For each (template, schema, class) row:

1. `synthesize_values(llm, schema)` invents one set of fake values,
2. `render(template, values, handwritten_fields=...)` rasterizes to a clean
   PDF (the "clear" variant),
3. `augment(clear_pdf, fields, profile="phone_photo")` produces the
   phone-photo variant.

In ~10% of rows (deterministic given `--seed`) the values are styled as
handwritten across every field; the other 90% are typed.

Outputs go to `/data/pile/documents_20260517/{class}_{augmentation}_{index:04}.pdf`
where `augmentation` is `clear` or `phone_photo` and `index` is the
per-class sequence number.

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    python -m datagen.builders.documents_20260517_cuad
    python -m datagen.builders.documents_20260517_cuad --limit 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
from dataclasses import asdict
from pathlib import Path

import pyarrow.parquet as pq

from pile.llm import LLM

from datagen.augment import augment
from datagen.render import render
from datagen.values import synthesize_values


_TEMPLATES_PARQUET = Path("/data/pile/templates/cuad_20260517.parquet")
_OUT_DIR = Path("/data/pile/documents_20260517")
_ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1/"
_MODEL = "claude-sonnet-4-6"
_CONCURRENCY = 10
_HANDWRITING_RATE = 0.10
_DEFAULT_SEED = 0
_AUGMENT_PROFILE = "phone_photo"
_RENDER_DPI = 200

_SAFE_NAME_RE = re.compile(r"[^a-z0-9_]+")


def _safe_class(cls: str) -> str:
    """Sanitize a class label for use in a filename component."""
    return _SAFE_NAME_RE.sub("_", cls.strip().lower()).strip("_") or "unknown"


def _load_rows(path: Path) -> list[dict]:
    table = pq.read_table(path)
    cols = {name: table.column(name).to_pylist() for name in table.column_names}
    n = table.num_rows
    return [
        {
            "template": cols["template"][i],
            "schema": list(cols["schema"][i]) if cols["schema"][i] else [],
            "class": cols["class"][i] or "unknown",
            "source": cols["source"][i],
        }
        for i in range(n)
    ]


async def _process_row(
    llm: LLM,
    sem: asyncio.Semaphore,
    row: dict,
    class_index: int,
    row_seed: int,
    handwritten: bool,
) -> None:
    cls = _safe_class(row["class"])
    label = f"[{cls}#{class_index:04}]"

    async with sem:
        try:
            values = await synthesize_values(llm, row["schema"])
        except Exception as e:
            print(f"  {label} SYNTH ERROR {e!r}", file=sys.stderr)
            return

        handwritten_fields = tuple(row["schema"]) if handwritten else ()
        try:
            clear_pdf, fields = await asyncio.to_thread(
                render,
                row["template"],
                values,
                handwritten_fields=handwritten_fields,
                seed=row_seed,
            )
        except Exception as e:
            print(f"  {label} RENDER ERROR {e!r}", file=sys.stderr)
            return

        fields_json = json.dumps(
            [asdict(f) for f in fields], ensure_ascii=False, indent=2
        ).encode("utf-8")

        clear_path = _OUT_DIR / f"{cls}_clear_{class_index:04}.pdf"
        clear_json = clear_path.with_suffix(".json")
        await asyncio.to_thread(clear_path.write_bytes, clear_pdf)
        await asyncio.to_thread(clear_json.write_bytes, fields_json)

        try:
            phone_pdf, phone_fields = await asyncio.to_thread(
                augment,
                clear_pdf,
                fields,
                profile=_AUGMENT_PROFILE,
                dpi=_RENDER_DPI,
                seed=row_seed,
            )
        except Exception as e:
            print(f"  {label} AUGMENT ERROR {e!r}", file=sys.stderr)
            return

        # Stage-A augmentation is pixel-only; bboxes are unchanged. Write a
        # matching sidecar for the phone-photo PDF too so each output is paired.
        phone_fields_json = json.dumps(
            [asdict(f) for f in phone_fields], ensure_ascii=False, indent=2
        ).encode("utf-8")
        phone_path = _OUT_DIR / f"{cls}_{_AUGMENT_PROFILE}_{class_index:04}.pdf"
        phone_json = phone_path.with_suffix(".json")
        await asyncio.to_thread(phone_path.write_bytes, phone_pdf)
        await asyncio.to_thread(phone_json.write_bytes, phone_fields_json)

        hw_tag = " hw" if handwritten else ""
        print(
            f"  {label}{hw_tag} -> {clear_path.name}+{clear_json.name}, "
            f"{phone_path.name}+{phone_json.name}"
        )


async def _run(limit: int, seed: int) -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("error: ANTHROPIC_API_KEY environment variable is not set")
    if not _TEMPLATES_PARQUET.exists():
        sys.exit(f"error: templates parquet not found: {_TEMPLATES_PARQUET}")

    rows = _load_rows(_TEMPLATES_PARQUET)
    if limit > 0:
        rows = rows[:limit]
    if not rows:
        sys.exit("error: templates parquet is empty")

    # Per-class sequence numbers and per-row deterministic flags.
    class_counter: dict[str, int] = {}
    plans: list[tuple[dict, int, int, bool]] = []
    rng = random.Random(seed)
    for r in rows:
        cls = _safe_class(r["class"])
        idx = class_counter.get(cls, 0)
        class_counter[cls] = idx + 1
        handwritten = rng.random() < _HANDWRITING_RATE
        plans.append((r, idx, rng.randint(0, 2**31 - 1), handwritten))

    n_hw = sum(1 for _, _, _, hw in plans if hw)
    print(
        f"rendering {len(plans)} template(s) -> {_OUT_DIR}/ "
        f"({n_hw} handwritten, profile={_AUGMENT_PROFILE!r}, seed={seed})"
    )
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    llm = LLM(
        base_url=_ANTHROPIC_BASE_URL,
        api_key=api_key,
        model=_MODEL,
        max_concurrency=_CONCURRENCY,
    )
    sem = asyncio.Semaphore(_CONCURRENCY)

    await asyncio.gather(
        *(
            _process_row(
                llm=llm,
                sem=sem,
                row=row,
                class_index=class_index,
                row_seed=row_seed,
                handwritten=handwritten,
            )
            for row, class_index, row_seed, handwritten in plans
        )
    )

    print(f"\ndone — outputs in {_OUT_DIR}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=0,
        help="Process only the first N templates (0 = all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_DEFAULT_SEED,
        help=f"Master seed for handwriting selection & per-row RNG (default: {_DEFAULT_SEED})",
    )
    args = parser.parse_args()
    asyncio.run(_run(limit=args.limit, seed=args.seed))


if __name__ == "__main__":
    main()
