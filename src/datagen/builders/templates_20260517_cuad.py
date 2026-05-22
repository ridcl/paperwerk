"""Build the CUAD template/schema parquet for the synthetic data pipeline.

For each of the first 1000 rows of `dvgodoy/CUAD_v1_Contract_Understanding_PDF`:
1. zero-shot classify the contract (`datagen.classifier.classify`),
2. derive a Jinja2 HTML template and field schema
   (`datagen.templates.make_template`).

Values are NOT synthesized here — downstream consumers fill them in
(typically via `datagen.values.synthesize_values`) when actually rendering
documents.

Writes one parquet row per successful contract to
`/data/pile/templates/cuad_20260517.parquet` with columns:
    template (str), schema (list[str]), class (str), source (str = "cuad").

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    python -m datagen.builders.templates_20260517_cuad
    python -m datagen.builders.templates_20260517_cuad --limit 100
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import os
import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

from pile.llm import LLM

from datagen.classifier import classify
from datagen.templates import make_template


_DATASET = "dvgodoy/CUAD_v1_Contract_Understanding_PDF"
_SPLIT = "train"
_PDF_COL = "pdf_bytes_base64"
_NAME_COL = "file_name"
_SOURCE = "cuad"
_OUT_PATH = Path("/data/pile/templates/cuad_20260517.parquet")
_ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1/"
_MODEL = "claude-sonnet-4-6"
_CONCURRENCY = 10
_DEFAULT_LIMIT = 1000


def _maybe_b64_decode(buf: bytes) -> bytes:
    s = buf.strip()
    if s.startswith(b"data:"):
        s = s.split(b",", 1)[1]
    s = b"".join(s.split())
    return base64.b64decode(s)


def _row_pdf_bytes(value) -> bytes:
    """Coerce the CUAD `pdf_bytes_base64` cell into raw PDF bytes.

    The column is base64 — sometimes surfaced as `str`, sometimes `bytes`.
    We accept either and fall back to b64-decoding when the value doesn't
    already start with `%PDF`.
    """
    if isinstance(value, str):
        raw: bytes = value.encode("ascii", errors="ignore")
    elif isinstance(value, bytes):
        raw = value
    else:
        raise RuntimeError(f"unsupported PDF cell type: {type(value).__name__}")

    if not raw.lstrip().startswith(b"%PDF"):
        raw = _maybe_b64_decode(raw)
    if not raw.lstrip().startswith(b"%PDF"):
        raise RuntimeError(
            f"decoded cell is not a PDF; first bytes={raw[:32]!r}"
        )
    return raw


async def _process_row(
    llm: LLM,
    sem: asyncio.Semaphore,
    i: int,
    row_idx: int,
    file_name: str,
    ds,
    tmp_root: Path,
) -> dict | None:
    async with sem:
        try:
            pdf_bytes = await asyncio.to_thread(
                lambda: _row_pdf_bytes(ds[row_idx][_PDF_COL])
            )
        except Exception as e:
            print(f"  [{i:04}] {file_name}: DECODE ERROR {e!r}", file=sys.stderr)
            return None

        tmp_pdf = tmp_root / f"{i:04}.pdf"
        try:
            await asyncio.to_thread(tmp_pdf.write_bytes, pdf_bytes)
            try:
                cls = await classify(llm, str(tmp_pdf))
            except Exception as e:
                print(f"  [{i:04}] {file_name}: CLASSIFY ERROR {e!r}", file=sys.stderr)
                return None
            try:
                template, schema = await make_template(llm, str(tmp_pdf))
            except Exception as e:
                print(f"  [{i:04}] {file_name}: TEMPLATE ERROR {e!r}", file=sys.stderr)
                return None
            print(
                f"  [{i:04}] {file_name}: cls={cls!r} "
                f"fields={len(schema)} tmpl={len(template)}B"
            )
            return {
                "template": template,
                "schema": schema,
                "class": cls,
                "source": _SOURCE,
            }
        finally:
            tmp_pdf.unlink(missing_ok=True)


async def _run(limit: int) -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("error: ANTHROPIC_API_KEY environment variable is not set")

    print(f"loading {_DATASET} ({_SPLIT}) from HuggingFace Hub")
    ds = load_dataset(_DATASET, split=_SPLIT)
    if _PDF_COL not in ds.column_names:
        sys.exit(f"error: expected column {_PDF_COL!r} in {ds.column_names}")

    n = min(limit, len(ds)) if limit > 0 else len(ds)
    names = ds.select(range(n))[_NAME_COL] if _NAME_COL in ds.column_names else None
    file_names = [str(names[i]) if names else f"row_{i}" for i in range(n)]
    print(f"building templates for {n} contract(s) -> {_OUT_PATH}")

    llm = LLM(
        base_url=_ANTHROPIC_BASE_URL,
        api_key=api_key,
        model=_MODEL,
        max_concurrency=_CONCURRENCY,
    )
    sem = asyncio.Semaphore(_CONCURRENCY)

    with tempfile.TemporaryDirectory(prefix="cuad_builder_") as tmpdir:
        tmp_root = Path(tmpdir)
        results = await asyncio.gather(
            *(
                _process_row(
                    llm=llm,
                    sem=sem,
                    i=i,
                    row_idx=i,
                    file_name=file_names[i],
                    ds=ds,
                    tmp_root=tmp_root,
                )
                for i in range(n)
            )
        )

    rows = [r for r in results if r is not None]
    if not rows:
        sys.exit("error: no rows succeeded; nothing to write")

    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "template": pa.array([r["template"] for r in rows], type=pa.string()),
            "schema": pa.array(
                [r["schema"] for r in rows], type=pa.list_(pa.string())
            ),
            "class": pa.array([r["class"] for r in rows], type=pa.string()),
            "source": pa.array([r["source"] for r in rows], type=pa.string()),
        }
    )
    pq.write_table(table, _OUT_PATH)
    print(f"\nwrote {_OUT_PATH}: {len(rows)}/{n} contract(s)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=_DEFAULT_LIMIT,
        help=f"Process only the first N rows (default: {_DEFAULT_LIMIT}; 0 = all)",
    )
    args = parser.parse_args()
    asyncio.run(_run(limit=args.limit))


if __name__ == "__main__":
    main()
