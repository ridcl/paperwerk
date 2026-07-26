"""Merge the kvp10k and CUAD VQA parquets into a single training file.

Both inputs follow the unified schema produced by `vqa_20260524_kvp10k.py`
and `vqa_20260524_cuad.py` (`images`, `queries`, `answers`, `source`,
`variant`, `page_start`, `page_end`). The kvp10k parquet carries an extra
`split` column from the source dataset; the CUAD parquet does not. This
script:

  - reads both parquets;
  - adds `split = --cuad-split` (default `"train"`) to every CUAD row;
  - aligns CUAD's column order to kvp10k's so the schemas match exactly;
  - concatenates and writes a single parquet.

Run:
    python -m datagen.builders.vqa_20260524_cuad_kvp10k
    python -m datagen.builders.vqa_20260524_cuad_kvp10k \\
        --kvp10k /data/paperwerk/vqa_20260524_kvp10k.parquet \\
        --cuad   /data/paperwerk/vqa_20260524_cuad.parquet \\
        -o /data/paperwerk/vqa_20260524.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

_DEFAULT_KVP10K = Path("/data/paperwerk/vqa_20260524_kvp10k.parquet")
_DEFAULT_CUAD = Path("/data/paperwerk/vqa_20260524_cuad.parquet")
_DEFAULT_OUTPUT = Path("/data/paperwerk/vqa_20260524.parquet")
_DEFAULT_CUAD_SPLIT = "train"


def _ensure_split(table: pa.Table, split: str) -> pa.Table:
    if "split" in table.column_names:
        return table
    return table.append_column(
        "split", pa.array([split] * table.num_rows, type=pa.string())
    )


def _run(
    kvp10k_path: Path,
    cuad_path: Path,
    output_path: Path,
    cuad_split: str,
) -> None:
    for p, label in ((kvp10k_path, "kvp10k"), (cuad_path, "cuad")):
        if not p.exists():
            sys.exit(f"error: {label} parquet not found: {p}")

    print(f"reading {kvp10k_path}")
    kvp_tbl = pq.read_table(kvp10k_path)
    print(f"  {kvp_tbl.num_rows} row(s), cols={kvp_tbl.column_names}")

    print(f"reading {cuad_path}")
    cuad_tbl = pq.read_table(cuad_path)
    print(f"  {cuad_tbl.num_rows} row(s), cols={cuad_tbl.column_names}")

    cuad_tbl = _ensure_split(cuad_tbl, cuad_split)
    missing = set(kvp_tbl.column_names) - set(cuad_tbl.column_names)
    if missing:
        sys.exit(
            f"error: cuad parquet is missing required column(s): {sorted(missing)}"
        )
    cuad_tbl = cuad_tbl.select(kvp_tbl.column_names)

    if cuad_tbl.schema != kvp_tbl.schema:
        sys.exit(
            "error: schemas differ after column alignment.\n"
            f"  kvp10k: {kvp_tbl.schema}\n"
            f"  cuad:   {cuad_tbl.schema}"
        )

    combined = pa.concat_tables([kvp_tbl, cuad_tbl])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(combined, output_path)
    print(
        f"\nwrote {output_path}: {combined.num_rows} row(s) "
        f"(kvp10k={kvp_tbl.num_rows}, cuad={cuad_tbl.num_rows})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--kvp10k",
        type=Path,
        default=_DEFAULT_KVP10K,
        help=f"kvp10k parquet (default: {_DEFAULT_KVP10K})",
    )
    parser.add_argument(
        "--cuad",
        type=Path,
        default=_DEFAULT_CUAD,
        help=f"CUAD parquet (default: {_DEFAULT_CUAD})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Output parquet (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--cuad-split",
        default=_DEFAULT_CUAD_SPLIT,
        help=(
            f"Split label to assign to CUAD rows that lack one "
            f"(default: {_DEFAULT_CUAD_SPLIT!r})"
        ),
    )
    args = parser.parse_args()
    _run(
        kvp10k_path=args.kvp10k,
        cuad_path=args.cuad,
        output_path=args.output,
        cuad_split=args.cuad_split,
    )


if __name__ == "__main__":
    main()
