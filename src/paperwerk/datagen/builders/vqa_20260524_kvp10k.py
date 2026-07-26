"""Convert KVP10K to the unified VQA datapoint schema.

Reads the existing `kvp10k-with-images.parquet` (one row per scanned page,
with the image bytes embedded and a `kvps` list of `{key, value, bounding_box}`
dicts) and emits the same schema as `vqa_20260524_cuad.py`, so the two
parquets can be mixed downstream.

Mapping per source row:
    images       <- [row.image]
    queries      <- unique kvp.key values in first-appearance order
    answers      <- one entry per kvp:
                       query        = kvp.key
                       value        = kvp.value
                       bounding_box = kvp.bounding_box
                       index        = 0           (kvp10k rows are 1 page)
    source       <- row.hash_name
    variant      <- "kvp10k"
    page_start   <- int(row.page_number)
    page_end     <- int(row.page_number)
    split        <- row.split  (only present in this parquet, not in the CUAD
                                one - downstream can drop it or fill the CUAD
                                side with a default split)

Run:
    python -m datagen.builders.vqa_20260524_kvp10k
    python -m datagen.builders.vqa_20260524_kvp10k --limit 100
    python -m datagen.builders.vqa_20260524_kvp10k --input /data/kvp10k-with-images.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_DEFAULT_INPUT = Path("/data/kvp10k-with-images.parquet")
_DEFAULT_OUTPUT = Path("/data/paperwerk/vqa_20260524_kvp10k.parquet")
_VARIANT = "kvp10k"

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
        ("split", pa.string()),
    ]
)


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _convert_row(row: pd.Series) -> dict | None:
    """Convert one kvp10k row to a unified VQA datapoint.

    Returns None if the row has no usable image or no validatable kvp - we
    don't emit empty datapoints.
    """
    image = row.get("image")
    if not isinstance(image, (bytes, bytearray)):
        return None
    kvps = row.get("kvps")
    if kvps is None or len(kvps) == 0:
        return None

    answers: list[dict] = []
    seen_keys: list[str] = []
    for kvp in kvps:
        if not isinstance(kvp, dict):
            continue
        key = kvp.get("key")
        if not isinstance(key, str):
            continue
        key = key.strip()
        if not key:
            continue
        value = kvp.get("value", "")
        if not isinstance(value, str):
            value = str(value)
        bbox = kvp.get("bounding_box")
        if bbox is None:
            continue
        try:
            bbox_list = [float(x) for x in bbox]
        except (TypeError, ValueError):
            continue
        if len(bbox_list) != 4:
            continue
        if key not in seen_keys:
            seen_keys.append(key)
        answers.append(
            {
                "query": key,
                "value": value,
                "bounding_box": bbox_list,
                "index": 0,
            }
        )

    if not answers:
        return None

    page = _coerce_int(row.get("page_number"))
    return {
        "images": [bytes(image)],
        "queries": list(seen_keys),
        "answers": answers,
        "source": str(row.get("hash_name", "")),
        "variant": _VARIANT,
        "page_start": page,
        "page_end": page,
        "split": str(row.get("split", "")),
    }


def _run(input_path: Path, output_path: Path, limit: int) -> None:
    if not input_path.exists():
        sys.exit(f"error: input parquet not found: {input_path}")

    print(f"reading {input_path}")
    df = pd.read_parquet(input_path)
    print(f"loaded {len(df)} rows; columns={list(df.columns)}")
    if limit > 0:
        df = df.head(limit)

    rows: list[dict] = []
    dropped = 0
    for _, src in df.iterrows():
        dp = _convert_row(src)
        if dp is None:
            dropped += 1
            continue
        rows.append(dp)

    if not rows:
        sys.exit("error: no rows survived conversion")

    print(f"converted {len(rows)} row(s), dropped {dropped}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=_PARQUET_SCHEMA)
    pq.write_table(table, output_path)
    print(f"wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=_DEFAULT_INPUT,
        help=f"Input kvp10k parquet (default: {_DEFAULT_INPUT})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Output parquet (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=0,
        help="Process only the first N rows (0 = all)",
    )
    args = parser.parse_args()
    _run(input_path=args.input, output_path=args.output, limit=args.limit)


if __name__ == "__main__":
    main()
