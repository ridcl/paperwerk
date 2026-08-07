"""Extract the CUAD rows from the combined VQA parquet and publish them to HF.

`/data/paperwerk/vqa_20260524.parquet` is the merged CUAD+kvp10k training file
produced by `vqa_20260524_cuad_kvp10k.py`. The two sources are distinguished by
`variant`: kvp10k rows carry `variant == "kvp10k"`, while the CUAD-synthetic rows
carry `variant in {"clear", "phone_photo"}` (the two pixel augmentations of each
logical contract). This script keeps only the CUAD rows and republishes them as a
standalone dataset on the HF Hub (`ridcl/cuad-synthetic-vqa`), so a training run
can combine it with `ridcl/xbrl-tables` without pulling in kvp10k.

The output schema is made identical to `ridcl/xbrl-tables` (the CUAD VQA schema
from `vqa_20260524_cuad.py`): `images`, `queries`, `answers`, `source`, `variant`,
`page_start`, `page_end`. The merge step had appended a `split` column; we drop it
here, since both the xbrl-tables dataset and the training script omit it (the
trainer does its own source-level split).

The combined parquet is one ~7.6 GB row group whose decoded `images` (a
`list<binary>` with 32-bit offsets) exceed pyarrow's 2 GiB offset limit, so it is
read in bounded batches (a single contiguous binary array per batch keeps the
nested conversion on the supported path) and written back the same way, one row
group per batch.

Run:
    # write the CUAD-only parquet locally:
    python -m datagen.builders.cuad_synthetic_vqa_20260622

    # build and push to the HF Hub (auth via HF_TOKEN or `huggingface-cli login`):
    python -m datagen.builders.cuad_synthetic_vqa_20260622 --push-to-hub
    python -m datagen.builders.cuad_synthetic_vqa_20260622 --push-to-hub \\
        --hub-repo my-org/cuad-synthetic-vqa
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

_DEFAULT_INPUT = Path("/data/paperwerk/vqa_20260524.parquet")
_DEFAULT_OUTPUT = Path("/data/paperwerk/cuad_synthetic_vqa_20260622.parquet")
_HUB_REPO = "cuad-synthetic-vqa"  # default HF Hub repo (under the logged-in namespace)

# CUAD rows are exactly the pixel-augmentation variants; kvp10k uses "kvp10k".
_CUAD_VARIANTS = frozenset({"clear", "phone_photo"})

_READ_BATCH_SIZE = 512

# Identical to `vqa_20260524_cuad.py` / `xbrl_tables_20260619.py` (no `split`).
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
    ]
)


def _push_to_hub(parquet_path: Path, repo: str, private: bool) -> None:
    """Upload the generated parquet to the HF Hub as a dataset repo.

    Authentication uses the standard huggingface_hub resolution (the `HF_TOKEN`
    env var or a cached `huggingface-cli login`). A bare `repo` lands under the
    logged-in namespace; pass `org/name` to target an organization.
    """
    from datasets import Dataset  # heavy import; only paid when actually pushing

    print(f"loading {parquet_path} and pushing to HF Hub dataset {repo!r} ...")
    ds = Dataset.from_parquet(str(parquet_path))
    ds.push_to_hub(repo, private=private)
    print(f"pushed {len(ds)} row(s) to HF Hub dataset {repo!r}")


def _run(
    input_path: Path,
    output_path: Path,
    push_hub: bool,
    hub_repo: str,
    hub_private: bool,
) -> None:
    if not input_path.exists():
        sys.exit(f"error: input parquet not found: {input_path}")

    cols = list(_PARQUET_SCHEMA.names)  # read only what we keep (drops `split`)
    pf = pq.ParquetFile(input_path)
    print(f"reading {input_path}: {pf.metadata.num_rows} row(s)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(output_path, _PARQUET_SCHEMA)
    n_read = 0
    n_written = 0
    try:
        for batch in pf.iter_batches(batch_size=_READ_BATCH_SIZE, columns=cols):
            n_read += batch.num_rows
            table = pa.Table.from_batches([batch]).cast(_PARQUET_SCHEMA)
            mask = pc.is_in(
                table.column("variant"),
                value_set=pa.array(sorted(_CUAD_VARIANTS), type=pa.string()),
            )
            cuad = table.filter(mask)
            if cuad.num_rows:
                writer.write_table(cuad)
                n_written += cuad.num_rows
    finally:
        writer.close()

    if n_written == 0:
        output_path.unlink(missing_ok=True)
        sys.exit("error: no CUAD rows found; nothing to write")

    print(
        f"\nwrote {output_path}: {n_written} CUAD row(s) "
        f"(of {n_read} total in {input_path.name})"
    )

    if push_hub:
        _push_to_hub(output_path, hub_repo, hub_private)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=_DEFAULT_INPUT,
        help=f"Combined CUAD+kvp10k parquet (default: {_DEFAULT_INPUT})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Output parquet for CUAD-only rows (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="After extracting, upload the parquet to the HF Hub as a dataset "
        "(auth via HF_TOKEN env or `huggingface-cli login`)",
    )
    parser.add_argument(
        "--hub-repo",
        default=_HUB_REPO,
        help=f"HF Hub dataset repo to push to; bare name uses your namespace, "
        f"or pass org/name (default: {_HUB_REPO})",
    )
    parser.add_argument(
        "--hub-private",
        action="store_true",
        help="Create/push the HF Hub dataset repo as private",
    )
    args = parser.parse_args()
    _run(
        input_path=args.input,
        output_path=args.output,
        push_hub=args.push_to_hub,
        hub_repo=args.hub_repo,
        hub_private=args.hub_private,
    )


if __name__ == "__main__" and "__file__" in globals():
    main()
