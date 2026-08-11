"""Build the `kvp10k-templates` VQA dataset: synthetic filled forms → unified schema.

Sibling of ``vqa_20260725_form_like.py``, but sourced from the **reworked
KVP10k template corpus** — the clean, fixed 10-class taxonomy regenerated under
``/data/paperwerk/assets/templates`` (invoice, tax_form, application_form,
datasheet, …) rather than the sprawling open-ended class set of the form-like
build. Everything else is the same: it renders a synthetic document from a
random template (filling every field with LLM-invented values), rasterizes each
page, and emits one datapoint per document in the shared VQA schema used by the
other builders (``vqa_20260524_kvp10k.py`` / ``vqa_20260524_cuad.py``), so the
parquets are mixable::

    images       <- every rendered page of the document (image bytes)
    queries      <- the datapoint's query strings, unique, first-appearance order
    answers      <- one per filled occurrence of a *selected* field:
                       query        = the field's query string
                       value        = synthesized value (as rendered)
                       bounding_box = [x0, y0, x1, y1], per-page, 0..1, top-left
                       index        = 0-based page index within `images`
    source       <- "<class>/<template>#<n>"
    variant      <- "kvp10k-templates"
    page_start   <- 0
    page_end     <- n_pages - 1
    split        <- --split (default "train")

Queries are not simply the field names. For each document the LLM selects
30-90% of its fields — the ones a user or a commercial extraction application
would actually want — and writes a query for each in ONE format chosen per
datapoint (never mixed within a datapoint): the field name (``account_number``),
the exact structured field name kept verbatim (``person[].first_name``), a short
search-box phrase (``account number``), or a full question (``What is the
account number?``). Only the selected fields appear in ``queries``/``answers``;
array-field cells share one logical query.

The classes drawn from are the ``CATEGORIES`` constant below — the current
template taxonomy. **Edit that list freely** to include/exclude categories; the
builder samples a random valid template from the listed categories at run time
(skipping parse-broken or near-empty templates).

Value generation runs on either a **locally-running Gemma-4** (vLLM, default)
or **Claude Sonnet 4.6** (``--backend sonnet``, needs ``ANTHROPIC_API_KEY``).
Signature-named fields are rendered as procedural SVG scrawls
(``paperwerk.datagen.signatures``) and excluded from queries/answers. With
``--augment-ratio`` a fraction of documents get scanner/phone augmentation.

Prerequisite for the default backend — Gemma-4 served locally by vLLM::

    docker run --gpus all --shm-size=16g -p 8000:8000 paperwerk-serve:latest \\
      /venv/bin/vllm serve google/gemma-4-E4B-it --max-model-len 65536 \\
      --enable-auto-tool-choice --tool-call-parser gemma4 --tensor-parallel-size 2

Output is a resumable *directory* of one-row parquet shards (``part_NNNNNN.parquet``,
one per doc). A completed shard is the cached result, so re-running the same
command skips finished slots and fills only the missing ones — safe to interrupt
and restart. Readers (pyarrow / HF ``datasets``) accept the directory directly;
pass ``--merge`` (or ``--merge-only``) to concatenate into a single parquet.

Run::

    python -m paperwerk.datagen.builders.vqa_20260727_kvp10k_templates -n 200
    python -m paperwerk.datagen.builders.vqa_20260727_kvp10k_templates -n 500 \\
        --augment-ratio 0.5 --merge
    python -m paperwerk.datagen.builders.vqa_20260727_kvp10k_templates -n 100 \\
        --backend sonnet --no-signatures -o /data/paperwerk/vqa_kvp10k_templates
    # re-run the SAME command after an interruption -> resumes; or just merge:
    python -m paperwerk.datagen.builders.vqa_20260727_kvp10k_templates \\
        --merge-only -o /data/paperwerk/vqa_kvp10k_templates
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
import sys
import traceback
from io import BytesIO
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from jinja2 import Environment
from pdf2image import convert_from_bytes

from paperwerk.async_utils import gather_limited
from paperwerk.llm import LLM

from paperwerk.datagen.augment import PROFILES, augment, augment_geometric
from paperwerk.datagen.render import render_sync
from paperwerk.datagen.signatures import inject_signatures
from paperwerk.datagen.templates import discover_fields
from paperwerk.datagen.values import random_values

# ---------------------------------------------------------------------------
# Template categories (EDITABLE).
#
# The reworked KVP10k corpus uses a fixed 10-class taxonomy (see
# templates_20260705_forms_kvp10k.py::DOCUMENT_TYPES). Each entry maps to a
# subdirectory of the templates root. Add or remove entries as you see fit;
# categories with no directory on disk are simply skipped with a warning.
# ---------------------------------------------------------------------------
CATEGORIES: list[str] = [
    "application_form",
    "authorization_form",
    "certificate",
    "datasheet",
    # "financial_statement",
    "invoice",
    "registration_form",
    "request_form",
    "tax_form",
    "worksheet",
]

# ---------------------------------------------------------------------------
# Value-generation backends. vLLM (local Gemma-4) is the default; Sonnet 4.6
# via Anthropic's OpenAI-compatible endpoint is available with --backend sonnet.
# vLLM ignores the API key but the OpenAI client requires a non-empty value.
# ---------------------------------------------------------------------------
_VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1/")
_VLLM_MODEL = os.environ.get("VLLM_MODEL", "google/gemma-4-E4B-it")
_ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1/"
_SONNET_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

_TEMPLATES_ROOT = Path(
    os.environ.get("PAPERWERK_TEMPLATES_ROOT", "/data/paperwerk/assets/templates")
)
_DEFAULT_OUTPUT = Path("/data/paperwerk/vqa_20260727_kvp10k_templates")  # shard dir
_VARIANT = "kvp10k-templates"

# ---------------------------------------------------------------------------
# Query generation. For each datapoint the LLM (a) selects 30-90% of the
# document's fields — the ones a user or a commercial extraction application
# would actually want to pull out — and (b) writes a query for each in ONE
# format chosen for the whole datapoint (never mixed within a datapoint):
#   field_name           -> the field name, LLM-echoed        ("account_number")
#   structured_field_name-> the exact logical field name, DETERMINISTIC, keeping
#                           `[]`/`.` structure                 ("person[].first_name")
#   short_query          -> a short search-box phrase          ("account number")
#   question             -> a full natural-language question   ("What is the account number?")
#
# For `structured_field_name` the LLM still SELECTS the subset, but the query
# string is the field name itself (never phrased by the model), so it can't be
# mangled and array structure is preserved verbatim.
# ---------------------------------------------------------------------------
_QUERY_FORMATS = ("field_name", "structured_field_name", "short_query", "question")
_QUERY_FRAC_MIN = 0.30
_QUERY_FRAC_MAX = 0.90

_FORMAT_DESC = {
    "field_name": (
        "the field name EXACTLY as given, unchanged (snake_case, with any `[]` "
        'and `.` intact), e.g. "account_number", "first_name"'
    ),
    "structured_field_name": (
        "the field name EXACTLY as given, unchanged (it will be used verbatim), "
        'e.g. "first_name", "person[].first_name", "address"'
    ),
    "short_query": (
        "a short keyword phrase like a user would type into a search box: "
        'lowercase words, no punctuation, e.g. "account number", "first name"'
    ),
    "question": (
        "a complete natural-language question, optionally capitalized and ending with '?', "
        'e.g. "What is the account number?", "Who is applying?", "How much is to pay"'
    ),
}

_QUERY_GEN_SCHEMA = {
    "title": "SelectedQueries",
    "type": "object",
    "properties": {
        "selected": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "field": {"type": "string"},
                    "query": {"type": "string"},
                },
                "required": ["field", "query"],
            },
        }
    },
    "required": ["selected"],
}

# Rendered field names carry a concrete row index (`items[0].qty`); the schema /
# value layer uses the logical `[]` form (`items[].qty`). Selection happens on
# the logical field, so every cell of an array field shares one query.
_INDEX_RE = re.compile(r"\[\d+\]")

# Same unified schema as vqa_20260524_kvp10k.py so the parquets are mixable.
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

_MAX_ATTEMPTS = 12  # per-slot template retries before giving up on that slot


def _make_llm(backend: str, concurrency: int) -> LLM:
    """Construct the value-generation LLM for the chosen backend."""
    if backend == "sonnet":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            sys.exit("error: --backend sonnet requires ANTHROPIC_API_KEY")
        return LLM(
            base_url=_ANTHROPIC_BASE_URL,
            api_key=api_key,
            model=_SONNET_MODEL,
            max_concurrency=concurrency,
        )
    return LLM(
        base_url=_VLLM_BASE_URL,
        api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
        model=_VLLM_MODEL,
        max_concurrency=concurrency,
    )


# A Jinja dict method (`.items`/`.values`/`.keys`/…) accessed WITHOUT calling it,
# inside a `{{ }}`/`{% %}` block: `{% for v in row.values %}` (missing `()`).
# `row` is a dict at render time, so `.values` is the bound method → the template
# always fails with "'builtin_function_or_method' object is not iterable". These
# are deterministically broken as generated, so drop them from the pool.
_BROKEN_JINJA_RE = re.compile(
    r"\{[{%][^}]*\.(?:items|values|keys|get|pop|update|setdefault)\b(?!\s*\()"
)


def build_template_pool(min_fields: int) -> list[tuple[str, Path]]:
    """Resolve CATEGORIES to concrete, usable (class, template) pairs."""
    env = Environment(autoescape=True)
    pool: list[tuple[str, Path]] = []
    missing: list[str] = []
    broken = 0
    for cls in CATEGORIES:
        class_dir = _TEMPLATES_ROOT / cls
        if not class_dir.is_dir():
            missing.append(cls)
            continue
        for tpl in sorted(class_dir.glob("*.html.j2")):
            if not tpl.with_suffix("").with_suffix(".json").exists():
                continue
            text = tpl.read_text()
            try:
                env.parse(text)
            except Exception:
                continue  # skip malformed-as-generated templates
            if _BROKEN_JINJA_RE.search(text):
                broken += 1
                continue  # skip templates with the uncalled-dict-method bug
            if len(discover_fields(text)) < min_fields:
                continue  # skip near-blank templates
            pool.append((cls, tpl))
    if missing:
        print(
            f"warning: {len(missing)} listed categor(y/ies) not found under "
            f"{_TEMPLATES_ROOT}: {missing}"
        )
    if broken:
        print(f"skipped {broken} template(s) with the uncalled-dict-method bug")
    return pool


def _augment_plan(rng: random.Random) -> dict:
    return {
        "profile": rng.choice(PROFILES),
        "quality": round(rng.uniform(0.25, 0.9), 3),
        "geometric": rng.random() < 0.5,
        "geo_quality": round(rng.uniform(0.4, 0.9), 3),
    }


def _encode(img, image_format: str, jpeg_quality: int) -> bytes:
    buf = BytesIO()
    rgb = img.convert("RGB")
    if image_format == "jpeg":
        rgb.save(buf, format="JPEG", quality=jpeg_quality)
    else:
        rgb.save(buf, format="PNG")
    return buf.getvalue()


def _render_to_datapoint(
    template_html: str,
    data: dict,
    plan: dict | None,
    *,
    seed: int,
    dpi: int,
    image_format: str,
    jpeg_quality: int,
) -> tuple[list[bytes], list, bool]:
    """Sync CPU stage: render (+optional augment) → (page images, fields, aug_ok).

    Augmentation is best-effort: augraphy occasionally trips an internal numba
    parfor bug (`AssertionError` in numba type inference) on certain images. If
    that happens we keep the CLEAN render rather than discarding the whole doc
    (and the value-gen call already spent on it). `aug_ok` reports whether the
    requested augmentation was actually applied.
    """
    pdf, fields = render_sync(template_html, data, seed=seed)
    aug_ok = False
    if plan is not None:
        try:
            if plan["geometric"]:
                pdf, fields = augment_geometric(
                    pdf, fields, quality=plan["geo_quality"], seed=seed
                )
            pdf, fields = augment(
                pdf, fields, profile=plan["profile"], quality=plan["quality"], seed=seed
            )
            aug_ok = True
        except Exception as e:  # noqa: BLE001 - augraphy/numba flakiness -> clean
            print(
                f"  augment failed ({plan['profile']}: {type(e).__name__}); using clean render"
            )
            pdf, fields = render_sync(template_html, data, seed=seed)
    pages = convert_from_bytes(pdf, dpi=dpi)
    images = [_encode(p, image_format, jpeg_quality) for p in pages]
    return images, fields, aug_ok


def _logical(name: str) -> str:
    """Collapse a rendered field name's row indices to the logical `[]` form."""
    return _INDEX_RE.sub("[]", name)


def _mechanical_query(field: str, fmt: str) -> str:
    """Format-consistent query for a field without an LLM (top-up fallback)."""
    if fmt in ("field_name", "structured_field_name"):
        return field
    words = " ".join(
        field.replace("[]", "").replace(".", " ").replace("_", " ").split()
    )
    if fmt == "short_query":
        return words.lower()
    return f"What is the {words.lower()}?"


def _distinct_filled_logical(fields: list, n_pages: int) -> list[str]:
    """Distinct logical field names with a rendered value, first-appearance order."""
    seen: set[str] = set()
    out: list[str] = []
    for f in fields:
        if not f.value.strip():
            continue  # skip empty / signature (SVG) fields
        if not (0 <= f.page < n_pages):
            continue
        lg = _logical(f.name)
        if lg not in seen:
            seen.add(lg)
            out.append(lg)
    return out


async def generate_queries(
    llm: LLM,
    category: str,
    fields: list[str],
    fmt: str,
    rng: random.Random,
    *,
    temperature: float,
) -> dict[str, str]:
    """LLM-select 30-90% of `fields` and write a query per field in `fmt`.

    Returns a {logical_field: query} map. Raises on unparseable output so the
    caller can retry. If the model returns too few/many, the selection is
    clamped into the 30-90% band (topping up with random fields if needed).
    """
    n = len(fields)
    lo = max(1, math.ceil(_QUERY_FRAC_MIN * n))
    hi = max(lo, math.floor(_QUERY_FRAC_MAX * n))
    listing = "\n".join(f"- {f}" for f in fields)
    prompt = (
        f"A user or a commercial data-extraction application is processing a "
        f"'{category}' document. These are the fields present in it "
        f"(snake_case; `[]` marks a repeated/array field, `.` marks nesting):\n\n"
        f"{listing}\n\n"
        f"Select between {lo} and {hi} of these fields — the ones a user or such an "
        f"application would most plausibly want to extract (identifiers, names, "
        f"dates, amounts, statuses, key attributes). Skip decorative, boilerplate, "
        f"or purely structural fields.\n\n"
        f"For each selected field, write `query` as {_FORMAT_DESC[fmt]}.\n"
        f"Set `field` to the EXACT string from the list above — never invent or "
        f"alter a field name."
    )
    # Scale the token budget to the field count: the JSON holds up to `hi`
    # {field, query} pairs, so a flat 4096 truncates field-heavy docs mid-string.
    max_tokens = min(20000, 2048 + 80 * n)
    resp = await llm.ainvoke(
        messages=[{"role": "user", "content": prompt}],
        schema=_QUERY_GEN_SCHEMA,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = resp.choices[0].message.content
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"query selection returned invalid JSON ({e})") from e

    valid = set(fields)
    out: dict[str, str] = {}
    for item in data.get("selected", []):
        field = item.get("field")
        if field not in valid or field in out:
            continue
        if fmt == "structured_field_name":
            out[field] = field  # deterministic: use the field name verbatim
            continue
        query = (item.get("query") or "").strip()
        if query:
            out[field] = query

    if len(out) > hi:  # clamp down (keep first hi in the model's order)
        out = dict(list(out.items())[:hi])
    if len(out) < lo:  # top up to the 30% floor with random remaining fields
        print("Mechanical query!")
        remaining = [f for f in fields if f not in out]
        rng.shuffle(remaining)
        for f in remaining[: lo - len(out)]:
            out[f] = _mechanical_query(f, fmt)
    return out


def _to_datapoint(
    images: list[bytes],
    fields: list,
    query_map: dict[str, str],
    *,
    source: str,
    split: str,
) -> dict | None:
    """Assemble a unified VQA datapoint; None if no selected field was filled.

    Only fields in `query_map` (the LLM-selected subset, keyed by logical name)
    become answers; each occurrence contributes one answer with the datapoint's
    chosen query string and its own per-page bounding box.
    """
    n_pages = len(images)
    answers: list[dict] = []
    queries: list[str] = []
    for f in fields:
        if not f.value.strip():
            continue  # skip empty / signature (SVG) fields
        if not (0 <= f.page < n_pages):
            continue
        query = query_map.get(_logical(f.name))
        if query is None:
            continue  # field not selected for extraction
        if query not in queries:
            queries.append(query)
        answers.append(
            {
                "query": query,
                "value": f.value,
                "bounding_box": [float(c) for c in f.bbox],
                "index": int(f.page),
            }
        )
    if not answers:
        return None
    return {
        "images": images,
        "queries": queries,
        "answers": answers,
        "source": source,
        "variant": _VARIANT,
        "page_start": 0,
        "page_end": n_pages - 1,
        "split": split,
    }


# ---------------------------------------------------------------------------
# Manual caching. The output is a *directory* of one-row parquet shards, one
# per slot: ``part_{slot:06d}.parquet``. A completed shard IS the persisted
# result, so a re-run simply skips slots whose shard already exists — LLM calls
# and rendering happen only for the missing slots. Each shard is written to a
# ``.tmp`` sibling and atomically renamed, so an interrupted run never leaves a
# partial/corrupt shard behind. ``_merge_shards`` concatenates them into a
# single parquet when a one-file artifact is wanted.
# ---------------------------------------------------------------------------
def _shard_path(out_dir: Path, slot: int) -> Path:
    return out_dir / f"part_{slot:06d}.parquet"


def _write_shard(path: Path, dp: dict) -> None:
    """Write a single-row shard atomically (temp file + rename)."""
    table = pa.Table.from_pylist([dp], schema=_PARQUET_SCHEMA)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, str(tmp))
    tmp.replace(path)  # atomic on POSIX


def _existing_slots(out_dir: Path) -> set[int]:
    """Slot indices already persisted as shards in `out_dir`."""
    slots: set[int] = set()
    for p in out_dir.glob("part_*.parquet"):
        try:
            slots.add(int(p.stem.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return slots


def _merge_shards(out_dir: Path, merged_path: Path) -> int:
    """Concatenate all shards in `out_dir` into a single parquet; returns count."""
    shards = sorted(out_dir.glob("part_*.parquet"))
    if not shards:
        print(f"merge: no shards found in {out_dir}")
        return 0
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(str(merged_path), _PARQUET_SCHEMA)
    try:
        for s in shards:
            writer.write_table(pq.read_table(s))
    finally:
        writer.close()
    print(f"merge: {len(shards)} shard(s) -> {merged_path}")
    return len(shards)


async def _build(
    *,
    limit: int = 100,
    output: Path = _DEFAULT_OUTPUT,
    backend: str = "vllm",
    augment_ratio: float = 0.0,
    signatures: bool = True,
    split: str = "train",
    seed: int = 20260727,
    dpi: int = 150,
    image_format: str = "png",
    jpeg_quality: int = 92,
    min_fields: int = 4,
    query_temperature: float = 0.7,
    concurrency: int = 10,
    cpu_concurrency: int = 4,
    merge: bool = False,
    merge_output: Path | None = None,
    merge_only: bool = False,
) -> None:
    """Generate the dataset. All parameters are keyword-only.

    `output` is a shard directory (one parquet per doc; resumable). When
    `merge_output` is None it defaults to ``<output>.parquet``.
    """
    if not 0.0 <= augment_ratio <= 1.0:
        raise ValueError("augment_ratio must be in [0, 1]")
    output = Path(output)
    merge_output = (
        output.with_name(output.name + ".parquet")
        if merge_output is None
        else Path(merge_output)
    )

    if merge_only:
        _merge_shards(output, merge_output)
        return

    pool = build_template_pool(min_fields)
    if not pool:
        sys.exit(
            f"error: no usable templates resolved from CATEGORIES under {_TEMPLATES_ROOT}"
        )
    print(
        f"pool: {len(pool)} template(s) across "
        f"{len({c for c, _ in pool})} categor(y/ies)"
    )

    rng = random.Random(seed)
    # Per-slot augmentation flags: exactly round(N * ratio) augmented, shuffled.
    n_aug = round(limit * augment_ratio)
    aug_flags = [True] * n_aug + [False] * (limit - n_aug)
    rng.shuffle(aug_flags)

    cpu_sem = asyncio.Semaphore(cpu_concurrency)

    llm = _make_llm(backend, concurrency)
    print(f"value generation via {llm!r} (backend={backend})")

    output.mkdir(parents=True, exist_ok=True)
    # Clear any leftover temp shards from a previously-killed run.
    for stale in output.glob("part_*.parquet.tmp"):
        stale.unlink()
    resumed = _existing_slots(output)
    if resumed:
        print(f"resuming: {len(resumed)} of {limit} shard(s) already present")
    done = 0

    async def _slot(slot: int) -> None:
        nonlocal done
        shard = _shard_path(output, slot)
        if slot in resumed:
            return  # already cached — skip all LLM/render work
        augmented = aug_flags[slot]
        slot_rng = random.Random(f"{seed}:{slot}")
        render_seed = seed + slot
        # Sample templates WITH replacement (a run of N docs typically far
        # exceeds the pool size); each pick gets fresh values + seed, so reuses
        # still differ. Each slot retries a bounded number of times to skip
        # templates that fail to render before giving up.
        for i in range(_MAX_ATTEMPTS):
            cls, tpl_path = slot_rng.choice(pool)
            print(f"[slot {slot:05d}] starting on class {cls}, template {tpl_path} ")
            try:
                template_html = tpl_path.read_text()
                schema = discover_fields(template_html)
                if signatures:
                    template_html, sig_fields = inject_signatures(
                        template_html, schema, slot_rng
                    )
                else:
                    sig_fields = []
                synth_fields = [f for f in schema if f not in sig_fields]
                if not synth_fields:
                    continue
                data = await random_values(llm, synth_fields, hint_rng=slot_rng)
                plan = _augment_plan(slot_rng) if augmented else None
                async with cpu_sem:
                    images, fields, aug_ok = await asyncio.to_thread(
                        _render_to_datapoint,
                        template_html,
                        data,
                        plan,
                        seed=render_seed,
                        dpi=dpi,
                        image_format=image_format,
                        jpeg_quality=jpeg_quality,
                    )
                # LLM picks the extractable subset and writes queries for it in
                # one format chosen for this whole datapoint.
                filled = _distinct_filled_logical(fields, len(images))
                if not filled:
                    print(f"[slot {slot:05d}] {cls} rendered no fields; retrying")
                    continue
                fmt = slot_rng.choice(_QUERY_FORMATS)
                query_map = await generate_queries(
                    llm, cls, filled, fmt, slot_rng, temperature=query_temperature
                )
            except Exception as e:  # noqa: BLE001 - retry with next template
                # Full traceback so opaque errors (e.g. bare AssertionError from
                # a bad template) are identifiable; one block per failed attempt.
                print(
                    f"[slot {slot:05d}] {cls} failed on {tpl_path.name} "
                    f"({type(e).__name__}: {e}); retrying\n"
                    + traceback.format_exc().rstrip()
                )
                continue

            source = f"{cls}/{tpl_path.stem.replace('.html', '')}#{slot:05d}"
            dp = _to_datapoint(images, fields, query_map, source=source, split=split)
            if dp is None:
                print(f"[slot {slot:05d}] {cls} produced no answers; retrying")
                continue
            await asyncio.to_thread(_write_shard, shard, dp)
            done += 1
            if plan and aug_ok:
                tag = f"aug:{plan['profile']}" + ("+geo" if plan["geometric"] else "")
            elif plan:
                tag = "clean(aug-failed)"
            else:
                tag = "clean"
            print(
                f"[slot {slot:05d}] ({done} new / {limit}) {cls} "
                f"[{tag}, fmt:{fmt}, {len(dp['images'])}p, "
                f"{len(dp['queries'])}/{len(filled)} fields, {len(dp['answers'])} qa]"
            )
            return
        print(f"[slot {slot:05d}] gave up after {_MAX_ATTEMPTS} failed attempt(s)")

    await gather_limited((_slot(i) for i in range(limit)), concurrency)

    total = len(_existing_slots(output))
    print(f"\ndone: wrote {done} new shard(s); {total}/{limit} total in {output}")
    if merge:
        _merge_shards(output, merge_output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=100,
        help="Number of documents to generate (default: 100)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=(
            "Output shard DIRECTORY (one parquet per doc; resumable). "
            f"Default: {_DEFAULT_OUTPUT}"
        ),
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="After generating, concatenate all shards into a single parquet",
    )
    parser.add_argument(
        "--merge-output",
        type=Path,
        default=None,
        help="Merged single-file path (default: <output>.parquet)",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Skip generation; just merge existing shards in --output and exit",
    )
    parser.add_argument(
        "--backend",
        choices=("vllm", "sonnet"),
        default="vllm",
        help="Value-generation backend (default: vllm / local Gemma-4)",
    )
    parser.add_argument(
        "--augment-ratio",
        type=float,
        default=0.0,
        help="Fraction of docs to scanner/photo-augment (default: 0.0)",
    )
    sig = parser.add_mutually_exclusive_group()
    sig.add_argument(
        "--signatures",
        dest="signatures",
        action="store_true",
        help="Render signature fields as SVG scrawls (default)",
    )
    sig.add_argument(
        "--no-signatures",
        dest="signatures",
        action="store_false",
        help="Leave signature fields as plain text values",
    )
    parser.set_defaults(signatures=True)
    parser.add_argument("--split", default="train", help="Split label (default: train)")
    parser.add_argument("--seed", type=int, default=20260727, help="RNG seed")
    parser.add_argument(
        "--dpi", type=int, default=150, help="Page raster DPI (default: 150)"
    )
    parser.add_argument(
        "--image-format",
        choices=("png", "jpeg"),
        default="png",
        help="Encoding for page images (default: png)",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=92,
        help="JPEG quality when --image-format jpeg (default: 92)",
    )
    parser.add_argument(
        "--min-fields",
        type=int,
        default=4,
        help="Skip templates with fewer discoverable fields (default: 4)",
    )
    parser.add_argument(
        "--query-temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for query selection/phrasing (default: 0.7)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Max concurrent value-gen requests (default: 10)",
    )
    parser.add_argument(
        "--cpu-concurrency",
        type=int,
        default=4,
        help="Max concurrent render/augment jobs (default: 4)",
    )
    args = parser.parse_args()
    if not 0.0 <= args.augment_ratio <= 1.0:
        sys.exit("error: --augment-ratio must be in [0, 1]")
    # argparse dest names match _build's keyword-only parameters one-to-one.
    asyncio.run(_build(**vars(args)))


if __name__ == "__main__" and "__file__" in globals():
    main()
