"""Generate version-controlled Jinja2 templates from the KVP10k dataset (per-document).

kvp10k stores one row per annotated page but its ``image_url`` field points
at the underlying multi-page source PDF; rows sharing a URL are pages of the
same document. This builder groups rows by ``image_url`` so every LLM call
sees the full document rather than a single page.

Pipeline: rows are grouped by ``image_url`` and pre-filtered to URLs with
``<= max_pages_per_doc`` annotated rows, then every document flows through
the full chain *independently and concurrently* — so template generation
(GPU/LLM) starts on the first downloaded PDF while the rest are still
downloading, keeping the GPU busy instead of idling through a global
download barrier. Per document:

  0a. ``download_pdf``   — fetch + cache the source PDF (retries=2).
  0b. ``get_page_count`` — cheap ``pdfinfo`` check; drop long-tail docs.
  1.  ``classify_pdf``   — first 3 PDF pages → one of the fixed
                            ``DOCUMENT_TYPES`` (or ``"other"``, which is
                            skipped), cached in
                            ``/data/paperwerk/cache/kvp10k/docs.json``. Cached
                            labels outside the current taxonomy are re-classified.
  2a. Per-class phash dedup on the first page, applied incrementally under a
      per-class lock as documents arrive. Because documents complete in
      download/LLM order rather than a fixed order, which member of a set of
      near-duplicates "wins" is no longer deterministic across runs — an
      accepted trade-off for overlapping download and generation.
  2b. ``build_template`` — full-document Jinja HTML + schema via
                            ``datagen.templates.make_template`` (which now
                            auto-chunks long docs internally, retries=1).
  3.  ``_persist``       — write ``<class>/kvp10k_<url_hash12>.html.j2`` +
                            sidecar under
                            ``src/datagen/assets/templates/``.

Concurrency is bounded per stage: downloads by ``_DOWNLOAD_CONCURRENCY``
(``_DL_SEM``) and LLM calls by the ``LLM`` client's ``max_concurrency``.

Running this triggers a Prefect flow. To see live per-task progress, start
a local Prefect server in another shell first::

    prefect server start
    prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api  # once

Then::

    from paperwerk.datagen.builders.templates_20260705_forms_kvp10k import main
    main()                                # local vLLM (default), full split
    main(backend="sonnet")                # Claude Sonnet 4.6 (needs ANTHROPIC_API_KEY)
    main(backend="sonnet", limit=200)     # first 200 unique URLs on Sonnet
    main(max_pages_per_doc=64)            # push the page cap higher
    main(limit=200, hamming_threshold=6)  # stricter dedup

Classification is restricted to the ``DOCUMENT_TYPES`` constant plus
``"other"`` (skipped) — edit that list to change the taxonomy.

The flow runs correctly without a server too — you'll just see per-task
progress in the console via Prefect's default logger instead of in the UI.
Templates whose target file already exists are skipped, so a re-run resumes
where the last one stopped without re-hitting the LLM.
"""

from __future__ import annotations

import asyncio
import hashlib
import http.client
import json
import os
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from pdf2image import convert_from_bytes, pdfinfo_from_path
from PIL import Image
from prefect import flow, get_run_logger, task

from paperwerk.llm import LLM

from paperwerk.datagen.classifier import classify
from paperwerk.datagen.templates import TemplateDeduper, make_template

_HF_DATASET = "ibm-research/KVP10k"
_HF_SPLIT = "train"

# ---------------------------------------------------------------------------
# LLM backend. Choose at run time via `backend=` ("vllm" | "sonnet"):
#   * "vllm"   — local Gemma served by vLLM on the host (default). We run inside
#     a Docker container on the host network, so the host's localhost is
#     reachable directly; override with VLLM_BASE_URL. vLLM ignores the API key,
#     but the OpenAI client still requires a non-empty value.
#   * "sonnet" — Anthropic's Claude Sonnet 4.6 via the OpenAI-compatible
#     endpoint; needs ANTHROPIC_API_KEY.
# ---------------------------------------------------------------------------
_VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1/")
_VLLM_MODEL = os.environ.get("VLLM_MODEL", "google/gemma-4-E4B-it")
_ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1/"
_SONNET_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
# _DEFAULT_BACKEND = "vllm"
_DEFAULT_BACKEND = "sonnet"

_LLM_CONCURRENCY = 10
_DOWNLOAD_CONCURRENCY = 16
_SOURCE = "kvp10k"

# ---------------------------------------------------------------------------
# Fixed classification vocabulary (EDITABLE).
#
# The classifier is constrained to exactly these snake_case types plus
# "other"; any document that doesn't fit is labelled "other" and skipped, so
# the corpus stays to a small, curated set of form-like types instead of the
# open-ended long tail. Add/remove types freely.
# ---------------------------------------------------------------------------
DOCUMENT_TYPES: list[str] = [
    "invoice",
    "application_form",
    "tax_form",
    "datasheet",
    "request_form",
    "authorization_form",
    "registration_form",
    "certificate",
    "worksheet",
    "financial_statement",
]
_OTHER = "other"
_CLASSIFY_CHOICES = DOCUMENT_TYPES + [_OTHER]  # what the classifier may return
_ALLOWED_TYPES = set(DOCUMENT_TYPES)  # kept types (everything else skipped)
_VALID_CLASSES = _ALLOWED_TYPES | {_OTHER}  # valid results under this taxonomy

_TEMPLATES_ROOT = Path("/data") / "paperwerk" / "assets" / "templates"
_CACHE_ROOT = Path("/data/paperwerk/cache/kvp10k")
_PDF_CACHE_DIR = _CACHE_ROOT / "pdfs"
_DOCS_CACHE_PATH = _CACHE_ROOT / "docs.json"

_DOWNLOAD_TIMEOUT = 60.0
_RENDER_DPI = 150
_MAX_FIRST_PAGE_SIDE = 1600
_MAX_PAGES_PER_DOC = 32

_DEFAULT_HAMMING_THRESHOLD = 8
_DEFAULT_HASH_SIZE = 8


# ---------------------------------------------------------------------------
# Module-level singletons
#
# The LLM client and the download semaphore both hold non-serialisable state.
# Passing them into a Prefect task as arguments would show up in task inputs
# / state and force serialization. Tasks fetch them from module scope
# instead.
# ---------------------------------------------------------------------------

_LLM: LLM | None = None
_BACKEND: str = _DEFAULT_BACKEND  # set by `main()` / the flow before tasks fan out
_DL_SEM = asyncio.Semaphore(_DOWNLOAD_CONCURRENCY)


def _get_llm() -> LLM:
    global _LLM
    if _LLM is None:
        if _BACKEND == "sonnet":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("backend='sonnet' requires ANTHROPIC_API_KEY")
            _LLM = LLM(
                base_url=_ANTHROPIC_BASE_URL,
                api_key=api_key,
                model=_SONNET_MODEL,
                max_concurrency=_LLM_CONCURRENCY,
            )
        else:
            # vLLM doesn't authenticate; any non-empty key satisfies the client.
            _LLM = LLM(
                base_url=_VLLM_BASE_URL,
                api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
                model=_VLLM_MODEL,
                max_concurrency=_LLM_CONCURRENCY,
            )
    return _LLM


# ---------------------------------------------------------------------------
# Plain helpers (no Prefect awareness)
# ---------------------------------------------------------------------------


def _url_hash(url: str) -> str:
    """Short stable filesystem-safe hash of a URL."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]


def _load_docs_cache() -> dict[str, dict]:
    if not _DOCS_CACHE_PATH.exists():
        return {}
    try:
        return json.loads(_DOCS_CACHE_PATH.read_text())
    except json.JSONDecodeError:
        return {}


def _save_docs_cache(cache: dict[str, dict]) -> None:
    _DOCS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _DOCS_CACHE_PATH.with_suffix(_DOCS_CACHE_PATH.suffix + ".tmp")
    tmp.write_text(json.dumps(cache, indent=2, sort_keys=True))
    tmp.replace(_DOCS_CACHE_PATH)


def _download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "paperwerk/0.1"})
    with urllib.request.urlopen(req, timeout=_DOWNLOAD_TIMEOUT) as resp:
        return resp.read()


def _pdf_page_count(pdf_path: Path) -> int:
    return pdfinfo_from_path(str(pdf_path))["Pages"]


def _shrink(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    s = max_side / max(w, h)
    return img.resize((round(w * s), round(h * s)), Image.LANCZOS)


def _first_page(pdf_path: Path) -> Image.Image | None:
    pages = convert_from_bytes(
        pdf_path.read_bytes(), dpi=_RENDER_DPI, first_page=1, last_page=1
    )
    if not pages:
        return None
    return _shrink(pages[0].convert("RGB"), _MAX_FIRST_PAGE_SIDE)


def _persist(templates_root: Path, r: dict, url: str, page_count: int) -> Path:
    class_dir = templates_root / r["class"]
    class_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{_SOURCE}_{r['url_hash']}"
    j2 = class_dir / f"{stem}.html.j2"
    meta = class_dir / f"{stem}.json"
    j2.write_text(r["template"])
    meta.write_text(
        json.dumps(
            {
                "url_hash": r["url_hash"],
                "image_url": url,
                "class": r["class"],
                "source": _SOURCE,
                "page_count": page_count,
                "schema": r["schema"],
            },
            indent=2,
        )
    )
    return j2


# ---------------------------------------------------------------------------
# Prefect tasks
# ---------------------------------------------------------------------------


@task(
    name="download-pdf",
    task_run_name="download {url_hash}",
    retries=2,
    retry_delay_seconds=10,
)
async def download_pdf(url_hash: str, url: str) -> Path | None:
    """Fetch and cache one source PDF; return local path or None on failure."""
    logger = get_run_logger()
    cached = _PDF_CACHE_DIR / f"{url_hash}.pdf"
    if cached.exists():
        return cached
    async with _DL_SEM:
        try:
            data = await asyncio.to_thread(_download, url)
        except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
            logger.warning(f"download failed {url}: {e!r}")
            return None
    if not data.lstrip().startswith(b"%PDF"):
        logger.warning(f"not a PDF {url} (magic={data[:8]!r})")
        return None
    _PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(cached.write_bytes, data)
    logger.info(f"cached {len(data)/1024:.0f} KB")
    return cached


@task(name="pdf-page-count", task_run_name="page-count {url_hash}")
async def get_page_count(url_hash: str, pdf_path: Path) -> int:
    del url_hash  # only used by task_run_name for the UI label
    return await asyncio.to_thread(_pdf_page_count, pdf_path)


@task(
    name="classify-pdf",
    task_run_name="classify {url_hash}",
    retries=1,
    retry_delay_seconds=5,
)
async def classify_pdf(url_hash: str, pdf_path: Path) -> str | None:
    logger = get_run_logger()
    try:
        cls = await classify(_get_llm(), str(pdf_path), classes=_CLASSIFY_CHOICES)
    except Exception as e:
        logger.error(f"[{url_hash}] classify failed: {e!r}")
        return None
    logger.info(f"[{url_hash}] -> {cls}")
    return cls


@task(
    name="build-template",
    task_run_name="template {url_hash} ({cls})",
    retries=1,
    retry_delay_seconds=5,
)
async def build_template(url_hash: str, cls: str, pdf_path: Path) -> dict | None:
    logger = get_run_logger()
    try:
        template, schema = await make_template(_get_llm(), str(pdf_path))
    except Exception as e:
        logger.error(f"make_template failed: {e!r}")
        return None
    logger.info(f"{len(schema)} fields, {len(template)} bytes")
    return {
        "url_hash": url_hash,
        "class": cls,
        "template": template,
        "schema": schema,
    }


# ---------------------------------------------------------------------------
# Prefect flow
# ---------------------------------------------------------------------------


# Flush the classification cache to disk every N new entries rather than on
# every call: rewriting the whole JSON per classification would be O(n^2)
# I/O over a long run, and the flow flushes any remainder on exit.
_CACHE_FLUSH_EVERY = 25


@flow(name="kvp10k-templates")
async def kvp10k_templates(
    limit: int = 0,
    hamming_threshold: int = _DEFAULT_HAMMING_THRESHOLD,
    hash_size: int = _DEFAULT_HASH_SIZE,
    max_pages_per_doc: int = _MAX_PAGES_PER_DOC,
    templates_root: Path = _TEMPLATES_ROOT,
    backend: str = _DEFAULT_BACKEND,
) -> None:
    global _BACKEND
    _BACKEND = backend
    logger = get_run_logger()

    # Initialise the LLM client before any tasks fan out.
    llm = _get_llm()
    logger.info(f"backend={backend!r} -> {llm!r}")
    logger.info(f"classifying into {len(DOCUMENT_TYPES)} types + 'other' (skipped)")

    logger.info(f"loading {_HF_DATASET} ({_HF_SPLIT}) from HuggingFace")
    ds = load_dataset(_HF_DATASET, split=_HF_SPLIT)
    logger.info(f"loaded {len(ds)} row(s)")

    row_counts = Counter(ds["image_url"])
    urls: list[tuple[str, str]] = []
    seen_hashes: set[str] = set()
    for url, n_rows in row_counts.items():
        if n_rows > max_pages_per_doc:
            continue
        h = _url_hash(url)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        urls.append((h, url))
    logger.info(
        f"pre-filter (kvp10k rows <= {max_pages_per_doc}): "
        f"{len(urls)}/{len(row_counts)} URL(s) pass"
    )
    if limit > 0:
        urls = urls[:limit]
        logger.info(f"limit: first {len(urls)} URL(s)")
    if not urls:
        return

    templates_root.mkdir(parents=True, exist_ok=True)

    # ---- Shared, mutable pipeline state ----
    #
    # Documents flow through the whole chain concurrently, so state touched by
    # more than one document is guarded by locks:
    #   * ``docs_cache`` — the classification cache (one global lock).
    #   * per-class ``TemplateDeduper`` — one lock per class; created lazily as
    #     classes are first seen, under ``class_guard``.
    docs_cache = _load_docs_cache()
    logger.info(f"loaded {len(docs_cache)} classification cache entrie(s)")
    cache_lock = asyncio.Lock()
    class_guard = asyncio.Lock()
    class_locks: dict[str, asyncio.Lock] = {}
    class_dedupers: dict[str, TemplateDeduper] = {}
    pending_cache_writes = 0

    async def _class_ctx(cls: str) -> tuple[asyncio.Lock, TemplateDeduper]:
        async with class_guard:
            if cls not in class_locks:
                class_locks[cls] = asyncio.Lock()
                class_dedupers[cls] = TemplateDeduper(
                    hash_size=hash_size, threshold=hamming_threshold
                )
            return class_locks[cls], class_dedupers[cls]

    async def _cached_class(url_hash: str) -> str | None:
        async with cache_lock:
            entry = docs_cache.get(url_hash)
            return entry.get("class") if entry else None

    async def _record_class(url_hash: str, url: str, n: int, cls: str) -> None:
        nonlocal pending_cache_writes
        async with cache_lock:
            entry = docs_cache.setdefault(url_hash, {})
            entry["url"] = url
            entry["page_count"] = n
            entry["class"] = cls
            pending_cache_writes += 1
            if pending_cache_writes >= _CACHE_FLUSH_EVERY:
                _save_docs_cache(docs_cache)
                pending_cache_writes = 0

    async def _process_doc(url_hash: str, url: str) -> str:
        """Full per-document chain; returns a status label for tallying."""
        # 0a. download
        pdf_path = await download_pdf(url_hash, url)
        if pdf_path is None:
            return "no_pdf"

        # 0b. page count
        n = await get_page_count(url_hash, pdf_path)
        if n > max_pages_per_doc:
            return "too_long"

        # 1. classify into one of the fixed types. Reuse the cache, but
        #    re-classify any label that isn't part of the current taxonomy
        #    (e.g. a stale open-ended label from a previous run). Anything the
        #    classifier can't place is bucketed as "other" and skipped.
        cls = await _cached_class(url_hash)
        if cls not in _VALID_CLASSES:
            cls = await classify_pdf(url_hash, pdf_path)
            if cls is None:
                return "classify_fail"
            if cls not in _ALLOWED_TYPES:  # off-list result → bucket as "other"
                cls = _OTHER
            await _record_class(url_hash, url, n, cls)
        if cls == _OTHER:
            return "off_type"

        # Skip documents whose template already exists (resume support).
        target = templates_root / cls / f"{_SOURCE}_{url_hash}.html.j2"
        if target.exists():
            return "exists"

        # 2a. per-class phash dedup on the first page
        first = await asyncio.to_thread(_first_page, pdf_path)
        if first is None:
            return "no_first_page"
        lock, deduper = await _class_ctx(cls)
        async with lock:
            if deduper.check_and_add(first):
                return "dedup"

        # 2b. build template (GPU/LLM), then persist
        r = await build_template(url_hash, cls, pdf_path)
        if r is None:
            return "build_fail"
        await asyncio.to_thread(_persist, templates_root, r, url, n)
        return "saved"

    logger.info(f"processing {len(urls)} document(s) end-to-end (pipelined)")
    try:
        statuses = await asyncio.gather(*(_process_doc(h, u) for h, u in urls))
    finally:
        async with cache_lock:
            if pending_cache_writes:
                _save_docs_cache(docs_cache)

    tally = Counter(statuses)
    logger.info(
        f"done: saved {tally['saved']}/{len(urls)} template(s); "
        + ", ".join(f"{k}={v}" for k, v in sorted(tally.items()) if k != "saved")
    )


def main(
    limit: int = 0,
    hamming_threshold: int = _DEFAULT_HAMMING_THRESHOLD,
    hash_size: int = _DEFAULT_HASH_SIZE,
    max_pages_per_doc: int = _MAX_PAGES_PER_DOC,
    templates_root: Path | str = _TEMPLATES_ROOT,
    backend: str = _DEFAULT_BACKEND,
) -> None:
    if backend not in ("vllm", "sonnet"):
        raise ValueError(f"backend must be 'vllm' or 'sonnet', got {backend!r}")
    asyncio.run(
        kvp10k_templates(
            limit=limit,
            hamming_threshold=hamming_threshold,
            hash_size=hash_size,
            max_pages_per_doc=max_pages_per_doc,
            templates_root=Path(templates_root),
            backend=backend,
        )
    )


if __name__ == "__main__" and "__file__" in globals():
    main()
