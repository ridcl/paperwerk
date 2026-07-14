"""Generate version-controlled Jinja2 templates from the KVP10k dataset (per-document).

kvp10k stores one row per annotated page but its ``image_url`` field points
at the underlying multi-page source PDF; rows sharing a URL are pages of the
same document. This builder groups rows by ``image_url`` so every LLM call
sees the full document rather than a single page.

Pipeline (each phase is a Prefect task fanning out per document):
  0.  Group kvp10k rows by ``image_url``; pre-filter to URLs with
      ``<= max_pages_per_doc`` annotated rows.
  0a. ``download_pdf``   — fetch + cache the source PDF (retries=2).
  0b. ``get_page_count`` — cheap ``pdfinfo`` check; drop long-tail docs.
  1.  ``classify_pdf``   — first 3 PDF pages → snake_case class label,
                            cached in ``/data/paperwerk/cache/kvp10k/docs.json``.
  2a. Per-class phash dedup on the first page (sequential, in-flow).
  2b. ``build_template`` — full-document Jinja HTML + schema via
                            ``datagen.templates.make_template`` (which now
                            auto-chunks long docs internally, retries=1).
  3.  ``_persist``       — write ``<class>/kvp10k_<url_hash12>.html.j2`` +
                            sidecar under
                            ``src/datagen/assets/templates/``.

Running this triggers a Prefect flow. To see live per-task progress, start
a local Prefect server in another shell first::

    prefect server start
    prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api  # once

Then::

    from datagen.builders.templates_20260705_forms_kvp10k import main
    main()                                # full train split, <=32-page docs
    main(limit=200)                       # first 200 unique URLs only
    main(max_pages_per_doc=64)            # push the cap higher
    main(limit=200, hamming_threshold=6)  # stricter dedup

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
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset
from pdf2image import convert_from_bytes, pdfinfo_from_path
from PIL import Image
from prefect import flow, get_run_logger, task

from paperwerk.llm import LLM

import datagen
from datagen.classifier import classify
from datagen.templates import TemplateDeduper, make_template

_HF_DATASET = "ibm-research/KVP10k"
_HF_SPLIT = "train"

_ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1/"
_MODEL = "claude-sonnet-4-6"
_LLM_CONCURRENCY = 10
_DOWNLOAD_CONCURRENCY = 16
_SOURCE = "kvp10k"

_TEMPLATES_ROOT = Path(datagen.__file__).resolve().parent / "assets" / "templates"
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
_DL_SEM = asyncio.Semaphore(_DOWNLOAD_CONCURRENCY)


def _get_llm() -> LLM:
    global _LLM
    if _LLM is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set")
        _LLM = LLM(
            base_url=_ANTHROPIC_BASE_URL,
            api_key=api_key,
            model=_MODEL,
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
        cls = await classify(_get_llm(), str(pdf_path))
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


@flow(name="kvp10k-templates")
async def kvp10k_templates(
    limit: int = 0,
    hamming_threshold: int = _DEFAULT_HAMMING_THRESHOLD,
    hash_size: int = _DEFAULT_HASH_SIZE,
    max_pages_per_doc: int = _MAX_PAGES_PER_DOC,
    templates_root: Path = _TEMPLATES_ROOT,
) -> None:
    logger = get_run_logger()

    # Fail fast if the key isn't set (before any tasks fan out).
    _get_llm()

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

    # ---- Phase 0a: download PDFs ----
    logger.info(f"phase 0a: downloading {len(urls)} PDF(s)")
    pdf_paths = await asyncio.gather(*(download_pdf(h, u) for h, u in urls))
    have_pdf = [(h, u, p) for (h, u), p in zip(urls, pdf_paths) if p is not None]
    logger.info(f"phase 0a: {len(have_pdf)}/{len(urls)} PDF(s) available")
    if not have_pdf:
        return

    # ---- Phase 0b: filter by actual page count ----
    logger.info("phase 0b: checking page counts")
    page_counts = await asyncio.gather(*(get_page_count(h, p) for h, _, p in have_pdf))
    kept = [
        (h, u, p, n)
        for (h, u, p), n in zip(have_pdf, page_counts)
        if n <= max_pages_per_doc
    ]
    logger.info(
        f"phase 0b: {len(kept)}/{len(have_pdf)} pass (<= {max_pages_per_doc} pages)"
    )
    if not kept:
        return

    # ---- Phase 1: classify (cached) ----
    docs_cache = _load_docs_cache()
    logger.info(f"phase 1: classifying ({len(docs_cache)} cache entries)")
    to_classify = [
        (h, u, p, n)
        for h, u, p, n in kept
        if h not in docs_cache or "class" not in docs_cache[h]
    ]
    if to_classify:
        results = await asyncio.gather(
            *(classify_pdf(h, p) for h, _, p, _ in to_classify)
        )
        for (h, u, p, n), cls in zip(to_classify, results):
            entry = docs_cache.setdefault(h, {})
            entry["url"] = u
            entry["page_count"] = n
            if cls is not None:
                entry["class"] = cls
        _save_docs_cache(docs_cache)
    else:
        logger.info("phase 1: all classifications cached")

    dist = Counter(
        docs_cache[h]["class"]
        for h, _, _, _ in kept
        if h in docs_cache and "class" in docs_cache[h]
    )
    logger.info(f"phase 1: {len(dist)} distinct class(es)")
    for cls, n in dist.most_common(30):
        logger.info(f"  {cls}: {n}")

    # ---- Phase 2a: per-class phash dedup (sequential, in-flow) ----
    logger.info("phase 2a: phash dedup")
    by_class: dict[str, list[tuple[str, str, Path, int]]] = defaultdict(list)
    for h, u, p, n in kept:
        entry = docs_cache.get(h) or {}
        cls = entry.get("class")
        if cls is None:
            continue
        by_class[cls].append((h, u, p, n))

    to_generate: list[tuple[str, str, str, Path, int]] = []
    for cls, entries in sorted(by_class.items()):
        existing_dir = templates_root / cls
        dedup = TemplateDeduper(hash_size=hash_size, threshold=hamming_threshold)
        n_new = 0
        for h, u, p, n in entries:
            target = existing_dir / f"{_SOURCE}_{h}.html.j2"
            if target.exists():
                continue
            first = _first_page(p)
            if first is None:
                continue
            if dedup.check_and_add(first):
                continue
            to_generate.append((cls, h, u, p, n))
            n_new += 1
        if entries:
            logger.info(f"  {cls}: {n_new} new / {len(entries)} candidate(s)")

    if not to_generate:
        logger.info("nothing new to generate.")
        return

    # ---- Phase 2b: template generation ----
    logger.info(f"phase 2b: generating {len(to_generate)} template(s)")
    results = await asyncio.gather(
        *(build_template(h, cls, p) for cls, h, _, p, _ in to_generate)
    )

    templates_root.mkdir(parents=True, exist_ok=True)
    saved = 0
    for r, (cls, h, u, p, n) in zip(results, to_generate):
        if r is None:
            continue
        _persist(templates_root, r, u, n)
        saved += 1
    logger.info(f"done: saved {saved}/{len(to_generate)} template(s)")


def main(
    limit: int = 0,
    hamming_threshold: int = _DEFAULT_HAMMING_THRESHOLD,
    hash_size: int = _DEFAULT_HASH_SIZE,
    max_pages_per_doc: int = _MAX_PAGES_PER_DOC,
    templates_root: Path | str = _TEMPLATES_ROOT,
) -> None:
    asyncio.run(
        kvp10k_templates(
            limit=limit,
            hamming_threshold=hamming_threshold,
            hash_size=hash_size,
            max_pages_per_doc=max_pages_per_doc,
            templates_root=Path(templates_root),
        )
    )


if __name__ == "__main__" and "__file__" in globals():
    main()
