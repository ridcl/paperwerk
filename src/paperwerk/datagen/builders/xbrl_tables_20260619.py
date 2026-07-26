"""Build a VQA-with-bounding-boxes dataset from SEC EDGAR inline XBRL filings.

SEC filings (10-K, 10-Q, 20-F, 40-F) are filed as *inline XBRL* (iXBRL): the
primary `.htm` is the human-readable report AND carries structured financial
facts tagged inline. Every numeric fact is wrapped in an `<ix:nonFraction>`
element with a stable `id`, a `contextRef` (period), a `unitRef` (currency),
and a `scale`/`sign`. Because the element *wraps the visible number*, its
on-page bounding box IS the value's location - no OCR or text search needed.

Crucially, the *complete* value comes from the metadata, not the display text:
a table may show "307,003" under a "thousands / millions" header, but the fact
carries `scale="6"` and `unitRef="usd"`, so the true value is `USD 307003000000`.
This builder emits that complete value as the answer while pointing the bounding
box at the displayed digits.

Pipeline:
  1. Harvest  - resolve tickers -> CIK via company_tickers.json, list recent
                filings via the submissions API, download each primary iXBRL
                `.htm`. Throttled to <=10 req/s with a descriptive User-Agent
                (SEC requirement); cached under /data/paperwerk/xbrl_cache/.
  2. Parse    - stdlib ElementTree extracts facts, contexts (periods +
                dimensional members), and units (currency). Each fact gets a
                canonical value computed as `sign * parse(text, format) * 10^scale`.
  3. Render   - Playwright renders the XHTML and splits it at the document's own
                page-break markers (`<hr style="page-break-after:always">`), so
                each "page" is a real logical page and financial tables stay
                intact (no arbitrary mid-table cuts). Each fact id is located to a
                page + normalized bbox (with its enclosing table-row text as
                context), and each needed page is screenshotted ONE AT A TIME via
                scroll + clip - the whole document is never rasterized at once, so
                a 500-page filing stays memory-flat. Pages with >10 facts are
                "dense"; consecutive dense pages are tiled into windows of <=5
                pages with 0-3 context pages each side.
  4. Query    - per window, the LLM turns up to 20 of the most-askable facts
                into natural queries (+ a few unanswerable ones). The answer
                VALUE is the canonical metadata value (never the LLM's); the
                bbox and page index come from the located fact.
  5. Emit     - one parquet row per window, schema-identical to the CUAD VQA
                builder (`vqa_20260524_cuad.py`).

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    python -m datagen.builders.xbrl_tables_20260619 --limit 5
    python -m datagen.builders.xbrl_tables_20260619 --tickers AAPL,MSFT --forms 10-K

    # build and push the dataset to the HF Hub (auth via HF_TOKEN or huggingface-cli login):
    python -m datagen.builders.xbrl_tables_20260619 --tickers AAPL,MSFT --push-to-hub
    python -m datagen.builders.xbrl_tables_20260619 --push-to-hub --hub-repo my-org/xbrl-tables
"""

from __future__ import annotations

import argparse
import asyncio
import bisect
import json
import math
import os
import random
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field as dc_field
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from playwright.sync_api import sync_playwright

from paperwerk.llm import LLM

# --- SEC endpoints / config ---------------------------------------------------
_CACHE_DIR = Path("/data/paperwerk/xbrl_cache")
_OUT_PATH = Path("/data/paperwerk/xbrl_tables_20260619.parquet")
_HUB_REPO = "xbrl-tables"  # default HF Hub dataset repo (under the logged-in namespace)
_USER_AGENT = "paperwerk-research andrei.zhabinski@datasnipper.com"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_PRIMARY_DOC_URL = (
    "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
)
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SEC_MIN_INTERVAL = 0.12  # seconds between SEC requests (<10 req/s)

# A curated default basket of large filers across sectors (resolved via tickers).
_DEFAULT_TICKERS = (
    "AAPL,MSFT,AMZN,GOOGL,META,NVDA,JPM,JNJ,XOM,WMT,"
    "PG,KO,PFE,INTC,CSCO,DIS,V,MA,HD,VZ"
)
_DEFAULT_FORMS = ("10-K", "10-Q", "20-F", "40-F")

# Local vLLM server (OpenAI-compatible), reachable via the devcontainer's
# network=host. Override with LOCAL_LLM_BASE_URL / LOCAL_LLM_MODEL if the host
# serves a different endpoint or model.
_LLM_BASE_URL = os.environ.get("LOCAL_LLM_BASE_URL", "http://localhost:8000/v1/")
_MODEL = os.environ.get("LOCAL_LLM_MODEL", "google/gemma-4-E4B-it")
_CONCURRENCY = 10
_RENDER_CONCURRENCY = 3

# Filings are processed in batches so memory stays bounded (only one batch is
# rendered/held at a time) and a `--max-datapoints` target can stop the run early.
_BATCH_SIZE = 25
# `--random N` samples N tickers from the largest this-many filers (by market
# cap, the order of company_tickers.json) - large enough to be diverse, while
# avoiding micro-cap shells that rarely carry substantive financial statements.
_RANDOM_UNIVERSE = 3000

# --- rendering / windowing config ---------------------------------------------
_PAGE_W = 794
_PAGE_H = 1123
_DENSE_FACT_THRESHOLD = 10  # a "dense" page has strictly more facts than this
_MAX_PAGES = 5  # hard cap on pages per window (matches existing VQA datasets)
_MAX_CORE_PAGES = 3  # dense pages per window core; leaves room for context
_MAX_CONTEXT_PAGES = 3  # context pages each side (further capped by _MAX_PAGES)

# Viewport height used for rendering/screenshots. Generous headroom so a single
# page always fits in one viewport clip; viewport *height* does not affect
# horizontal reflow, so pagination is stable. SEC pages run ~900px (max ~1750).
_VIEWPORT_H = 2048

# Minimum height for a real page; shorter regions between two page-break markers
# are spacers/empty and get merged into the preceding page.
_MIN_PAGE_H = 50

# --- query-gen config ---------------------------------------------------------
_MAX_FACTS_IN_PROMPT = 20
_MAX_UNANSWERABLE = 3
_MAX_ROW_CONTEXT_CHARS = 200
_DEFAULT_SEED = 0

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

# XBRL namespaces (instance + inline).
_XBRLI = "{http://www.xbrl.org/2003/instance}"
_IX = "{http://www.xbrl.org/2013/inlineXBRL}"
_XBRLDI = "{http://xbrl.org/2006/xbrldi}"


# ==============================================================================
# Phase 1: harvest
# ==============================================================================
class _Throttle:
    """Enforce a minimum wall-clock interval between SEC requests."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._last = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last = time.monotonic()


_THROTTLE = _Throttle(_SEC_MIN_INTERVAL)


def _http_get(url: str) -> bytes:
    _THROTTLE.wait()
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def _cached_get(url: str, cache_path: Path) -> bytes:
    if cache_path.exists():
        return cache_path.read_bytes()
    data = _http_get(url)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(data)
    return data


def _load_ticker_map() -> dict[str, int]:
    """Return {TICKER -> CIK}. company_tickers.json is keyed by arbitrary index."""
    raw = _cached_get(_TICKERS_URL, _CACHE_DIR / "company_tickers.json")
    data = json.loads(raw)
    out: dict[str, int] = {}
    for entry in data.values():
        out[entry["ticker"].upper()] = int(entry["cik_str"])
    return out


@dataclass
class Filing:
    cik: int
    ticker: str
    form: str
    accession: str  # no dashes
    document: str  # primary doc filename
    filing_date: str

    @property
    def slug(self) -> str:
        who = self.ticker or f"cik{self.cik}"
        return f"sec_{who}_{self.accession}"

    @property
    def base_url(self) -> str:
        return _PRIMARY_DOC_URL.format(
            cik=self.cik, accession=self.accession, document=""
        )

    @property
    def doc_url(self) -> str:
        return _PRIMARY_DOC_URL.format(
            cik=self.cik, accession=self.accession, document=self.document
        )


def _list_filings(
    cik: int, ticker: str, forms: tuple[str, ...], per_company: int
) -> list[Filing]:
    """Pick the most recent `per_company` filings of the requested forms."""
    raw = _cached_get(
        _SUBMISSIONS_URL.format(cik=cik), _CACHE_DIR / f"CIK{cik:010d}.json"
    )
    recent = json.loads(raw).get("filings", {}).get("recent", {})
    forms_col = recent.get("form", [])
    acc_col = recent.get("accessionNumber", [])
    doc_col = recent.get("primaryDocument", [])
    date_col = recent.get("filingDate", [])
    out: list[Filing] = []
    for form, acc, doc, date in zip(forms_col, acc_col, doc_col, date_col):
        if form not in forms or not doc or not doc.lower().endswith((".htm", ".html")):
            continue
        out.append(
            Filing(
                cik=cik,
                ticker=ticker,
                form=form,
                accession=acc.replace("-", ""),
                document=doc,
                filing_date=date,
            )
        )
        if len(out) >= per_company:
            break
    return out


def _download_filing(filing: Filing) -> str:
    """Download (cached) the primary iXBRL document; return its text."""
    cache_path = _CACHE_DIR / f"{filing.accession}_{filing.document}"
    raw = _cached_get(filing.doc_url, cache_path)
    return raw.decode("utf-8", errors="replace")


# ==============================================================================
# Phase 2: parse XBRL facts
# ==============================================================================
@dataclass
class Fact:
    fact_id: str
    name: str  # QName, e.g. "us-gaap:NetIncomeLoss"
    concept: str  # humanized, e.g. "Net Income Loss"
    period: str | None  # human-readable period label
    members: list[str]  # humanized dimensional member labels (disambiguators)
    value: str  # canonical value, e.g. "USD 307003000000"
    raw_text: str  # the displayed text, e.g. "307,003"
    # Filled in during the render phase:
    page: int = -1
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    row_text: str = ""


def _humanize_qname(qname: str) -> str:
    local = qname.split(":", 1)[-1]
    # Strip a trailing disambiguation suffix XBRL often appends.
    local = re.sub(r"(Member|Axis|Domain)$", "", local)
    return _CAMEL_RE.sub(" ", local).strip()


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _parse_contexts(root: ET.Element) -> dict[str, dict]:
    contexts: dict[str, dict] = {}
    for ctx in root.iter(_XBRLI + "context"):
        cid = ctx.get("id")
        if cid is None:
            continue
        period = ctx.find(_XBRLI + "period")
        label = None
        if period is not None:
            inst = period.find(_XBRLI + "instant")
            if inst is not None:
                label = f"as of {inst.text}"
            else:
                sd = period.find(_XBRLI + "startDate")
                ed = period.find(_XBRLI + "endDate")
                if sd is not None and ed is not None:
                    label = f"{sd.text} to {ed.text}"
        members = [
            _humanize_qname((m.text or "").strip())
            for m in ctx.iter(_XBRLDI + "explicitMember")
            if (m.text or "").strip()
        ]
        contexts[cid] = {"period": label, "members": members}
    return contexts


def _parse_units(root: ET.Element) -> dict[str, list[str]]:
    units: dict[str, list[str]] = {}
    for u in root.iter(_XBRLI + "unit"):
        uid = u.get("id")
        if uid is not None:
            units[uid] = [m.text for m in u.iter(_XBRLI + "measure") if m.text]
    return units


def _currency_of(unit_measures: list[str]) -> str | None:
    for m in unit_measures:
        if m.startswith("iso4217:"):
            return m.split(":", 1)[1]
    return None


def _format_number(d: Decimal) -> str:
    """Render a Decimal as a plain string: integer if whole, else trimmed."""
    d = d.normalize()
    if d == d.to_integral_value():
        return str(d.quantize(Decimal(1)))
    return format(d, "f")


def _canonical_value(
    raw_text: str,
    scale: str | None,
    sign: str | None,
    fmt: str | None,
    unit_measures: list[str],
) -> str | None:
    """Compute the complete value: `sign * parse(text, format) * 10^scale`.

    Monetary facts become "<ISO> <full_number>" (no thousands separators);
    per-share facts add " per share"; share counts add " shares"; bare numbers
    are returned as-is. Returns None for unparseable text (e.g. a stray dash).
    """
    t = (raw_text or "").strip().replace(" ", "").replace(" ", "")
    if fmt and "comma-decimal" in fmt:  # European: 1.234,56
        t = t.replace(".", "").replace(",", ".")
    else:
        t = t.replace(",", "")
    t = t.replace("%", "").lstrip("$")
    if t in ("", "-", "—", "–"):
        return None
    try:
        d = Decimal(t)
    except InvalidOperation:
        return None
    if scale:
        try:
            d = d * (Decimal(10) ** int(scale))
        except (ValueError, InvalidOperation):
            return None
    if sign == "-":
        d = -d

    num = _format_number(d)
    currency = _currency_of(unit_measures)
    is_shares = any(m == "xbrli:shares" for m in unit_measures)
    if currency and is_shares:  # e.g. usdPerShare (divide unit)
        return f"{currency} {num} per share"
    if currency:
        return f"{currency} {num}"
    if is_shares:
        return f"{num} shares"
    return num


def _parse_facts(html: str) -> list[Fact]:
    """Extract numeric (and visible text) facts with canonical values."""
    root = ET.fromstring(html)
    contexts = _parse_contexts(root)
    units = _parse_units(root)

    facts: list[Fact] = []
    for el in root.iter(_IX + "nonFraction"):
        fid = el.get("id")
        name = el.get("name")
        if not fid or not name:
            continue
        unit_measures = units.get(el.get("unitRef"), [])
        value = _canonical_value(
            el.text, el.get("scale"), el.get("sign"), el.get("format"), unit_measures
        )
        if value is None:
            continue
        ctx = contexts.get(el.get("contextRef"), {})
        facts.append(
            Fact(
                fact_id=fid,
                name=name,
                concept=_humanize_qname(name),
                period=ctx.get("period"),
                members=ctx.get("members", []),
                value=value,
                raw_text=(el.text or "").strip(),
            )
        )
    return facts


# ==============================================================================
# Phase 3: render + locate + window
# ==============================================================================
# Locate every fact id in one round-trip. Coordinates are document-relative
# (rect + scroll offset) and already in the post-`zoom` space - getBoundingClientRect
# returns zoomed coordinates, consistent with scrollHeight, so we do NOT rescale.
# An `ix:nonFraction` element that itself has no box (rare; e.g. a transform wraps
# inline children) falls back to the union of its children's boxes. Truly hidden
# facts (ix:hidden / display:none -> no boxed descendant) are skipped.
_LOCATE_JS = r"""
(ids) => {
  const out = {};
  for (const id of ids) {
    const el = document.getElementById(id);
    if (!el) continue;
    let r = el.getBoundingClientRect();
    if (r.width === 0 && r.height === 0) {
      let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
      for (const c of el.querySelectorAll('*')) {
        const cr = c.getBoundingClientRect();
        if (cr.width > 0 && cr.height > 0) {
          minx = Math.min(minx, cr.left); miny = Math.min(miny, cr.top);
          maxx = Math.max(maxx, cr.right); maxy = Math.max(maxy, cr.bottom);
        }
      }
      if (minx === Infinity) continue;  // genuinely hidden
      r = {left: minx, top: miny, width: maxx - minx, height: maxy - miny};
    }
    const tr = el.closest('tr');
    const rowText = tr ? tr.innerText : '';
    out[id] = {
      x: r.left + window.pageXOffset,
      y: r.top + window.pageYOffset,
      w: r.width,
      h: r.height,
      row: rowText.replace(/\s+/g, ' ').trim(),
    };
  }
  return out;
}
"""

# Find the document's own page boundaries: the y-coordinates of explicit page-
# break markers. SEC iXBRL uses `<hr style="page-break-after:always">` between
# pages (verified on Apple/Microsoft/JPMorgan filings); a `page-break-after`
# ends the current page at the marker's bottom, `page-break-before` starts a new
# page at its top. Returns the sorted boundary y-list (document-relative).
_PAGE_BREAK_JS = r"""
() => {
  const ys = [];
  for (const e of document.querySelectorAll('[style*="page-break-after"], [style*="break-after"]')) {
    const r = e.getBoundingClientRect();
    ys.push(r.bottom + window.pageYOffset);
  }
  for (const e of document.querySelectorAll('[style*="page-break-before"], [style*="break-before"]')) {
    const r = e.getBoundingClientRect();
    ys.push(r.top + window.pageYOffset);
  }
  return ys.sort((a, b) => a - b);
}
"""


def _inject_base(html: str, base_url: str) -> str:
    """Add <base href> so relative CSS/images resolve over the network."""
    return re.sub(
        r"(<head[^>]*>)",
        r'\1<base href="' + base_url + '">',
        html,
        count=1,
        flags=re.IGNORECASE,
    )


def _window_plan(dense_pages: list[int], n_pages: int) -> list[tuple[int, int]]:
    """Tile consecutive dense pages into <=`_MAX_PAGES` windows with context.

    Maximal runs of dense pages are split into cores of at most
    `_MAX_CORE_PAGES`; each core is padded with context pages (balanced across
    the available budget, capped per side and clamped to the document) until it
    reaches `_MAX_PAGES`. Duplicate windows are removed.
    """
    if not dense_pages:
        return []
    dense = sorted(set(dense_pages))
    runs: list[list[int]] = []
    for p in dense:
        if runs and p == runs[-1][-1] + 1:
            runs[-1].append(p)
        else:
            runs.append([p])

    windows: list[tuple[int, int]] = []
    for run in runs:
        for i in range(0, len(run), _MAX_CORE_PAGES):
            core = run[i : i + _MAX_CORE_PAGES]
            start, end = core[0], core[-1]
            budget = _MAX_PAGES - (end - start + 1)
            # Pad alternately after then before, respecting per-side + doc limits.
            after = before = 0
            while budget > 0:
                grew = False
                if after < _MAX_CONTEXT_PAGES and end + after + 1 < n_pages:
                    after += 1
                    budget -= 1
                    grew = True
                if budget > 0 and before < _MAX_CONTEXT_PAGES and start - before - 1 >= 0:
                    before += 1
                    budget -= 1
                    grew = True
                if not grew:
                    break
            windows.append((start - before, end + after))

    seen: set[tuple[int, int]] = set()
    unique: list[tuple[int, int]] = []
    for w in windows:
        if w not in seen:
            seen.add(w)
            unique.append(w)
    return unique


@dataclass
class RenderedFiling:
    facts: list[Fact]  # only located (visible) facts, with page/bbox/row_text
    windows: list[tuple[int, int]]
    page_png: dict[int, bytes]  # PNG bytes for every page referenced by a window


def _detect_page_boxes(page, total_h: int) -> list[dict]:
    """Return page boxes (document-relative, sorted by y).

    Splits the document at its OWN page-break markers, so each box is a real
    logical page (financial tables stay intact instead of being cut mid-table by
    an arbitrary slice). Regions shorter than `_MIN_PAGE_H` between two markers
    are spacers and get merged into the preceding page. Only if a document has no
    page-break markers (not seen in SEC iXBRL) do we fall back to fixed A4 slices.
    Either way the boxes drive BOTH fact->page assignment and screenshots, so the
    page image and the bboxes on it are consistent.
    """
    break_ys = page.evaluate(_PAGE_BREAK_JS)
    bounds = [0.0]
    for y in break_ys:
        if y - bounds[-1] >= _MIN_PAGE_H and y <= total_h - _MIN_PAGE_H:
            bounds.append(float(y))
    bounds.append(float(total_h))

    if len(bounds) >= 3:  # at least two real pages detected
        return [
            {
                "x": 0.0,
                "y": bounds[i],
                "width": float(_PAGE_W),
                "height": bounds[i + 1] - bounds[i],
            }
            for i in range(len(bounds) - 1)
        ]

    n_pages = max(1, math.ceil(total_h / _PAGE_H))
    return [
        {
            "x": 0.0,
            "y": float(i * _PAGE_H),
            "width": float(_PAGE_W),
            "height": float(min(_PAGE_H, total_h - i * _PAGE_H)),
        }
        for i in range(n_pages)
    ]


def _normalize_bbox(c: dict, box: dict) -> tuple[float, float, float, float]:
    """Element box (document coords) -> [0,1] bbox relative to its page box."""
    bw = box["width"] or 1.0
    bh = box["height"] or 1.0
    lx, ly = c["x"] - box["x"], c["y"] - box["y"]
    return (
        _clamp01(lx / bw),
        _clamp01(ly / bh),
        _clamp01((lx + c["w"]) / bw),
        _clamp01((ly + c["h"]) / bh),
    )


def _screenshot_page(page, box: dict) -> bytes:
    """Screenshot a single page (slice or page-div) via scroll + viewport clip.

    Capturing one page at a time keeps memory flat regardless of document length
    (a 500-page filing never materializes as one giant image). `scrollTo` clamps
    near the document end, so the clip is offset by the actual scroll position.
    Driven by the same y-sorted box used for fact assignment, so the image and
    its bboxes always correspond.
    """
    top = box["y"]
    page.evaluate("y => window.scrollTo(0, y)", top)
    page.wait_for_timeout(30)
    offset_y = page.evaluate("window.pageYOffset")
    offset_x = page.evaluate("window.pageXOffset")
    clip_y = top - offset_y
    return page.screenshot(
        clip={
            "x": box["x"] - offset_x,
            "y": clip_y,
            "width": box["width"],
            # Clamp to what remains of the viewport so the clip never exceeds it.
            "height": min(box["height"], _VIEWPORT_H - clip_y),
        }
    )


def _render_and_locate(html: str, base_url: str) -> RenderedFiling:
    """Render the iXBRL doc, locate every fact, plan windows, screenshot pages.

    Runs a single synchronous Playwright session (call from a worker thread).
    Screenshots are taken one page at a time - the whole document is never
    rendered to a single image.
    """
    facts = _parse_facts(html)
    fact_by_id = {f.fact_id: f for f in facts}
    html2 = _inject_base(html, base_url)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            ctx = browser.new_context(
                user_agent=_USER_AGENT,
                viewport={"width": _PAGE_W, "height": _VIEWPORT_H},
                device_scale_factor=1,
            )
            page = ctx.new_page()
            page.set_content(html2, wait_until="load", timeout=90000)
            page.wait_for_timeout(500)
            scroll_w = page.evaluate("Math.ceil(document.documentElement.scrollWidth)")
            zoom = min(1.0, _PAGE_W / scroll_w) if scroll_w > _PAGE_W else 1.0
            if zoom < 1.0:
                page.evaluate("z => {document.documentElement.style.zoom = z}", zoom)
                page.wait_for_timeout(300)
            total_h = page.evaluate("Math.ceil(document.documentElement.scrollHeight)")

            page_boxes = _detect_page_boxes(page, total_h)
            starts = [b["y"] for b in page_boxes]

            coords = page.evaluate(_LOCATE_JS, list(fact_by_id.keys()))

            located: list[Fact] = []
            per_page: Counter[int] = Counter()
            for fid, c in coords.items():
                f = fact_by_id.get(fid)
                if f is None:
                    continue
                idx = bisect.bisect_right(starts, c["y"]) - 1
                if idx < 0:
                    idx = 0  # above the first page box (e.g. a header band)
                box = page_boxes[idx]
                f.page = idx
                f.bbox = _normalize_bbox(c, box)
                f.row_text = c["row"][:_MAX_ROW_CONTEXT_CHARS]
                located.append(f)
                per_page[idx] += 1

            dense = [pg for pg, n in per_page.items() if n > _DENSE_FACT_THRESHOLD]
            windows = _window_plan(dense, len(page_boxes))

            needed = sorted({pg for s, e in windows for pg in range(s, e + 1)})
            page_png: dict[int, bytes] = {}
            for pg in needed:
                page_png[pg] = _screenshot_page(page, page_boxes[pg])
        finally:
            browser.close()

    return RenderedFiling(facts=located, windows=windows, page_png=page_png)


# ==============================================================================
# Phase 4: query generation (LLM)
# ==============================================================================
_PROMPT_TEMPLATE = """\
You are building a Visual Question Answering training set from a company's SEC \
financial filing ({form}). Below are structured financial facts extracted from \
page(s) {pages_label} of the filing, each with a stable id, the accounting \
concept, the reporting period, the complete value, and the table-row text where \
it appears.

A user of this dataset is a finance professional or investor looking up specific \
figures: revenues, expenses, balances, share counts, per-share amounts, dates, \
and the like.

Facts:
{fact_lines}

Task:
  1) Pick up to 10 of the MOST USEFUL figures a real user would actually search \
for - revenues, expenses, balances, share counts, per-share amounts, and the \
like. Each becomes ONE query.

  2) For each, write ONE query the way a real user would type it. Phrase it as a \
natural question or a short search phrase, and do NOT include the raw value.
     - MOST queries (about 3 out of every 4) must be PRECISE: include the period \
or segment so the query identifies a SINGLE figure (e.g. "total net sales for \
the six months ended March 2026", "cash and cash equivalents at March 28, 2026"). \
This reflects users who phrase their search carefully or iterate until it is \
exact.
     - A MINORITY (only about 1 in 5, i.e. 10-25% of queries) should be \
deliberately UNDER-SPECIFIED, the way a hurried user types: the concept alone \
with NO period or segment (e.g. just "total net sales", "cash and cash \
equivalents"). Choose these for concepts that appear under SEVERAL periods or \
segments, so the query is genuinely ambiguous.
     - The reader already knows which company's filing this is, so do NOT name \
the filing entity in the query: write "total net sales", NOT "Apple's total net \
sales". Only include a company/entity name when the figure belongs to a \
DIFFERENT named entity mentioned on the page (e.g. a subsidiary, an equity-method \
investee, or an acquired company) and the name is needed to say which one the \
figure is about.

  3) For each query, list the ids of EVERY fact it matches:
     - A PRECISE query that pins down one period/segment -> exactly ONE id.
     - An UNDER-SPECIFIED query that matches the same concept across several \
periods or segments -> list the ids of ALL of them together. The dataset returns \
every plausible match so downstream logic can pick the most probable answer, \
raise, or ask the user.

  4) Also produce 1-{max_unanswerable} "unanswerable" queries: plausible \
questions about a filing like this that the facts above do NOT answer. They \
should look natural, not adversarial.

Return one JSON object inside a single ```json``` fence and nothing else. Use \
the fact `id` values verbatim; do NOT include values (we supply those from \
metadata):

```json
{{
  "answerable": [
    {{"query": "precise query naming the period", "fact_ids": ["f-1"]}},
    {{"query": "under-specified query, no period", "fact_ids": ["f-2", "f-3"]}}
  ],
  "unanswerable": ["..."]
}}
```
"""


def _rank_facts(facts: list[Fact]) -> list[Fact]:
    """Order facts by how likely a user is to ask about them, then cap.

    Monetary facts rank above bare numbers; within each, larger magnitude
    first. Exact (concept, period, value) duplicates are dropped so the prompt
    isn't wasted on repeats of the same figure.
    """

    def magnitude(f: Fact) -> float:
        m = re.search(r"-?\d+(?:\.\d+)?", f.value.replace(",", ""))
        try:
            return abs(float(m.group())) if m else 0.0
        except ValueError:
            return 0.0

    seen: set[tuple[str, str | None, str]] = set()
    deduped: list[Fact] = []
    for f in facts:
        key = (f.name, f.period, f.value)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(f)

    deduped.sort(
        key=lambda f: (0 if " " in f.value else 1, -magnitude(f)),
    )
    return deduped[:_MAX_FACTS_IN_PROMPT]


def _fact_line(f: Fact) -> str:
    parts = [f"[{f.fact_id}]", f.concept]
    if f.members:
        parts.append(f"({'; '.join(f.members)})")
    if f.period:
        parts.append(f"| {f.period}")
    parts.append(f'| value: "{f.value}"')
    if f.row_text:
        parts.append(f'| row: "{f.row_text}"')
    return "  - " + " ".join(parts)


async def _ask_llm(
    llm: LLM, form: str, page_start: int, page_end: int, facts: list[Fact]
) -> dict | None:
    pages_label = (
        f"{page_start + 1}"
        if page_start == page_end
        else f"{page_start + 1}-{page_end + 1}"
    )
    prompt = _PROMPT_TEMPLATE.format(
        form=form,
        pages_label=pages_label,
        fact_lines="\n".join(_fact_line(f) for f in facts),
        max_unanswerable=_MAX_UNANSWERABLE,
    )
    resp = await llm.ainvoke(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4000,
        temperature=0.7,
    )
    text = resp.choices[0].message.content
    m = _JSON_FENCE_RE.search(text)
    blob = (m.group(1) if m else text).strip()
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("answerable"), list):
        return None
    return data


def _assemble_datapoint(
    llm_out: dict,
    facts_in_window: dict[str, Fact],
    window_start: int,
    rng_shuffle,
) -> tuple[list[str], list[dict]] | None:
    """Turn validated LLM output into (queries, answers); answer VALUE is metadata.

    Returns None when no answerable query survives validation (we never emit a
    datapoint without grounded evidence).
    """
    answers: list[dict] = []
    answerable_queries: list[str] = []
    for item in llm_out.get("answerable", []):
        if not isinstance(item, dict):
            continue
        query = item.get("query")
        ids = item.get("fact_ids")
        if not (isinstance(query, str) and isinstance(ids, list)):
            continue
        per_query: list[dict] = []
        seen_ids: set[str] = set()
        for fid in ids:
            if not isinstance(fid, str) or fid in seen_ids:
                continue
            f = facts_in_window.get(fid)
            if f is None:
                continue
            seen_ids.add(fid)
            per_query.append(
                {
                    "query": query,
                    "value": f.value,  # canonical metadata value, NOT the LLM's
                    "bounding_box": [float(v) for v in f.bbox],
                    "index": f.page - window_start,
                }
            )
        if not per_query:
            continue
        answers.extend(per_query)
        answerable_queries.append(query)

    if not answers:
        return None

    unanswerable = [
        q for q in llm_out.get("unanswerable", []) if isinstance(q, str)
    ][:_MAX_UNANSWERABLE]
    queries = answerable_queries + unanswerable
    rng_shuffle(queries)
    return queries, answers


# ==============================================================================
# Output schema (identical to vqa_20260524_cuad.py)
# ==============================================================================
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


# ==============================================================================
# Orchestration
# ==============================================================================
@dataclass
class WindowTask:
    filing: Filing
    win_start: int
    win_end: int
    facts: list[Fact]  # facts located within [win_start, win_end]
    rendered: RenderedFiling = dc_field(repr=False, default=None)  # type: ignore


def _chunk(seq: list, size: int):
    """Yield successive `size`-length chunks of `seq`."""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _sample_tickers(n: int, seed: int) -> list[str]:
    """Randomly sample `n` tickers from the largest `_RANDOM_UNIVERSE` filers."""
    raw = _cached_get(_TICKERS_URL, _CACHE_DIR / "company_tickers.json")
    entries = json.loads(raw)
    # company_tickers.json is ordered by market cap; keys are stringified indices.
    ordered = [
        entries[k]["ticker"].upper()
        for k in sorted(entries, key=int)[:_RANDOM_UNIVERSE]
    ]
    rng = random.Random(seed)
    return rng.sample(ordered, min(n, len(ordered)))


def _filings_for_tickers(
    tickers: list[str],
    ticker_map: dict[str, int],
    forms: tuple[str, ...],
    per_company: int,
) -> list[Filing]:
    """Resolve a batch of tickers to recent filings (submissions API)."""
    filings: list[Filing] = []
    for t in tickers:
        cik = ticker_map.get(t.upper())
        if cik is None:
            print(f"  ticker {t!r} not found in SEC map; skipping", file=sys.stderr)
            continue
        try:
            filings.extend(_list_filings(cik, t.upper(), forms, per_company))
        except Exception as e:  # noqa: BLE001
            print(f"  [{t}] submissions ERROR {e!r}", file=sys.stderr)
    return filings


async def _render_filing(
    filing: Filing, render_sem: asyncio.Semaphore
) -> RenderedFiling | None:
    async with render_sem:
        try:
            html = await asyncio.to_thread(_download_filing, filing)
        except Exception as e:  # noqa: BLE001
            print(f"  [{filing.slug}] DOWNLOAD ERROR {e!r}", file=sys.stderr)
            return None
        try:
            return await asyncio.to_thread(_render_and_locate, html, filing.base_url)
        except Exception as e:  # noqa: BLE001
            print(f"  [{filing.slug}] RENDER ERROR {e!r}", file=sys.stderr)
            return None


async def _llm_task(
    llm: LLM, sem: asyncio.Semaphore, task: WindowTask
) -> dict | None:
    async with sem:
        try:
            return await _ask_llm(
                llm, task.filing.form, task.win_start, task.win_end, task.facts
            )
        except Exception as e:  # noqa: BLE001
            print(
                f"  [{task.filing.slug} p{task.win_start}-{task.win_end}] "
                f"LLM ERROR {e!r}",
                file=sys.stderr,
            )
            return None


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


def _build_window_tasks(
    filings: list[Filing], rendered: list[RenderedFiling | None]
) -> list[WindowTask]:
    """Flatten rendered filings to one LLM task per (filing, window)."""
    tasks: list[WindowTask] = []
    for filing, r in zip(filings, rendered):
        if r is None or not r.windows:
            continue
        for win_start, win_end in r.windows:
            in_window = [f for f in r.facts if win_start <= f.page <= win_end]
            ranked = _rank_facts(in_window)
            if not ranked:
                continue
            tasks.append(
                WindowTask(
                    filing=filing,
                    win_start=win_start,
                    win_end=win_end,
                    facts=ranked,
                    rendered=r,
                )
            )
    return tasks


def _assemble_row(
    task: WindowTask, out: dict | None, rng: random.Random, win_counter: Counter
) -> dict | None:
    """Build one parquet row from an LLM result, or None if it doesn't qualify."""
    if out is None:
        return None
    facts_in_window = {f.fact_id: f for f in task.facts}
    assembled = _assemble_datapoint(out, facts_in_window, task.win_start, rng.shuffle)
    if assembled is None:
        return None
    queries, answers = assembled
    page_png = task.rendered.page_png
    if any(p not in page_png for p in range(task.win_start, task.win_end + 1)):
        return None
    images = [page_png[p] for p in range(task.win_start, task.win_end + 1)]
    win_idx = win_counter[task.filing.slug]
    win_counter[task.filing.slug] += 1
    return {
        "images": images,
        "queries": queries,
        "answers": answers,
        "source": f"{task.filing.slug}_{win_idx}",
        "variant": "clear",
        "page_start": task.win_start,
        "page_end": task.win_end,
    }


async def _run(
    tickers: list[str],
    forms: tuple[str, ...],
    per_company: int,
    limit: int,
    seed: int,
    output_path: Path,
    max_datapoints: int = 0,
    push_hub: bool = False,
    hub_repo: str = _HUB_REPO,
    hub_private: bool = False,
) -> None:
    # The local vLLM server ignores the API key, but the OpenAI client requires
    # a non-empty string. Fall back to a placeholder when none is set.
    api_key = os.environ.get("LOCAL_LLM_API_KEY") or os.environ.get(
        "ANTHROPIC_API_KEY"
    ) or "EMPTY"

    ticker_map = _load_ticker_map()
    llm = LLM(
        base_url=_LLM_BASE_URL,
        api_key=api_key,
        model=_MODEL,
        max_concurrency=_CONCURRENCY,
    )
    render_sem = asyncio.Semaphore(_RENDER_CONCURRENCY)
    sem = asyncio.Semaphore(_CONCURRENCY)
    rng = random.Random(seed)
    win_counter: Counter[str] = Counter()

    target = str(max_datapoints) if max_datapoints else "all"
    print(
        f"processing up to {len(tickers)} ticker(s) in batches of {_BATCH_SIZE}, "
        f"forms={forms}, target datapoints={target}"
    )

    # Process tickers batch-by-batch: only one batch is rendered/held at a time
    # (bounded memory), and we stop as soon as the datapoint target is reached.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(output_path, _PARQUET_SCHEMA)
    n_written = 0
    n_filings = 0
    try:
        for tbatch in _chunk(tickers, _BATCH_SIZE):
            filings = _filings_for_tickers(tbatch, ticker_map, forms, per_company)
            if limit > 0:
                filings = filings[: max(0, limit - n_filings)]
            if not filings:
                if limit > 0 and n_filings >= limit:
                    break
                continue
            n_filings += len(filings)

            rendered = await asyncio.gather(
                *(_render_filing(f, render_sem) for f in filings)
            )
            tasks = _build_window_tasks(filings, rendered)
            llm_outputs = (
                await asyncio.gather(*(_llm_task(llm, sem, t) for t in tasks))
                if tasks
                else []
            )
            for task, out in zip(tasks, llm_outputs):
                row = _assemble_row(task, out, rng, win_counter)
                if row is None:
                    continue
                writer.write_table(pa.Table.from_pylist([row], schema=_PARQUET_SCHEMA))
                n_written += 1
                if max_datapoints and n_written >= max_datapoints:
                    break

            print(f"  {n_filings} filing(s) processed, {n_written} datapoint(s) written")
            if max_datapoints and n_written >= max_datapoints:
                break
            if limit > 0 and n_filings >= limit:
                break
    finally:
        writer.close()

    if n_written == 0:
        output_path.unlink(missing_ok=True)
        sys.exit("error: no datapoints produced; nothing to write")
    print(f"\nwrote {output_path}: {n_written} datapoint(s) from {n_filings} filing(s)")

    if push_hub:
        _push_to_hub(output_path, hub_repo, hub_private)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--tickers",
        default=_DEFAULT_TICKERS,
        help="Comma-separated tickers to harvest (default: a basket of large filers)",
    )
    parser.add_argument(
        "--random",
        type=int,
        default=0,
        metavar="N",
        help=f"Instead of --tickers, randomly sample N companies (seeded by "
        f"--seed) from the largest {_RANDOM_UNIVERSE} SEC filers",
    )
    parser.add_argument(
        "--max-datapoints",
        type=int,
        default=0,
        metavar="M",
        help="Stop once M datapoints have been written (0 = no cap)",
    )
    parser.add_argument(
        "--forms",
        default=",".join(_DEFAULT_FORMS),
        help=f"Comma-separated SEC form types (default: {','.join(_DEFAULT_FORMS)})",
    )
    parser.add_argument(
        "--filings-per-company",
        type=int,
        default=1,
        help="Most-recent matching filings to take per company (default: 1)",
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=0,
        help="Process at most N filings total (0 = all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_DEFAULT_SEED,
        help=f"Seed for query shuffling (default: {_DEFAULT_SEED})",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=str(_OUT_PATH),
        help=f"Output parquet path (default: {_OUT_PATH})",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="After building, upload the parquet to the HF Hub as a dataset "
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
    forms = tuple(f.strip() for f in args.forms.split(",") if f.strip())
    if args.random > 0:
        tickers = _sample_tickers(args.random, args.seed)
        print(f"sampled {len(tickers)} random ticker(s) (seed={args.seed})")
    else:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        sys.exit("error: no tickers (use --tickers or --random N)")
    asyncio.run(
        _run(
            tickers=tickers,
            forms=forms,
            per_company=args.filings_per_company,
            limit=args.limit,
            seed=args.seed,
            output_path=Path(args.output),
            max_datapoints=args.max_datapoints,
            push_hub=args.push_to_hub,
            hub_repo=args.hub_repo,
            hub_private=args.hub_private,
        )
    )


if __name__ == "__main__" and "__file__" in globals():
    main()
