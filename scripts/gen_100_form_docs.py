"""Generate 100 synthetic documents from form-sounding templates.

Templates are selected only from classes whose *name* reads like a fill-in
form (``*_form``, ``*_application``, ``*_request``, ``*_worksheet``,
``*_questionnaire``, ``*_claim``, ``application_for_*`` …). For each of 100
output slots we:

  1. synthesize one realistic-but-synthetic record for the template's schema
     (``datagen.values.synthesize_values``),
  2. render it to a clean PDF (``datagen.render.render``),
  3. for a random half of the slots, apply scanner/photo augmentation
     (``datagen.augment`` — Stage A pixel pipeline, optionally preceded by a
     Stage B geometric warp), the other half stay clean,
  4. write ``doc_{n:03d}.pdf`` plus a ``doc_{n:03d}.json`` sidecar (fields +
     augmentation metadata) under ``output/100_sample_docs``.

Exactly 50 slots are clean and 50 augmented. Each slot retries with a fresh
template if one fails to render, so the run yields all 100 documents.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path

from jinja2 import Environment

from paperwerk.llm import LLM

import datagen
from datagen.augment import PROFILES, augment, augment_geometric
from datagen.render import render_sync
from datagen.signatures import inject_signatures
from datagen.templates import discover_fields
from datagen.values import random_values

# Skip templates that would render nearly blank (too few fillable fields).
_MIN_FIELDS = 3

_ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1/"
_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

_TEMPLATES_ROOT = Path(datagen.__file__).resolve().parent / "assets" / "templates"
_OUT_DIR = Path("output/100_sample_docs")
_N = 100
_SEED = 20260719

# Concurrency caps. LLM has its own semaphore; the CPU stage (Chromium render
# + augraphy) is heavy, so bound it separately to avoid launching 100 browsers
# / augraphy pipelines at once.
_LLM_CONCURRENCY = 8
_CPU_CONCURRENCY = 4

# Class names that read like a fill-in form.
_FORM_SUFFIXES = (
    "_form",
    "_forms",
    "_application",
    "_worksheet",
    "_questionnaire",
    "_checklist",
    "_request",
    "_claim",
    "_authorization",
    "_registration",
    "_enrollment",
    "_waiver",
    "_petition",
    "_requisition",
    "_consent",
    "_ballot",
    "_intake",
)
_FORM_EXACT = {
    "form",
    "application",
    "worksheet",
    "questionnaire",
    "checklist",
    "request",
    "claim",
    "petition",
    "affidavit_form",
    "registration",
}
_FORM_PREFIXES = ("request_for_", "application_for_", "petition_for_")


def _is_form_name(cls: str) -> bool:
    return (
        cls in _FORM_EXACT
        or cls.endswith(_FORM_SUFFIXES)
        or cls.startswith(_FORM_PREFIXES)
    )


def _form_template_pool() -> list[Path]:
    """Parseable form-sounding templates that have a schema sidecar."""
    env = Environment(autoescape=True)
    pool: list[Path] = []
    for class_dir in _TEMPLATES_ROOT.iterdir():
        if not class_dir.is_dir() or not _is_form_name(class_dir.name):
            continue
        for tpl in class_dir.glob("*.html.j2"):
            if not tpl.with_suffix("").with_suffix(".json").exists():
                continue
            text = tpl.read_text()
            try:
                env.parse(text)
            except Exception:
                continue  # skip malformed-as-generated templates
            if len(discover_fields(text)) < _MIN_FIELDS:
                continue  # skip near-blank templates (too few fillable fields)
            pool.append(tpl)
    return pool


def _augment_plan(rng: random.Random) -> dict:
    """Random augmentation recipe for one document."""
    return {
        "profile": rng.choice(PROFILES),
        "quality": round(rng.uniform(0.25, 0.9), 3),
        "geometric": rng.random() < 0.5,
        "geo_quality": round(rng.uniform(0.4, 0.9), 3),
    }


def _render_and_augment(template: str, data: dict, plan: dict | None, seed: int):
    """Sync CPU stage: render, then optionally warp + scanner-augment."""
    pdf, fields = render_sync(template, data, seed=seed)
    if plan is None:
        return pdf, fields
    if plan["geometric"]:
        pdf, fields = augment_geometric(
            pdf, fields, quality=plan["geo_quality"], seed=seed
        )
    pdf, fields = augment(
        pdf, fields, profile=plan["profile"], quality=plan["quality"], seed=seed
    )
    return pdf, fields


async def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("error: ANTHROPIC_API_KEY is not set")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(_SEED)
    pool = _form_template_pool()
    if len(pool) < 20:
        sys.exit(f"error: only {len(pool)} form templates found")
    print(f"pool: {len(pool)} form-sounding templates")

    # Exactly 50 clean / 50 augmented, shuffled across slots.
    augmented_flags = [True] * (_N // 2) + [False] * (_N - _N // 2)
    rng.shuffle(augmented_flags)

    # A shuffled fallback order of candidate templates, handed out on demand so
    # a slot that hits a render failure can retry with the next candidate.
    candidates = pool[:]
    rng.shuffle(candidates)
    next_idx = 0
    idx_lock = asyncio.Lock()
    cpu_sem = asyncio.Semaphore(_CPU_CONCURRENCY)
    done = 0

    async def _next_candidate() -> Path | None:
        nonlocal next_idx
        async with idx_lock:
            if next_idx >= len(candidates):
                return None
            c = candidates[next_idx]
            next_idx += 1
            return c

    llm = LLM(
        base_url=_ANTHROPIC_BASE_URL,
        api_key=api_key,
        model=_MODEL,
        max_concurrency=_LLM_CONCURRENCY,
    )

    async def _slot(slot: int) -> dict | None:
        nonlocal done
        augmented = augmented_flags[slot]
        slot_rng = random.Random(f"{_SEED}:{slot}")
        seed = _SEED + slot
        while True:
            tpl_path = await _next_candidate()
            if tpl_path is None:
                print(f"[slot {slot:03d}] exhausted candidates; giving up")
                return None
            meta = json.loads(tpl_path.with_suffix("").with_suffix(".json").read_text())
            schema = meta["schema"]
            cls = meta.get("class", tpl_path.parent.name)
            try:
                # Swap signature-named fields for procedural SVG scrawls; the
                # rest get synthesized text values.
                template_html, sig_fields = inject_signatures(
                    tpl_path.read_text(), schema, slot_rng
                )
                synth_fields = [f for f in schema if f not in sig_fields]
                data = await random_values(llm, synth_fields)
                plan = _augment_plan(slot_rng) if augmented else None
                async with cpu_sem:
                    pdf, fields = await asyncio.to_thread(
                        _render_and_augment, template_html, data, plan, seed
                    )
            except Exception as e:  # noqa: BLE001 - retry with next template
                print(f"[slot {slot:03d}] {cls} failed ({e!r}); retrying")
                continue

            stem = f"doc_{slot:03d}"
            (_OUT_DIR / f"{stem}.pdf").write_bytes(pdf)
            record = {
                "doc": stem,
                "class": cls,
                "template": tpl_path.name,
                "augmented": augmented,
                "augmentation": plan,
                "signature_fields": sig_fields,
                "n_pages": (max((f.page for f in fields), default=0) + 1),
                "n_field_occurrences": len(fields),
                "data": data,
                "fields": [asdict(f) for f in fields],
            }
            (_OUT_DIR / f"{stem}.json").write_text(json.dumps(record, indent=2))
            done += 1
            tag = (
                f"aug:{plan['profile']} q={plan['quality']}"
                + (" +geo" if plan["geometric"] else "")
                if plan
                else "clean"
            )
            print(f"[slot {slot:03d}] ({done}/{_N}) {cls} [{tag}]")
            return {
                k: record[k]
                for k in (
                    "doc",
                    "class",
                    "template",
                    "augmented",
                    "augmentation",
                    "n_pages",
                )
            }

    results = await asyncio.gather(*(_slot(i) for i in range(_N)))
    ok = [r for r in results if r]
    manifest = {
        "count": len(ok),
        "clean": sum(1 for r in ok if not r["augmented"]),
        "augmented": sum(1 for r in ok if r["augmented"]),
        "model": _MODEL,
        "seed": _SEED,
        "documents": ok,
    }
    (_OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"\ndone: {manifest['count']} docs "
        f"({manifest['clean']} clean, {manifest['augmented']} augmented) "
        f"-> {_OUT_DIR}"
    )


if __name__ == "__main__":
    asyncio.run(main())
