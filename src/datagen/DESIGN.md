# Synthetic document generator — design plan

## Goal

Generate a large, diverse training set of synthetic documents covering many
real-world types (CVs, contracts, certificates, IDs, bank statements,
invoices, etc.). Each generated sample must include the rendered file
(PDF/PNG), the underlying data record, and per-field bounding boxes for
KV/VQA training.

## Core architecture

Two-stage pipeline, with a clear split between **layout** (PII-free,
audited, reusable) and **content** (per-sample, scenario-driven):

1. **Templates** are checked-in `.html` assets. Each template is a
   self-contained HTML file with `<span data-field="X">{{X}}</span>`
   placeholders, no real PII anywhere (not even in static text). Templates
   are generated once from real reference documents, scrubbed by the
   `_force_placeholder_content` guardrail, and reviewed by a human before
   being committed.

2. **Scenarios** are rows in a JSON manifest. Each scenario binds a
   specific template to a specific combination of axis values:
   `{ "doc_type": "cv", "template_id": "cv_two_column_v1", "axes": { ... } }`
   Scenarios are the unit of diversity control: balancing the manifest
   balances the dataset. The render step takes a scenario, asks the LLM
   for a synthetic data record matching that scenario's axes, fills the
   template, and writes outputs.

Splitting like this means the expensive LLM call (template generation
from a real doc) happens once per template, not once per sample. The
hot path is template + small data-synthesis call → local Playwright
render. At scale, ~$0.01–0.02 per generated sample.

## Repository layout

```
src/datagen/
  generator.py              # current LLM-driven generator (will be refactored)
  render.py                 # shared Playwright render + annotate (extract from generator.py)
  catalog.py                # corpus cataloging (LLM-assisted, see step 1)
  scenarios.py              # scenario manifest builder + sampler
  synthesize.py             # data synthesis from a scenario
  templates/
    cv/
      axes.yaml             # axes config: industry, seniority, region, name_origin, ...
      single_column.html
      two_column_sidebar.html
      academic.html
      ...
    employment_contract/
      axes.yaml             # jurisdiction, term_type, role_seniority, ...
      basic_3page.html
      ...
    bank_statement/
      axes.yaml             # account_type, currency, txn_volume_tier, period_length, ...
      ...
    passport/
    marriage_certificate/
    ...
  scenarios.json            # generated manifest of (template, axes) tuples
```

Templates and `axes.yaml` files are first-class committed assets — never
regenerated as part of a render run.

## Build order

### Step 0 — refactor shared rendering

Extract the Playwright render + per-page screenshot slicing + bbox
annotation logic out of `generator.py` (and ideally
`employment_contract.py`) into `render.py`. Single source of truth for
the `<div class="page">` 794×1123 contract. New modules call
`render.render_template(html, data) -> (pdf, png, fields)` and
`render.annotate(png, fields) -> list[bytes]`.

### Step 1 — catalog the real corpus

Inventory the real reference documents in `/data/Documents/` (or wherever
the user points). For each document, ask Claude (vision) to identify:
- Document type (controlled vocabulary; if a new type, add it).
- Observable layout variants (single-column / table-heavy / multi-page / etc.).
- Field set: every variable text region, named in snake_case.

Output: `catalog.json` keyed by source path, listing detected type +
variant + fields. This becomes the input plan for steps 2 and 3.

Implementation: `python -m datagen.catalog /data/Documents/ -o catalog.json`.

### Step 2 — generate templates per (type, variant)

For each (type, variant) combination identified in the catalog:
- Pick a representative real document of that type+variant.
- Call the existing template-generation pipeline (now in
  `generator.py` → will be moved into `templates.py` or similar) with an
  extra system-prompt slot describing the desired variant
  ("single-column chronological CV", "two-column CV with left sidebar",
  etc.) so variants don't collapse into near-duplicates.
- Apply `_force_placeholder_content` scrubber.
- Write to `templates/<type>/<variant>.html`.
- **Human review gate**: open each template, verify no PII has leaked
  into static text (section headings, footers, watermarks). Only
  reviewed templates count as released.

Reuse existing `_TEMPLATE_SYSTEM_PROMPT` but extend with a
`{variant_description}` slot.

### Step 3 — author per-type axes configs

For each document type, hand-write `templates/<type>/axes.yaml` listing
the relevant diversity axes and value pools. Examples:

```yaml
# templates/cv/axes.yaml
industry: [technology, finance, healthcare, education, retail, ...]
seniority: [recent_graduate, early_career, mid_career, senior, executive]
region: [na, uk, nordic, dach, latam, south_asia, east_asia, mena, ...]
name_origin: [anglo, hispanic, slavic, south_asian, east_asian, ...]
```

```yaml
# templates/passport/axes.yaml
issuing_country: [us, uk, de, fr, in, jp, ng, br, ...]
doc_age_years: [new, mid, near_expiry]
gender: [m, f, x]
name_origin: [...]   # tied to issuing_country in a few cases
```

```yaml
# templates/bank_statement/axes.yaml
account_type: [checking, savings, joint, business]
currency: [usd, eur, gbp, jpy, inr, ...]
txn_volume_tier: [low_5_10, medium_20_40, high_60_100]
period_length: [monthly, quarterly]
```

The CV-specific axis lists currently hardcoded in `generator.py` move into
`templates/cv/axes.yaml`. The `_diversity_hint()` builder becomes
generic: load axes config, draw one value per axis, format into a profile
string. The English-language descriptions for each axis value live in the
YAML or in a per-type `hint_template.txt` so the LLM can get rich phrasing
without us inlining it in code.

### Step 4 — build the scenario manifest

`scenarios.py` builds `scenarios.json` by:
- Loading per-type axes configs.
- For each type, picking a target sample count.
- For each sample slot, choosing (a) a template_id from
  `templates/<type>/*.html`, (b) one value per axis, with a strategy:
  - `uniform`: independent uniform per axis (simple, may
    under-represent rare combinations).
  - `latin_hypercube`: stratified — guarantees each axis value appears
    at least once per template before any value repeats.
  - `manifest`: an explicit hand-authored list (for the highest-control,
    smallest case).
- Writing `{ "scenario_id": "...", "doc_type": "...", "template_id": "...",
   "axes": { ... } }` rows.

Pre-binding template at scenario time (not at render time) means the
manifest is a complete description of the dataset.

CLI: `python -m datagen.scenarios --total 5000 --strategy latin_hypercube
-o scenarios.json`.

### Step 5 — render samples from scenarios

A worker over `scenarios.json` that:
1. Loads the bound template HTML.
2. Loads the type's axes config and converts the scenario's axis values
   into a hint string.
3. Calls `synthesize.synthesize_data(field_names, llm, hint=...)`.
4. Calls `render.render_template(template, data)`.
5. Writes outputs: `output/<doc_type>/<scenario_id>.pdf`,
   `<scenario_id>.json`, `<scenario_id>_p{N}_annotated.png`.

`render.py` is unchanged from step 0. Synthesis prompt is the existing
`_DATA_PROMPT`. Concurrency: use `paperwerk.async_utils.gather_limited` over
`LLM.ainvoke` for the synthesis calls; Playwright stays sync (browser
launch is the heavy part — pool browsers if rendering becomes a
bottleneck).

CLI: `python -m datagen.render_scenarios scenarios.json -o output/`.

### Step 6 (later) — augmentation pipeline

Real-world documents are scanned, photographed, rotated, JPEG-compressed,
shadowed. Add a post-render augmentation stage that takes
`(rendered_png, fields)` and produces
`(augmented_png, transformed_fields)` by applying:
- Random rotation (±5°), perspective warp.
- Lighting/shadow gradient.
- JPEG re-encoding at varying quality.
- Optional: paper texture overlay, ink bleeding, fold lines.

Bboxes must be transformed through the same affine/projective transform.
Store augmented variants alongside originals so the trainer can sample
either.

This is local-only, no LLM, embarrassingly parallel.

## Quality control

- **Per-template smoke test**: render once with placeholder data (e.g.
  `{name: "Lorem Ipsum", ...}`), assert every `data-field` produces a
  non-None bbox, assert no rendered text overflows the `.page` div.
- **Per-scenario validation on render**: same bbox-not-None check on
  every produced sample; warn (not fail) on overflow so a few weird
  scenarios don't tank a batch run.
- **Human spot-check sampling**: render a small uniform sample of the
  manifest at the start of each large run before committing to the
  full N.

## Open questions

- **Source corpus scope**: which directory holds the reference docs, and
  do we have the rights to use them as layout references? (Layout-only
  use is generally fine; templates carry no PII by construction.)
- **Distribution targets**: do we want the synthetic distribution to
  match the expected real-world inference distribution, or to be
  uniform over types for training balance? (Affects scenarios target
  counts.)
- **Multi-page templates with variable page count**: e.g. a bank
  statement that's 1 page for low_volume but 4 pages for high_volume.
  Either author one template per length bucket, or use a parameterized
  template that the synth step expands. Start with one template per
  length bucket — simpler.

## Cost (Sonnet 4.6, see prior estimate)

- One-time per template: ~$0.09 (template gen). With ~3–5 templates per
  type × 10 types = 30–50 templates → ~$3–5 total, one-time.
- Per generated sample: ~$0.013 (synthesis only, template is local).
- 5,000 samples: ~$65 in synthesis + ~$5 in templates = ~$70 end-to-end.
- Augmentation: free (local).
