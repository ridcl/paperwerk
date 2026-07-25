"""Enrich form templates by tagging blank fill-areas that have no data-field.

Templates generated from *blank* source forms often reproduce every label
and underline but mark only a handful of ``data-field`` regions, so most of
the form renders empty (e.g. a credit-card request form with 30 blank lines
and 5 fields). This pass finds the untagged blank fill-areas and turns them
into bound fields so data synthesis can populate them:

  * empty underline elements — a ``<div>``/``<span>`` whose class is
    ``input-line`` or whose inline style sets ``border-bottom`` and whose
    content is empty — get ``data-field="<name>"`` and a ``{{ <name> }}``
    binding;
  * bare ``<input type="text">`` (or untyped) without ``data-field`` get
    ``data-field="<name>" value="{{ <name> }}"`` (checkboxes/radios skipped).

The field ``<name>`` is derived from the nearest short preceding visible text
(the label), snake-cased and de-duplicated per template; when no plausible
label is nearby it falls back to ``field_N``.

After rewriting a template the sidecar ``.json`` schema is refreshed via
``discover_fields`` so downstream synthesis sees the new fields.

Usage:
    python scripts/enrich_form_fields.py            # enrich in place
    python scripts/enrich_form_fields.py --dry-run  # report only
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from jinja2 import Environment

import datagen
from datagen.templates import discover_fields

_TEMPLATES_ROOT = Path(datagen.__file__).resolve().parent / "assets" / "templates"

_FORM_SUFFIXES = (
    "_form", "_forms", "_application", "_worksheet", "_questionnaire",
    "_checklist", "_request", "_claim", "_authorization", "_registration",
    "_enrollment", "_waiver", "_petition", "_requisition", "_consent",
    "_ballot", "_intake",
)
_FORM_EXACT = {
    "form", "application", "worksheet", "questionnaire", "checklist",
    "request", "claim", "petition", "affidavit_form", "registration",
}
_FORM_PREFIXES = ("request_for_", "application_for_", "petition_for_")


def _is_form_name(cls: str) -> bool:
    return (
        cls in _FORM_EXACT
        or cls.endswith(_FORM_SUFFIXES)
        or cls.startswith(_FORM_PREFIXES)
    )


# A blank underline div/span: class contains input-line OR style has
# border-bottom OR its class maps to a border-bottom rule in <style>; inner
# content empty and no data-field already present.
_BLANK_EL_RE = re.compile(
    r'<(?P<tag>div|span)\b(?P<attrs>[^>]*)>(?P<inner>\s*)</(?P=tag)>',
    re.DOTALL,
)
_STYLE_BLOCK_RE = re.compile(r"<style[^>]*>(.*?)</style>", re.DOTALL | re.IGNORECASE)
_CSS_RULE_RE = re.compile(r"([^{}]+)\{([^{}]*)\}", re.DOTALL)
# A simple class selector (optionally tag-prefixed), NOT a descendant selector.
_SIMPLE_CLASS_RE = re.compile(r"^[a-z0-9]*\.([\w-]+)$")
_CLASS_ATTR_RE = re.compile(r'class="([^"]*)"')


def _border_bottom_classes(html: str) -> set[str]:
    """Class names whose <style> rule sets border-bottom via a simple selector.

    Descendant selectors (e.g. ``.table td``) are ignored so we never tag a
    container div whose *children* carry the underline.
    """
    classes: set[str] = set()
    for block in _STYLE_BLOCK_RE.findall(html):
        for sel, body in _CSS_RULE_RE.findall(block):
            if "border-bottom" not in body:
                continue
            for part in sel.split(","):
                m = _SIMPLE_CLASS_RE.match(part.strip())
                if m:
                    classes.add(m.group(1))
    return classes
# Bare input without data-field.
_INPUT_RE = re.compile(r'<input\b(?P<attrs>(?![^>]*\bdata-field=)[^>]*)>')

_TEXT_RUN_RE = re.compile(r">([^<>]{1,60})<")
_JINJA_RE = re.compile(r"\{\{.*?\}\}|\{%.*?%\}")
_PAREN_RE = re.compile(r"\(.*?\)")
_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _looks_blank_underline(attrs: str, bb_classes: set[str]) -> bool:
    if "data-field=" in attrs:
        return False
    if "input-line" in attrs:
        return True
    # style="...border-bottom..." (an underline used as a write-on line)
    if "border-bottom" in attrs:
        return True
    # class maps to a border-bottom rule in <style>
    m = _CLASS_ATTR_RE.search(attrs)
    if m:
        return any(c in bb_classes for c in m.group(1).split())
    return False


def _skip_input(attrs: str) -> bool:
    if "value=" in attrs:
        return True
    m = re.search(r'type\s*=\s*"([^"]*)"', attrs)
    if m and m.group(1).lower() in ("checkbox", "radio", "hidden", "submit", "button", "file"):
        return True
    return False


def _derive_name(pre_html: str) -> str | None:
    """Nearest short preceding visible text -> snake_case field name."""
    for text in reversed(_TEXT_RUN_RE.findall(pre_html[-400:])):
        t = _JINJA_RE.sub("", text)
        t = _PAREN_RE.sub("", t).strip(" :#*.-\t\n")
        words = _WORD_RE.findall(t)
        if 1 <= len(words) <= 6 and any(c.isalpha() for c in t):
            return "_".join(w.lower() for w in words)[:40].strip("_")
    return None


def _unique(base: str | None, used: set[str], counter: list[int]) -> str:
    if not base:
        counter[0] += 1
        base = f"field_{counter[0]}"
    name = base
    i = 2
    while name in used:
        name = f"{base}_{i}"
        i += 1
    used.add(name)
    return name


def enrich_text(text: str) -> tuple[str, int]:
    """Return (new_text, n_fields_added)."""
    used = set(re.findall(r'data-field="([^"]*)"', text))
    counter = [0]
    added = 0
    out: list[str] = []
    pos = 0

    # Interleave the two element kinds by scanning positionally so label
    # derivation always sees the up-to-date preceding text.
    bb_classes = _border_bottom_classes(text)
    events = []
    for m in _BLANK_EL_RE.finditer(text):
        if _looks_blank_underline(m.group("attrs"), bb_classes):
            events.append((m.start(), "blank", m))
    for m in _INPUT_RE.finditer(text):
        if not _skip_input(m.group("attrs")):
            events.append((m.start(), "input", m))
    events.sort(key=lambda e: e[0])

    for _, kind, m in events:
        out.append(text[pos:m.start()])
        name = _unique(_derive_name(text[:m.start()]), used, counter)
        if kind == "blank":
            tag, attrs = m.group("tag"), m.group("attrs")
            out.append(f'<{tag}{attrs} data-field="{name}">{{{{ {name} }}}}</{tag}>')
        else:  # input
            attrs = m.group("attrs").rstrip()
            out.append(f'<input{attrs} data-field="{name}" value="{{{{ {name} }}}}">')
        added += 1
        pos = m.end()
    out.append(text[pos:])
    return "".join(out), added


def main(dry_run: bool = False) -> None:
    env = Environment(autoescape=True)
    files = sorted(
        f
        for d in _TEMPLATES_ROOT.iterdir()
        if d.is_dir() and _is_form_name(d.name)
        for f in d.glob("*.html.j2")
    )
    total_added = 0
    files_changed = 0
    for f in files:
        text = f.read_text()
        try:
            env.parse(text)
        except Exception:
            continue  # don't touch malformed-as-generated templates
        new_text, added = enrich_text(text)
        if not added:
            continue
        try:
            env.parse(new_text)  # guard: never write a template we just broke
        except Exception:
            print(f"  skip {f.name}: enrichment would break Jinja parse")
            continue
        files_changed += 1
        total_added += added
        if not dry_run:
            f.write_text(new_text)
            sidecar = f.with_suffix("").with_suffix(".json")
            if sidecar.exists():
                meta = json.loads(sidecar.read_text())
                meta["schema"] = discover_fields(new_text)
                sidecar.write_text(json.dumps(meta, indent=2))
    verb = "would add" if dry_run else "added"
    print(
        f"{len(files)} form templates scanned; {verb} {total_added} field(s) "
        f"across {files_changed} file(s)"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    main(dry_run=args.dry_run)
