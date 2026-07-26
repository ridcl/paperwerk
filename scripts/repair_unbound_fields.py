"""Repair `data-field` elements that were emitted without a Jinja expression.

The template builder (`make_template`) requires every element carrying a
``data-field`` attribute to bind a ``{{ ... }}`` expression, but a large
fraction of the generated KVP10k templates violate this: the element names
the field yet never references its value. The synthesized value then never
reaches the page — the field renders blank or shows a constant leaked from
the source document (a PII leak).

Three carrier shapes are repaired, keyed off the ``data-field`` attribute:

1. Text containers (``span``, ``a``, ``td``, ``li``, ``sup``, ``div``,
   ``textarea``, headings, …) whose inner content is text-only (no nested
   tag) and contains no ``{{``::

       <span data-field="full_name"></span>
           -> <span data-field="full_name">{{ full_name }}</span>
       <a data-field="website">http://old.example</a>
           -> <a data-field="website">{{ website }}</a>

2. ``<input>`` void elements without a ``value=`` attribute get one::

       <input type="text" data-field="ssn">
           -> <input type="text" data-field="ssn" value="{{ ssn }}">

Array-indexed fields keep the index expression already present in the
attribute; it is unwrapped to a plain subscript for the binding, e.g.
``line_items[{{ loop.index0 }}].amount`` ->
``{{ line_items[loop.index0].amount }}``. Inside the enclosing
``{% for %}`` that subscript indexes the iterated collection back to the
current item, so the loop variable name is never needed.

Elements whose inner content (or ``value=``) already references ``{{`` are
left untouched, so the pass is idempotent. Containers with *nested tags*
inside are skipped (the real field is the inner leaf), as are ``<select>``
controls (they need an ``<option>`` set, out of scope here).

Usage:
    python scripts/repair_unbound_fields.py            # repair in place
    python scripts/repair_unbound_fields.py --dry-run  # report only
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import datagen

_TEMPLATES_ROOT = Path(datagen.__file__).resolve().parent / "assets" / "templates"

# Text-holding elements whose inner content is the field value. `input`,
# `select` and other void/control elements are handled separately.
_CONTAINER_TAGS = (
    "span|a|td|th|li|sup|sub|div|p|h1|h2|h3|h4|h5|h6|strong|b|em|i|label|textarea"
)

# A container element with TEXT-ONLY inner content (`[^<]*` — no nested tag),
# so replacing the inner can never delete a nested field. Tag name is captured
# and backreferenced so open/close always agree.
_CONTAINER_RE = re.compile(
    r'(?P<open><(?P<tag>' + _CONTAINER_TAGS + r')\b[^>]*'
    r'\bdata-field=("|\')(?P<name>.*?)\3[^>]*>)'
    r'(?P<inner>[^<]*)'
    r'(?P<close></(?P=tag)>)',
    re.DOTALL,
)

# An <input> that carries data-field but has no value= attribute yet.
_INPUT_RE = re.compile(
    r'<input\b(?![^>]*\bvalue=)(?P<attrs>[^>]*\bdata-field=("|\')(?P<name>.*?)\2[^>]*)>',
)

# Unwrap `{{ ... }}` occurring inside the data-field attribute (the array
# index) down to the bare expression, so it becomes a valid subscript.
_JINJA_UNWRAP_RE = re.compile(r"\{\{\s*(.*?)\s*\}\}")


def _expr_for(name: str) -> str:
    """Canonical Jinja expression for a data-field name.

    `full_name`                        -> `full_name`
    `employer.name`                    -> `employer.name`
    `posts[{{ loop.index0 }}].title`   -> `posts[loop.index0].title`
    """
    return _JINJA_UNWRAP_RE.sub(r"\1", name)


def repair_text(text: str) -> tuple[str, Counter]:
    """Return (repaired_text, per-kind counts of elements rebound)."""
    counts: Counter = Counter()

    def fix_container(m: re.Match) -> str:
        if "{{" in m.group("inner"):
            return m.group(0)  # already bound
        counts[m.group("tag").lower()] += 1
        expr = _expr_for(m.group("name"))
        return f"{m.group('open')}{{{{ {expr} }}}}{m.group('close')}"

    def fix_input(m: re.Match) -> str:
        counts["input"] += 1
        expr = _expr_for(m.group("name"))
        return f'<input {m.group("attrs").strip()} value="{{{{ {expr} }}}}">'

    text = _CONTAINER_RE.sub(fix_container, text)
    text = _INPUT_RE.sub(fix_input, text)
    return text, counts


def main(dry_run: bool = False) -> None:
    files = sorted(_TEMPLATES_ROOT.rglob("*.html.j2"))
    total: Counter = Counter()
    files_changed = 0
    for f in files:
        text = f.read_text()
        repaired, counts = repair_text(text)
        if counts:
            files_changed += 1
            total.update(counts)
            if not dry_run:
                f.write_text(repaired)
    verb = "would rebind" if dry_run else "rebound"
    n = sum(total.values())
    print(
        f"{len(files)} templates scanned; {verb} {n} element(s) "
        f"across {files_changed} file(s)"
    )
    if total:
        print("  by tag: " + ", ".join(f"{k}={v}" for k, v in total.most_common()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    main(dry_run=args.dry_run)
