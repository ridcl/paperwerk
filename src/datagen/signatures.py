"""Procedural handwritten-signature synthesis as inline SVG.

`signature_svg` returns a self-contained ``<svg>`` string containing a
random, handwriting-looking scrawl — no external assets or fonts, fully
deterministic given a seed. It is used to fill signature fields in rendered
documents so a synthetic form looks *signed* rather than blank.

`inject_signatures` rewrites a Jinja template in place: for every
``data-field`` element whose name looks like a signature (``signature``,
``signed_by``, ``authorized_signature`` …) it swaps the ``{{ ... }}`` binding
for a freshly generated SVG scrawl, and reports which field names it consumed
so the caller can drop them from data synthesis (a signature carries no text
value).

The scrawl is built from a chain of cubic Bézier segments along a jittered
baseline with occasional loops, an oversized initial flourish, and a trailing
underline — enough structure to read as a signature at a glance while varying
widely across seeds.
"""

from __future__ import annotations

import random
import re

# Field names that should be rendered as a signature rather than as text.
_SIGNATURE_HINTS = (
    "signature",
    "signed_by",
    "sign_here",
    "autograph",
    "authorized_sign",
)


def is_signature_field(name: str) -> bool:
    n = name.lower()
    return any(h in n for h in _SIGNATURE_HINTS)


def _bezier_chain(rng: random.Random, width: float, height: float) -> str:
    """Build the SVG path `d` for one signature stroke."""
    left = width * 0.05
    right = width * 0.95
    baseline = height * 0.62
    amp = height * 0.34

    n = rng.randint(5, 8)
    step = (right - left) / n

    x = left
    y = baseline + rng.uniform(-amp * 0.3, amp * 0.3)
    # Oversized initial flourish (like a capital letter loop).
    d = [f"M {x:.1f} {y:.1f}"]
    first_amp = amp * rng.uniform(1.3, 1.9)
    c1x, c1y = x + step * 0.3, y - first_amp
    c2x, c2y = x + step * 0.7, y + first_amp * 0.4
    x, y = x + step, baseline + rng.uniform(-amp * 0.2, amp * 0.2)
    d.append(f"C {c1x:.1f} {c1y:.1f} {c2x:.1f} {c2y:.1f} {x:.1f} {y:.1f}")

    direction = -1
    for _ in range(n - 1):
        c1x = x + step * rng.uniform(0.1, 0.4)
        c1y = y + direction * amp * rng.uniform(0.6, 1.2)
        c2x = x + step * rng.uniform(0.6, 0.9)
        c2y = y + direction * amp * rng.uniform(0.3, 0.9)
        x = x + step * rng.uniform(0.8, 1.15)
        y = baseline + rng.uniform(-amp * 0.35, amp * 0.35)
        d.append(f"C {c1x:.1f} {c1y:.1f} {c2x:.1f} {c2y:.1f} {x:.1f} {y:.1f}")
        direction *= -1

    return " ".join(d)


def signature_svg(
    seed: int | str | None = None,
    *,
    width: int = 240,
    height: int = 72,
    ink: str = "#12245e",
) -> str:
    """Return an inline `<svg>` string with a random signature scrawl.

    Deterministic given `seed`. `ink` is any CSS color; the default is a
    dark blue-black "pen" tone. The SVG has no width/height attributes so it
    scales to its container; a `viewBox` sets the coordinate system.
    """
    rng = random.Random(seed)
    stroke_w = rng.uniform(1.8, 2.8)
    tilt = rng.uniform(-4.0, 4.0)

    main = _bezier_chain(rng, width, height)

    # Trailing underline flourish beneath the scrawl.
    uy = height * 0.82
    ux0 = width * rng.uniform(0.08, 0.2)
    ux1 = width * rng.uniform(0.8, 0.96)
    dip = height * rng.uniform(0.05, 0.14)
    flourish = (
        f"M {ux0:.1f} {uy:.1f} "
        f"C {width * 0.35:.1f} {uy + dip:.1f} {width * 0.65:.1f} {uy + dip:.1f} "
        f"{ux1:.1f} {uy - dip * 0.4:.1f}"
    )

    # Optional detached mark (dot / cross-stroke) for extra realism.
    extra = ""
    if rng.random() < 0.6:
        mx = width * rng.uniform(0.3, 0.7)
        my = height * rng.uniform(0.12, 0.28)
        mlen = width * rng.uniform(0.04, 0.1)
        extra = (
            f'<path d="M {mx:.1f} {my:.1f} L {mx + mlen:.1f} {my - mlen * 0.3:.1f}" '
            f'stroke="{ink}" stroke-width="{stroke_w:.1f}" '
            f'fill="none" stroke-linecap="round"/>'
        )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="display:inline-block;width:100%;max-width:{width}px;height:auto;'
        f'overflow:visible;transform:rotate({tilt:.1f}deg)">'
        f'<path d="{main}" stroke="{ink}" stroke-width="{stroke_w:.1f}" fill="none" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
        f'<path d="{flourish}" stroke="{ink}" stroke-width="{stroke_w * 0.8:.1f}" '
        f'fill="none" stroke-linecap="round"/>'
        f"{extra}</svg>"
    )


# data-field element whose inner content is a single Jinja expression; we only
# rebind the innermost text-only ones (no nested tags), matching enrichment.
def _field_re(name: str) -> re.Pattern:
    esc = re.escape(name)
    return re.compile(
        r'(<(?P<tag>span|div|td|p)\b[^>]*\bdata-field="' + esc + r'"[^>]*>)'
        r'(?P<inner>(?:(?!</?(?P=tag)\b).)*?)'
        r"(</(?P=tag)>)",
        re.DOTALL,
    )


def inject_signatures(
    template_html: str,
    field_names: list[str],
    rng: random.Random,
) -> tuple[str, list[str]]:
    """Replace signature fields' bindings with SVG scrawls.

    Returns `(new_html, consumed)` where `consumed` is the list of field
    names that were turned into signatures (and therefore no longer need a
    synthesized text value). Each signature gets its own seed so multiple
    signatures on one page differ.
    """
    consumed: list[str] = []
    html = template_html
    for name in field_names:
        if not is_signature_field(name) or "[]" in name:
            continue
        pat = _field_re(name)
        if not pat.search(html):
            continue
        svg = signature_svg(seed=rng.randint(0, 2**31))
        html = pat.sub(lambda m: f"{m.group(1)}{svg}{m.group(4)}", html, count=1)
        consumed.append(name)
    return html, consumed
