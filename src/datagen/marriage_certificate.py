"""Synthetic marriage certificate generator.

Renders an HTML/CSS template to PNG via Playwright, captures per-field
bounding boxes by querying `[data-field="..."]` locators, and emits an
annotated overlay alongside the raw image and field JSON.

Run:
    python -m datagen.marriage_certificate --out-dir output/datagen --n 3
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright


FIELD_NAMES = [
    "certificate_no",
    "groom_name",
    "groom_dob",
    "bride_name",
    "bride_dob",
    "marriage_date",
    "location",
    "officiant_name",
]


@dataclass
class Field:
    name: str
    value: str
    bbox: tuple[int, int, int, int]  # (x0, y0, x1, y1) in page coords


_GROOM_FIRST = [
    "James", "Robert", "Michael", "William", "David", "Richard",
    "Joseph", "Thomas", "Charles", "Christopher", "Daniel", "Matthew",
    "Anthony", "Mark", "Steven", "Andrew", "Kenneth", "Paul",
]
_BRIDE_FIRST = [
    "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara",
    "Susan", "Jessica", "Sarah", "Karen", "Nancy", "Margaret",
    "Lisa", "Betty", "Dorothy", "Sandra", "Ashley", "Donna",
]
_LAST = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
]
_OFFICIANT_TITLES = ["Rev.", "Father", "Pastor", "Judge", "Minister"]
_LOCATIONS = [
    "St. Mary's Church, Boston, MA",
    "City Hall, San Francisco, CA",
    "Trinity Cathedral, Portland, OR",
    "Grace Chapel, Austin, TX",
    "First Baptist Church, Nashville, TN",
    "County Clerk's Office, Denver, CO",
    "Sacred Heart Parish, Chicago, IL",
    "Old North Church, Philadelphia, PA",
    "Riverside Wedding Hall, Seattle, WA",
]


def _random_date(rng: random.Random, start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=rng.randint(0, delta))


def random_data(rng: random.Random) -> dict[str, str]:
    groom_dob = _random_date(rng, date(1960, 1, 1), date(2000, 12, 31))
    bride_dob = _random_date(rng, date(1960, 1, 1), date(2000, 12, 31))
    marriage = _random_date(rng, date(2015, 1, 1), date(2025, 12, 31))
    return {
        "certificate_no": f"MC-{rng.randint(100000, 999999)}",
        "groom_name": f"{rng.choice(_GROOM_FIRST)} {rng.choice(_LAST)}",
        "groom_dob": groom_dob.strftime("%B %d, %Y"),
        "bride_name": f"{rng.choice(_BRIDE_FIRST)} {rng.choice(_LAST)}",
        "bride_dob": bride_dob.strftime("%B %d, %Y"),
        "marriage_date": marriage.strftime("%B %d, %Y"),
        "location": rng.choice(_LOCATIONS),
        "officiant_name": (
            f"{rng.choice(_OFFICIANT_TITLES)} "
            f"{rng.choice(_GROOM_FIRST + _BRIDE_FIRST)} {rng.choice(_LAST)}"
        ),
    }


# System fonts only — no Google Fonts @import so the renderer works offline.
_HTML_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{
    margin: 0;
    padding: 50px 70px;
    background: #fdfaf0;
    font-family: "DejaVu Serif", "Liberation Serif", Georgia, serif;
    color: #2a1f10;
  }}
  .frame {{
    border: 6px double #8b6914;
    padding: 40px 50px;
    background: #fffef5;
  }}
  h1 {{
    text-align: center;
    font-size: 40px;
    letter-spacing: 4px;
    margin: 0 0 6px 0;
    font-variant: small-caps;
  }}
  .subtitle {{
    text-align: center;
    font-style: italic;
    font-size: 17px;
    margin-bottom: 32px;
    color: #5a4422;
  }}
  .cert-no {{
    text-align: right;
    font-size: 14px;
    margin-bottom: 22px;
  }}
  .cert-no .value {{
    min-width: 160px;
  }}
  .field-row {{
    margin: 16px 0;
    font-size: 18px;
    line-height: 1.5;
  }}
  .label {{
    display: inline-block;
    width: 200px;
    font-weight: bold;
  }}
  .value {{
    border-bottom: 1px solid #8b6914;
    padding: 0 6px;
    display: inline-block;
    min-width: 320px;
  }}
  .signoff {{
    margin-top: 50px;
    text-align: center;
    font-size: 16px;
  }}
  .signoff .value {{
    min-width: 280px;
  }}
</style>
</head>
<body>
<div class="frame">
  <h1>Certificate of Marriage</h1>
  <div class="subtitle">This is to certify that the persons named herein were lawfully joined in matrimony.</div>

  <div class="cert-no">Certificate No: <span class="value" data-field="certificate_no">{certificate_no}</span></div>

  <div class="field-row"><span class="label">Groom:</span> <span class="value" data-field="groom_name">{groom_name}</span></div>
  <div class="field-row"><span class="label">Date of Birth:</span> <span class="value" data-field="groom_dob">{groom_dob}</span></div>
  <div class="field-row"><span class="label">Bride:</span> <span class="value" data-field="bride_name">{bride_name}</span></div>
  <div class="field-row"><span class="label">Date of Birth:</span> <span class="value" data-field="bride_dob">{bride_dob}</span></div>
  <div class="field-row"><span class="label">Date of Marriage:</span> <span class="value" data-field="marriage_date">{marriage_date}</span></div>
  <div class="field-row"><span class="label">Place of Marriage:</span> <span class="value" data-field="location">{location}</span></div>

  <div class="signoff">
    Solemnized by <span class="value" data-field="officiant_name">{officiant_name}</span>
  </div>
</div>
</body>
</html>
"""


def render(data: dict[str, str]) -> tuple[bytes, list[Field]]:
    """Render the certificate to a PNG and capture per-field bboxes.

    Bounding boxes are in page (document) coordinates, matching the
    `full_page` screenshot's coordinate system.
    """
    html = _HTML_TEMPLATE.format(**data)
    fields: list[Field] = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page(
                viewport={"width": 1000, "height": 1300},
                device_scale_factor=1,
            )
            page.set_content(html, wait_until="networkidle")

            for name in FIELD_NAMES:
                box = page.locator(f'[data-field="{name}"]').bounding_box()
                if box is None:
                    raise RuntimeError(
                        f"Field {name!r} has no bounding box (element not visible)"
                    )
                x0 = round(box["x"])
                y0 = round(box["y"])
                x1 = round(box["x"] + box["width"])
                y1 = round(box["y"] + box["height"])
                fields.append(Field(name=name, value=data[name], bbox=(x0, y0, x1, y1)))

            png_bytes = page.screenshot(type="png", full_page=True)
        finally:
            browser.close()

    return png_bytes, fields


def annotate(png_bytes: bytes, fields: list[Field]) -> bytes:
    """Overlay red bounding boxes and field names on the rendered PNG."""
    img = Image.open(BytesIO(png_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14
        )
    except OSError:
        font = ImageFont.load_default()

    for f in fields:
        x0, y0, x1, y1 = f.bbox
        draw.rectangle((x0, y0, x1, y1), outline=(220, 30, 30), width=2)
        draw.text((x0 + 2, max(0, y0 - 16)), f.name, fill=(220, 30, 30), font=font)

    out = BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def main(out_dir: str = "output/datagen", n: int = 3, seed: int = 0) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    for i in range(n):
        data = random_data(rng)
        png, fields = render(data)
        annotated = annotate(png, fields)

        base = out / f"marriage_{i:02d}"
        base.with_suffix(".png").write_bytes(png)
        out.joinpath(f"marriage_{i:02d}_annotated.png").write_bytes(annotated)
        base.with_suffix(".json").write_text(
            json.dumps(
                {"data": data, "fields": [asdict(f) for f in fields]},
                indent=2,
            )
        )
        print(f"wrote {base}.png (+ .json, _annotated.png)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="output/datagen")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(out_dir=args.out_dir, n=args.n, seed=args.seed)
