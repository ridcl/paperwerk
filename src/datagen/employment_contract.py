"""Synthetic employment contract generator: multi-page PDF + bbox JSON + per-page annotated PNGs.

Renders a three-page contract via Playwright at A4 (96 DPI, 794×1123 CSS px),
emits a PDF through Chromium's print pipeline with explicit page-break-after
boundaries on fixed-height `.page` divs, and captures every `[data-field=...]`
occurrence with its page index and within-page bounding box.

Each field may appear multiple times (e.g. signature_date in both signature
cells); every occurrence is recorded as its own entry in `fields[]`.

Bboxes are in CSS pixels with top-left origin, scoped to the page they appear
on. Rasterize the PDF at 96 DPI for direct alignment, or multiply by 0.75 to
convert to PDF points.

Run:
    python -m datagen.employment_contract --out-dir output/datagen --n 3
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright


FIELD_NAMES = [
    "contract_no",
    "contract_date",
    "employer_name",
    "employer_address",
    "employee_name",
    "employee_address",
    "position",
    "start_date",
    "term",
    "salary",
    "working_hours",
    "employer_signature",
    "employee_signature",
    "signature_date",
]


@dataclass
class Field:
    name: str
    value: str
    page: int
    bbox: tuple[int, int, int, int]  # (x0, y0, x1, y1) in within-page CSS px


# A4 at 96 DPI: 210mm × 297mm → ~794 × 1123 CSS px.
_A4_WIDTH_PX = 794
_A4_HEIGHT_PX = 1123


_FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "Michael", "Jennifer", "William",
    "Linda", "David", "Elizabeth", "Sarah", "Daniel", "Karen", "Matthew",
    "Susan", "Andrew", "Margaret", "Steven", "Lisa", "Paul",
]
_LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Wilson", "Anderson",
    "Taylor", "Moore", "Jackson", "Martin", "Lee", "Thompson", "White",
]
_COMPANY_BASES = [
    "Apex", "Northbridge", "Pacific Crest", "Summit", "Riverbend",
    "Crestwood", "Ironclad", "Stellar", "Lakeside", "Vanguard",
    "Pinnacle", "Cobalt", "Greystone", "Harborline",
]
_COMPANY_SUFFIXES = [
    "Industries", "Holdings", "Solutions LLC", "Analytics Inc.",
    "Partners", "Group", "Technologies", "Logistics Co.", "Ventures",
]
_STREETS = [
    "Main St", "Oak Ave", "Cedar Ln", "Elm St", "Park Rd", "Broadway",
    "Market St", "Washington Ave", "Lincoln Blvd", "Highland Dr",
]
_CITIES = [
    ("Springfield", "IL", "62701"),
    ("Madison", "WI", "53703"),
    ("Boulder", "CO", "80302"),
    ("Asheville", "NC", "28801"),
    ("Salem", "OR", "97301"),
    ("Burlington", "VT", "05401"),
    ("Lexington", "KY", "40507"),
    ("Tacoma", "WA", "98402"),
]
_POSITIONS = [
    "Senior Software Engineer",
    "Product Manager",
    "Marketing Director",
    "Account Executive",
    "Data Analyst",
    "Operations Lead",
    "UX Designer",
    "Financial Controller",
    "Customer Success Manager",
]
_TERMS = [
    "for an indefinite period",
    "for a fixed term of one (1) year",
    "for a fixed term of two (2) years",
    "for a fixed term of three (3) years",
]
_HOURS = ["40 hours per week", "37.5 hours per week", "35 hours per week"]


def _random_date(rng: random.Random, start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=rng.randint(0, delta))


def _random_address(rng: random.Random) -> str:
    city, state, zip_ = rng.choice(_CITIES)
    return f"{rng.randint(100, 9999)} {rng.choice(_STREETS)}, {city}, {state} {zip_}"


def random_data(rng: random.Random) -> dict[str, str]:
    contract_date = _random_date(rng, date(2023, 1, 1), date(2025, 12, 31))
    start = contract_date + timedelta(days=rng.randint(7, 60))
    sig_date = contract_date + timedelta(days=rng.randint(0, 4))
    salary = rng.randint(55, 220) * 1000
    return {
        "contract_no": f"EMP-{rng.randint(10000, 99999)}",
        "contract_date": contract_date.strftime("%B %d, %Y"),
        "employer_name": f"{rng.choice(_COMPANY_BASES)} {rng.choice(_COMPANY_SUFFIXES)}",
        "employer_address": _random_address(rng),
        "employee_name": f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}",
        "employee_address": _random_address(rng),
        "position": rng.choice(_POSITIONS),
        "start_date": start.strftime("%B %d, %Y"),
        "term": rng.choice(_TERMS),
        "salary": f"${salary:,}",
        "working_hours": rng.choice(_HOURS),
        "employer_signature": "[signature]",
        "employee_signature": "[signature]",
        "signature_date": sig_date.strftime("%B %d, %Y"),
    }


def _signature_svg(rng: random.Random, width: int = 220, height: int = 56) -> str:
    """Generate a smooth random squiggle that resembles a handwritten signature."""
    margin = 8
    n = rng.randint(7, 11)
    step = (width - 2 * margin) / n
    points: list[tuple[float, float]] = []
    for i in range(n + 1):
        px = margin + i * step + rng.uniform(-step * 0.15, step * 0.15)
        # Combine a slow oscillation with smaller random jitter for an
        # organic-looking baseline.
        wave = math.sin(i * rng.uniform(0.6, 1.1) + rng.uniform(0, math.pi)) * height * 0.25
        py = height / 2 + wave + rng.uniform(-height * 0.12, height * 0.12)
        points.append((px, py))

    # Quadratic curves through midpoints for a smooth path.
    parts = [f"M {points[0][0]:.1f},{points[0][1]:.1f}"]
    for i in range(1, len(points) - 1):
        mid_x = (points[i][0] + points[i + 1][0]) / 2
        mid_y = (points[i][1] + points[i + 1][1]) / 2
        parts.append(f"Q {points[i][0]:.1f},{points[i][1]:.1f} {mid_x:.1f},{mid_y:.1f}")
    parts.append(f"T {points[-1][0]:.1f},{points[-1][1]:.1f}")
    path_d = " ".join(parts)

    return (
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" '
        f'style="display:block;margin:0 auto;">'
        f'<path d="{path_d}" stroke="#1a3a8a" stroke-width="2.2" '
        f'fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
        f'</svg>'
    )


# Three fixed-height `.page` divs with `page-break-after: always` so that the
# print pipeline produces exactly one PDF page per div, and the full-page
# screenshot tiles cleanly into N × 1123-px slices.
_HTML_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
  @page {{ size: A4; margin: 0; }}
  html, body {{ margin: 0; padding: 0; background: #fff; }}
  body {{
    font-family: "DejaVu Serif", "Liberation Serif", Georgia, serif;
    font-size: 11pt;
    line-height: 1.45;
    color: #111;
  }}
  .page {{
    width: 794px;
    height: 1123px;
    page-break-after: always;
    overflow: hidden;
    padding: 60px 70px;
    box-sizing: border-box;
    position: relative;
  }}
  .page:last-child {{ page-break-after: auto; }}
  h1 {{
    text-align: center;
    font-size: 22pt;
    letter-spacing: 3px;
    margin: 0 0 4px;
  }}
  .preamble {{
    text-align: center;
    font-style: italic;
    color: #555;
    margin-bottom: 22px;
  }}
  .meta {{
    display: flex;
    justify-content: space-between;
    font-size: 10pt;
    margin-bottom: 18px;
    padding-bottom: 8px;
    border-bottom: 1px solid #999;
  }}
  .value {{
    display: inline-block;
    white-space: nowrap;
    border-bottom: 1px solid #888;
    padding: 0 4px;
    min-width: 160px;
  }}
  .meta .value {{ min-width: 130px; }}
  h2 {{
    font-size: 12pt;
    margin: 16px 0 6px;
    border-bottom: 1px solid #ccc;
    padding-bottom: 2px;
  }}
  .field-row {{ margin: 6px 0; }}
  .field-row .label {{
    display: inline-block;
    width: 170px;
    font-weight: bold;
  }}
  p {{ margin: 6px 0 10px; text-align: justify; }}
  .signature-row {{
    display: flex;
    justify-content: space-around;
    margin-top: 36px;
    gap: 20px;
  }}
  .sig-cell {{ text-align: center; width: 280px; }}
  .sig-image {{
    height: 56px;
    margin: 0 auto 2px;
    width: 220px;
  }}
  .sig-line {{
    border-top: 1px solid #000;
    padding-top: 4px;
    font-size: 10pt;
  }}
  .sig-line .value {{ min-width: 130px; }}
  .footer {{
    position: absolute;
    bottom: 26px;
    left: 70px;
    right: 70px;
    text-align: center;
    font-size: 9pt;
    color: #888;
    border-top: 1px solid #ddd;
    padding-top: 4px;
  }}
</style>
</head>
<body>

<div class="page">
  <h1>EMPLOYMENT AGREEMENT</h1>
  <div class="preamble">This Agreement is entered into by and between the parties identified below.</div>

  <div class="meta">
    <div>Contract No: <span class="value" data-field="contract_no">{contract_no}</span></div>
    <div>Date: <span class="value" data-field="contract_date">{contract_date}</span></div>
  </div>

  <h2>Article I &mdash; The Parties</h2>
  <p>This Employment Agreement (the &ldquo;Agreement&rdquo;) is made as of the date noted above between the Employer and the Employee identified in this Article. Each party represents that it has the legal capacity and authority to enter into this Agreement.</p>
  <div class="field-row"><span class="label">Employer:</span><span class="value" data-field="employer_name">{employer_name}</span></div>
  <div class="field-row"><span class="label">Registered office:</span><span class="value" data-field="employer_address">{employer_address}</span></div>
  <div class="field-row"><span class="label">Employee:</span><span class="value" data-field="employee_name">{employee_name}</span></div>
  <div class="field-row"><span class="label">Residential address:</span><span class="value" data-field="employee_address">{employee_address}</span></div>

  <h2>Article II &mdash; Schedule of Key Terms</h2>
  <p>The Employer hereby engages the Employee, and the Employee hereby accepts engagement, on the terms set out below and elaborated in the articles that follow.</p>
  <div class="field-row"><span class="label">Position:</span><span class="value" data-field="position">{position}</span></div>
  <div class="field-row"><span class="label">Start date:</span><span class="value" data-field="start_date">{start_date}</span></div>
  <div class="field-row"><span class="label">Term:</span><span class="value" data-field="term">{term}</span></div>
  <div class="field-row"><span class="label">Annual salary:</span><span class="value" data-field="salary">{salary}</span></div>
  <div class="field-row"><span class="label">Working hours:</span><span class="value" data-field="working_hours">{working_hours}</span></div>

  <h2>Article III &mdash; Recitals</h2>
  <p>WHEREAS, the Employer is engaged in the business of providing professional services and wishes to retain the skills and labor of the Employee; and</p>
  <p>WHEREAS, the Employee represents that they possess the requisite qualifications, experience, and willingness to perform the duties of the Position;</p>
  <p>NOW, THEREFORE, in consideration of the mutual promises and covenants set forth herein, the parties agree as follows in the articles that follow.</p>

  <div class="footer">Page 1 of 3 &mdash; Contract {contract_no}</div>
</div>

<div class="page">
  <h2>Article IV &mdash; Duties and Responsibilities</h2>
  <p>The Employee shall faithfully perform the duties and responsibilities customarily associated with the Position, together with such additional duties consistent with the Position as may be assigned from time to time by the Employer. The Employee shall devote their full professional time, attention, and best efforts to the performance of these duties and shall not, without the prior written consent of the Employer, engage in any other business activity that materially interferes with the performance of their obligations hereunder.</p>

  <h2>Article V &mdash; Compensation and Benefits</h2>
  <p>The Employer shall pay the Employee the Annual Salary stated in Article II, payable in regular installments in accordance with the Employer&rsquo;s standard payroll practices and subject to applicable tax withholdings. In addition to base salary, the Employee shall be eligible to participate in the Employer&rsquo;s standard benefit programs, including health insurance, retirement plan contributions, and paid time off, as those programs may be amended from time to time. The Employee shall be reimbursed for reasonable and necessary business expenses incurred in the course of their duties, subject to the Employer&rsquo;s expense reimbursement policies.</p>

  <h2>Article VI &mdash; Confidentiality</h2>
  <p>The Employee acknowledges that, during the term of this Agreement, they may have access to and become acquainted with confidential information belonging to the Employer, including but not limited to trade secrets, customer and prospect lists, financial data, business strategies, pricing information, and proprietary technical information (collectively, &ldquo;Confidential Information&rdquo;). The Employee agrees that they shall not, during or after the term of this Agreement, disclose any Confidential Information to any third party or use any Confidential Information for any purpose other than the performance of their duties hereunder, except as expressly authorized in writing by the Employer or as required by law.</p>

  <h2>Article VII &mdash; Intellectual Property</h2>
  <p>All inventions, works of authorship, designs, processes, and other intellectual property created by the Employee, alone or jointly with others, in the course of their employment or using the resources of the Employer, shall be the sole and exclusive property of the Employer. The Employee hereby assigns to the Employer all right, title, and interest in such intellectual property and agrees to execute, at the Employer&rsquo;s expense, any documents reasonably necessary to perfect such assignment.</p>

  <div class="footer">Page 2 of 3 &mdash; Contract {contract_no}</div>
</div>

<div class="page">
  <h2>Article VIII &mdash; Non-Competition and Non-Solicitation</h2>
  <p>For a period of twelve (12) months following the termination of employment for any reason, the Employee shall not, directly or indirectly, engage in any business that competes with the Employer within the geographic territory in which the Employer conducts business, nor solicit any customer or employee of the Employer for the purpose of diverting their business or employment to a competing entity.</p>

  <h2>Article IX &mdash; Termination</h2>
  <p>Employment under this Agreement may be terminated by the Employer for cause, including but not limited to material breach of this Agreement, willful misconduct, or persistent failure to perform assigned duties. Either party may terminate this Agreement without cause upon thirty (30) days&rsquo; written notice to the other party. Upon termination, the Employee shall return all property of the Employer in their possession and shall continue to be bound by the confidentiality, non-competition, and non-solicitation provisions of this Agreement.</p>

  <h2>Article X &mdash; Governing Law and Dispute Resolution</h2>
  <p>This Agreement shall be governed by and construed in accordance with the laws of the jurisdiction in which the Employer&rsquo;s registered office is located, without regard to its conflict of laws principles. Any dispute arising out of or relating to this Agreement shall be resolved by binding arbitration administered under the rules of a recognized arbitration body, with judgment on the award entered in any court of competent jurisdiction.</p>

  <h2>Article XI &mdash; Acceptance</h2>
  <p>IN WITNESS WHEREOF, the parties have executed this Agreement as of the date first written above. Each party represents that they have read, understood, and voluntarily agreed to the terms hereof.</p>

  <div class="signature-row">
    <div class="sig-cell">
      <div class="sig-image" data-field="employer_signature">{signature_employer_svg}</div>
      <div class="sig-line">
        <strong>{employer_name}</strong><br>
        Authorized Representative<br>
        Date: <span class="value" data-field="signature_date">{signature_date}</span>
      </div>
    </div>
    <div class="sig-cell">
      <div class="sig-image" data-field="employee_signature">{signature_employee_svg}</div>
      <div class="sig-line">
        <strong>{employee_name}</strong><br>
        Employee<br>
        Date: <span class="value" data-field="signature_date">{signature_date}</span>
      </div>
    </div>
  </div>

  <div class="footer">Page 3 of 3 &mdash; Contract {contract_no}</div>
</div>

</body>
</html>
"""


def _capture_fields(page, data: dict[str, str]) -> list[Field]:
    """Collect every `[data-field]` occurrence with page index and within-page bbox."""
    fields: list[Field] = []
    for name in FIELD_NAMES:
        locators = page.locator(f'[data-field="{name}"]').all()
        if not locators:
            raise RuntimeError(f"Field {name!r} not present in document")
        for loc in locators:
            box = loc.bounding_box()
            if box is None:
                raise RuntimeError(
                    f"Field {name!r} has no bounding box (element not visible)"
                )
            x = round(box["x"])
            y = round(box["y"])
            w = round(box["width"])
            h = round(box["height"])
            page_idx = y // _A4_HEIGHT_PX
            local_y = y - page_idx * _A4_HEIGHT_PX
            fields.append(Field(
                name=name,
                value=data[name],
                page=page_idx,
                bbox=(x, local_y, x + w, local_y + h),
            ))
    return fields


def render(
    data: dict[str, str], extras: dict[str, str]
) -> tuple[bytes, bytes, list[Field]]:
    """Render the contract; return (pdf_bytes, png_bytes, fields).

    `extras` carries non-field render-only content (e.g. signature SVGs).
    The PNG is a single full-page screenshot covering all pages stacked
    vertically; slice by `_A4_HEIGHT_PX` to get individual pages.
    """
    html = _HTML_TEMPLATE.format(**data, **extras)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page(
                viewport={"width": _A4_WIDTH_PX, "height": _A4_HEIGHT_PX},
                device_scale_factor=1,
            )
            page.emulate_media(media="print")
            page.set_content(html, wait_until="networkidle")
            fields = _capture_fields(page, data)
            png_bytes = page.screenshot(type="png", full_page=True)
            pdf_bytes = page.pdf(
                width=f"{_A4_WIDTH_PX}px",
                height=f"{_A4_HEIGHT_PX}px",
                margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
                print_background=True,
            )
        finally:
            browser.close()
    return pdf_bytes, png_bytes, fields


def annotate(png_bytes: bytes, fields: list[Field]) -> list[bytes]:
    """Slice the full-page screenshot per page and overlay bboxes on each slice."""
    img = Image.open(BytesIO(png_bytes)).convert("RGB")
    n_pages = max(1, math.ceil(img.height / _A4_HEIGHT_PX))
    by_page: dict[int, list[Field]] = {}
    for f in fields:
        by_page.setdefault(f.page, []).append(f)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11
        )
    except OSError:
        font = ImageFont.load_default()

    outputs: list[bytes] = []
    for p in range(n_pages):
        top = p * _A4_HEIGHT_PX
        bottom = min(img.height, top + _A4_HEIGHT_PX)
        slice_img = img.crop((0, top, img.width, bottom)).copy()
        draw = ImageDraw.Draw(slice_img)
        for f in by_page.get(p, []):
            x0, y0, x1, y1 = f.bbox
            draw.rectangle((x0, y0, x1, y1), outline=(220, 30, 30), width=2)
            draw.text((x0 + 2, max(0, y0 - 13)), f.name, fill=(220, 30, 30), font=font)
        out = BytesIO()
        slice_img.save(out, format="PNG")
        outputs.append(out.getvalue())
    return outputs


def main(out_dir: str = "output/datagen", n: int = 3, seed: int = 0) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    for i in range(n):
        data = random_data(rng)
        extras = {
            "signature_employer_svg": _signature_svg(rng),
            "signature_employee_svg": _signature_svg(rng),
        }
        pdf, png, fields = render(data, extras)
        pages = annotate(png, fields)

        base = out / f"contract_{i:02d}"
        base.with_suffix(".pdf").write_bytes(pdf)
        base.with_suffix(".json").write_text(
            json.dumps(
                {"data": data, "fields": [asdict(f) for f in fields]},
                indent=2,
            )
        )
        for p, png_bytes in enumerate(pages):
            out.joinpath(f"contract_{i:02d}_p{p}_annotated.png").write_bytes(png_bytes)
        print(f"wrote {base}.pdf ({len(pages)} pages, {len(fields)} field occurrences)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="output/datagen")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(out_dir=args.out_dir, n=args.n, seed=args.seed)
