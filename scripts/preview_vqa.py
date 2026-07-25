"""Render sample rows of a VQA parquet to annotated PDFs.

Picks N random rows, decodes every page image, overlays each answer's
bounding box with its query label, and writes one multi-page PDF per row to
``output/vqa_form_like_{i}.pdf``.

Usage:
    python scripts/preview_vqa.py
    python scripts/preview_vqa.py --parquet /data/paperwerk/vqa_20260725_form_like.parquet -n 10
"""

from __future__ import annotations

import argparse
import random
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

_DEFAULT_PARQUET = Path("/data/paperwerk/vqa_20260725_form_like.parquet")
_OUT_DIR = Path("output")
_COLOR = (220, 20, 20)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
        )
    except OSError:
        return ImageFont.load_default()


def _annotate_page(img: Image.Image, answers: list[dict]) -> Image.Image:
    canvas = img.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size
    font = _font(max(12, h // 95))
    for a in answers:
        x0, y0, x1, y1 = a["bounding_box"]
        px0, py0, px1, py1 = x0 * w, y0 * h, x1 * w, y1 * h
        draw.rectangle((px0, py0, px1, py1), outline=_COLOR, width=2)
        label = a["query"]
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        # Label sits just above the box (or just below if near the top edge).
        ly = py0 - th - 4 if py0 - th - 4 >= 0 else py1 + 2
        draw.rectangle((px0, ly, px0 + tw + 4, ly + th + 4), fill=_COLOR)
        draw.text((px0 + 2, ly + 1), label, fill=(255, 255, 255), font=font)
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", type=Path, default=_DEFAULT_PARQUET)
    ap.add_argument("-n", "--count", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    table = pq.read_table(args.parquet)
    n = min(args.count, table.num_rows)
    idx = random.Random(args.seed).sample(range(table.num_rows), n)
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    for i, ridx in enumerate(idx):
        # Slice a single row: take() would concatenate the large binary image
        # column across rows and overflow the 32-bit list offset.
        row = table.slice(ridx, 1).to_pylist()[0]
        by_page: dict[int, list[dict]] = {}
        for a in row["answers"]:
            by_page.setdefault(int(a["index"]), []).append(a)
        pages = [
            _annotate_page(Image.open(BytesIO(b)), by_page.get(p, []))
            for p, b in enumerate(row["images"])
        ]
        out = _OUT_DIR / f"vqa_form_like_{i}.pdf"
        pages[0].save(out, "PDF", save_all=True, append_images=pages[1:])
        print(
            f"{out}  <- row {ridx} [{row['source']}] "
            f"{len(pages)} page(s), {len(row['answers'])} boxes"
        )


if __name__ == "__main__":
    main()
