"""Stage A — pixel-space scanner-emulation augmentation for rendered PDFs.

`augment` rasterizes a rendered PDF at the requested DPI, runs each page
through a profile-driven augraphy pipeline (paper texture, lighting, ink
bleed, JPEG compression, fax-style binarization, …), and re-packs the
augmented pages back into a single image-only PDF — i.e. the kind of PDF
you'd get from scanning paper or stitching phone photos together.

Stage A is pixel-space only — no geometric warps — so the per-field
bounding boxes (stored in normalized [0, 1] per-page coordinates) are
unchanged and pass straight through. Stage B (rotation, perspective,
barrel) will require bbox transformation and lives separately.
"""

from __future__ import annotations

import random
from io import BytesIO
from typing import Callable

import numpy as np
from augraphy import (
    AugraphyPipeline,
    BadPhotoCopy,
    Brightness,
    ColorPaper,
    DirtyDrum,
    Faxify,
    InkBleed,
    Jpeg,
    LightingGradient,
    LowInkPeriodicLines,
    NoiseTexturize,
    ShadowCast,
    SubtleNoise,
)
from pdf2image import convert_from_bytes
from PIL import Image

from datagen.render import Field


_DEFAULT_DPI = 200
_DEFAULT_QUALITY = 0.7


def _lerp(q: float, worst: float, best: float) -> float:
    """Linear interp: q=0 → worst, q=1 → best. Clamps q to [0, 1]."""
    q = max(0.0, min(1.0, q))
    return worst + (best - worst) * q


def _lerp_int(q: float, worst: int, best: int) -> int:
    return int(round(_lerp(q, worst, best)))


def _lerp_range(
    q: float, worst: tuple[float, float], best: tuple[float, float]
) -> tuple[float, float]:
    return (_lerp(q, worst[0], best[0]), _lerp(q, worst[1], best[1]))


def _lerp_int_range(
    q: float, worst: tuple[int, int], best: tuple[int, int]
) -> tuple[int, int]:
    return (_lerp_int(q, worst[0], best[0]), _lerp_int(q, worst[1], best[1]))


def _clean_scan_pipeline(_rng: random.Random, q: float) -> AugraphyPipeline:
    """Flatbed-scanner output. At q=1 it's near-pristine; at q=0 it's a tired scanner."""
    return AugraphyPipeline(
        ink_phase=[
            InkBleed(
                intensity_range=_lerp_range(q, (0.2, 0.4), (0.0, 0.1)),
                p=_lerp(q, 0.7, 0.2),
            )
        ],
        paper_phase=[
            ColorPaper(
                hue_range=(0, 30),
                saturation_range=_lerp_int_range(q, (15, 35), (5, 15)),
                p=0.8,
            )
        ],
        post_phase=[
            SubtleNoise(subtle_range=_lerp_int(q, 25, 6), p=_lerp(q, 1.0, 0.3)),
            Jpeg(quality_range=_lerp_int_range(q, (55, 75), (88, 98)), p=0.9),
        ],
    )


def _phone_photo_pipeline(rng: random.Random, q: float) -> AugraphyPipeline:
    """Phone snapshot of a printed page.

    At q=1: gentle lighting (min_brightness 220), JPEG 88-98, no shadow,
    minimal noise — looks like a modern phone in even light.
    At q=0: harsh lighting (min_brightness 110), aggressive JPEG 40-60,
    visible shadow cast, more noise.
    """
    return AugraphyPipeline(
        ink_phase=[
            InkBleed(
                intensity_range=_lerp_range(q, (0.2, 0.5), (0.0, 0.1)),
                p=_lerp(q, 0.6, 0.1),
            )
        ],
        paper_phase=[
            ColorPaper(
                hue_range=(0, 40),
                saturation_range=_lerp_int_range(q, (20, 50), (5, 18)),
                p=_lerp(q, 0.95, 0.6),
            )
        ],
        post_phase=[
            LightingGradient(
                light_position=(rng.randint(100, 700), rng.randint(100, 700)),
                direction=rng.randint(0, 359),
                max_brightness=255,
                min_brightness=_lerp_int(q, 110, 220),
                mode="gaussian",
                p=_lerp(q, 0.95, 0.5),
            ),
            ShadowCast(p=_lerp(q, 0.8, 0.05)),
            SubtleNoise(subtle_range=_lerp_int(q, 30, 6), p=_lerp(q, 1.0, 0.2)),
            Jpeg(quality_range=_lerp_int_range(q, (40, 60), (88, 98)), p=1.0),
        ],
    )


def _old_photocopy_pipeline(_rng: random.Random, q: float) -> AugraphyPipeline:
    """Cheap photocopier output: even at q=1 still has banding and drum noise."""
    return AugraphyPipeline(
        ink_phase=[
            InkBleed(
                intensity_range=_lerp_range(q, (0.3, 0.6), (0.1, 0.3)),
                p=_lerp(q, 0.9, 0.5),
            )
        ],
        paper_phase=[
            ColorPaper(
                hue_range=(0, 60),
                saturation_range=_lerp_int_range(q, (30, 60), (10, 30)),
                p=1.0,
            )
        ],
        post_phase=[
            DirtyDrum(p=_lerp(q, 0.9, 0.3)),
            LowInkPeriodicLines(p=_lerp(q, 0.7, 0.2)),
            BadPhotoCopy(p=_lerp(q, 0.8, 0.3)),
            NoiseTexturize(p=_lerp(q, 0.9, 0.4)),
            Brightness(
                brightness_range=_lerp_range(q, (0.6, 0.85), (0.9, 1.0)),
                p=_lerp(q, 0.9, 0.4),
            ),
            SubtleNoise(subtle_range=_lerp_int(q, 35, 10), p=1.0),
            Jpeg(quality_range=_lerp_int_range(q, (40, 60), (70, 88)), p=1.0),
        ],
    )


def _fax_pipeline(_rng: random.Random, q: float) -> AugraphyPipeline:
    """Fax-style output. Faxify always fires (that's the profile's identity);
    quality scales noise and JPEG only."""
    return AugraphyPipeline(
        ink_phase=[
            InkBleed(
                intensity_range=_lerp_range(q, (0.4, 0.7), (0.2, 0.4)),
                p=_lerp(q, 0.9, 0.5),
            )
        ],
        paper_phase=[],
        post_phase=[
            Faxify(p=1.0),
            NoiseTexturize(p=_lerp(q, 0.8, 0.3)),
            Jpeg(quality_range=_lerp_int_range(q, (35, 55), (60, 80)), p=1.0),
        ],
    )


_PROFILES: dict[str, Callable[[random.Random, float], AugraphyPipeline]] = {
    "clean_scan": _clean_scan_pipeline,
    "phone_photo": _phone_photo_pipeline,
    "old_photocopy": _old_photocopy_pipeline,
    "fax": _fax_pipeline,
}


PROFILES: tuple[str, ...] = tuple(_PROFILES)


def _images_to_pdf(images: list[Image.Image], *, dpi: int) -> bytes:
    """Pack a list of PIL images into a single image-only PDF, one image per page.

    `dpi` sets the embedded resolution so the PDF page size in points matches
    the source — e.g. a 1654×2339 raster at 200 DPI lands on an A4 page
    (8.27 × 11.69 in), not a 23 × 32 in monster.
    """
    if not images:
        raise ValueError("no images to pack into PDF")
    rgb = [img.convert("RGB") for img in images]
    buf = BytesIO()
    rgb[0].save(
        buf,
        format="PDF",
        save_all=True,
        append_images=rgb[1:],
        resolution=dpi,
    )
    return buf.getvalue()


def augment(
    pdf_bytes: bytes,
    fields: list[Field],
    *,
    profile: str = "phone_photo",
    quality: float = _DEFAULT_QUALITY,
    dpi: int = _DEFAULT_DPI,
    seed: int | None = None,
) -> tuple[bytes, list[Field]]:
    """Apply Stage-A scanner-style augmentation to a rendered PDF.

    The PDF is rasterized at `dpi`, each page is run through the profile's
    augraphy pipeline, and the results are re-packed into a single
    image-only PDF (one image per page) — matching what you'd get from
    scanning paper or stitching phone photos together.

    `quality` is a float in `[0, 1]` (clamped). At `quality=1.0` each
    profile produces its cleanest possible variant (mild noise, gentle
    lighting, high JPEG quality); at `quality=0.0` it produces its
    nastiest (heavy noise, harsh artifacts, aggressive JPEG). The default
    target is "readable phone photo / decent scan".

    Fields are returned unchanged: Stage A is pixel-space only, and the
    normalized [0, 1] bbox coordinates are independent of raster DPI.
    """
    if profile not in _PROFILES:
        raise ValueError(
            f"unknown profile {profile!r}; choose one of {sorted(_PROFILES)}"
        )
    q = max(0.0, min(1.0, quality))
    rng = random.Random(seed)
    # Augraphy uses numpy.random internally; seed it for reproducibility too.
    if seed is not None:
        np.random.seed(seed)
    pipeline = _PROFILES[profile](rng, q)

    pages = convert_from_bytes(pdf_bytes, dpi=dpi)
    augmented: list[Image.Image] = []
    for img in pages:
        arr = np.array(img.convert("RGB"))
        out = pipeline(arr)
        augmented.append(Image.fromarray(out))

    return _images_to_pdf(augmented, dpi=dpi), fields
