"""PaddleOCR-style two-panel OCR visualisation.

This reimplements the renderer behind the PP-OCRv6 announcement's comparison
figures, which show per sample:

    [ original ] [ translucent coloured polygons ] [ text redrawn in outlines ]

The code lives in PaddleX, not PaddleOCR: `OCRResult._to_img()` in
`paddlex/inference/pipelines/ocr/result.py`. PaddleOCR 3.7.x — the release that
shipped PP-OCRv6 — delegates all inference and visualisation to PaddleX and
contains no drawing code of its own. The published case1/2/3.jpg figures are
hand-assembled three-column posters whose per-model cells are each one
unmodified two-panel `ocr_res_img`.

The construction is specific and worth stating, because it is what makes those
figures readable at a glance:

  * Left panel: each region is filled with an OPAQUE random colour on a copy of
    the page, and the copy is then blended 50/50 with the original. There is no
    per-shape alpha. Blending once at the end (rather than drawing each polygon
    translucently) means overlapping regions do not compound into darker
    patches — every region reads at the same strength.
    Four-point quads are fill-only; upstream strokes an outline (width 8) only
    for N-point polygons, which is the seal/curved-text path PP-OCRv6 does not
    take.
  * Colours come from `random.seed(0)` + three `randint(0, 255)` calls per
    region, so the palette is deterministic: the same page always produces the
    same colours, and two models' outputs are visually comparable.
  * Right panel: starts white. Each region's text is rendered into an
    upright image sized to that box, then perspective-warped onto the canvas
    through the box's own corners — so rotated text lands rotated. Panels are
    combined with `bitwise_and`, which keeps white as identity and lets each
    region punch its dark pixels through without erasing its neighbours.
  * Font size is fitted to the box: start at 99% of box height, then shrink
    proportionally if the string is wider than the box.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# A font with broad Latin coverage. PaddleOCR ships simfang.ttf; on Windows we
# fall back through fonts that are always present.
_FONT_CANDIDATES = [
    Path("C:/Windows/Fonts/arial.ttf"),
    Path("C:/Windows/Fonts/segoeui.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
]


def resolve_font() -> str:
    for candidate in _FONT_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"no usable TTF found; tried {_FONT_CANDIDATES}")


def _fit_font(text: str, box_w: int, box_h: int, font_path: str) -> ImageFont.FreeTypeFont:
    """Largest size that fits the box: 99% of height, shrunk to fit width."""
    size = max(int(box_h * 0.99), 8)
    font = ImageFont.truetype(font_path, size, encoding="utf-8")
    if not text:
        return font
    length = font.getlength(text)
    if length > box_w and length > 0:
        size = max(int(size * box_w / length), 6)
        font = ImageFont.truetype(font_path, size, encoding="utf-8")
    return font


def _draw_text_in_box(canvas_size: tuple[int, int], quad: np.ndarray, text: str, font_path: str) -> np.ndarray:
    """Render `text` upright, then warp it into `quad` on a white canvas."""
    w = int(math.hypot(quad[0][0] - quad[1][0], quad[0][1] - quad[1][1]))
    h = int(math.hypot(quad[0][0] - quad[3][0], quad[0][1] - quad[3][1]))
    w, h = max(w, 1), max(h, 1)

    # A tall, narrow box is vertical text: lay it out horizontally, then rotate.
    vertical = h > 2 * w and h > 30
    tile_w, tile_h = (h, w) if vertical else (w, h)

    tile = Image.new("RGB", (tile_w, tile_h), (255, 255, 255))
    if text:
        ImageDraw.Draw(tile).text(
            (0, 0), text, fill=(0, 0, 0), font=_fit_font(text, tile_w, tile_h, font_path)
        )
    if vertical:
        tile = tile.transpose(Image.ROTATE_270)

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    matrix = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    return cv2.warpPerspective(
        np.array(tile, dtype=np.uint8), matrix, canvas_size,
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )


def draw_ocr_panels(
    image_bgr: np.ndarray,
    quads: list[np.ndarray],
    texts: list[str] | None = None,
    scores: list[float] | None = None,
    drop_score: float = 0.5,
    font_path: str | None = None,
) -> np.ndarray:
    """Return the side-by-side BGR panel image (2*W x H)."""
    font_path = font_path or resolve_font()
    base = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    h, w = base.height, base.width

    overlay = base.copy()
    draw_overlay = ImageDraw.Draw(overlay)
    right = np.ones((h, w, 3), dtype=np.uint8) * 255

    texts = texts if texts and len(texts) == len(quads) else [""] * len(quads)
    random.seed(0)  # deterministic palette — the same page always looks the same
    for idx, quad in enumerate(quads):
        if scores is not None and idx < len(scores) and scores[idx] < drop_score:
            continue
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw_overlay.polygon([tuple(p) for p in quad.tolist()], fill=color)

        tile = _draw_text_in_box((w, h), quad, texts[idx], font_path)
        cv2.polylines(tile, [quad.astype(np.int32).reshape((-1, 1, 2))], True, color, 1)
        # bitwise_and keeps white (255) as identity, so each region only darkens
        # its own pixels rather than overwriting previously drawn neighbours.
        right = cv2.bitwise_and(right, tile)

    left = Image.blend(base, overlay, 0.5)
    combined = Image.new("RGB", (w * 2, h), (255, 255, 255))
    combined.paste(left, (0, 0))
    combined.paste(Image.fromarray(right), (w, 0))
    return cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)


def label_strip(image: np.ndarray, label: str, height: int = 46) -> np.ndarray:
    """Stack a titled banner above an image (mimics the figure's blue tabs)."""
    strip = np.full((height, image.shape[1], 3), 255, dtype=np.uint8)
    cv2.rectangle(strip, (8, 6), (min(430, image.shape[1] - 8), height - 8), (214, 116, 92), -1)
    cv2.putText(strip, label, (24, height - 17), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([strip, image])
