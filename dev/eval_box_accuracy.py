#!/usr/bin/env python3
"""Box-accuracy evaluator for searchable PDFs — provider-neutral.

Measures how well a PDF's invisible text layer covers the printed ink,
so different OCR providers (PaddleKnife, Tesseract/OCRmyPDF, Surya,
ABBYY, ...) and languages can be compared on the same documents.

Per page:
  ink_recall    fraction of ink pixels that fall inside some text-layer
                word box — "how much of the printed text is selectable".
                Deleted/missed words lower this.
  box_precision fraction of box-union area that lies on or near ink
                (ink dilated by ~half a line height) — boxes floating
                over whitespace or wildly oversized lower this.

Usage:
  python dev/eval_box_accuracy.py out.pdf [--pages 1-10] [--dpi 150]
                                  [--hotspots dir]   # PNGs of uncovered ink

Caveat: illustrations and display/calligraphic type count as "ink", so
cover/art pages legitimately score low — read those via --hotspots
images, and compare providers on body pages (e.g. --pages 3-49).

Requires: pymupdf, numpy (pip install pymupdf numpy, or
pip install -r dev/requirements.txt).
"""
import argparse
import sys

import fitz  # pymupdf
import numpy as np


def parse_pages(spec: str, n: int):
    if not spec:
        return range(n)
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a) - 1, min(int(b), n)))
        else:
            out.append(int(part) - 1)
    return [p for p in out if 0 <= p < n]


def page_ink_mask(page: fitz.Page, dpi: int) -> np.ndarray:
    """Boolean ink mask from the rendered page (text layer is invisible,
    so the render is exactly the original raster)."""
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    # Otsu threshold (no cv2 dependency).
    hist = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    sum_all = (np.arange(256) * hist).sum()
    w_b = cum_b = best_t = 0.0
    max_var = -1.0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0 or w_b == total:
            continue
        cum_b += t * hist[t]
        m_b = cum_b / w_b
        m_f = (sum_all - cum_b) / (total - w_b)
        var = w_b * (total - w_b) * (m_b - m_f) ** 2
        if var > max_var:
            max_var, best_t = var, t
    return img < best_t  # dark = ink


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    """Binary dilation with a (2r+1) square, via shifted ORs."""
    out = mask.copy()
    for axis in (0, 1):
        acc = out.copy()
        for s in range(1, r + 1):
            acc |= np.roll(out, s, axis=axis)
            acc |= np.roll(out, -s, axis=axis)
        out = acc
    return out


def eval_page(page: fitz.Page, dpi: int):
    scale = dpi / 72.0
    ink = page_ink_mask(page, dpi)
    h, w = ink.shape

    words = page.get_text("words")
    boxes = np.zeros_like(ink)
    line_heights = []
    for x0, y0, x1, y1, *_ in words:
        xa, ya = int(x0 * scale), int(y0 * scale)
        xb, yb = int(np.ceil(x1 * scale)), int(np.ceil(y1 * scale))
        xa, ya = max(xa, 0), max(ya, 0)
        xb, yb = min(xb, w), min(yb, h)
        if xb > xa and yb > ya:
            boxes[ya:yb, xa:xb] = True
            line_heights.append(yb - ya)

    ink_px = int(ink.sum())
    if ink_px == 0:
        return None  # blank page
    covered = int((ink & boxes).sum())
    recall = covered / ink_px

    if boxes.any():
        r = max(2, int(np.median(line_heights) * 0.5)) if line_heights else 4
        near_ink = dilate(ink, r)
        precision = float((boxes & near_ink).sum()) / float(boxes.sum())
    else:
        precision = 0.0
    return {
        "words": len(words),
        "ink_px": ink_px,
        "ink_recall": recall,
        "box_precision": precision,
        "uncovered": ink & ~boxes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--pages", default="", help="e.g. 1-10,15")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--hotspots", default="", help="dir for uncovered-ink PNGs")
    args = ap.parse_args()

    doc = fitz.open(args.pdf)
    pages = parse_pages(args.pages, len(doc))
    rows, blanks = [], 0
    for pno in pages:
        r = eval_page(doc[pno], args.dpi)
        if r is None:
            blanks += 1
            continue
        rows.append((pno + 1, r))
        if args.hotspots:
            import os

            os.makedirs(args.hotspots, exist_ok=True)
            pix = doc[pno].get_pixmap(dpi=args.dpi)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )[:, :, :3].copy()
            img[r["uncovered"]] = [255, 0, 0]
            fitz.Pixmap(
                fitz.csRGB, pix.width, pix.height, img.tobytes(), False
            ).save(f"{args.hotspots}/p{pno + 1:03}_uncovered.png")

    if not rows:
        print("no non-blank pages evaluated")
        sys.exit(1)
    print(f"{'page':>5} {'words':>6} {'ink_recall':>11} {'box_precision':>14}")
    for pno, r in rows:
        flag = "  <-- check" if r["ink_recall"] < 0.85 else ""
        print(
            f"{pno:>5} {r['words']:>6} {r['ink_recall']:>11.3f}"
            f" {r['box_precision']:>14.3f}{flag}"
        )
    recalls = [r["ink_recall"] for _, r in rows]
    precs = [r["box_precision"] for _, r in rows]
    print(
        f"\n{args.pdf}\n  pages evaluated: {len(rows)} (+{blanks} blank)"
        f"\n  ink_recall    mean {np.mean(recalls):.3f}  min {min(recalls):.3f}"
        f"\n  box_precision mean {np.mean(precs):.3f}  min {min(precs):.3f}"
    )


if __name__ == "__main__":
    main()
