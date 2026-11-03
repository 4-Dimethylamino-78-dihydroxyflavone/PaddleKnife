#!/usr/bin/env python
"""Compare PP-OCR model generations side by side on real pages.

Usage:
    python dev/compare_ocr.py IMAGE_OR_PDF [more...] [--out DIR] [--pages N]
    python dev/compare_ocr.py page.png --preset v5-mobile --preset v6-medium

Produces, per input page and per preset:
    <out>/<stem>__<preset>.png    two-panel visualisation (boxes | text)
and prints a comparison table of region counts, mean confidence, and latency.

The point is to answer "is PP-OCRv6 actually better on MY documents" with
evidence rather than with the vendor's benchmark table.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ppocr_onnx import DetConfig, TextDetector, TextRecognizer, crop_quad  # noqa: E402
from ppocr_viz import draw_ocr_panels, label_strip  # noqa: E402

MODELS = Path(__file__).resolve().parents[1] / "models"

# Each preset is a (detector, recogniser, dictionary) triple. Detection and
# recognition are independent — any detector can feed any recogniser, because
# the only thing crossing the boundary is a cropped image. That is what makes
# a mixed "v6 detector + v5 script recogniser" configuration legal.
PRESETS: dict[str, dict[str, str]] = {
    "v5-mobile":  {"det": "pp-ocrv5_mobile_det.onnx",  "rec": "pp-ocrv5_mobile_rec.onnx", "dict": "ppocrv5_dict.txt"},
    "v6-tiny":    {"det": "pp-ocrv6_tiny_det.onnx",    "rec": "pp-ocrv6_tiny_rec.onnx",   "dict": "ppocrv6_tiny_dict.txt"},
    "v6-small":   {"det": "pp-ocrv6_small_det.onnx",   "rec": "pp-ocrv6_small_rec.onnx",  "dict": "ppocrv6_dict.txt"},
    "v6-medium":  {"det": "pp-ocrv6_medium_det.onnx",  "rec": "pp-ocrv6_medium_rec.onnx", "dict": "ppocrv6_dict.txt"},
    # Mixed: v6's stronger detector driving a v5 script recogniser. This is the
    # configuration that keeps Devanagari (and Cyrillic/Korean) working, since
    # the v6 dictionary contains none of those scripts.
    "v6det-v5rec": {"det": "pp-ocrv6_medium_det.onnx", "rec": "pp-ocrv5_mobile_rec.onnx", "dict": "ppocrv5_dict.txt"},
    "v6det-deva":  {"det": "pp-ocrv6_medium_det.onnx", "rec": "devanagari_PP-OCRv5_rec.onnx", "dict": "ppocrv5_devanagari_dict.txt"},
}


def load_pages(path: Path, max_pages: int, dpi: int, start: int = 1) -> list[tuple[str, np.ndarray]]:
    """Load an image, or rasterise `max_pages` pages of a PDF from `start` (1-based)."""
    if path.suffix.lower() != ".pdf":
        img = cv2.imread(str(path))
        if img is None:
            raise SystemExit(f"could not read image: {path}")
        return [(path.stem, img)]

    try:
        import pypdfium2 as pdfium
    except ImportError:
        raise SystemExit("PDF input needs pypdfium2 — `pip install pypdfium2`, or pass an image")

    doc = pdfium.PdfDocument(str(path))
    pages = []
    for i in range(start - 1, min(len(doc), start - 1 + max_pages)):
        pil = doc[i].render(scale=dpi / 72).to_pil().convert("RGB")
        pages.append((f"{path.stem}_p{i + 1}", cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)))
    return pages


def run_preset(name: str, spec: dict[str, str], img: np.ndarray, cfg: DetConfig, recognise: bool) -> dict:
    for key in ("det", "rec", "dict"):
        if not (MODELS / spec[key]).exists():
            return {"preset": name, "error": f"missing {spec[key]}"}

    t0 = time.perf_counter()
    detector = TextDetector(MODELS / spec["det"], cfg)
    regions = detector.detect(img)
    t_det = time.perf_counter() - t0

    quads = [r["quad"] for r in regions]
    texts: list[str] = []
    t_rec = 0.0
    if recognise and quads:
        recogniser = TextRecognizer(MODELS / spec["rec"], MODELS / spec["dict"])
        t1 = time.perf_counter()
        pairs = recogniser.recognise([crop_quad(img, q) for q in quads])
        t_rec = time.perf_counter() - t1
        texts = [t for t, _ in pairs]
        rec_conf = float(np.mean([c for _, c in pairs])) if pairs else 0.0
    else:
        rec_conf = float("nan")

    page_area = img.shape[0] * img.shape[1]
    covered = sum(cv2.contourArea(q.astype(np.float32)) for q in quads)
    return {
        "preset": name,
        "regions": len(regions),
        "det_conf": float(np.mean([r["score"] for r in regions])) if regions else 0.0,
        "rec_conf": rec_conf,
        "coverage": covered / page_area,
        "chars": sum(len(t) for t in texts),
        "t_det": t_det,
        "t_rec": t_rec,
        "quads": quads,
        "texts": texts,
        "scores": [r["score"] for r in regions],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("dev/out"))
    ap.add_argument("--preset", action="append", choices=list(PRESETS), default=None)
    ap.add_argument("--pages", type=int, default=2, help="max PDF pages per file")
    ap.add_argument("--start-page", type=int, default=1, help="first PDF page (1-based)")
    ap.add_argument("--dpi", type=int, default=200, help="PDF rasterisation DPI")
    ap.add_argument("--no-rec", action="store_true", help="detection only (script-agnostic)")
    ap.add_argument("--box-thresh", type=float, default=0.6)
    ap.add_argument("--unclip", type=float, default=1.5)
    ap.add_argument("--limit-side-len", type=int, default=960)
    args = ap.parse_args()

    presets = args.preset or ["v5-mobile", "v6-small", "v6-medium"]
    cfg = DetConfig(box_thresh=args.box_thresh, unclip_ratio=args.unclip, limit_side_len=args.limit_side_len)
    args.out.mkdir(parents=True, exist_ok=True)

    for path in args.inputs:
        for stem, img in load_pages(path, args.pages, args.dpi, args.start_page):
            print(f"\n=== {stem}  ({img.shape[1]}x{img.shape[0]} px) "
                  f"[box_thresh={cfg.box_thresh} unclip={cfg.unclip_ratio} side={cfg.limit_side_len}]")
            print(f"{'preset':<14}{'regions':>8}{'det_conf':>10}{'rec_conf':>10}"
                  f"{'cover%':>9}{'chars':>8}{'det_s':>8}{'rec_s':>8}")
            for name in presets:
                res = run_preset(name, PRESETS[name], img, cfg, not args.no_rec)
                if "error" in res:
                    print(f"{name:<14}  -- skipped: {res['error']}")
                    continue
                print(f"{name:<14}{res['regions']:>8}{res['det_conf']:>10.3f}"
                      f"{res['rec_conf']:>10.3f}{res['coverage'] * 100:>9.2f}"
                      f"{res['chars']:>8}{res['t_det']:>8.2f}{res['t_rec']:>8.2f}")

                panel = draw_ocr_panels(img, res["quads"], res["texts"], res["scores"], drop_score=0.0)
                dest = args.out / f"{stem}__{name}.png"
                cv2.imwrite(str(dest), label_strip(panel, f"{name}  ({res['regions']} regions)"))
    print(f"\nvisualisations -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
