#!/usr/bin/env python
"""Measure how PP-OCR detection degrades as page quality drops.

Motivation: vendor benchmarks are run on clean corpora, but the workloads that
matter here are receipts, badly scanned pages, and phone photos. This applies
controlled, realistic degradations to a clean page and measures how much of the
clean-page detection each model retains.

Metric — "survival recall". We have no human-labelled ground truth, so we use
each model's OWN detection on the CLEAN page as its reference set, then report
what fraction of those regions it still finds after degradation (IoU >= 0.5).
This deliberately measures *robustness*, not absolute accuracy: a model that
finds little on the clean page is not rewarded, because its reference set is
its own. Absolute region counts are printed alongside so a model that simply
detects less overall is still visible.

Usage:
    python dev/degrade_bench.py page.png
    python dev/degrade_bench.py book.pdf --start-page 25 --dpi 200
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from compare_ocr import MODELS, PRESETS, load_pages  # noqa: E402
from ppocr_onnx import DetConfig, TextDetector  # noqa: E402


# ---------------------------------------------------------------- degradations

def blur(img, k=5):
    return cv2.GaussianBlur(img, (k, k), 0)


def downscale(img, factor=0.5):
    """Low-DPI scan: lose real detail, then resize back to compare like-for-like."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(int(w * factor), 1), max(int(h * factor), 1)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def jpeg(img, quality=25):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


def noise(img, sigma=18):
    rng = np.random.default_rng(0)  # fixed seed: reruns are comparable
    out = img.astype(np.float32) + rng.normal(0, sigma, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def fade(img, strength=0.45):
    """Faded thermal receipt / weak toner: compress the dynamic range upward."""
    return np.clip(255 - (255 - img.astype(np.float32)) * (1 - strength), 0, 255).astype(np.uint8)


def skew(img, degrees=3.0):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)


def bad_photo(img):
    """Everything at once — the realistic phone-snap-of-a-receipt case."""
    return noise(jpeg(blur(downscale(img, 0.6), 5), 30), 12)


DEGRADATIONS: dict[str, callable] = {
    "clean":       lambda i: i,
    "blur-5":      lambda i: blur(i, 5),
    "blur-9":      lambda i: blur(i, 9),
    "half-res":    lambda i: downscale(i, 0.5),
    "third-res":   lambda i: downscale(i, 0.33),
    "jpeg-25":     lambda i: jpeg(i, 25),
    "jpeg-10":     lambda i: jpeg(i, 10),
    "noise":       lambda i: noise(i, 18),
    "faded":       lambda i: fade(i, 0.45),
    "skew-3deg":   lambda i: skew(i, 3.0),
    "bad-photo":   bad_photo,
}


# ------------------------------------------------------------------- geometry

def quad_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two convex quads, via cv2.intersectConvexConvex."""
    area_a = cv2.contourArea(a.astype(np.float32))
    area_b = cv2.contourArea(b.astype(np.float32))
    if area_a <= 0 or area_b <= 0:
        return 0.0
    inter, _ = cv2.intersectConvexConvex(a.astype(np.float32), b.astype(np.float32))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def survival_recall(reference: list[np.ndarray], found: list[np.ndarray], iou_thresh: float = 0.5) -> float:
    """Fraction of reference regions matched by some found region (greedy, 1:1)."""
    if not reference:
        return float("nan")
    unused = list(range(len(found)))
    matched = 0
    for ref in reference:
        best_i, best_iou = -1, 0.0
        for i in unused:
            score = quad_iou(ref, found[i])
            if score > best_iou:
                best_i, best_iou = i, score
        if best_iou >= iou_thresh:
            matched += 1
            unused.remove(best_i)
    return matched / len(reference)


# ---------------------------------------------------------------------- driver

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("--preset", action="append", choices=list(PRESETS), default=None)
    ap.add_argument("--start-page", type=int, default=1)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--box-thresh", type=float, default=0.6)
    ap.add_argument("--save-samples", type=Path, default=None, help="dump degraded images here")
    args = ap.parse_args()

    presets = args.preset or ["v5-mobile", "v6-tiny", "v6-small", "v6-medium"]
    cfg = DetConfig(box_thresh=args.box_thresh)
    stem, img = load_pages(args.input, 1, args.dpi, args.start_page)[0]

    if args.save_samples:
        args.save_samples.mkdir(parents=True, exist_ok=True)
        for name, fn in DEGRADATIONS.items():
            cv2.imwrite(str(args.save_samples / f"{stem}__{name}.png"), fn(img))

    detectors = {}
    for name in presets:
        path = MODELS / PRESETS[name]["det"]
        if path.exists():
            detectors[name] = TextDetector(path, cfg)
        else:
            print(f"skip {name}: missing {path.name}")

    # Reference = each model's own detection on the clean page.
    variants = {name: fn(img) for name, fn in DEGRADATIONS.items()}
    references = {name: [r["quad"] for r in det.detect(variants["clean"])] for name, det in detectors.items()}

    print(f"\n=== {stem}  ({img.shape[1]}x{img.shape[0]} px)  IoU>={args.iou}")
    print("survival recall %  (regions found)   — reference = each model's own clean-page detection\n")
    header = f"{'degradation':<13}" + "".join(f"{n:>20}" for n in detectors)
    print(header)
    print("-" * len(header))

    timings = {n: 0.0 for n in detectors}
    for deg_name, page in variants.items():
        cells = []
        for model_name, det in detectors.items():
            t0 = time.perf_counter()
            quads = [r["quad"] for r in det.detect(page)]
            timings[model_name] += time.perf_counter() - t0
            recall = survival_recall(references[model_name], quads, args.iou)
            cells.append(f"{recall * 100:>13.1f} ({len(quads):>3})")
        print(f"{deg_name:<13}" + "".join(f"{c:>20}" for c in cells))

    print("-" * len(header))
    print(f"{'ref regions':<13}" + "".join(f"{len(references[n]):>20}" for n in detectors))
    print(f"{'total det s':<13}" + "".join(f"{timings[n]:>20.2f}" for n in detectors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
