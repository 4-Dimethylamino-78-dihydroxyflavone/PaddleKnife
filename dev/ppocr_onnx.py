"""Minimal, faithful PP-OCR ONNX pipeline (DBNet detect + CTC recognise).

Why this exists: PaddleKnife's Rust path goes through oar-ocr, which hides the
pre/post-processing. To *evaluate* PP-OCRv6 against PP-OCRv5 — and to see what
each model actually emits before committing to an upgrade — we need a reference
implementation whose knobs are all visible and tweakable.

This is deliberately dependency-light (onnxruntime + numpy + cv2 + pyclipper)
and mirrors PaddleOCR's own defaults so the numbers are comparable to published
benchmarks. It is a *development* tool, not part of the shipped product.

Contracts verified directly from the ONNX graphs:
    detection    input  [N, 3, H, W]  (H, W dynamic, must be /32)
                 output [N, 1, H, W]  probability map in [0, 1]
    recognition  input  [N, 3, 48, W] (W dynamic)
                 output [N, T, C]     C = len(dict) + 2
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pyclipper
from shapely.geometry import Polygon

# ImageNet statistics — PaddleOCR normalises detection input with these.
_DET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_DET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


@dataclass
class DetConfig:
    """DBNet post-processing knobs, defaulted to PaddleOCR's own values.

    These are the levers that matter for the user's hard cases:
      * `unclip_ratio` grows each detected polygon outward. Too small clips
        ascenders/descenders; too large merges adjacent receipt lines.
      * `box_thresh` filters low-confidence regions — lower it for faint scans.
      * `limit_side_len` caps the longest side. PaddleOCR only ever *shrinks*
        by default (`limit_type="max"`), so a small receipt photo is fed at
        native resolution and a 300-DPI book page is downscaled to 960px.
    """

    limit_side_len: int = 960
    limit_type: str = "max"  # "max" shrinks big images; "min" upscales small ones
    thresh: float = 0.3  # binarisation threshold on the probability map
    box_thresh: float = 0.6  # mean-score threshold per candidate region
    unclip_ratio: float = 1.5
    min_size: int = 3
    max_candidates: int = 1000
    use_dilation: bool = False
    score_mode: str = "fast"  # "fast" = bbox mean, "slow" = polygon mean


def _make_session(model_path: str | Path, providers: list[str] | None = None) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(model_path),
        sess_options=opts,
        providers=providers or ["CPUExecutionProvider"],
    )


class TextDetector:
    """DBNet text detector — works unchanged for PP-OCRv3/v4/v5/v6 weights."""

    def __init__(self, model_path: str | Path, cfg: DetConfig | None = None, providers=None):
        self.sess = _make_session(model_path, providers)
        self.cfg = cfg or DetConfig()
        self.input_name = self.sess.get_inputs()[0].name

    def _resize(self, img: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Scale by the configured rule, then pad each side up to a multiple of 32.

        The /32 constraint comes from the backbone's 5 downsampling stages; a
        non-conforming size makes the FPN's upsample-and-add shapes disagree.
        """
        h, w = img.shape[:2]
        limit = self.cfg.limit_side_len
        if self.cfg.limit_type == "max":
            ratio = limit / max(h, w) if max(h, w) > limit else 1.0
        else:
            ratio = limit / min(h, w) if min(h, w) < limit else 1.0

        rh, rw = int(h * ratio), int(w * ratio)
        rh = max(int(round(rh / 32) * 32), 32)
        rw = max(int(round(rw / 32) * 32), 32)
        resized = cv2.resize(img, (rw, rh))
        # Return the *actual* achieved ratios — rounding to /32 perturbs them.
        return resized, rh / h, rw / w

    def _unclip(self, box: np.ndarray) -> np.ndarray | None:
        """Grow a polygon outward by `unclip_ratio`, per the DB paper.

        DB shrinks text regions during training, so inference must expand them
        back. Offset distance = area * ratio / perimeter.
        """
        poly = Polygon(box)
        if poly.length < 1e-6:
            return None
        distance = poly.area * self.cfg.unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box.astype(np.int64).tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = offset.Execute(distance)
        if not expanded:
            return None
        return np.array(expanded[0], dtype=np.float32)

    @staticmethod
    def _min_area_box(contour: np.ndarray) -> tuple[np.ndarray, float]:
        """Minimum-area rotated rect as 4 points ordered clockwise from top-left."""
        rect = cv2.minAreaRect(contour)
        pts = sorted(cv2.boxPoints(rect).tolist(), key=lambda p: p[0])
        # Left pair / right pair, then order each vertically.
        left = sorted(pts[:2], key=lambda p: p[1])
        right = sorted(pts[2:], key=lambda p: p[1])
        box = np.array([left[0], right[0], right[1], left[1]], dtype=np.float32)
        return box, min(rect[1])

    @staticmethod
    def _box_score_fast(prob: np.ndarray, box: np.ndarray) -> float:
        """Mean probability inside the box's axis-aligned bound (PaddleOCR 'fast')."""
        h, w = prob.shape
        xs, ys = box[:, 0], box[:, 1]
        x0, x1 = np.clip([math.floor(xs.min()), math.ceil(xs.max())], 0, w - 1).astype(int)
        y0, y1 = np.clip([math.floor(ys.min()), math.ceil(ys.max())], 0, h - 1).astype(int)
        mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=np.uint8)
        shifted = box.copy()
        shifted[:, 0] -= x0
        shifted[:, 1] -= y0
        cv2.fillPoly(mask, [shifted.astype(np.int32)], 1)
        region = prob[y0 : y1 + 1, x0 : x1 + 1]
        if region.size == 0 or mask.sum() == 0:
            return 0.0
        return float(cv2.mean(region, mask)[0])

    def detect(self, img_bgr: np.ndarray) -> list[dict]:
        """Return [{quad: (4,2) float32 in ORIGINAL pixels, score: float}, ...]."""
        resized, ratio_h, ratio_w = self._resize(img_bgr)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = ((rgb - _DET_MEAN) / _DET_STD).transpose(2, 0, 1)[None, ...]

        prob = self.sess.run(None, {self.input_name: chw})[0][0, 0]
        binary = (prob > self.cfg.thresh).astype(np.uint8)
        if self.cfg.use_dilation:
            binary = cv2.dilate(binary, np.array([[1, 1], [1, 1]], dtype=np.uint8))

        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        out: list[dict] = []
        for contour in contours[: self.cfg.max_candidates]:
            box, side = self._min_area_box(contour)
            if side < self.cfg.min_size:
                continue
            score = self._box_score_fast(prob, box)
            if score < self.cfg.box_thresh:
                continue
            expanded = self._unclip(box)
            if expanded is None or len(expanded) < 4:
                continue
            box, side = self._min_area_box(expanded.reshape(-1, 1, 2))
            if side < self.cfg.min_size + 2:
                continue
            # Undo the detection-time resize to land back in original pixels.
            box[:, 0] = np.clip(box[:, 0] / ratio_w, 0, img_bgr.shape[1])
            box[:, 1] = np.clip(box[:, 1] / ratio_h, 0, img_bgr.shape[0])
            out.append({"quad": box, "score": score})

        # Reading order: top-to-bottom, then left-to-right within a band.
        out.sort(key=lambda d: (d["quad"][:, 1].min(), d["quad"][:, 0].min()))
        return out


def crop_quad(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Perspective-rectify a (possibly rotated) quad into an upright crop.

    This mirrors PaddleOCR's `get_rotate_crop_image`. oar-ocr switched to this
    same routine in v0.7.2 (PR #142), replacing a plain rotated-bbox crop — so
    tilted lines are now sampled along their own axis instead of being cut from
    an axis-aligned rectangle that includes neighbouring text.
    """
    quad = quad.astype(np.float32)
    w = int(max(np.linalg.norm(quad[0] - quad[1]), np.linalg.norm(quad[2] - quad[3])))
    h = int(max(np.linalg.norm(quad[0] - quad[3]), np.linalg.norm(quad[1] - quad[2])))
    w, h = max(w, 1), max(h, 1)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    crop = cv2.warpPerspective(
        img, cv2.getPerspectiveTransform(quad, dst), (w, h),
        borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC,
    )
    # A crop taller than it is wide is a vertical line — rotate it upright.
    if h / w >= 1.5:
        crop = np.rot90(crop)
    return np.ascontiguousarray(crop)


class TextRecognizer:
    """CRNN/SVTR CTC recogniser — same contract for PP-OCRv5 and PP-OCRv6.

    Every PP-OCR recogniser inspected takes [N, 3, 48, W] and emits
    [N, T, len(dict) + 2]. Index 0 is the CTC blank; dictionary entry *i* maps
    to class *i + 1*. The final class is a trailing space PaddleOCR appends.
    """

    def __init__(self, model_path: str | Path, dict_path: str | Path, providers=None, img_h: int = 48):
        self.sess = _make_session(model_path, providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.img_h = img_h
        chars = Path(dict_path).read_text(encoding="utf-8").split("\n")
        if chars and chars[-1] == "":
            chars = chars[:-1]
        # blank at 0, dictionary shifted by 1, trailing space to match C = n + 2.
        self.charset = ["<blank>"] + chars + [" "]
        n_classes = self.sess.get_outputs()[0].shape[-1]
        if isinstance(n_classes, int) and n_classes != len(self.charset):
            raise ValueError(
                f"dict/model mismatch: model emits {n_classes} classes, "
                f"dict '{Path(dict_path).name}' yields {len(self.charset)} "
                f"({len(chars)} entries + blank + space). Wrong dictionary?"
            )

    def _preprocess(self, crop: np.ndarray, target_w: int) -> np.ndarray:
        h, w = crop.shape[:2]
        new_w = min(int(math.ceil(self.img_h * w / h)), target_w)
        resized = cv2.resize(crop, (new_w, self.img_h)).astype(np.float32)
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # PaddleOCR recognition normalisation: (x/255 - 0.5) / 0.5  ->  [-1, 1]
        resized = (resized / 255.0 - 0.5) / 0.5
        padded = np.zeros((self.img_h, target_w, 3), dtype=np.float32)
        padded[:, :new_w] = resized
        return padded.transpose(2, 0, 1)

    def recognise(self, crops: list[np.ndarray], batch_size: int = 16) -> list[tuple[str, float]]:
        """Batch by aspect ratio so padding waste stays low (PaddleOCR does this too)."""
        if not crops:
            return []
        order = sorted(range(len(crops)), key=lambda i: crops[i].shape[1] / max(crops[i].shape[0], 1))
        results: list[tuple[str, float]] = [("", 0.0)] * len(crops)

        for start in range(0, len(order), batch_size):
            chunk = order[start : start + batch_size]
            # One width for the whole batch: the widest aspect ratio in it.
            max_ratio = max(crops[i].shape[1] / max(crops[i].shape[0], 1) for i in chunk)
            target_w = max(int(math.ceil(self.img_h * max_ratio)), self.img_h)
            batch = np.stack([self._preprocess(crops[i], target_w) for i in chunk])
            logits = self.sess.run(None, {self.input_name: batch})[0]
            for row, idx in enumerate(chunk):
                results[idx] = self._ctc_decode(logits[row])
        return results

    def _ctc_decode(self, logits: np.ndarray) -> tuple[str, float]:
        """Greedy best-path decode: argmax, drop blanks, collapse repeats."""
        best = logits.argmax(axis=1)
        conf = logits.max(axis=1)
        chars, scores = [], []
        for t, cls in enumerate(best):
            if cls == 0:  # blank
                continue
            if t > 0 and best[t - 1] == cls:  # repeat collapse
                continue
            if cls < len(self.charset):
                chars.append(self.charset[cls])
                scores.append(float(conf[t]))
        return "".join(chars), float(np.mean(scores)) if scores else 0.0
