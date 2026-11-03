# Development tools

Python scripts for evaluating OCR models and output. They are not part of
the shipped product and the Rust CLI never calls them. Install their
dependencies with `pip install -r dev/requirements.txt` and run them from
the repository root. The model-driven tools read weights from `models/`
(see `models/README.md`; `paddleknife models pull` fetches most of them)
and skip any preset whose files are missing. Scratch output goes to
`dev/out/`, which is gitignored.

## `compare_ocr.py`

Runs PP-OCR model presets side by side on your own images or PDF pages:
PP-OCRv5 mobile, PP-OCRv6 tiny/small/medium, and mixed presets pairing
the v6 detector with a v5 recognizer (`v6det-v5rec`, and `v6det-deva`
for Devanagari). For each page it prints regions found, mean detection
and recognition confidence, the share of the page the boxes cover,
character count and latency, and writes a two-panel PNG per preset to
`dev/out/`. It defaults to `v5-mobile`, `v6-small` and `v6-medium`;
`--no-rec` runs detection only.

```
python dev/compare_ocr.py page.png book.pdf [--preset v6-medium ...] [--pages 2] [--start-page 1] [--dpi 200] [--no-rec]
```

## `degrade_bench.py`

Measures how detection holds up as page quality drops. It applies blur,
downscaling, JPEG compression, noise, fading, skew and a combined "bad
phone photo" to one clean page, and reports each model's survival
recall: the share of its own clean-page regions it still finds at
IoU >= 0.5. This measures robustness, not absolute accuracy, so raw
region counts are printed alongside. Detection only.

```
python dev/degrade_bench.py page.png [--preset v6-small ...] [--start-page 25] [--dpi 200] [--iou 0.5] [--save-samples DIR]
```

## `eval_box_accuracy.py`

Scores the invisible text layer of any searchable PDF, from PaddleKnife
or any other OCR provider, against the printed ink. Per page:
`ink_recall` is the share of ink pixels inside some word box, and
`box_precision` is the share of box area on or near ink. Pages with
recall below 0.85 are flagged. `--hotspots` writes PNGs marking
uncovered ink in red. Illustrations count as ink, so compare providers
on body pages.

```
python dev/eval_box_accuracy.py out.pdf [--pages 3-49] [--dpi 150] [--hotspots DIR]
```

## `ppocr_onnx.py`

Library module, no command line. A minimal PP-OCR pipeline on
`onnxruntime` (DBNet detection, CTC recognition) whose pre- and
post-processing knobs are all visible in `DetConfig` and default to
PaddleOCR's own values. It exists because oar-ocr hides those steps.
`compare_ocr.py` and `degrade_bench.py` are built on it.

```
from ppocr_onnx import DetConfig, TextDetector, TextRecognizer, crop_quad
```

## `ppocr_viz.py`

Library module, no command line. Reimplements PaddleX's two-panel OCR
figure: the page with translucent coloured regions on the left, and each
region's text redrawn into its box on the right. Colours come from a
fixed seed, so two models' panels for the same page are directly
comparable. `compare_ocr.py` uses it for its PNGs.

```
from ppocr_viz import draw_ocr_panels, label_strip
```
