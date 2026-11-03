# Third-party notices

PaddleKnife itself is licensed under AGPL-3.0-or-later (see [`LICENSE`](LICENSE)).
The repository also redistributes the third-party components below under their
own licences. The full Apache License 2.0 text is in
[`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt).

## oar-ocr 0.8.1 — `vendor/oar-ocr/`

- Copyright 2025 Wang Xin (GreatV)
- Licence: Apache-2.0 ([`vendor/oar-ocr/LICENSE`](vendor/oar-ocr/LICENSE))
- Source: <https://github.com/greatv/oar-ocr>
- **Modified.** Every change is marked `// PaddleKnife patch` in the source
  and listed in [`vendor/oar-ocr/PADDLEKNIFE_PATCHES.md`](vendor/oar-ocr/PADDLEKNIFE_PATCHES.md).

## PaddleOCR models and dictionaries — `models/`

- `devanagari_PP-OCRv5_rec.onnx` — PP-OCRv5 Devanagari recognizer,
  **converted** from PaddlePaddle inference format to ONNX.
- `ppocrv5_devanagari_dict.txt`, `ppocrv5_dict.txt` — recognizer dictionaries,
  unmodified.
- Copyright PaddlePaddle Authors
- Licence: Apache-2.0
- Sources: <https://github.com/PaddlePaddle/PaddleOCR> (`ppocr/utils/dict/`),
  <https://huggingface.co/PaddlePaddle>

Models that the CLI downloads on demand come from the same upstreams under the
same licence; see [`models/README.md`](models/README.md).

## Tesseract `pdf.ttf` — embedded in `crates/paddleknife-core/src/output/searchable_pdf/glyphless.rs`

- Copyright the Tesseract OCR authors
- Licence: Apache-2.0
- Source: <https://github.com/tesseract-ocr/tesseract/blob/main/tessdata/pdf.ttf>
- **Modified.** Glyph 1's advance and outline were changed and the checksums
  repaired; the doc comment on `FONT` in `glyphless.rs` lists every change.

## PDFium — not redistributed

Downloaded separately into `vendor/pdfium/` (see
[`vendor/pdfium/README.md`](vendor/pdfium/README.md)). BSD-3-Clause, from
<https://github.com/bblanchon/pdfium-binaries>.
