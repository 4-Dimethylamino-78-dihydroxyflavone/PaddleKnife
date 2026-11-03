# Models

ONNX weights and dictionaries the Rust CLI runs through oar-ocr and ONNX
Runtime. Only this README, the Devanagari recognizer, its dictionary and
`ppocrv5_dict.txt` are committed. Every other file here is gitignored,
and the CLI downloads what it needs into this directory on demand.
Downloads are atomic (a `.part` file renamed on success).

```
paddleknife models list                             # every tier and script, present or missing
paddleknife models pull latin-cjk --quality small   # one script at one tier
paddleknife models pull --orientation               # both orientation classifiers
paddleknife models pull --all                       # every script at the default (medium) tier, plus the classifiers
```

`models pull` takes a script, `--orientation` or `--all`; `--quality`
(default `medium`) picks the tier, which sets the detector and the
Latin/CJK recognizer that come with it. `paddleknife ocr` fetches
any missing detector, recognizer or dictionary for the selected
`--script` and `--quality`, and any orientation classifier its flags ask
for, before it starts. `paddleknife models list`
prints the exact file names; the registry behind all of this is
`crates/paddleknife-core/src/model.rs` (`QUALITIES` for tiers,
`DESCRIPTORS` for scripts).

A recognizer paired with the wrong dictionary emits plausible garbage
rather than an error, so the two are always selected and downloaded
together. Never swap one without the other.

## Detectors and general recognizers: PP-OCRv6 tiers (Plan 9)

The default. `--quality` selects the tier, and the tier always selects
the detector: detection is script-agnostic, so every script uses it.
PP-OCRv6's general recognizer covers Latin, CJK and Greek, so for
`--script latin-cjk` (aliases `chinese`, `japanese`, `english`) the tier
picks the recognizer and dictionary too.

| `--quality` | Detector | Recognizer | Dictionary | Size (det + rec) |
|---|---|---|---|---|
| `tiny` | `pp-ocrv6_tiny_det.onnx` | `pp-ocrv6_tiny_rec.onnx` | `ppocrv6_tiny_dict.txt` | 1.7 + 4.3 MiB |
| `small` | `pp-ocrv6_small_det.onnx` | `pp-ocrv6_small_rec.onnx` | `ppocrv6_dict.txt` | 9.4 + 20.2 MiB |
| `medium` (default) | `pp-ocrv6_medium_det.onnx` | `pp-ocrv6_medium_rec.onnx` | `ppocrv6_dict.txt` | 59.2 + 73.0 MiB |

Source: oar-ocr's GitHub release `v0.7.0`
(`https://github.com/GreatV/oar-ocr/releases/download/v0.7.0/<file>`).
Tiny's 6,904-entry dictionary has no kana, so tiny cannot read Japanese.

## Script recognizers: PP-OCRv5 (Plan 6)

`ppocrv6_dict.txt` contains no Devanagari, Cyrillic or Hangul, and it
lacks 46 Vietnamese tonal characters. These scripts therefore keep a
PP-OCRv5 recognizer at every tier; `--quality` changes only their
detector. That pairing is legal because every PP-OCR recognizer takes
the same 48-pixel-high crop.

| Script (`--script` aliases) | Recognizer | Dictionary | Source repo (`PaddlePaddle/…`) | Size |
|---|---|---|---|---|
| `devanagari` / `hindi` / `nepali` / `sanskrit` / `marathi` | `devanagari_PP-OCRv5_rec.onnx` | `ppocrv5_devanagari_dict.txt` | `devanagari_PP-OCRv5_mobile_rec_onnx` | ~7.6 MiB |
| `latin` / `czech` / `polish` / `slovak` / `vietnamese` | `latin_PP-OCRv5_mobile_rec.onnx` | `ppocrv5_latin_dict.txt` | `latin_PP-OCRv5_mobile_rec_onnx` | ~7.7 MiB |
| `cyrillic` | `cyrillic_PP-OCRv5_mobile_rec.onnx` | `ppocrv5_cyrillic_dict.txt` | `cyrillic_PP-OCRv5_mobile_rec_onnx` | ~7.7 MiB |
| `eslav` / `russian` / `ukrainian` / `belarusian` | `eslav_PP-OCRv5_mobile_rec.onnx` | `ppocrv5_eslav_dict.txt` | `eslav_PP-OCRv5_mobile_rec_onnx` | ~7.5 MiB |
| `korean` | `korean_PP-OCRv5_mobile_rec.onnx` | `ppocrv5_korean_dict.txt` | `korean_PP-OCRv5_mobile_rec_onnx` | ~12.8 MiB |

Recognizers download from Hugging Face
(`https://huggingface.co/PaddlePaddle/<repo>/resolve/main/inference.onnx`)
and dictionaries from PaddleOCR's GitHub (`ppocr/utils/dict/`); both are
Apache-2.0. `--script auto` (the default) reads the first page with text
(of the first three) with the Latin/CJK recognizer and, if that reads
poorly, tries these specialists in turn, downloading each as needed.

The Devanagari pair is committed to the repository: it is the model the
Python tool used, converted to ONNX from PaddleOCR's PP-OCRv5 Devanagari
recognizer (© PaddlePaddle Authors, Apache-2.0; see
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md)), and the Rust CLI loads it for every Devanagari script alias
(with the extra Nepali/Devanagari clean-up pass). If it is missing,
`paddleknife models pull devanagari` fetches the official ONNX export
under the same name.

## Orientation classifiers (Plans 2.8 and 7)

Two optional PP-LCNet classifiers from oar-ocr's GitHub release `v0.3.0`.
Neither is required unless its flag is used. When a flag asks for one
that is absent, the run downloads it on first use and checks it against
the SHA-256 in oar-ocr-core 0.8.1's model registry; a run whose
classifier can't be fetched stops before OCR starts.

| File | Source | Size | Used by |
|------|--------|------|---------|
| `pp-lcnet_x1_0_textline_ori.onnx` | https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-lcnet_x1_0_textline_ori.onnx | ~6.5 MiB | `--detect-line-orientation` (Plan 2.8) |
| `pp-lcnet_x1_0_doc_ori.onnx` | https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-lcnet_x1_0_doc_ori.onnx | ~6.5 MiB | `--detect-page-orientation` (Plan 7) |

- **Text-line** (Plan 2.8): a 2-class, 0° / 180° classifier run on each
  detected region before recognition. It catches upside-down lines only,
  not 90° rotation; sideways text is handled by quad ordering in
  `ocr::normalize_quad`.
- **Document** (Plan 7): a 4-class, 0° / 90° / 180° / 270° page
  classifier run before detection. It turns fully rotated pages
  (sideways tables, landscape scans) upright and maps every box back to
  the original page. It misreads screenshots and some Devanagari book
  pages as upside-down, so any page that then reads poorly is re-read
  turned 180° from the classifier's verdict and the better read kept
  (Plan 10).

`paddleknife models pull --orientation` fetches both ahead of time, for
a machine that will run offline; `paddleknife models list` shows whether
they are present.

## Legacy PP-OCRv5 mobile files

The pre-Plan 9 defaults, from oar-ocr's GitHub release `v0.3.0`. The OCR
path no longer loads them: Latin/CJK now uses the tier's PP-OCRv6
recognizer. `dev/compare_ocr.py`'s `v5-mobile` preset and the
`smoke_textline_ori` example still read them.

| File | Source | Size |
|------|--------|------|
| `pp-ocrv5_mobile_det.onnx` | https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-ocrv5_mobile_det.onnx | ~4.6 MiB |
| `pp-ocrv5_mobile_rec.onnx` | https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-ocrv5_mobile_rec.onnx | ~15.8 MiB |
| `ppocrv5_dict.txt` (committed) | https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/ppocrv5_dict.txt | small |
