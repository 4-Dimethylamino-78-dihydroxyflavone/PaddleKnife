# PaddleKnife

Command-line OCR for scanned PDFs and images. PaddleKnife turns a scan into
a **searchable PDF** (the original page image plus an invisible text layer
aligned to the printed characters) or into plain text, JSON or Markdown.

It is written in Rust and runs PaddleOCR's PP-OCR models through ONNX Runtime
([oar-ocr](https://github.com/greatv/oar-ocr) + [ort](https://ort.pyke.io)).
On Windows it uses the GPU by default through DirectML. There is no Python and
no PaddlePaddle runtime.

## Features

- **Searchable PDFs that select correctly.** Each grapheme cluster (a letter,
  a Devanagari conjunct, a CJK ideograph) is placed at its detected position
  with its measured width. Selection, search and copy land on the printed
  text in any script, and on tilted, curved and sideways lines too.
- **Text exports:** `txt`, `json` (lines, word boxes and per-character
  geometry) and `md`.
- **Multi-script recognition with auto-detect.** Latin, Chinese, Japanese,
  Devanagari, Cyrillic, East Slavic and Korean. `--script auto` samples the
  first page with text and picks the recognizer. Models download on first
  use.
- **PP-OCRv6 quality tiers:** `tiny`, `small` and `medium` (the default).
- **GPU by default.** DirectML on Windows (any DX12 GPU: NVIDIA, AMD or Intel),
  CoreML on macOS, CUDA as a build option, CPU as the fallback.
- **Resilient GPU batching.** Recognizer batches are bounded by padded area,
  not just count. A page that still runs out of GPU memory is retried with a
  smaller batch, then on CPU. A PDF page that fails everywhere is kept,
  image included, without text; it doesn't take the rest of the book with
  it. An image file that fails is reported as a failed file, so `--merge`
  leaves it out of the merged document.
- **Orientation handling.** `--detect-page-orientation` straightens sideways
  and upside-down pages, and double-checks the classifier by re-reading
  poorly-read pages the other way up. `--detect-line-orientation` fixes
  upside-down lines.
- **Directory input.** Pass a folder of PDFs and images, optionally
  recursive. Files are processed in natural order (`page2` before `page10`).
  You get one output per file, or one merged document with `--merge`.
- **Slim, faithful PDFs.** When a page is a plain scan, its original
  compressed image is copied into the output byte-for-byte. Other pages are
  re-encoded at the source's own resolution (grayscale when there's no
  colour), so output stays close to input size.

## Quick start

```powershell
# 1. Build (needs rustup; see "Build from source")
cargo build --release --locked

# 2. Put PDFium where PaddleKnife can find it (PDF input only; see "PDFium")

# 3. OCR. Models download on first use.
.\target\release\paddleknife.exe ocr scan.pdf -f pdf
```

This writes `scan_paddleknife_OCR.pdf` next to `scan.pdf`.

## Build from source

Requirements:

- [rustup](https://rustup.rs). `rust-toolchain.toml` pins the toolchain
  (currently 1.96.0), and rustup installs it automatically on the first
  build.
- Windows 10/11 x64 is the primary and regularly tested platform.

```powershell
git clone https://github.com/4-Dimethylamino-78-dihydroxyflavone/PaddleKnife.git
cd PaddleKnife
cargo build --release --locked
```

The binary is `target/release/paddleknife(.exe)`. `--locked` builds against
the committed `Cargo.lock`. A fresh dependency resolve can pick up
`oar-ocr`/`ort` versions that don't compile together.

The first build downloads a prebuilt ONNX Runtime for the selected execution
provider. On Windows it is cached under `%LOCALAPPDATA%\ort.pyke.io`.

### Execution-provider features

| Cargo feature | Execution provider | Selected with |
|---|---|---|
| `directml` (default) | DirectML: Windows, any DX12 GPU | `--device auto` (on Windows) or `--device dml` |
| `cuda` | CUDA: NVIDIA only | `--device cuda` |
| `coreml` | CoreML: macOS | `--device auto` (on macOS) |
| `tensorrt`, `openvino` | compiled into ONNX Runtime | no `--device` value selects them yet |

```powershell
cargo build --release --locked --features cuda      # DirectML + CUDA
cargo build --release --locked --no-default-features --features coreml   # macOS
```

Every execution-provider feature also pulls in the prebuilt ONNX Runtime
download and the SIMD CPU kernels. Don't build with `--no-default-features`
alone: without an execution-provider feature, the build has no ONNX
Runtime binaries to link against.

## PDFium

Reading PDFs needs the PDFium shared library. It is not checked in. Image
inputs and every output format work without it. To set it up, follow
[`vendor/pdfium/README.md`](vendor/pdfium/README.md): download a
[pdfium-binaries](https://github.com/bblanchon/pdfium-binaries) archive and
extract it into `vendor/pdfium/`.

PaddleKnife looks for the library in this order:

1. The directory named by the `PDFIUM_LIB_PATH` environment variable.
2. The directory holding `paddleknife.exe`. For a standalone install, copy
   `pdfium.dll` next to the executable.
3. `vendor/pdfium/bin/` in the source tree the binary was built from.

On Linux and macOS the library is `libpdfium.so` or `libpdfium.dylib`, and
the pdfium-binaries archive puts it in `lib/`, not `bin/`: set
`PDFIUM_LIB_PATH` to that directory, or copy the library next to the
executable.

## Models

PP-OCR splits OCR into a **detector**, which finds text regions, and a
**recognizer** plus dictionary, which reads them. `--quality` picks the
detector tier. `--script` picks the recognizer.

```powershell
paddleknife models list                       # what's present, what's missing
paddleknife models pull english               # one script at the default tier (medium)
paddleknife models pull korean --quality small
paddleknife models pull --orientation         # the two orientation classifiers
paddleknife models pull --all                 # every script and both classifiers
```

- **On-demand download.** You rarely need `models pull`. Before an `ocr`
  run starts, it downloads whatever the chosen script and tier are missing.
  While `--script auto` is sampling a page, it may download candidate
  recognizers too. `models pull` is for pre-warming, or for a machine that
  will run offline.
- **Orientation classifiers.** `--detect-page-orientation` and
  `--detect-line-orientation` each use a small PP-LCNet classifier (about
  6.5 MB). It downloads the first time you use the flag.
- **Sources.** PP-OCRv6 detectors and recognizers, and the orientation
  classifiers, come from oar-ocr's GitHub release assets. PP-OCRv5 script
  recognizers come from PaddlePaddle's HuggingFace repositories, and
  dictionaries from PaddleOCR's GitHub. Downloads write to `<name>.part`
  and are renamed only when complete. The classifiers are also checked
  against a SHA-256. The Devanagari recognizer ships in the repository.
- **Location.** PaddleKnife uses the first of these that applies:
  1. The folder named by `PADDLEKNIFE_MODELS_DIR`.
  2. The first `models/` folder that already holds a model file. It looks
     beside the executable and up to three levels above it (which covers
     `target/release/` in a source checkout), then in the current directory.
  3. A new `models/` folder next to the executable.

  `paddleknife models list` prints the folder in use.

[`models/README.md`](models/README.md) lists every model file with its
source and size.

| `--quality` | Detector + recognizer size | Notes |
|---|---|---|
| `tiny` | 6.0 MiB | Fastest. Its dictionary has no kana, so it can't read Japanese. |
| `small` | 29.6 MiB | Near-medium accuracy. The sweet spot on CPU only. |
| `medium` (default) | 132.2 MiB | Best accuracy and tightest boxes. On a GPU it measured no slower than the others. |

## Usage

```powershell
paddleknife ocr scan.pdf                          # -> scan_paddleknife_OCR.txt
paddleknife ocr scan.pdf -f pdf                   # searchable PDF
paddleknife ocr receipt.jpg -s english -f json    # pin the script, skip auto-detect
paddleknife ocr book.pdf -f pdf --detect-page-orientation
paddleknife ocr ./scans/ -r -o ./out/             # every file under ./scans, results mirrored into ./out
paddleknife ocr ./book_pages/ --merge -f pdf      # one PDF from a folder of page scans
paddleknife ocr notes.pdf --device cpu --quality small
```

Run `paddleknife` with no arguments for a short summary. Run
`paddleknife ocr --help` for full flag help.
[`docs/OCR_RECIPES.md`](docs/OCR_RECIPES.md) has measured settings for
receipts, books, handwriting, rotated inserts and non-Latin scripts.

## Flag reference

`paddleknife ocr <INPUT> [OPTIONS]`, where `INPUT` is a file (PDF or image)
or a directory.

**Input and output**

| Flag | Default | Meaning |
|---|---|---|
| `-f, --format <txt\|pdf\|json\|md>` | `txt` | Output format. |
| `-o, --output <PATH>` | beside the input | The output file (for a single file or `--merge`), or the output directory (for a directory without `--merge`). |
| `-r, --recursive` | off | Descend into sub-directories. |
| `--merge` | off | Treat every discovered file as one document with continuous page numbers. |
| `--fail-fast` | off | Stop at the first failure (a file, or a page that fails on every retry) instead of reporting it and carrying on. |

**Recognition**

| Flag | Default | Meaning |
|---|---|---|
| `-s, --script <NAME>` | `auto` | Recognizer script, or `auto` to detect it. See [Scripts and languages](#scripts-and-languages). |
| `--script-per-file` | off | Re-run auto-detection for every file (for folders that mix languages). Ignored when `--script` is pinned. |
| `--quality <tiny\|small\|medium>` | `medium` | PP-OCRv6 size tier. |
| `-d, --device <auto\|cpu\|dml\|cuda>` | `auto` | Execution provider. `auto` = DirectML on Windows, CoreML on macOS, else CPU, with CPU fallback. `cuda` needs a `--features cuda` build. |
| `--detect-page-orientation` | off | Rotate 90°/180°/270° pages upright before detection. Pages that read poorly (mean confidence below 0.80, or no text) are re-read turned 180°, and the better read is kept. The classifier downloads on first use. |
| `--detect-line-orientation` | off | Fix upside-down lines (0° vs 180° only; it does not detect sideways text). Adds about 15% latency. The classifier downloads on first use. |

**GPU memory and batching** (advanced; usually handled automatically)

| Flag | Default | Meaning |
|---|---|---|
| `--region-batch-size <N>` | 64 on GPU | Recognizer batch size, the main memory knob. Try 16 on a 4 GB card. |
| `--region-ref-width <PX>` | 896 on GPU | Reference line width for the padded-area batch budget. `0` turns the budget off (for benchmarking only). |
| `--image-batch-size <N>` | oar-ocr's | Detector image batch size. Currently has no effect: each page goes to the detector on its own. |

**Searchable-PDF output**

| Flag | Default | Meaning |
|---|---|---|
| `--pdf-image <auto\|reencode>` | `auto` | `auto` copies a plain scan page's original image bytes through. `reencode` always re-encodes from the OCR raster. |
| `--jpeg-quality <1-100>` | 85 | JPEG quality for re-encoded page images. |
| `--embed-dpi <DPI>` | the source page's own DPI | For PDF input, an upper limit on the resolution of re-encoded page images; use it to shrink output. It lowers a page's measured source resolution but never raises it. A page with no measured resolution is stored at this DPI. Pages copied through are unaffected, and image inputs are always embedded at their own resolution. |

**Other:** `-v, --verbose` raises PaddleKnife's own log output to debug
level; `RUST_LOG`, when set, takes precedence. `-h, --help` works on every
command, and `paddleknife --version` prints the version.

**`paddleknife models`**

| Command | Meaning |
|---|---|
| `models list` | Print the models folder, then each tier's detector, each script's recognizer and each orientation classifier, with whether it is present. |
| `models pull <SCRIPT> [--quality TIER]` | Download one script's recognizer and dictionary, plus the tier's detector if it's missing. `--quality` defaults to `medium`. |
| `models pull --orientation` | Download the orientation classifiers. |
| `models pull --all [--quality TIER]` | Download every script and both orientation classifiers. |

**Environment variables**

| Variable | Effect |
|---|---|
| `PADDLEKNIFE_MODELS_DIR` | Use this folder for models. |
| `PDFIUM_LIB_PATH` | Directory containing the PDFium library. |
| `RUST_LOG` | Log filter. Default `info,ort=warn`. Examples: `RUST_LOG=debug`, `RUST_LOG=paddleknife_core::source_pdf=debug`. |

## Output

**File names.** Each output is `<input stem>_paddleknife_OCR.<ext>`:

- For a single file, the output goes next to the input, or wherever `-o` names.
- For a directory without `--merge`, you get one output per input file.
  Each lands next to its input, or under the `-o` directory with the
  input's sub-folder layout mirrored (so `a/page.png` and `b/page.png` don't
  collide).
- For a directory with `--merge`, the output is
  `<directory name>_paddleknife_OCR.<ext>`, next to the directory.

Existing outputs are overwritten. When PaddleKnife scans a folder, it skips
files whose name ends in `_paddleknife_OCR`, so re-running over a folder
never OCRs its own output.

**Formats.**

| Format | Content |
|---|---|
| `txt` | One line of text per detected line, with a `--- Page N ---` header per page. |
| `pdf` | The original page image plus an invisible text layer. |
| `json` | An array of pages. Each page has its size, confidence metrics and lines; each line has its words, and each word has its box, quad, confidence and per-character quads. |
| `md` | A `# Page N` heading per page, then the lines. |

For Devanagari, the text exports (`txt`/`json`/`md`) get a Nepali clean-up
pass. It strips Latin misreads and page noise, and converts digits to
Devanagari. The PDF text layer is never cleaned up, because it must mirror
the page.

**Console summary.** After each document, PaddleKnife reports the page
count, line count and average confidence. It lists pages with no text and
flags pages with low confidence. It also reports pages that needed a retry
after a GPU failure, and pages whose orientation was corrected. If the run had
to settle on a smaller batch, it suggests a `--region-batch-size` for next
time.

**Exit status** is non-zero when any input file failed, or when any page was
written without text. Everything that succeeded is still written.

**Supported inputs:** PDF, plus the raster formats the `image` crate decodes
in this build: PNG, JPEG, TIFF, WebP, BMP, GIF and others. When scanning a
folder, other files, dot-files and symlinked directories are skipped.

## Scripts and languages

Pass any alias to `--script` (case-insensitive). The first alias in each row
is the canonical name that `models list` and `models pull` use.

| `--script` aliases | Recognizer | Reads |
|---|---|---|
| `latin-cjk`, `chinese`, `japanese`, `english` | PP-OCRv6 general, at the chosen tier | English, most European Latin-script languages (German, French, Spanish, Italian, Portuguese, Dutch, Polish, Czech, Slovak, Hungarian, Romanian, Turkish, Nordic), Chinese, Japanese |
| `latin`, `czech`, `polish`, `slovak`, `vietnamese` | PP-OCRv5 Latin | Full Latin diacritics, including Vietnamese tone marks and typographic punctuation that v6 dropped |
| `devanagari`, `hindi`, `nepali`, `sanskrit`, `marathi` | PP-OCRv5 Devanagari (bundled) | Hindi, Nepali, Sanskrit, Marathi |
| `cyrillic` | PP-OCRv5 Cyrillic | Broad Cyrillic: Serbian, Bulgarian, Mongolian and others |
| `eslav`, `russian`, `ukrainian`, `belarusian` | PP-OCRv5 East Slavic | Russian, Ukrainian, Belarusian (tuned tighter than `cyrillic`) |
| `korean` | PP-OCRv5 Korean | Hangul, Hanja and English |

PP-OCRv6's dictionary has no Devanagari, Cyrillic or Hangul. Those scripts
keep their PP-OCRv5 recognizer at every tier. For them, `--quality` only
changes the detector.

**How `--script auto` decides.** It reads the first page that has any
text, trying up to three pages, with the broad `latin-cjk` recognizer. A
confident read keeps that recognizer, or switches to `latin` when the page
is heavy in accented letters. An unsure read, or one that looks like
Cyrillic misread as Latin, triggers a race between the specialist
recognizers. Each one downloads if needed, and the best score wins.

The verdict is reused for the rest of the run unless you pass
`--script-per-file`. If none of the sampled pages has text, the file is read
with `latin-cjk` and the next file gets its own chance to decide.

Arabic, Tamil, Telugu and Thai have no recognizer yet.

## Troubleshooting

**"Not enough memory resources" / `8007000E` on a GPU with plenty of free
VRAM.** On Windows, every GPU allocation must also fit in *system commit*
(RAM + pagefile). A machine whose commit is nearly used up refuses GPU memory
however much VRAM is free. PaddleKnife already retries the page with a
smaller batch, then on CPU, and names low commit in the log when it sees it.
To skip the failed attempts, pass `--region-batch-size 16` (or 8). To check
commit, run:

```powershell
Get-Counter '\Memory\Committed Bytes','\Memory\Commit Limit'
```

If the two numbers are close, close other applications or enlarge the
pagefile. The *Memory* section of [`docs/OCR_RECIPES.md`](docs/OCR_RECIPES.md)
has the measurements.

**The output PDF is much larger than the input.** Run with
`RUST_LOG=paddleknife_core::source_pdf=debug`. The log gives the reason
each page's original image couldn't be copied through, and whether its
source resolution was found. Pages with visible annotations (stamps,
signatures, filled form fields) are deliberately re-encoded at full raster
resolution, so the annotations stay sharp. Don't use `--pdf-image reencode`
unless you need it. `--jpeg-quality` and `--embed-dpi` trade size for
fidelity on re-encoded pages.

**`pdfium.dll not found`.** See [PDFium](#pdfium). Set `PDFIUM_LIB_PATH`, or
put the library next to the executable.

**Models download into the wrong folder, or can't download at all.** Run
`paddleknife models list`: its first line is the models folder in use. Set
`PADDLEKNIFE_MODELS_DIR` to pick another. For a machine without internet
access, run `paddleknife models pull <script>` (plus `--orientation` if you
use the orientation flags) on a connected machine, and copy the folder
across. Auto-detect always reads with `latin-cjk` first, so either pull
`latin-cjk` too, at the same `--quality`, or pin `--script <script>` on the
offline machine.

**Sideways pages come out garbled.** Use `--detect-page-orientation`.
`--detect-line-orientation` only fixes upside-down lines.

**The wrong script was detected.** Auto-detect decides from one page: the
first with text among the first three. A page of mixed or unusual text can
mislead it. Pin the language with `--script`.

**Build fails to link (`OrtGetApiBase`, `DMLCreateDevice1`), or cargo dies
with `os error 1455` (paging file too small).** Build with `--locked` and at
least one execution-provider feature. On a machine low on commit, build and
test with `-j 2`, and don't run GPU jobs at the same time.

## Development

| Path | What |
|---|---|
| `crates/paddleknife-core/` | The library: rasterising, OCR adapter, layout, writers, model registry, retry logic. |
| `crates/paddleknife/` | The CLI: argument parsing, script routing, batch runs, model downloads. |
| `vendor/oar-ocr/` | oar-ocr 0.8.1 with PaddleKnife's patches, wired in through `[patch.crates-io]`. The patches are listed in `PADDLEKNIFE_PATCHES.md`. |
| `dev/` | Python evaluation tools (model comparison, degradation sweeps, box-coverage scoring). See [`dev/README.md`](dev/README.md); their Python dependencies are in `dev/requirements.txt`. |
| `tests/fixtures/` | Shared test inputs. |
| `docs/` | Design and history. Start at [`docs/README.md`](docs/README.md). |

```powershell
cargo test --workspace -j 2                        # -j 2 keeps memory use down on Windows
cargo clippy --workspace --all-targets
cargo test --manifest-path vendor/oar-ocr/Cargo.toml   # the vendored crate's own tests
```

- Integration tests that run real OCR expect the default models (`latin-cjk`
  at `medium`) in the repository's `models/` folder, and tests that read
  PDFs expect PDFium. Tests that touch PDFium are serialised with
  `serial_test`, because its C bindings are not thread-safe.
- **The text-layer gate** is
  `crates/paddleknife-core/tests/alignment_metrics.rs`. It writes synthetic
  pages with known character boxes (Latin, Devanagari, CJK, RTL Arabic,
  tilted, arched), reads them back through PDFium and checks every
  character's position and width. Run it before touching anything that
  positions text.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) describes the pipeline,
  the contracts between stages and the invariants the tests guard.

## License

AGPL-3.0-or-later.

Third-party components keep their own licences:

- oar-ocr, vendored: Apache-2.0.
- PaddleOCR models: Apache-2.0.
- The glyphless text-layer font, Tesseract's `pdf.ttf`: Apache-2.0.
- PDFium, downloaded separately: BSD-3-Clause.

Copyright holders, sources and modifications are listed in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md); the Apache-2.0 text is in
[`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt).
