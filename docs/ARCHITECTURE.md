# PaddleKnife architecture

The living design reference: stages, the contracts between them, and the
invariants tests guard. Update it in the same commit as any behaviour it
describes. History and rejected options live in `docs/plans/`,
`docs/superpowers/`, `docs/adr/` and the audit ([`docs/README.md`](README.md)).

Paths are relative to `crates/paddleknife-core/src/` unless they start with
`crates/`, `vendor/` or `docs/`. `tests/…` means `crates/paddleknife-core/tests/`.

**Crates.** `crates/paddleknife` is the CLI: `cli.rs` (clap), `main.rs`
(discovery, script routing, the page loop, output paths, the summary) and
`models.rs` (`models list|pull`, on-demand fetching). `crates/paddleknife-core`
is everything else. `vendor/oar-ocr` is oar-ocr 0.8.1 with small patches via
`[patch.crates-io]`, each explained in `vendor/oar-ocr/PADDLEKNIFE_PATCHES.md`.

## Pipeline

```
INPUT (file or directory)
 └─ input::discover          natural order; skips *_paddleknife_OCR*, dot-files, symlinked dirs
     └─ script routing        once per run (per file with --script-per-file)   crates/paddleknife/src/main.rs
         └─ for each page:
             rasterise        pdf::for_each_page_with at 300 DPI within a pixel budget (pdfium) | image_io::load_image
             probe            source_pdf::SourcePdf::probe_page (lopdf, read-only; -f pdf only)
             OCR              resilient::ResilientEngine → ocr::OarOcrBackend → oar-ocr:
                                [page orientation] → DBNet detection → [line orientation] → CTC recognition
             adapt            ocr.rs: reading-order quads, fullwidth fold, CTC char boxes → CharSpans, word blocks
             postproc         postproc::nepali (Devanagari; txt/json/md only)
             layout           layout::PageMetrics::from_blocks, layout::band_sort → Vec<TextLine>
             encode           output::searchable_pdf::encode_page_opts (-f pdf only)
 └─ write                     output::{text, json, markdown} | output::searchable_pdf::write_searchable_pdf_encoded
```

- **Streaming.** One page bitmap at a time; PDF output keeps only each
  page's encoded image (under 1 MB against ~25 MB). `DocPage`s stay in
  memory, and the writer runs once per document.
- **The OCR raster is 300 DPI** (`DEFAULT_DPI`, `main.rs`) whatever the
  source holds, lowered only for pages beyond `pdf::MAX_RENDER_PIXELS`
  (100 MP) or `MAX_RENDER_SIDE` (20,000 px). The embed logic below keeps
  it from inflating output.
- **Failure scope.** A page that fails on every retry rung is kept without
  text and listed (non-zero exit); a file where every page fails is a failed
  file. `--fail-fast` turns both into immediate aborts.

## Geometry contract

- `geometry::Quad = [[f32; 2]; 4]`: corners clockwise from the quad's
  top-left, in raster pixels, origin top-left, y down. The one px→pt
  transform (`pt = px × 72/dpi`, y flipped, `dpi` = the page's own render
  DPI, `EncodedPage::dpi`) lives in the PDF writer.
- `ocr::OcrBlock` is one whitespace-delimited word: `quad` is its slice of
  the detector's region polygon, and `chars: Vec<CharSpan>` has one entry
  per grapheme cluster, cut from oar-ocr's CTC character boxes and
  re-sliced along the region quad (`geometry::quad_slice`) so tilt survives.
  `chars_synthesized` marks a uniform split when the boxes were missing or
  didn't match. `bbox: [i32; 4]` is derived, for legacy consumers.
- **Invariant:** the concatenated `chars` text equals `text` (whitespace
  aside). Code that edits `text` must keep that true or call
  `OcrBlock::synthesize_chars`. So `postproc::fullwidth`, a strict 1:1
  codepoint swap, runs inside the adapter before clustering, for every
  format. The Nepali pass deletes and reorders characters, so it runs only
  for text exports (`apply_script_postproc` in `main.rs`).
- **Quads are in reading order.** `normalize_quad` (`ocr.rs`) relabels any
  region at least 1.5× taller than wide as `[tr, br, bl, tl]`, mirroring
  oar-ocr's `get_rotate_crop_image`, which rotates such crops 270° before
  recognition. `ROTATED_CROP_ASPECT` must equal oar-ocr's 1.5, or every
  character slice of a sideways line collapses into a 1-px sliver.
- **`layout::band_sort`** takes each block's reading axis from its advance
  vector `tl→tr`, not from bbox aspect (which would misclassify a lone "l").
  Horizontal text bands by y and orders by x; vertical text bands by x and
  orders by y; horizontal lines come out first. The band threshold is
  0.5 × the median line extent, and columns split at gaps over 2.5 × the
  band's own median. Majority-RTL lines are reversed into logical order.
  Blocks are joined with a space, except between two unspaced-script
  characters (`script::is_unspaced_script`).

## Searchable-PDF text layer

Hand-written with `pdf-writer` (`output/searchable_pdf/`), because
pdfium's object API cannot express per-glyph advances.

- **Glyphless font** (`glyphless.rs`): Tesseract's 572-byte `pdf.ttf`,
  vendored as bytes. Each distinct (grapheme cluster, 25‰-em width bucket)
  pair gets its own CID, whose `/W` entry is the measured advance. A
  ToUnicode `bfchar` maps each CID back to the cluster's full codepoint
  sequence, so conjuncts copy out intact. Ascent is 1 em and descent 0,
  so a glyph at size `h` on a band's bottom edge selects exactly that band.
- **No `Tz` anywhere.** Per-instance `Tz` advances, the earlier design, are
  mis-scaled by pdf.js into sliver-width selection boxes. `/W` is honoured
  identically by every viewer.
- **pdfium gotchas, both load-bearing:**
  1. `CIDToGIDMap` sends every CID to **glyph 1**, never glyph 0. pdfium's
     text page drops characters whose glyph has no outline.
  2. Glyph 1 is locally patched: its outline shrank from 1024×2048 to
     50×2048 units, and its hmtx advance went from 1024 to 0. pdfium's loose
     character box is `max(/W advance, outline width)`. 50 units (24‰ em)
     is below the smallest `/W` bucket (25‰), so `/W` always wins, yet tall
     enough that pdfium doesn't treat the glyph as empty. The test
     `vendored_font_is_uncorrupted` guards the bytes.
- **Placement** (`place_line`, `emit_line`). One `BT…ET` per `TextLine`:
  `geometry::fit_baseline` fits a baseline through its cluster quads, `Tm`
  carries its rotation and start, the font size is the band height, and
  `Td` walks the pen to each cluster's measured start. The baseline is
  canonicalised to `dir[0] ≥ 0`, so RTL is decreasing positions, never a
  flipped frame (a leftward `Tm` is a 180° rotation). Words of spaced
  scripts get an explicit space CID across the measured gap (≥ 0.2 × line
  height), so extracted text equals `TextLine.text`. Curved lines (centres
  straying > 0.5 × band height from one baseline) get one frame per
  cluster, and bands overhanging the page are trimmed.

## Page images: passthrough, re-encode, DPI hint

`encode_page_opts` (`output/searchable_pdf/mod.rs`) tries two paths in order:

1. **Passthrough.** Under `--pdf-image auto`, if the probe found a
   qualifying image and its aspect matches the raster within 4 %, the source
   stream's still-encoded bytes are copied verbatim. Accepted encodings are
   DCT, Flate, and CCITT with its `DecodeParms`, in DeviceGray or DeviceRGB.
2. **Re-encode** as JPEG (`--jpeg-quality`). `embed_target_dpi` takes the
   source's measured resolution (the larger of the probe's `effective_dpi`
   and `RasterPage::native_dpi`), lowered to `--embed-dpi` if that is
   smaller; with no measurement, `--embed-dpi` alone. The raster is
   downscaled (Lanczos3, floor 50 DPI) only when the target is below 0.95 ×
   its render DPI, never upsampled, and stored as grayscale when ≥ 99.5 % of
   sampled pixels have a channel spread ≤ 8. Image inputs are never scaled.

The probe (`source_pdf.rs`) uses lopdf strictly read-only; `pdf-writer`
remains the only PDF author. It logs every decline at debug level
(`RUST_LOG=paddleknife_core::source_pdf=debug`) and runs in **two stages**,
so a passthrough-only rejection never costs the DPI hint:

- **`single_image_draw`** is the geometry stage, and it feeds the hint. It
  needs a finite, non-degenerate MediaBox and exactly one image in
  `/Resources /XObject`. The content stream is size-capped before and while
  inflating (untrusted input), pre-lexed to the whitelist `q Q cm Do` plus
  plain numbers and names, then parsed to a single draw of that image with a
  finite matrix. `effective_dpi` is set when the drawn size is ½–1½× the
  MediaBox on each axis, unless a visible annotation (anything but a Link or
  Popup, not Hidden/NoView) is drawn over the scan at full render DPI.
- **`passthrough_gates` + `extract_image`** decide whether the bytes can be
  copied. The page needs a fully visible Parent chain, no `/Rotate`, a
  non-cropping CropBox and no annotations. The draw must be axis-aligned,
  unflipped, within ±3 % of the page size and ±1 % of the origin. The image
  needs no SMask, Mask, Decode or Interpolate, and a self-contained encoding.
- **The lexer stays narrower than lopdf**, whose content parser silently
  returns only the prefix it managed to parse. Integer parts are capped at
  12 digits, because only lopdf's i64 path can overflow.
- `main.rs` opens the probe even under `--pdf-image reencode` (for the hint),
  and disables it when lopdf and pdfium disagree on the page count.
- **`native_dpi`** (`pdf.rs`, `image_only_page_dpi`) is the backstop. It
  reads pdfium's object model: an image-only page gets its sharpest image's
  DPI, and any visible non-image object or annotation gives `None`. It
  covers pages the probe refuses, such as clip-path wrappers or `gs`.

## Models, registry and script routing

- **Two orthogonal axes** (`model.rs`). `Quality` (`QUALITIES`: tiny,
  small, medium) picks the PP-OCRv6 detector. `RecognizerScript`
  (`DESCRIPTORS`, six rows) picks the recognizer and dictionary. They meet
  only in `RecognizerScript::rec_and_dict_for`. Only `LatinCjk` is
  `v6_general`, because PP-OCRv6's dictionary has no Devanagari, Cyrillic
  or Hangul. A v6 detector with a v5 recognizer is legal: every PP-OCR
  recognizer takes `[N, 3, 48, W]` crops. **A recognizer and its dictionary
  always travel together**: a mismatch emits plausible garbage, not an error.
- **Downloads** (`download.rs`, `crates/paddleknife/src/models.rs`).
  `missing_files` lists the detector, recognizer and dictionary a (script,
  tier) pair lacks. Each is fetched to `<name>.part` and renamed after a
  byte-count check against `Content-Length`. Sources: PP-OCRv6 from
  oar-ocr's v0.7.0 release, PP-OCRv5 recognizers from PaddlePaddle's
  HuggingFace repos, dictionaries from PaddleOCR's GitHub.
- **Orientation classifiers** (`CLASSIFIERS`, `model.rs`) download when an
  `ocr` flag first asks for one (`requested_classifiers` in `main.rs`), and
  are checked against a registry SHA-256 before the rename. The detectors
  and recognizers have no checksum yet.
- **One models-folder resolver** serves `ocr` and `models`
  (`select_models_dir`, `main.rs`): `PADDLEKNIFE_MODELS_DIR`; else the first
  `models/` holding a known model file, from the executable's folder up
  three levels, then the working directory; else a new `models/` by the exe.
- **Auto-detect** (`resolve_script`, `classify_first_pass`, `race_winner`;
  currently in `crates/paddleknife/src/main.rs`) reads, with `LatinCjk`, the
  first page with text among the first `SAMPLE_MAX_PAGES` (3); with none, it
  keeps `LatinCjk` uncached, so the next file decides. Confidence ≥ 0.7 keeps
  `LatinCjk` (or `Latin` when over 6 % of letters are accented). Otherwise,
  or when over 8 % of letters are mid-word capitals (Cyrillic misread as
  Latin), specialists race (Devanagari, East Slavic, Korean, Cyrillic,
  Latin), downloading on demand; ≥ 0.95 ends the race, the best score wins.

## GPU memory and resilience

- **The memory law.** A batch is zero-padded to its widest crop, so it
  costs `batch × widest`, not crop count. The CTC head alone is
  `batch × (width/8) × vocab × 4` bytes, held about 3× (vocab 18,710). One
  50:1 line once padded a 47-crop page to ~2,480 px: ≥ 9 GiB.
- **The area budget** is vendor patch 1. A chunk needs both
  `count ≤ region_batch_size` and
  `count × widest ≤ region_batch_size × region_ref_width`. The GPU defaults
  are 64 and 896 (`OarOcrBackend` constants, `ocr.rs`); 896 keeps a 64-crop
  logits tensor at 511.6 MiB, inside DirectML's 512 MiB bucket. Measured
  peaks: 3.6 / 2.2 / 1.5 / 1.2 GiB at batch 64 / 32 / 16 / 8.
- **On Windows the ceiling is often system commit, not VRAM.** Under WDDM
  every GPU allocation is backed by commit (RAM + pagefile). ONNX Runtime
  reports a refusal as `8007000E`. `sysmem::low_commit_hint` names the cause
  when less than 8 GiB of commit is available.
- **The retry ladder** (`resilient.rs`) is `base → base/2 → 8 → CPU`,
  skipping rungs that wouldn't shrink the batch. A failed session is
  dropped before retrying, because DirectML's allocator caches freed
  buffers per session. Every page starts from the top; after 2 pages that
  a smaller GPU batch rescued, the run stays on that rung. CPU-only rescues
  don't count, so CPU (with its own default tuning) serves single pages and
  never becomes the floor. Errors that fail identically on every rung (a
  missing model, an unreadable image) are not retried;
  `OcrError::is_out_of_memory` only picks the summary's wording.
- **The orientation check.** oar-ocr (patch 4) ignores page-orientation
  verdicts below 0.5. On screenshots and Devanagari book pages the
  classifier finds the text axis but often the wrong direction, at scores up
  to 0.91. So `ResilientEngine::ocr_page` re-reads a poorly-read or empty
  page at the opposite orientation (`PageOrientation::Forced`, patch 2),
  from the rung the first read succeeded on. Reads are weighed per
  recognised character (`PageOcr::read_quality`): the flipped read wins only
  with ≥ 0.05 higher per-character confidence *and* at least half as many
  characters (`flipped_read_wins`).

## Invariants and the tests that gate them

| Invariant | Guarded by |
|---|---|
| Text-layer characters land on their quads (centre within 15 % of line height, width within 20 %), with an exact text round-trip. Covers Latin, Devanagari, CJK, RTL Arabic, tilted, arched and legacy input. | `tests/alignment_metrics.rs`, **the gate**. Run it before touching any positioning code. |
| Multi-codepoint clusters survive ToUnicode | `tests/devanagari_pdf_unicode.rs`, `glyphless.rs` unit tests |
| Rotated quads come out in reading order; the 1.5 threshold matches oar-ocr | `normalize_quad_*` tests in `ocr.rs` |
| Axis-aware banding, RTL order, CJK joins | `band_sort.rs` unit tests |
| Probe gates, a lexer narrower than lopdf, Parent-chain poisoning | `source_pdf.rs` unit tests, `tests/passthrough_probe.rs` |
| Passthrough bytes are verbatim and render; `native_dpi` caps image-only pages | `tests/passthrough_roundtrip.rs`, `tests/native_dpi_fallback.rs` |
| Ladder shape, per-page reset, stickiness, CPU never sticky, orientation re-check | `resilient.rs` unit tests (a fake engine through `with_factory`) |
| Recognizer and dictionary pairing; how the two axes combine | `model.rs` unit tests |
| The vendored oar-ocr patches | tests marked `PaddleKnife patch` in `vendor/oar-ocr/src` |

## Known limits

- **No 90° text-layer fixture.** A rotated-quad regression shows up only
  as upright-shaped character boxes on a sideways page.
- **Auto-detect decides once, from one page**; mixed folders need `--script-per-file`.
- **The 300-DPI raster ignores source resolution.** Low-resolution sources
  are upsampled for recognition.
- **65,535 CIDs per document** (`glyphless.rs`). The cap is hit only at
  write time, after all OCR; very large CJK books are at risk.
- **Annotated pages never pass through**; visible annotations (not Link,
  Popup, Hidden, NoView) also lift the DPI cap: bigger output, sharp stamps.
- **PP-OCRv5 downloads are unpinned** (`main`-branch URLs, no checksum).
- **`tensorrt` and `openvino`** compile in, but `--device` can't select them.
