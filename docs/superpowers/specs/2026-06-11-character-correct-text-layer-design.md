# Character-correct searchable-PDF text layer — ground-up restructure

**Date:** 2026-06-11
**Status:** Approved for implementation (Plan 5)
**Supersedes:** the pdfium text-object writer (Plans 2 → 2.8 lineage) and, by design
compatibility, the unmerged Plan 2.9/3.0 polygon work on `claude/admiring-murdock-c5a931`.

## Problem

The invisible text layer in searchable PDFs does not align with the printed
characters, and the misalignment compounds across scripts. Five stacked fixes
(Plans 2.6–2.8) each "verified" visually; the structural causes remain:

1. **Character positions are synthesized, never measured.** The writer emits one
   text object per line and force-fits its width with a single anisotropic
   `scale_x` (`searchable_pdf.rs:159-170`). Intra-line character positions therefore
   come from *Noto Sans advance widths*, not from the page. The recognizer's
   per-character boxes (CTC column positions — real signal) are unioned into word
   bboxes in `ocr.rs:326-347` and then ignored entirely by the writer.
2. **Geometry is destroyed at every stage.** Detector 4-point polygon → axis-aligned
   bbox (tilt lost); f32 → `as i32` truncation; word boxes → line union bbox;
   DPI carried by convention, not type.
3. **Font-coverage and shaping holes.** "LatinCjk" embeds a Noto Sans with **no CJK
   ideographs** (`.notdef` widths feed `scale_x`); no shaping for Devanagari;
   Arabic/Thai/Hebrew aren't expressible in the two-variant `RecognizerScript`.
4. **Vertical placement is eyeballed constants** (`em_over_bbox` 1.20/1.00,
   `DESCENDER_RATIO` 0.20) rather than metrics.
5. **No quantitative alignment test exists** — all tests are content round-trips.

## Approaches considered

**A. Evolve the pdfium writer to per-word objects** (the unmerged Plan 3.0 direction):
word-anchored placement, real font metrics, polygon baselines. Rejected as the
primary fix: characters inside each word are still positioned by substitute-font
advances; CJK needs a ~20 MB embedded font to even have widths; Arabic needs
shaping; pdfium's API cannot express per-glyph advances (no `TJ`/`Tz` control).

**B. Per-character pdfium text objects.** Same coverage/shaping problems as A,
plus thousands of objects per page. Rejected.

**C. Ground-up text-layer writer: measured geometry end-to-end + glyphless CID
font with per-character placement.** This is the Tesseract/OCRmyPDF architecture,
upgraded with the per-char boxes Tesseract doesn't have. **Chosen.**

## Design (Approach C)

Character-correct by construction: every character's selection rectangle is
*written from its detected box*, with zero dependence on any real font's metrics,
coverage, or shaping — which is what makes it work uniformly across scripts.

### 1. `geometry` module — typed, lossless geometry

- `Quad` = `[[f32; 2]; 4]` clockwise from top-left, in raster-pixel space
  (compatible with Plan 2.9's polygon contract; f32, no truncation).
- Helpers: `bbox_of`, `from_bbox`, `union`, baseline/midline, axis-aligned tests.
- One invertible transform owns px↔pt: built from `dpi` in exactly one place in
  the writer (`72/dpi` scale + Y-flip).

### 2. Recognition contract — per-character geometry survives

`OcrBlock` (serde-compatible: old fields keep meaning, new fields default):

- keep `bbox: [i32; 4]`, `text`, `confidence`, `rotation_deg`;
- add `quad: Quad` — detector polygon read from `region.bounding_box.points`
  (public in oar-ocr 0.6; currently unread) with bbox-rect fallback;
- add `chars: Vec<CharSpan>` where `CharSpan { text: String, quad: Quad }`.
  One span per **grapheme cluster** (`unicode-segmentation`), its quad the union
  of the member chars' CTC boxes — Devanagari conjuncts/matras stay atomic.
- add `chars_synthesized: bool` — on char↔box count mismatch the spans are
  synthesized by proportional split and the flag is set, with a `warn!` log.
  Never a silent line-level fallback (replaces the debug-only telemetry).

Whitespace in recognizer output keeps its box → becomes the measured advance of
the space CID in the PDF (extractors get real word breaks; the
`surya_injector.py` space-glyph trick, but measured).

### 3. Line assembly — grouping without destruction

`TextLine` keeps `blocks` (already present) and gains `quad` (union of block
quads). Two correctness fixes in `band_sort`:

- **join policy:** no space inserted between adjacent blocks whose boundary
  chars are CJK/kana (fixes "中 国" corruption in txt/md/PDF text);
- **RTL ordering:** when a line's text is majority-RTL, order blocks by
  descending x so logical text order is preserved for Arabic/Hebrew.

Banding heuristics otherwise unchanged (vertical CJK explicitly out of scope).

### 4. New searchable-PDF writer (`pdf-writer` crate; pdfium leaves the output path)

- **Glyphless CID font** (~1 KB, generated in code or vendored): `Type0` +
  `CIDFontType2`, `Identity-H`, every CID → GID 0, fixed 500/1000 default
  advance. CIDs allocated per distinct grapheme cluster per document;
  **ToUnicode CMap** maps each CID to the cluster's full UTF-16BE sequence
  (multi-codepoint `bfchar`). One font covers *every* script — no coverage,
  no shaping, no per-script fonts, and `vendor/fonts` leaves the build.
- **Per-line placement:** baseline anchor + angle from a linear fit through char
  span centers (works for continuous tilt and RTL automatically; falls back to
  `quad`, then `rotation_deg`, then bbox). `Tm` carries rotation + translation;
  font size = line height in points.
- **Per-character placement:** for each `CharSpan`, position the pen at the span's
  projected start (via `TJ` adjustments) and set `Tz` so the advance equals the
  span's width. Selection rectangles match detected boxes exactly — for Latin,
  Devanagari clusters, CJK, Arabic (logical order in the content stream, visual
  positions from the boxes), and Thai alike.
- **Vertical metrics:** font `Ascent`/`Descent` (single pair of constants)
  calibrated once against the measurement harness — replaces `em_over_bbox` and
  `DESCENDER_RATIO`.
- **Page raster:** JPEG (DCTDecode) XObject via the `image` crate; page size =
  px × 72/dpi. Content streams Flate-compressed.
- Public API: `write_searchable_pdf(output, doc_pages, raster_pages, dpi)` —
  the `script` parameter disappears (font no longer depends on it).

### 5. Alignment measurement harness — lands FIRST, gates everything

`tests/alignment_metrics.rs`: synthetic `DocPage`s per script family
(Latin proportional, Devanagari with multi-codepoint clusters, CJK no-space,
Arabic RTL, a ~5° tilted line) → write PDF → re-open with pdfium's text-page
API → per-character rectangles compared against the input char quads.
Asserts: text round-trip exact; per-char center offset ≤ 0.15 × line height;
width error ≤ 20 %. This replaces eyeball verification as the acceptance gate
and is the regression net the previous five fixes never had.

### 6. Script handling decoupled

`RecognizerScript` now selects only the recognizer model. Fonts, em-ratios and
per-script writer constants are deleted. Adding Arabic/Thai/etc. later =
dropping in a rec model + keys file; the text layer already handles the script.

## Out of scope (recorded, not forgotten)

- Vertical CJK column layout; bidi *mixed-direction* extraction order within a
  single line (positions are correct; order follows recognizer output).
- New recognizer models (Arabic, Thai, …) — model acquisition is orthogonal.
- Making the destructive Nepali postproc opt-in; banding/reading-order rework.
- Streaming page-at-a-time PDF assembly (API unchanged; can come later).

## Testing

- Unit: geometry quads, grapheme grouping, CID/ToUnicode builder, join policy.
- Integration: alignment harness (above); existing pdfium-extraction round-trip
  tests stay as cross-validation (our writer, pdfium's reader).
- Manual artifact: MunaMadan_full.pdf first pages + an English scan, regenerated
  for visual confirmation.

## Implementation order

1. Measurement harness scaffolding (red).
2. `geometry` module.
3. `ocr.rs` char spans + polygon capture.
4. Layout: `TextLine.quad`, join policy, RTL order.
5. Glyphless font + ToUnicode builder.
6. Content emission + document assembly; harness goes green.
7. Output-format compat (json/md/txt), delete `fonts.rs` + vendored fonts from
   the build, docs, real-document smoke artifacts.
