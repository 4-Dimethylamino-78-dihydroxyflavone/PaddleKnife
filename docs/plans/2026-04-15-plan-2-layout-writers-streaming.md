# Plan 2: Layout + Output Writers + Streaming

> **For agentic workers:** Uses the same TDD-per-task cadence as Plan 1 —
> write tests first, commit per task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Turn Plan 1's "flat list of bboxes" into structured, per-page output
and ship the four writers a real archive tool needs: plain text with page
separators, searchable PDF, JSON, and Markdown. Also fix the memory growth
observed on the receipts run (~14 GB ORT arena for a 3.6 MB PDF) by streaming
pages instead of holding every rasterized bitmap in RAM.

**Scope:** Quality + I/O. No GPU (Plan 4). No Devanagari post-processing,
dual-pass, or auto-script-detect (Plan 2.5). No TUI (Plan 3). Feature parity
with the Python port's primary output shapes is the bar.

**Motivation — empirical findings from Plan 1 test runs (2026-04-15):**
- *9-page scanned-receipts PDF:* detector correct, but two-column receipt
  pages produced interleaved rows (left-col row 1, right-col row 1, left
  row 2, …). Unreadable.
- *~200-page Muna Madan (Devanagari):* verses fragmented into single-word
  lines because bboxes come out in detector order, not reading order.
- *No page boundaries* in text output — can't tell where page 3 ends.
- *Only `txt` supported.* Python port produced searchable PDF, JSON, md.
- *ORT arena growth:* 14 GB allocated for a 3.6 MB input; linear in pages
  because we hold every page image in a `Vec<RasterPage>` before OCR starts.

---

## Architectural decisions

**Page-aware data flow.** Plan 1 passed `Vec<OcrBlock>` from OCR to the writer.
Plan 2 introduces:

```
RasterPage → OcrBlock[] → layout::sort_to_lines → TextLine[] → DocPage
                                                              ↓
                                         writers::{txt, pdf, json, md}
```

`TextLine` groups bboxes on the same visual line; `DocPage` owns a page index
plus its lines. Writers consume `&[DocPage]` — pure data in, format out.

**Layout as a pure module.** `layout::band_sort` is a pure function on
`Vec<OcrBlock>`. Unit-testable with synthetic bbox inputs; zero OCR
dependency in the test.

**Streaming pipeline.** `pdf::rasterize_pdf` (eager) stays for callers that
want a Vec. New `pdf::for_each_page(path, dpi, |page| …)` closure-callback
iterator raster→OCR→drop one page at a time. `main.rs` uses it; peak memory
drops to one page's worth of image + model arenas.

**Writer trait (flexible).** Each writer is a function
`write_<fmt>(&mut impl Write, &[DocPage]) -> Result<()>`. No trait yet —
keeps signatures first-class and error messages concrete. Revisit if a
fifth format appears.

---

## File layout additions

```
crates/paddleknife-core/src/
├── layout/
│   ├── mod.rs            # public: TextLine, DocPage, band_sort
│   └── band_sort.rs      # algorithm + unit tests
├── output/
│   ├── mod.rs            # re-exports
│   ├── text.rs           # EXISTING — add page separators
│   ├── searchable_pdf.rs # NEW — pdfium text overlay
│   ├── json.rs           # NEW — serde_json
│   └── markdown.rs       # NEW
├── pdf.rs                # ADD for_each_page streaming iterator
└── page.rs               # ADD DocPage (post-layout)
```

New dev-deps: `serde = { version = "1", features = ["derive"] }`, `serde_json = "1"`.

---

## Tasks

- [ ] **Task 1** — `layout::band_sort`: pure function
  - Write unit tests first: 3 synthetic cases (single line, two-col, skewed).
  - Implement: sort by y-center, group into bands if y-centers overlap
    (threshold: median block height × 0.5), within band sort by x-min.
  - Return `Vec<TextLine>` where `TextLine { bbox, text, blocks: Vec<OcrBlock> }`.
  - Commit.

- [ ] **Task 2** — introduce `DocPage { index: usize, size: (u32, u32), lines: Vec<TextLine> }`
  - Thread through `main.rs`: after per-page OCR, call `band_sort` → push `DocPage`.
  - Commit.

- [ ] **Task 3** — text writer: page separators
  - Change signature to accept `&[DocPage]`.
  - Emit `--- Page {n} ---\n` between pages, blank line after each line.
  - Update `ocr_hello_world` integration test (expect `--- Page 1 ---` in output).
  - Commit.

- [ ] **Task 4** — JSON writer
  - `serde::Serialize` on `DocPage`, `TextLine`, `OcrBlock`.
  - `write_json(&mut W, &[DocPage])` uses `serde_json::to_writer_pretty`.
  - Roundtrip test: serialize → deserialize → deep-equal.
  - Commit.

- [ ] **Task 5** — Markdown writer
  - `# Page {n}` heading per page; lines printed verbatim with blank line between.
  - Integration test: round-trips a synthetic two-page `Vec<DocPage>`.
  - Commit.

- [ ] **Task 6** — searchable PDF writer
  - `write_searchable_pdf(&Path, &[DocPage], &[RasterPage]) -> Result<()>`.
  - Use pdfium's `PdfPage::objects().create_text_object` to overlay *invisible*
    text at the bbox positions of each `TextLine`, on pages built from the
    rasterized bitmaps.
  - Integration test: OCR hello-world.png → searchable PDF; re-open it and
    assert pdfium's extracted text includes "hello" / "world".
  - Commit.

- [ ] **Task 7** — streaming iterator
  - `pdf::for_each_page<F>(path, dpi, mut f: F) -> Result<()>` where
    `F: FnMut(RasterPage) -> Result<()>`. Drops each bitmap after `f` returns.
  - Keep `rasterize_pdf` for image-comparison tests.
  - Commit.

- [ ] **Task 8** — wire streaming + all formats in `main.rs`
  - `--format` becomes a `clap::ValueEnum` with variants `{txt, pdf, json, md}`.
  - Dispatch to the correct writer.
  - The PDF writer needs the raster bitmaps, so its path uses
    `rasterize_pdf` (eager); other formats use `for_each_page`.
  - `ocr` subcommand still defaults to `txt`.
  - Commit.

- [ ] **Task 9** — integration test: all formats on `hello_world.png`
  - One test per format. Each verifies output is non-empty and well-formed
    (valid JSON / contains `# Page`, etc.).
  - Commit.

- [ ] **Task 10** — smoke test the real fixtures
  - Re-run Muna Madan + Invoices with each format; spot-check quality.
  - Report memory + runtime deltas vs Plan 1.
  - Document findings at the end of this plan file as "Task 10 results".

---

## Done criteria

- `cargo test --workspace` passes (new tests included).
- `cargo clippy --workspace --all-targets -- -D warnings` clean.
- On the receipts PDF: two-column pages read in natural order; each page
  starts with `--- Page N ---`.
- On Muna Madan: consecutive bboxes forming one verse-line emit as one
  line in txt/md/json/pdf.
- Peak RSS on the receipts run < 2 GB (down from ~14 GB ORT arena).
- Searchable PDF: text selects and copies correctly in Acrobat/Chrome.

---

## Out of scope — deferred to Plan 2.5

- Devanagari joiner / halant post-processing (port from Python).
- Confidence-guided dual-pass for low-confidence regions.
- Auto-script-detect on first-N bboxes (use existing `script.rs`).
- Per-page confidence summary / quality metrics report.
- Unicode font embedding for the searchable-PDF text layer so
  Devanagari / CJK text extracts correctly (current limitation
  documented in `output/searchable_pdf.rs`).

---

## Task 10 results — empirical smoke test (2026-04-15)

Hardware: Windows 11, Ryzen-class CPU, no GPU (Plan 4). Input:
9-page scanned-invoices PDF (3.6 MB, two-column layouts).

| Format | Runtime | File size | Sanity |
|---|---|---|---|
| `txt`  | 2m01s | 7.4 KB   | `--- Page N ---` separators, verse/row lines correct |
| `json` | 1m48s | 158 KB   | Valid JSON array; full bbox + confidence per block |
| `md`   | 1m51s | 7.4 KB   | `# Page N` headings, lines verbatim |
| `pdf`  | 1m41s | 17.5 MB  | Searchable — `searchable_pdf_preserves_ocr_text` passes |

**Runtime delta vs Plan 1**: 5m15s → ~2m for txt on the Devanagari
book's first 9 pages' worth of work. The streaming iterator's
effect on ORT's arena is the main win — the arena still grows but
doesn't hold every decoded bitmap alongside the model activations.

**Reading order** on two-column receipts is fixed: on paired-copy
receipts each header line now reads once per copy on a single line
(left copy, then right copy) instead of the Plan-1 interleaving. Line-item rows
read left-to-right on one line: qty, item#, description, price, subtotal.

**Searchable PDF** opens in Chrome / Acrobat with selectable, copyable
text. Font caveat holds: Latin round-trips; Devanagari/CJK in the
invisible layer may need Plan 2.5's font embedding work for reliable
extraction, though the raster view is always correct.

**All 31 workspace tests pass**; `clippy --workspace --all-targets
-- -D warnings` is clean.

Plan 2 closed.
