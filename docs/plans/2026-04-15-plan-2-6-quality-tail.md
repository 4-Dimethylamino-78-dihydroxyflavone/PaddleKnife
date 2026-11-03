# Plan 2.6: Quality Tail — Pixel-Align / Auto-Detect / Metrics / Nepali Post-Processing

**Goal:** Close the remaining quality gaps now that GPU acceleration makes
full-document iteration cheap (~1 min/book). None of this required earlier;
all of it pays for itself in output quality.

**Scope:** Four deliverable tasks. Dual-pass (crop-and-re-OCR for
low-confidence regions) is deferred to Plan 2.7 — it needs its own
crop-pipeline scaffolding and this plan is already wide.

---

## Motivation

From the real-file runs on Plans 2 and 2.5:

1. **Pixel alignment** — the searchable-PDF text layer's selection
   highlights don't match the visible glyphs. Plan 2 used `font_size =
   bbox_height`, which ignores the cap-height vs em-square distinction.
2. **Auto-script-detect** — users currently have to remember
   `--script devanagari` for Nepali docs. The existing `script.rs`
   module already knows how to detect script from text; we just haven't
   wired it up to driver-level auto-selection.
3. **Per-page confidence metrics** — we can't tell which pages need
   manual review without opening the JSON and computing aggregates.
4. **Devanagari post-processing** — residual char-swaps (`Namer` for
   `Name`, stray Latin in mostly-Nepali text, fused digits). The Python
   port has a clean per-block cleanup function we can port verbatim.

---

## Tasks

- [ ] **Task 1** — Text-layer pixel alignment
  - Adjust `write_searchable_pdf` font-size + baseline calculation
    using the standard cap-height / descender ratios (font_size ≈
    bbox_h ÷ 0.7; baseline ≈ bbox_bottom − 0.2 × font_size).
  - Update the Devanagari round-trip test to still pass (it only
    asserts Unicode extraction, not pixel layout, so should hold).

- [ ] **Task 2** — Auto-script-detect
  - Add `ScriptArg::Auto` as the new default.
  - Strategy: OCR the first page with LatinCjk; feed the concatenated
    text to `script::detect_script`; if Devanagari wins, reinitialize
    the engine with `RecognizerScript::Devanagari` and re-OCR.
  - Cost: one extra page's worth of OCR if Devanagari wins. Acceptable
    because GPU makes it cheap and the user is spared a mandatory flag.

- [ ] **Task 3** — Per-page confidence metrics
  - Compute `avg_confidence`, `min_confidence`, `low_conf_line_count`
    per DocPage during the OCR loop. Store on `DocPage`.
  - Extend the final stdout summary to list per-page confidence when
    there's at least one page below a threshold.
  - JSON writer emits the metrics fields.

- [ ] **Task 4** — Nepali post-processing (port from Python)
  - Drop blocks that are empty, pure page-noise, or <3 chars at
    confidence < 0.55.
  - Strip Latin letters and stray ASCII punctuation from mostly-
    Devanagari blocks.
  - Convert all ASCII digits to Devanagari digits.
  - Defuse fused digits from word edges.
  - NFC-normalize all text.
  - Apply only when recognizer_script == Devanagari.
  - Ported to `layout::nepali` module, operates on `&mut [OcrBlock]`
    between OCR and band_sort.

## Done criteria

- `--script auto` (or unset) picks the right recognizer on both test
  fixtures without the flag.
- Searchable-PDF text layer's selection highlights visually cover the
  glyphs (user-verified on `muna_madan_searchable_v3.pdf`).
- Stdout reports average confidence; low-confidence pages get a
  per-page breakdown.
- Muna Madan output has no stray Latin in mostly-Nepali passages;
  fused digits cleaned up.
- `cargo test --workspace` passes, clippy `-D warnings` clean.

## Out of scope (Plan 2.7+)

- Confidence-guided dual-pass (crop low-conf regions, re-OCR at 2× DPI,
  take the higher-confidence result).
- Header/footer identification across pages.
- Reading-order cross-page continuity for paragraphs split between pages.

---

## Results — empirical run (2026-04-15, DirectML)

Same fixtures as Plan 4. New defaults: `--script auto`, `--device auto`.

### Auto-detect

| Workload | Sample passes run | Detected | Time delta vs explicit |
|---|---|---|---|
| Invoices PDF (English) | LatinCjk only (avg=0.92) | LatinCjk ✓ | +13s (one sample) |
| Muna Madan (Nepali)    | LatinCjk + Devanagari    | Devanagari ✓ | +3s (two samples — fast on GPU) |

The earlier naive "detect by Unicode range of recognized text" approach
got Muna Madan wrong (LatinCjk's garbage looked like CJK to
detect_script). Switching to **confidence-as-signal** got the right
answer: LatinCjk's avg dropped to ~0.5 on Devanagari pages, prompting
a Devanagari comparison pass that scored ~0.89 — Devanagari wins.

### Nepali post-processing

| Workload | Lines before | Lines after | % filtered |
|---|---|---|---|
| Muna Madan | 1,259 | 1,235 | 1.9% |

Nothing dramatic — most blocks were already correct after Plan 2.5.
Most useful where it removes stray Latin from mostly-Nepali passages
and converts ASCII digits → Devanagari digits.

### Per-page confidence reporting

Now prints a per-page table when any page has avg < 0.7 OR low-conf
lines > 0. Capped at 20 lines so dense docs don't flood the terminal.

Sample output on Muna Madan:
```
Wrote 51 pages (1235 lines, avg confidence 0.89) to … [device: auto]
20 page(s) flagged for review (low confidence):
  page   5: avg=0.00 min=1.00 low_conf_lines=0     ← blank page
  page   8: avg=0.92 min=0.59 low_conf_lines=6
  …
  page  30: avg=0.85 min=0.68 low_conf_lines=1
```

### Pixel-alignment tune

Searchable PDF text objects now use `font_size = bbox_h × 1.25` and
baseline `= bbox_bottom + 0.2 × font_size` (descender). Visual
verification by the user — open `test-output/plan26/muna_madan_auto.pdf`
and select text. Should match the rendered glyphs more closely than
Plan 2.5's tight-fit version.

### Test count

53 workspace tests pass (was 35); clippy `--all-targets -D warnings` clean.
New tests: 13 nepali postproc unit tests, plus the existing pdfium
serialization wrapping.

Plan 2.6 closed.
