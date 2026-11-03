# Universal PDFs, local translation, and the in-place translated overlay

**Date:** 2026-06-13
**Status:** Approved design — implementation phased (1 → 4), not yet started
**Builds on:** glyphless text layer (Plan 5), script registry + download-on-demand (Plan 6), curved/rotated placement (Plan 7/7b/7c)

## Motivation & diagnosis

A PaddleKnife output (`MunaMadan_full_paddleknife_OCR.pdf`) failed in Kagi
Translate's document mode with a frontend `JSON.parse` error. Investigation
showed the text layer is **not** the problem:

- pypdf 6.0, PyMuPDF 1.27, and pdfminer 20250506 all extract identical,
  clean Devanagari from the file — the glyphless CID layer is universally
  parseable.
- The real anomaly is size: **1.8 MB source → 34.3 MB output** (51 pages).
  Source pages are ~75–150 effective-DPI DeviceGray JPEGs (~10–40 KB);
  the writer re-rasterizes every page at 300 DPI **DeviceRGB** JPEG q85
  (~300–920 KB). ~19× blowup, zero added information. Oversized uploads
  are exactly what makes web backends return HTML error pages that
  frontends then fail to `JSON.parse`.

So "make the PDF more universal" decomposes into **(1)** stop the size
blowup, and the user's larger ask adds **(2)** a fully local translation
pipeline and **(3)** a Google-Lens-style in-place translated overlay —
each phase ships independently and stacks on the previous one.

Decisions taken with the user:

- Scope: all three, phased.
- Translation backend: pluggable trait; **local-LLM endpoint first**
  (Ollama / LM Studio / llama.cpp, OpenAI-compatible), built-in NLLB
  ONNX later for the zero-external-software path.
- Overlay semantics: translated text visible **and** original text kept
  as the invisible searchable layer — the output is bilingual under
  Ctrl-F.

---

## Phase 1 — Slim, faithful searchable PDFs

Per-page decision in the searchable-PDF writer:

### 1a. Passthrough (new; default when safe)

If the input is a PDF and the page qualifies as a *plain scan page*,
copy the original compressed image bytes verbatim into the output's
image XObject. Qualification (all required):

- exactly one image XObject drawn covering ≳95 % of the MediaBox,
  no other meaningful content operators;
- filter ∈ {`DCTDecode`, `CCITTFaxDecode`, `FlateDecode`} with
  self-contained `DecodeParms`;
- colorspace ∈ {`DeviceGray`, `DeviceRGB`} (no Indexed/ICC/Separation).

Source inspection uses a new **read-only `lopdf` dependency** (we never
write with it; `pdf-writer` remains the sole author). Any anomaly —
encryption, JBIG2, multi-image pages, parse errors — falls back to 1b.
OCR still rasterizes via pdfium at full `--dpi`; passthrough affects
only which bytes get *embedded*.

Expected result on the motivating file: ~1.9 MB (input size + ~100 KB
text layer).

### 1b. Re-encode (today's path, improved)

- **Grayscale auto-detect:** if the raster is effectively neutral
  (max channel spread below threshold for ~all pixels), encode L8 JPEG
  in `DeviceGray` (~2–3× smaller than RGB for gray scans).
- **Embed-DPI cap:** downscale the embedded image (Lanczos3) to the
  source page's effective image DPI when that is below the OCR DPI —
  never upsample beyond what the scan contains. Image inputs keep their
  native resolution.
- **Quality knob:** `--jpeg-quality N` (default 85, unchanged).

### CLI

Better defaults, no breaking change: `--pdf-image auto|reencode`
(default `auto`), `--jpeg-quality N`, `--embed-dpi N` (default: auto
cap). Text-layer geometry is untouched — quads are already mapped in
page points independent of the embedded bytes.

---

## Phase 2 — Translation core + local-LLM backend (text output)

New `translate/` module in `paddleknife-core` around an object-safe
trait:

```rust
pub trait Translator {
    fn translate(&self, block: &str, ctx: &BlockCtx) -> Result<String>;
}
```

`BlockCtx` carries source/target language labels and the neighboring
blocks (previous/next snippets) for cross-block coherence.

### LlmTranslator (first backend)

- OpenAI-compatible `POST /v1/chat/completions` over the existing
  `ureq` (blocking, rustls — no async runtime). Works with Ollama
  (default URL `http://localhost:11434/v1`), LM Studio, llama.cpp
  server. 100 % on-machine.
- `--llm-url` (default above), `--llm-model` (required), temperature
  0.3, built-in literary-aware system prompt (translate only, preserve
  line breaks, no commentary), overridable via `--prompt-file`.
- Per-block retry with backoff; a block that still fails is recorded as
  untranslated, never silently dropped.

### Paragraph grouping

`layout` gains a block-grouping step: cluster band-sorted `TextLine`s
into paragraphs by vertical gap + indent heuristics (reusable later by
the markdown writer). The **block** is the translation unit.

### Translation cache (mandatory)

Sidecar `<output stem>.tlcache.json`, keyed by
`(block text hash, model id, target lang)` → values are translated
strings. A 51-page book through a local LLM is slow; interrupt/resume
and instant re-render must work. Cache hits skip the network entirely.

### CLI

`paddleknife translate <input> --to English [--from auto] -f md|txt`
→ translated markdown (default) with `## Page N` structure, or plain
text. `--from` defaults to the script auto-detector's verdict.

---

## Phase 3 — The "plaster" overlay PDF (`translate … -f pdf`)

In-place visible translation, kept lossless:

### Patching at the PDF level, not the raster

The original page image stays byte-identical (Phase 1 passthrough still
applies). Each translated block gets a **filled vector rectangle** over
the original text, colored with the **median of a border ring** sampled
around the block in the OCR raster — on clean scans the patch is
visually seamless, and the original pixels survive underneath.

### Visible translated text

- Vendored OFL **Noto Sans** (Latin + Cyrillic target scripts in v1),
  subset-embedded with `subsetter`, metrics via `ttf-parser` (both
  pure-Rust, same ecosystem as `pdf-writer`).
- Per block: greedy wrap + shrink-to-fit into the block bbox. Start
  size = block's median original line height; floor at 55 %.
- Text color = sampled dominant glyph (dark-pixel) color of the block,
  so patched text keeps the scan's ink tone instead of harsh black.

### Bilingual layers

The glyphless invisible layer still carries the original text at its
measured positions; the visible translated text is real-font text and
therefore searchable too. Ctrl-F finds either language.

### Failure honesty

- Untranslated block → **no patch**, original stays visible; end-of-run
  summary lists page/block.
- Translation that cannot fit even at the 55 % floor → rendered at
  floor and flagged in the summary; never silently truncated.

---

## Phase 4 — Built-in NLLB backend (zero external software)

`NllbTranslator` behind the same trait:

- **NLLB-200-distilled-600M**, int8 ONNX (encoder + decoder-with-past),
  run on the existing ort runtime; downloaded on demand through the
  Plan-6 infra (`paddleknife models pull translator`, ~600 MB).
- HF `tokenizers` crate (native Rust) for the SentencePiece model.
- Greedy decode with KV cache in v1; beam search later if quality
  demands it.
- Source language token derived from the script detector
  (`npi_Deva`, `rus_Cyrl`, …); target from `--to`.
- Selected with `--backend nllb` (no LLM flags needed). Expectation set
  honestly: functional/literal quality — fine for documents, weaker
  than a local LLM for poetry.

---

## Error handling (cross-phase)

- lopdf parse failure or any passthrough doubt → re-encode fallback
  (current behavior), warn at `-v`.
- LLM endpoint unreachable → fail fast with a message naming the URL
  and how to start Ollama; mid-run failures retry then mark blocks
  untranslated.
- Cache file corrupt → ignore + rebuild (it is derived data).

## Testing

- **Phase 1:** integration test on a fixture scan asserting passthrough
  output ≤ ~1.3× source size; `alignment_metrics` gate stays green;
  cross-engine extraction parity (pypdf + pdfminer) added to the Python
  eval harness.
- **Phase 2:** mock-`Translator` unit tests (prompt assembly, cache
  hit/miss, resume); live Ollama integration test `#[ignore]`d.
- **Phase 3:** text-fitting unit tests (measured line widths ≤ bbox);
  original-layer metrics unchanged; acceptance = MunaMadan translated
  overlay < 3 MB opening cleanly in Acrobat and pdf.js.
- **Phase 4:** tokenizer round-trip vectors vs HF reference; greedy
  decode golden outputs vs Python `transformers` on short sentences.

## Out of scope (future work)

JBIG2 passthrough (globals-stream complexity), complex-script *target*
rendering (needs rustybuzz shaping + per-script fonts), ML inpainting
for textured/photo backgrounds, toggleable OCG layers, serif/font-style
auto-matching, batched multi-document translation.
