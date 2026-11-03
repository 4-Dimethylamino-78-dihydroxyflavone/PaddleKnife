# PaddleKnife patches to oar-ocr 0.8.1

This directory is a vendored copy of the [`oar-ocr`](https://github.com/greatv/oar-ocr)
0.8.1 library crate (Apache-2.0, © Wang Xin), wired in through
`[patch.crates-io]` in the workspace `Cargo.toml`. Only the top-level crate is
vendored; `oar-ocr-core` and `oar-ocr-derive` still come from crates.io.
`oar-ocr-core` is pinned to `=0.8.1` in `Cargo.toml`: the patches depend on its
exact behaviour (patch 3 mirrors its recognizer width cap), and a
semver-compatible release could change that without anything failing to build.

`src/` is the published crate source with the modifications below. `examples/`,
`docs/` and the dev-dependencies that served them are not vendored. Each
modified file — `src/oarocr/ocr.rs`, `src/oarocr/preprocess.rs` and
`src/oarocr/table_analyzer.rs` — carries a `// PaddleKnife patch` notice at the
top.

Every patch is **opt-in**: with none of the new knobs set, behaviour is
identical to upstream — except patch 3, which is a pure bug fix.

## 1. Area-budgeted recognition batching — `src/oarocr/ocr.rs`

**Problem.** `recognize_global` sorts crops by aspect ratio and cuts them into
fixed-*count* chunks. Each chunk is zero-padded to its widest crop, so a chunk's
memory is `count × tensor_width`, and the CTC head alone materialises
`count × (tensor_width / 8) × vocab × 4` bytes about three times over. On a page
with one 50:1 line among 46 ordinary ones, upstream pads all 47 crops to
~2480 px: ≥ 9 GiB on DirectML with PP-OCRv6's 18,710-class head, and the run
dies with `E_OUTOFMEMORY` in `MatMul`/`Add` of the head. `region_batch_size`
cannot fix this in general — it bounds the wrong quantity.

**Patch.** `plan_recognition_chunks` replaces `crops.chunks(batch_size)`. A
chunk must satisfy `count ≤ region_batch_size` **and**
`count × tensor_width(widest) ≤ region_batch_size × region_ref_width`. The input
is already sorted ascending, so the widest crop of a chunk is its last and one
greedy pass is optimal. Narrow crops still fill the count cap (invoice-style
pages batch exactly as before); wide ones land in proportionally smaller chunks.
Because the budget is a multiple of `region_batch_size`, halving the batch size
halves peak recognizer memory for every page shape.

New API: `OAROCRBuilder::region_ref_width(px)` (`0`/unset = upstream behaviour).

## 2. Per-call overrides — `src/oarocr/ocr.rs`, `src/oarocr/preprocess.rs`

`OAROCR::predict_with(images, &PredictOptions)`; `predict` delegates to it with
`PredictOptions::default()`.

* `region_batch_size`, `region_ref_width` — retry a page under a tighter memory
  budget without rebuilding the ONNX sessions.
* `document_orientation: DocumentOrientation::{Auto, Skip, Forced(quarter_turns)}`
  — re-run a page at a chosen orientation through the *same* rotate-back
  geometry. Needed because the PP-LCNet doc-orientation classifier is unreliable
  off-distribution (UI screenshots, sparse poetry): it finds the text axis but
  flips the direction, with scores (≤ 0.91 measured) as high as its correct
  verdicts. Recognition confidence is the better judge, and the caller owns it.

## 3. Word-box ratio clamp (bug fix) — `src/oarocr/ocr.rs`

`ctc_word_boxes` undoes batch padding as `wh_ratio / max_wh_ratio`, but
`CRNNModel::preprocess` caps the tensor at `DEFAULT_MAX_IMG_WIDTH` (3200 px,
ratio 66.67). When a chunk's widest crop exceeds that ratio, every *other* crop
in the chunk is scaled against a tensor wider than the real one and its
per-character boxes are stretched (a 50:1 crop beside a 100:1 crop: 1.5×).
`clamp_rec_ratio` applies the recognizer's own clamp to both terms.

## 4. Orientation score floor — `src/oarocr/preprocess.rs`, `src/oarocr/table_analyzer.rs`

`OAROCRBuilder::document_orientation_min_score(f32)`: a top-1 verdict scoring
below the floor is treated as "no opinion" (page left untouched) instead of
being applied unconditionally. Default `0.0` = upstream. This is a weak filter
by design — see patch 2 for why it cannot be the whole answer.

Follow-on: the floor is a new `min_score` parameter of the crate-internal
`correct_image_orientation`, whose only other caller is the table-orientation
step in `table_analyzer.rs`. It passes `0.0`, so table crops behave exactly as
upstream.

## Not patched here (reported upstream instead)

`oar-ocr-core` 0.8.1: `CRNNModelBuilder::max_img_w()` is a no-op —
`build()` calls `OCRResize::new(Some(shape), None)`, discarding
`preprocess_config.max_img_w`, so the 3200 px cap cannot be changed through the
public API. Patch 1 makes this irrelevant for memory safety, so `oar-ocr-core`
is left unvendored. Draft issue text: `docs/upstream/oar-ocr-issues.md`.

## Updating

To move to a newer oar-ocr: replace `src/` with the new release, re-apply the
patches (each is small and self-contained; the tests at the bottom of `ocr.rs`
and `preprocess.rs` marked `PaddleKnife patch` pin the behaviour), bump
`version` and the `oar-ocr-core` pin in `Cargo.toml` to match, and run
`cargo test --manifest-path vendor/oar-ocr/Cargo.toml --lib`.
