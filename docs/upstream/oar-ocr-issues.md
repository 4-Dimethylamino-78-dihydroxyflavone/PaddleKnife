# Draft upstream issues for oar-ocr

Found while diagnosing a DirectML out-of-memory failure (Plan 10). Nothing here
has been filed — these are ready-to-paste drafts for
<https://github.com/greatv/oar-ocr/issues>. Issues 1–3 are patched locally in
`vendor/oar-ocr` (see `vendor/oar-ocr/PADDLEKNIFE_PATCHES.md`); issue 4 is not,
because the local fix for issue 1 makes it moot for us.

All line numbers refer to the 0.8.1 crates as published on crates.io.

---

## 1. Recognition batches are bounded by crop count, but memory scales with padded area

**Where:** `oar-ocr` `src/oarocr/ocr.rs`, `OAROCR::recognize_global`.

**What happens.** Crops are sorted by width/height ratio and cut with
`crops.chunks(region_batch_size)`. `CRNNModel::preprocess_refs` then pads every
crop in a chunk to the chunk's widest member
(`tensor_width = min(48 × max_wh_ratio, 3200)`). A chunk's memory is therefore
`count × tensor_width`, not `count`, and `region_batch_size` bounds the wrong
quantity. When a page has fewer crops than `region_batch_size`, the sort does
not help at all: there is a single chunk, padded to the page's widest line.

**Why it bites.** The CTC head materialises `[B, W/8, vocab]` roughly three
times in a row (final `MatMul` → bias `Add` → `Softmax`). With PP-OCRv6's
18,710-class dictionary that is 9.1 KiB per tensor column, per copy.

Reproduction: a page with 46 ordinary lines (ratio ≈ 5) and one 51.6:1 line —
a desktop chat screenshot whose header lists the group's members. With the
default accelerator batch of 64, all 47 crops go out as one
`47 × 3 × 48 × 2476` tensor:

| | padded columns per batch | largest logits tensor | DirectML peak (measured) |
|---|---|---|---|
| upstream, one chunk of 47 | 116,372 | 1.01 GiB × ~3 | ≥ 9.2 GiB → `E_OUTOFMEMORY` |
| area-budgeted: 39 + 8 | ≤ 57,344 (the 8 widest: 19,816) | ≤ 0.50 GiB × ~3 | 3.6 GiB over the whole document |

```
Non-zero status code returned while running FusedMatMul node.
Name:'MatMul.12/MatmulTransposeFusion/' ... 8007000E Not enough memory resources
adapter execution failed: TextRecognitionAdapter: forward (batch_size=47, return_word_box=true)
```

The failing node alternates between `MatMul.12` and `Add.216` from run to run;
they are consecutive nodes of the head allocating the same-sized tensor.

**Suggested fix.** Budget chunks by padded area as well as count. Because the
crops are already sorted ascending, the widest member of a chunk is its last,
so a single greedy pass is optimal:

```rust
fn plan_recognition_chunks(wh_ratios: &[f32], max_count: usize,
                           column_budget: Option<usize>) -> Vec<Range<usize>> {
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < wh_ratios.len() {
        let mut end = start + 1;
        while end < wh_ratios.len() && end - start < max_count {
            let padded = (end - start + 1) * rec_tensor_width(wh_ratios[end]);
            if column_budget.is_some_and(|b| padded > b) { break; }
            end += 1;
        }
        chunks.push(start..end);
        start = end;
    }
    chunks
}
```

We express the budget as `region_batch_size × ref_width` (we use 896), so that
halving `region_batch_size` halves peak recognizer memory for every page shape
and narrow-crop pages batch exactly as they do today. An `Option` keeps current
behaviour as the default. A patch with tests is in our tree and we are happy to
open a PR.

---

## 2. `ctc_word_boxes` ignores the recognizer's width cap

**Where:** `oar-ocr` `src/oarocr/ocr.rs`, the call to `ctc_word_boxes` in
`recognize_global`.

Padding is undone as `effective_cols = seq_len × wh_ratio / chunk_max_wh_ratio`.
But `CRNNModel::preprocess_refs` caps the tensor at `DEFAULT_MAX_IMG_WIDTH`
(3200 px = ratio 66.67) and squeezes wider crops into it. When a chunk's widest
crop exceeds that ratio, every *other* crop in the chunk is scaled against a
tensor wider than the real one.

Example: a 50:1 crop sharing a chunk with a 100:1 crop occupies
`2400 / 3200 = 75 %` of the tensor (300 of 400 CTC columns). The formula gives
`50 / 100 = 50 %` (200 columns), so every character box on that line is
stretched 1.5× to the right.

**Fix:** clamp both ratios the way the recognizer does —
`r.min(DEFAULT_MAX_IMG_WIDTH as f32 / img_h as f32)` — before dividing.

---

## 3. Document-orientation verdict is applied unconditionally; no way to override per call

**Where:** `oar-ocr` `src/oarocr/preprocess.rs`, `correct_image_orientation`.

The top-1 class is applied whatever its score, and `predict` offers no way to
run a page at a caller-chosen orientation.

On scanned Latin documents `pp-lcnet_x1_0_doc_ori` is excellent (0.87–0.93,
always right in our sample, all four rotations). Off-distribution it is not,
and — importantly — **its score does not tell you when**:

| material | verdicts wrong | score of wrong verdicts |
|---|---|---|
| desktop chat screenshots, upright (9 pages) | 3 of 9 called 180° | 0.34, 0.46, 0.70 |
| Devanagari poetry, 4 rotations × 3 pages | 9 of 12 | up to **0.91** |
| a sparse scan, 4 rotations | 4 of 4 | 0.47–0.81 |

Correct verdicts in the same sample: median 0.92, min 0.47. No threshold
separates the two populations. What *is* consistent: 14 of the 15 wrong
verdicts were exact 180° flips (upright ↔ 180°, 90° ↔ 270°) — the model finds
the text axis and misses the direction. Pages read upside-down come back as
confident-looking garbage (mean recognition confidence ≈ 0.65 vs ≈ 0.97).

**Suggested changes.**

1. `PredictOptions { document_orientation: Auto | Skip | Forced(class_id) }` on
   a `predict_with` entry point, routed through the existing
   `apply_orientation_from_class_id` so rotate-back geometry is unchanged. This
   lets a caller that *can* judge the result (by recognition confidence) re-run
   a page flipped and keep the better read. That is what we do, and it fixed
   every case above at the cost of one extra pass on the affected pages only.
2. Optionally, a `document_orientation_min_score` builder knob so very weak
   verdicts are treated as "no opinion". Useful, but by the table above it
   cannot be the whole answer.

---

## 4. `CRNNModelBuilder::max_img_w()` has no effect

**Where:** `oar-ocr-core` `src/models/recognition/crnn.rs`, `CRNNModelBuilder::build`
(line 366).

```rust
let resizer = OCRResize::new(Some(self.preprocess_config.model_input_shape), None);
//                                                                           ^^^^
```

`preprocess_config.max_img_w` — set by `CRNNModelBuilder::max_img_w()` and by
`TextRecognitionAdapterBuilder::max_img_w()` — is never read, so `OCRResize`
always falls back to `DEFAULT_MAX_IMG_WIDTH` (3200). The cap cannot be changed
through the public API.

**Fix:** pass `self.preprocess_config.max_img_w` as the second argument.

Note for whoever fixes this: lowering the cap squeezes long lines horizontally
(it trades accuracy on exactly those lines for memory), and it makes issue 2
reachable at lower ratios — the two should land together.
