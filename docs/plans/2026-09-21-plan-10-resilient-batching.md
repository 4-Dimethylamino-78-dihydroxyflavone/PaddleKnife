# Plan 10: Memory-safe recognition batching, page-level recovery, verified orientation

**Status:** done — 2026-09-21
**Trigger:** a 9-page PDF of desktop WhatsApp screenshots died on page 9 with
`8007000E Not enough memory resources`, on a 16 GB GPU with 12 GB free, and the
eight pages already OCR'd were thrown away.

---

## What was actually wrong

Six separate holes. The first is the root cause; the rest are what made it
costly or were found while chasing it.

### 1. The recognizer batch budget was in the wrong unit

oar-ocr sorts a page's crops by aspect ratio and cuts them into chunks of
`region_batch_size` (64 on GPU). Each chunk is zero-padded to its **widest**
crop, so a chunk costs `count × width`. The CTC head alone is
`count × (width / 8) × vocab × 4` bytes, held ~3× (final `MatMul` → bias `Add` →
`Softmax`), and PP-OCRv6's vocabulary is 18,710.

Every page of the trigger document has one 51:1 line — the chat header listing
the group's members — among lines with a median ratio of 5. With ≤ 64 crops on a
page there is a single chunk, and that one line pads all of them to ~2,480 px:

| page | crops | padded to | result (count-only batching) |
|---|---|---|---|
| 8 | 44 | 2478 px | pass, 5.7 GiB dedicated + 1.1 GiB readback |
| 9 | 47 | 2477 px | **fail**, ≥ 9.2 GiB at the point of failure |

The two failing runs died at "different" nodes (`MatMul.12`, `Add.216`). They
are consecutive nodes of the head allocating the same tensor; the second run
simply had marginally more headroom.

Things that were *not* the cause, each checked: the GPU (16 GB, 12 free); the
iGPU being picked by mistake (the 4060 Ti drives the display, so it is DXGI
adapter 0); the Plan 9 model swap (PP-OCRv5 mobile has an 18,385-class head and
the same peak within 17 %); the 300-DPI rasterisation (crops are resized to
48 px high, so aspect ratio — and tensor width — is resolution-independent);
memory ratcheting across pages (page 9 alone, in a fresh session, fails the same
way); the detector (capped at 960 px, one image per call).

`DEFAULT_GPU_REGION_BATCH = 64` came from the Plan 4.5 sweep, which only ever saw
invoices and a Devanagari book — narrow lines. Its `256 → FAIL` row was this same
mechanism; the sweep found a symptom threshold, not the law.

### 2. The ceiling was system commit, not VRAM

Under WDDM every GPU allocation must be backed by system commit (RAM +
pagefile). The machine was at **93 % commit — 6.8 GiB available of 104.8** — so
~7–9 GiB was all the GPU memory a process could obtain, whatever the card had
free. This is why failure was non-monotonic in batch area and drifted during the
day: `--region-batch-size 32` passed three times in the morning and failed on
page 1 that afternoon, with identical free VRAM. During this work the same
exhaustion broke a `cargo test` build (`os error 1455: The paging file is too
small`).

The lesson for the design: the limit is a moving target set by everything else
on the machine, so budgeting alone cannot be enough — recovery has to exist too.

### 3. A failed page cost the whole document

`for_each_page` propagated the first error; eight good pages were dropped and the
run ended with `no pages were OCR'd — every input failed`, which was false.
Running out of accelerator memory under dynamic shapes is an *expected* failure
with a mechanical remedy, and it was being treated as fatal.

### 4. Script auto-detect ignored the batch settings

`sample_confidence` built its engine with `OcrTuning::default()`, so
`--region-batch-size` could not rescue a document whose *first* page was the
heavy one.

### 5. `--detect-page-orientation` silently destroyed pages

The classifier's top-1 verdict was applied unconditionally. It is reliable on
clean Latin scans (0.87–0.93, always right in our sample) and unreliable off
them — and its score does not say when:

| material | wrong | score when wrong |
|---|---|---|
| upright chat screenshots (9 pp) | 3 called 180° | 0.34, 0.46, 0.70 |
| *Muna Madan*, 3 pp × 4 rotations | 9 of 12 | up to **0.91** |
| correct verdicts, same sample | — | median 0.92, min 0.47 |

With the old binary, 3 of 4 upright *Muna Madan* pages fell from ~150 text blocks
to **0 or 1**. `OCR_RECIPES.md` recommended this flag in its "universal command".
What *is* consistent: 14 of 15 wrong verdicts were exact 180° flips — the model
finds the text axis and misses the direction — and a page read upside-down scores
0.48–0.52 mean recognition confidence against 0.85–0.91 the right way up.

### 6. Upstream: two latent bugs

`ctc_word_boxes` undoes padding as `wh_ratio / max_wh_ratio` without the
recognizer's 3200 px clamp, stretching boxes 1.5× for a 50:1 crop batched with a
100:1 crop. And `CRNNModelBuilder::max_img_w()` is a no-op (`build()` passes
`None`). Drafts for both, plus the batching and orientation issues, are in
`docs/upstream/oar-ocr-issues.md`. Nothing has been filed.

---

## What changed

**Vendored oar-ocr** (`vendor/oar-ocr`, via `[patch.crates-io]`; top crate only,
414 KB of source; every patch opt-in except the box fix; ledger in
`vendor/oar-ocr/PADDLEKNIFE_PATCHES.md`):

- `plan_recognition_chunks` — a chunk must satisfy `count ≤ batch` **and**
  `count × tensor_width(widest) ≤ batch × ref_width`. Input is already sorted, so
  one greedy pass is optimal. Narrow-crop pages batch exactly as before.
- `OAROCR::predict_with(images, &PredictOptions)` — per-call batch size and
  `DocumentOrientation::{Auto, Skip, Forced(quarter_turns)}`, the latter through
  the existing rotate-back geometry.
- `clamp_rec_ratio` — the word-box fix.
- `document_orientation_min_score` — a floor below which a verdict is "no
  opinion".

**`paddleknife-core`**

- `ocr.rs`: `OcrTuning::region_ref_width`, `PageOptions`, `PageOrientation`,
  `PageOcr`, `OcrBackend::ocr_page`. One number drives everything: the area
  budget is `region_batch_size × ref_width`, so halving the batch halves the
  memory ceiling for every page shape.
- `resilient.rs`: `ResilientEngine`.
  - Ladder `base → base/2 → 8 → CPU`. A failed session is dropped before the
    retry — ONNX Runtime's DirectML allocator caches freed buffers per session —
    while stepping back *up* for the next page needs no rebuild.
  - Per-page reset, turning sticky after 2 degraded pages so a systematically
    heavy document (or a small GPU) stops paying a failed attempt per page. CPU
    serves single pages only and is never adopted.
  - Orientation: floor of 0.5 on the classifier, then any page reading below
    0.80 mean confidence (or reading empty) is re-read flipped 180°; the flipped
    read must win by 0.05. (As landed, "win" compared per-block means; it now
    compares per-character quality — see the follow-ups at the end.)
- `sysmem.rs`: reads available commit and names it when a GPU failure coincides
  with < 8 GiB of it.

**CLI**: page-level salvage (a page that fails on every rung is kept without
text, listed, exit non-zero; a file where *every* page fails is a failed file);
honest messages; auto-detect shares the run's tuning and ladder;
`--region-ref-width`; corrected `--region-batch-size` help (it claimed a default
of 32).

---

## Verification

Hardware: RTX 4060 Ti 16 GB, DirectML, PP-OCRv6 medium, Windows 10, machine at
~93 % commit throughout (so timings are noisy; see below).

| check | result |
|---|---|
| both original failing commands, no workaround flags | pass, 9/9 pages, 0.98 avg |
| whole-process GPU memory, trigger document | **3.6 GiB** (was ≥ 9.2 GiB and a crash) |
| … at `--region-batch-size` 32 / 16 / 8 | 2.2 / 1.5 / 1.2 GiB (≈ 0.85 GiB floor: detector + weights) |
| batch plan, page 9 | 39 + 8 crops, not 47 × 2477 px; ≤ 2 batches on every page |
| ladder on real hardware (`--region-ref-width 0`, 6.0 GiB commit) | p1 fail@64 → pass@32; p2 fail@64, fail@32 → pass@8; sticky; 9/9 pages |
| in-process session rebuild after a DirectML OOM | works (was the one open risk) |
| screenshots + `--detect-page-orientation` | pp 2, 3, 5: 0.68/0.70/0.62 → 0.97 |
| *Muna Madan*, 4 upright + 3 rotated pages, flag on | old: 0/1/1 blocks on three upright pages, 0 and 1 on two rotated; new: best achievable on all 7 |
| box geometry vs old binary (1,587 word blocks, 7,224 char quads) | median shift 0 px; p99 43 px blocks / 8 px chars, vs 31 / 7 px between two runs of the *old* binary |
| text vs unbatched CPU reading | 0.9922 (old GPU b32: 0.9934; new b8: 0.9985); no text lost |
| unit tests | vendored oar-ocr 111 (96 upstream + 15 new); `resilient` 20; `sysmem` 1 |
| full workspace suite + `clippy --all-targets` | 229 pass, 1 pre-existing ignore, incl. all 7 `alignment_metrics`; clippy clean |

Page 2 of the ladder run is the design argument in one line: with count-only
batching, halving 64 → 32 changes nothing for a 24-crop page, so it failed twice.
With the area budget each rung really is half the memory.

**Timing — old vs new on the primary workload.** *Muna Madan*, 51 pages,
`--script devanagari`, three interleaved rounds:

| binary | runs (s) | median | output |
|---|---|---|---|
| old (count-only batching) | 50.8 56.1 50.9 | 50.9 s | 51 pp, 1243 lines, conf 0.87 |
| new (Plan 10) | 48.3 43.4 41.6 | **43.4 s** | 51 pp, 1243 lines, conf 0.87 |

~15 % faster with identical output: less padding means fewer columns computed.

**Default `ref_width`.** Count cap held at 64, three interleaved rounds each:

| `ref_width` | screenshots, 9 pp (wide lines) | *Muna Madan*, 51 pp |
|---|---|---|
| **896** (chosen) | 25.4 18.2 15.5 — median 18.2 s | 40.5 43.2 45.7 — median **43.2 s** |
| 448 | 15.6 13.9 14.1 — median 14.1 s | 51.1 39.2 49.4 — median 49.4 s |
| 320 | 14.3 12.6 13.0 — median 13.0 s | 57.7 39.4 57.2 — median 57.2 s |

The two document types pull opposite ways: a smaller budget helps wide-line
pages and hurts book pages. Books of 100+ pages are the main workload, so 896
stays. Treat this as "no evidence for changing it", not as a tuned optimum: the
machine was paging throughout, the book runs are bimodal (≈ 39 s or ≈ 50–57 s),
and every setting's *best* book run is within a second of the others. 896 also
has a structural reason: at batch 64 it keeps the 18,710-class logits tensor at
511.6 MiB, inside DirectML's 512 MiB power-of-two bucket.

**Fidelity observation.** Padding perturbs a CTC recognizer slightly, mostly as
dropped spaces. The less padding, the closer to the unbatched reading, which is
why batch 8 scores highest above. n = 1 document.

---

## Not done / follow-ups

- ~~**Pages that miss PDF passthrough are embedded at 12–13× their source
  size**~~ (the fixed 300-DPI raster of a 96-DPI screenshot). Fixed by c759316
  ("never embed the inflated OCR raster when the source DPI is known"), merged
  2026-09-30.
- **Fixed 300-DPI rasterisation** upsamples low-resolution sources 3.1× for no
  gain in recognition, and nothing caps raster dimensions for very large pages.
- **Padding-aware chunking for fidelity**: bounding the ratio *spread* within a
  chunk would cut padding drift further, at the cost of more distinct tensor
  shapes (which DirectML pays for). Worth a measurement, not a guess.
- **Upstreaming** the oar-ocr patches so `vendor/oar-ocr` can go away.

### Done since landing (2026-09-30 audit, `docs/AUDIT-2026-09-30.md` §4.1)

- **The flip is decided per character.** The per-block mean let a sparse
  flipped read (two fragments at 0.9) replace a dense correct one (40 lines at
  0.72), stripping the page's text layer. The flipped read must now beat the
  first on `PageOcr::read_quality` — confidence per recognised grapheme, by
  0.05 — *and* recover at least half its graphemes. Mean confidence remains the
  re-check trigger only.
- **The re-read starts on the rung the first read succeeded on**, not the run's
  floor, which had just failed on the same lines.
- **Stickiness counts pages, not reads**, and only pages a smaller *GPU* batch
  rescued: a page whose re-check also stepped down counts once, and a page only
  CPU could rescue says nothing about batch size.
- **The low-commit hint is read before the failed session is freed**, so it
  sees the memory state the allocation failed in.
- **Honest summary wording**: "exceeded GPU memory" only when every failure
  behind a retry reads as an allocation failure (`OcrError::is_out_of_memory`),
  otherwise "hit GPU errors". Retry policy is unchanged: every engine error is
  retried.
- `oar-ocr-core` is pinned to `=0.8.1` under the vendored crate, and
  `PADDLEKNIFE_PATCHES.md` now lists the `table_analyzer.rs` follow-on.
