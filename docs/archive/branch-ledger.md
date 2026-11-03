# Branch & worktree ledger

What every non-`main` line of work held, what it taught, and what happened
to it. Written 2026-09-30 during the worktree audit so that branches,
worktrees and stashes can be pruned without losing their lessons. Update
this file whenever a branch is merged, archived or deleted.

`main` @ `46f5d97` (2026-08-02) is the baseline for every "ahead/behind"
count below.

## Unmerged — still carry code that is not on `main`

### `claude/paddleknife-ocr-pipeline-1bdb54` — Plan 10: resilient batching (3 ahead, 0 behind)

- `cc0bcc6` vendors oar-ocr 0.8.1 under `vendor/oar-ocr` via
  `[patch.crates-io]` and adds area-budgeted recognition batching
  (budget = `region_batch_size × region_ref_width`).
- `ef0848f` adds `resilient::ResilientEngine` (retry ladder
  `base → base/2 → 8 → CPU`, per-page reset, sticky after 2 degraded
  pages) and `sysmem.rs` (Windows commit-headroom probe), plus the
  doc-orientation 180° verification re-read.
- `c6acce2` Plan 10 record, `OCR_RECIPES.md` memory recipe, drafts of
  upstream oar-ocr issues (`docs/upstream/oar-ocr-issues.md`, not filed).
- Lesson: recognizer memory ∝ B × W_max × vocab, and on Windows the
  ceiling is **system commit**, not VRAM (`8007000E` reads like VRAM,
  isn't).
- Review 2026-09-30: mergeable as-is; see `docs/AUDIT-2026-09-30.md` §4.1.
- Worktree: `.claude/worktrees/paddleknife-ocrv6-upgrade-d5d781` (name is
  historical — the worktree was reused for this branch).

### `claude/amazing-bhabha-1c74e8` — embedded-raster size regression (1 ahead, 0 behind)

- `c759316` splits `source_pdf::probe_inner` into `single_image_draw`
  (feeds the DPI hint) and `passthrough_gates`, fixes the lexer's
  12-digit cap (integer part only), makes `--pdf-image reencode` open
  `SourcePdf`, and adds a `RasterPage.native_dpi` backstop from pdfium's
  object model.
- Lesson: "the re-encode fallback is cheap" held only while the DPI hint
  survived; the root issue is the CLI rasterising at a fixed 300 DPI.
  Real-file cause was inferred, not verified.
- Review 2026-09-30: mergeable after two small fixes; see audit §4.2.
- Merges cleanly with the Plan 10 branch (`git merge-tree` dry run).

### `claude/admiring-murdock-c5a931` — Plan 2.9 i32 polygons + Plan 4.0 TrOCR via ort (27 ahead, 46 behind)

- Plan 2.9: `OcrBlock.polygon: [[i32; 2]; 4]`. **Superseded** by Plan 5's
  f32 `Quad` + `CharSpan` contract on `main` (i32 truncation was one root
  cause of the alignment bugs). Do not port.
- Plan 4.0 T1–T12: TrOCR (ViT encoder + merged-with-past decoder, Xenova
  ONNX) behind `--engine {paddle,trocr,auto}`, `--trocr-variant`,
  perspective-warp dewarp, HF tokenizer, KV-cache decode loop.
- Findings worth keeping:
  - TrOCR encoder **and** merged decoder segfault on DirectML during
    graph optimisation (SIGSEGV in the quantized-MatMul fusion path).
    Decoder must run on the CPU EP (or CUDA). ADR-0002 carries this forward.
  - Page-level TrOCR is wrong; it needs **line-level** crops with
    aspect-preserving resize (`cbad83e`).
  - Padding polygons before TrOCR made its receipt bias *worse*
    (reverted in `6de46e2`).
- Status: archival. Any future handwriting work starts from ADR-0002 on
  top of the Plan 5 contract, not from this branch.

### Tag `archive/trocr-force-cpu-a8915b2` (formerly only reachable from a stash)

`a8915b2` "fix(trocr): force CPU for TrOCR sessions — DirectML
segfaults" is a sibling of `22452db` on `admiring-murdock` — the branch
went the CUDA-pinned way instead. It was anchored only by the stash
below, so it was tagged before the stash was dropped.

## Dropped stash — candle TrOCR experiment (2026-05-22)

`stash@{0}` `2659f5b2`, "WIP on claude/admiring-murdock-c5a931: a8915b2".
Dropped 2026-09-30. It was created with untracked files included
(`git stash -u`), which swept a **1.27 GB `model.safetensors`** into
`.git` and pinned it there for four months.

What the WIP did (−716 / +154 lines tracked, plus untracked `trocr/model.rs`):

- Replaced the ort encoder/decoder (`trocr/{encoder,decoder,config}.rs`,
  ~560 lines) with `candle-transformers`' native TrOCR/ViT
  (`trocr/model.rs`, 157 lines, modelled on candle's
  `candle-examples/examples/trocr`), loading safetensors from
  `models/trocr-base-handwritten/` via `hf-hub`.
- Added `candle-core/nn/transformers 0.9` + `hf-hub 0.4` and a
  `trocr-cuda` feature made **default** (CPU path ≈ 5 min/page).
- `build_with_msvc.bat`: `vcvars64.bat` + `cargo check`, a workaround for
  the CUDA toolchain needing MSVC on PATH.

Why it was abandoned: candle has no DirectML backend (issue #344), so the
default Windows build would need a CUDA toolkit at build time. That
contradicts the zero-install DirectML default. ADR-0002 later reached the
same verdict independently and chose TrOCR via ort with the decoder on the
CPU EP.

Rule learned: **never `git stash -u` in this repo while `models/` holds
unignored weights.** Use a WIP commit instead. The stash stack is shared by
every worktree, so one bad stash inflates all of them.

## Merged — worktrees removed 2026-09-30 (branches kept)

| Branch | Held | Lesson pointer |
|---|---|---|
| `claude/thirsty-cannon-0a793c` | Plans 5–7: f32 quad + CharSpans, glyphless text layer, multi-script registry, doc-orientation | `docs/ARCHITECTURE.md` (searchable-PDF text layer; models and script routing) |
| `claude/zealous-franklin-e0bb6f` | Plan 8 spec + Amendment C (DPI hint decoupled from passthrough) | `docs/superpowers/plans/2026-06-13-plan-8-slim-faithful-pdfs.md` |
| `claude/inspiring-dewdney-88cd11` | Plan 8 merge (slim faithful searchable PDFs) | same |
| `claude/handwriting-surya` | directory input (single/mixed formats) + ADR-0002 research, which was **never committed** there; rescued to `docs/adr/` on 2026-09-30 | `docs/adr/0002-handwriting-recognition-backends.md` |

Also fully merged with no worktree: `claude/upbeat-elbakyan-e22b5a`
(Plan 2.8 `--detect-line-orientation`),
`claude/paddleknife-ocrv6-upgrade-d5d781` (Plan 9 PP-OCRv6),
`claude/paddleknife-directory-formats-fb4805`.
