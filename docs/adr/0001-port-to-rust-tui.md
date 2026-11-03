# ADR-0001: Port PaddleKnife to Rust with a Ratatui TUI

**Status:** Superseded (2026-09-30). Accepted 2026-04-14; see
[What actually shipped](#what-actually-shipped).
**Date:** 2026-04-14
**Deciders:** @Nicholas-EW (maintainer)

## Context

PaddleKnife is a 1,910-line Python OCR tool (plus ~440 lines across 5 helper
scripts) built on RapidOCR (ONNX Runtime) with custom PP-OCRv5 Devanagari
support, PyMuPDF for PDF I/O, and Rich for CLI output. It ships as a CLI
(`ocr` / `batch` / `info` sub-commands) invoked through a `.bat` launcher on
Windows.

The goal is a Rust rewrite with:

- A Ratatui TUI as the primary interface (CLI retained as a secondary path).
- File drag-and-drop from the OS **and** in-TUI mouse interactions (reorder
  queue, resize panes, drag ranges).
- Path editing, keyboard input, and live progress per file and per page.
- GPU acceleration at the dominant bottlenecks (detection + recognition
  inference).
- Full feature parity with all Python scripts **except** `translate_book.py`
  (LLM-dependent, not a good Rust port target).

The backend decision is pre-committed to [`rust-paddle-ocr`][rpo] (also
published as [`ocr-rs`][ocrs]) per the user's direction. Both are the same
crate by @zibo-chen, backed by the MNN inference framework.

[rpo]: https://github.com/zibo-chen/rust-paddle-ocr
[ocrs]: https://docs.rs/ocr-rs/latest/ocr_rs/

### Key constraints

| Constraint | Impact |
|---|---|
| Windows 10/11 is the primary target | MNN Vulkan/OpenCL GPU must work on Windows; DirectML is not an MNN path |
| Devanagari PP-OCRv5 custom ONNX model exists | Must re-acquire/convert to MNN, or lose the 84.96% custom-converted model |
| Terminal "file drop" is bracketed-paste text | Cannot assume structured file-drop events; must parse paths from pasted input |
| `rust-paddle-ocr` Cargo.toml exposes **no GPU feature flags** | GPU backend selection must come from MNN runtime API, which the crate does not document — this is the single biggest unknown in the plan |
| Single-binary deploy preferred | Rules out heavy DLL sets; favors statically-linked MNN |

### Pipeline hotspots (for GPU targeting)

Estimated hot path for a typical multi-page PDF (unmeasured — to be
validated in Action Item 1):

1. **PDF rasterization at 300 DPI** — non-trivial fraction (MuPDF, CPU)
2. **Text detection inference (PP-OCRv5 det)** — dominant cost
   (ONNX → MNN, GPU-able)
3. **Text recognition inference (PP-OCRv5 rec, per crop)** — second-largest
   cost (ONNX → MNN, GPU-able, batchable)
4. Image preprocessing (resize/normalize) — small (CPU)
5. Devanagari post-processing + quality metrics — small (CPU)
6. PDF write with invisible text layer — small (MuPDF, CPU)

"GPU at traditional bottlenecks" therefore means MNN-GPU for (2) and (3).
The other steps stay CPU and are not materially improved by GPU.
Profile the Python version end-to-end before the port to establish
baseline numbers worth beating.

## Decision

Build `paddleknife-rs` as a **Cargo workspace** with two crates and a
**single unified binary** named `paddleknife.exe`:

1. `paddleknife-core` — library: OCR pipeline, model manager, PDF I/O,
   Devanagari post-processing, output writers.
2. `paddleknife` — single binary with mode dispatch:
   - Invoked with no args (e.g. double-clicked from File Explorer) → **TUI mode**.
   - Invoked with sub-command (`paddleknife ocr file.pdf`) → **CLI mode**.
   - Invoked with `paddleknife batch ./dir --tui` → CLI mode that launches the
     TUI pre-populated with the directory.

The double-click-to-TUI behavior is a hard requirement: the resulting
`paddleknife.exe` must open a usable TUI session when launched from
Explorer on Windows. Dispatch happens in `main()` before `clap` parsing
— if `env::args().count() == 1` and stdout is a TTY, enter TUI; otherwise
delegate to the clap sub-command router.

Backend: **MNN via `rust-paddle-ocr`** with a runtime-controlled GPU selector.
Because the crate exposes no GPU feature flags or public API for backend
choice, we will **fork `rust-paddle-ocr` into `vendor/`** and add a
`BackendConfig { Cpu, Vulkan, OpenCl }` constructor that routes through
`mnn::ScheduleConfig`. Upstream a PR in parallel; drop the fork when merged.

PDF: **`mupdf-rs`** (same MuPDF library as PyMuPDF) to preserve the exact
invisible-text-layer technique `output_searchable_pdf()` uses.

Concurrency: **`tokio`** with a blocking-threadpool for OCR batches, so the
Ratatui render loop stays at 30 fps while pages process in parallel.

## Options Considered

### Decision 1: Workspace structure

**Chosen: Option A-modified — two-crate workspace with one unified binary.**
The original Option A proposed two binaries (`paddleknife-cli` and
`paddleknife-tui`). User requirement added on 2026-04-14: double-clicking
the `.exe` on Windows must launch the TUI. Simplest path is a single
binary that dispatches on `argv.len() == 1`. Two crates remain: `core`
(library) and `paddleknife` (binary).

#### Option A: Three-crate workspace (core / cli / tui)

| Dimension | Assessment |
|---|---|
| Complexity | Medium |
| Binary size | Two binaries, ~15 MB each |
| Reusability | Core usable from scripts, tests, future crates |
| Team familiarity | Standard Rust layout |

**Pros:** Clean boundaries, core is independently testable, CLI and TUI
evolve without stepping on each other, lib crate can be `cargo publish`ed
later if useful.
**Cons:** Three Cargo.toml files to maintain; slight compile-time overhead.

#### Option B: Single crate with `[[bin]]` targets

| Dimension | Assessment |
|---|---|
| Complexity | Low |
| Binary size | Same as A |
| Reusability | Harder — lib and bins share a namespace |
| Team familiarity | Standard for small projects |

**Pros:** One Cargo.toml, simpler mental model for a solo project.
**Cons:** Library code and binary code mix; scales poorly past ~5k LoC,
and the Python version is already 2,300+ LoC — Rust port will be larger.

#### Option C: Monolithic single-binary with feature flag for TUI

**Pros:** Smallest build.
**Cons:** Loses library reusability; `--features tui` vs default split is
awkward when both are first-class interfaces. Not recommended.

### Decision 2: PDF library

#### Option A: `mupdf-rs` — **recommended**

| Dimension | Assessment |
|---|---|
| Feature parity with PyMuPDF | Very high (same underlying C library) |
| Searchable-PDF text layer | Supported via TextWriter API |
| Build complexity | Medium — MuPDF C build, but messense ships prebuilt |
| License | AGPL (same as PyMuPDF) — user already on this license |

**Pros:** 1:1 port of `output_searchable_pdf()` and `surya_injector.py`;
same invisible-text technique, same glyph measurement.
**Cons:** AGPL locks the project's redistribution terms — but PaddleKnife
is already effectively AGPL via PyMuPDF, so this is a no-op.

#### Option B: `pdfium-render`

| Dimension | Assessment |
|---|---|
| Feature parity with PyMuPDF | Partial — render + extract good, text injection less polished |
| Searchable-PDF text layer | Workable but different API |
| License | Apache-2.0 / BSD (more permissive) |

**Pros:** Looser license if monetization ever matters.
**Cons:** Rewrite of the PDF write path; invisible-text technique needs
re-engineering for PDFium's text-object model. Adds risk.

#### Option C: Pure-Rust (`lopdf` + `printpdf` + `pdf-extract`)

**Pros:** No C deps.
**Cons:** None of these individually match MuPDF's feature set; combining
them is a significant engineering project in itself. Not recommended.

### Decision 3: Concurrency model

#### Option A: Tokio async + `spawn_blocking` for OCR — **recommended**

Ratatui event loop on tokio; OCR pages dispatched to
`tokio::task::spawn_blocking` (or `rayon` pool behind a channel). Progress
updates flow back through `mpsc` to the render loop.

**Pros:** TUI stays responsive during OCR; clean cancellation;
parallelism across pages.
**Cons:** One more concept to learn; care needed not to starve the pool.

#### Option B: Thread-per-task with `crossbeam` channels

**Pros:** Simpler mental model; no async runtime.
**Cons:** Cancellation is harder; TUI frame pacing gets coupled to OCR
throughput if not carefully separated.

#### Option C: Synchronous OCR with TUI on a separate thread

**Pros:** Simplest.
**Cons:** Progress granularity limited to per-page; no parallelism.

### Decision 4: MNN GPU backend strategy — **highest-risk decision**

#### Option A: Fork `rust-paddle-ocr`, add backend selector — **recommended**

Maintain `vendor/rust-paddle-ocr` with a public `Det::with_backend(cfg)` /
`Rec::with_backend(cfg)` API that forwards to MNN's `ScheduleConfig`.
Upstream the change; drop the fork on merge.

**Pros:** Full control; Vulkan/OpenCL selectable; CPU fallback explicit;
surfacing the choice in CLI/TUI (`--gpu=vulkan`) is straightforward.
**Cons:** Fork maintenance overhead until PR lands; one-time effort
to trace MNN's `ScheduleConfig` through the crate's abstractions.

#### Option B: Ship CPU-only initially, GPU as phase 2

**Pros:** Fastest to first working build; no fork.
**Cons:** "GPU acceleration at traditional bottlenecks" is a stated goal;
punting it defeats a core requirement. Users will see CPU OCR as a
regression vs current Python + DirectML.

#### Option C: Bypass `rust-paddle-ocr`, bind MNN directly

**Pros:** No abstraction; maximum control.
**Cons:** Reimplements detection/recognition pre/post-processing from
scratch. ~2k LoC of work that `rust-paddle-ocr` already solved. Not
recommended unless Option A fork proves infeasible.

### Decision 5: TUI layout

#### Option A: Four-pane single-screen layout — **recommended**

```
┌─ Queue ───────────────┬─ Settings ────────────────┐
│ > scan_001.pdf  [ne]  │ Lang:   [auto ▾]          │
│   scan_002.pdf  [ne]  │ Format: [pdf  ▾]          │
│   invoice.png   [en]  │ GPU:    [vulkan ▾]        │
│                       │ Coords: [ ]               │
│ [drop files here]     │ Output: /path/to/out/     │
├───────────────────────┴───────────────────────────┤
│ Progress ──────────────────────────────────────── │
│ scan_001.pdf  ██████████░░░░░░░  142/248  01:23  │
│ Overall       ████░░░░░░░░░░░░░   1/3                │
├───────────────────────────────────────────────────┤
│ Log                                               │
│ 14:22:03 Initializing MNN (vulkan)                │
│ 14:22:05 Loaded PP-OCRv5 det + rec               │
└───────────────────────────────────────────────────┘
```

- Queue pane: click to select, drag to reorder, `x` to remove, `space`
  to toggle pin.
- Settings pane: Tab to focus, arrows to change, enter to apply.
- Path editor is a modal overlay (`a` to add, opens input with Tab
  completion backed by `tokio::fs`).
- OS drag-drop: bracketed-paste sequence triggers a path parser; if the
  pasted text matches `^[\"']?([A-Z]:[\\/]|/)[^\"']+[\"']?$` (Windows abs
  or POSIX abs), it's added to the queue silently.

**Pros:** Single screen = low cognitive load; mouse + keyboard parity;
matches the "progress bar for OCR" requirement directly.
**Cons:** Cramped on 80-column terminals; needs responsive layout rules.

#### Option B: Multi-screen wizard (file select → settings → run)

**Pros:** Clear step-by-step flow.
**Cons:** More modal, slower for batch users; harder to show live
progress while adding more files.

## Trade-off Analysis

The ordering of risks from highest to lowest:

1. **MNN GPU selector** (Decision 4). `rust-paddle-ocr` does not expose a
   documented GPU backend API. If the MNN `ScheduleConfig` route does
   not thread cleanly through the crate's abstractions, the fork could
   balloon into a vendored reimplementation. **Mitigation:** spike this
   in week 1 — build a minimal `Det` that selects Vulkan on a sample
   image. Go/no-go gate before committing to Option A.
2. **PP-OCRv5 Devanagari model conversion** (implicit). The custom
   ONNX-converted model must be re-converted to MNN, or an equivalent
   MNN model must be sourced from upstream PaddleOCR. **Mitigation:**
   upstream PaddleOCR repo ships MNN variants; if the existing custom
   dict embedding technique is lost, fall back to loading dict from
   `ppocrv5_devanagari_dict.txt` alongside the `.mnn` model.
3. **OS file-drop UX on Windows Terminal.** Drag-from-Explorer yields
   pasted text, not structured events. Path parsing must handle quotes
   and spaces. **Mitigation:** ship a thorough path regex + unit tests
   against drag payloads from wt, conhost, Alacritty, Wezterm.
4. **AGPL license propagation from `mupdf-rs`.** No change from current
   posture but should be noted in README.
5. **Crate ecosystem maturity.** Rust OCR space is thinner than Python.
   Upstream bugs will land on this project. **Mitigation:** vendoring
   `rust-paddle-ocr` already creates the patch path.

## Consequences

### Becomes easier

- Single-binary deploy: no Python DLL/alias-conflict workarounds
  (`paddleknife.bat`, `_add_nvidia_dll_paths`, NVIDIA DLL path surgery
  all disappear).
- Startup latency: no ONNX Runtime C++ warm-up, no rapidocr import cost.
- Interactive batch ergonomics: queue + live progress instead of
  terminal-locking `typer` invocations.
- Cross-platform GPU via Vulkan (Linux/Mac get GPU for the first time).
- Memory bounds: no Python gc.collect() between pages needed.

### Becomes harder

- First-run model download UX: must match HF Hub's current implicit
  caching behavior; needs a `hf-hub` crate or `reqwest`-based fetcher.
- Font handling for Devanagari searchable PDF: Rust font ecosystem
  (`ttf-parser`, `fontdue`) is less batteries-included than PyMuPDF's
  `insert_font`.
- Debugging GPU path: MNN's error messages are terser than ONNX Runtime's.
- Development velocity drops short-term: the Python rewrite cycle is
  faster than Rust compile times for the first few weeks.

### Must revisit

- **Backend lock-in to MNN.** If MNN GPU proves unreliable on Windows,
  the ONNX path via `ort` crate remains as a fallback (kept as a design
  option under a `backend-ort` Cargo feature, not implemented in v1).
- **TUI mouse protocol support.** If users report missing events in
  specific terminals (e.g., legacy conhost, tmux-in-ssh), fall back to
  keyboard-only mode with a status-bar indicator.

## Action Items

1. [ ] **Spike MNN GPU selector.** Vendor `rust-paddle-ocr`, add
       `Det::with_backend(BackendConfig::Vulkan)`, run on a 10-page PDF
       on this Windows machine. Document the MNN API surface needed.
       **This is a go/no-go gate for the rest of the plan** — if MNN
       GPU on Windows does not work cleanly, revisit backend choice
       (fall back to `ort` with DirectML) before building out the rest.
2. [ ] **Model acquisition plan.** Confirm PP-OCRv5 `.mnn` availability
       for Devanagari in upstream PaddleOCR MNN releases; document the
       re-conversion fallback path.
3. [ ] **Workspace scaffold.** `cargo new --lib paddleknife-core`;
       `paddleknife-cli`; `paddleknife-tui`. Wire workspace root.
       `git init` (the project is not currently a git repo — initialize
       before the first commit).
4. [ ] **Core pipeline in `paddleknife-core`.** Port in this order to
       preserve incremental testability: PDF load → image preprocess
       → det → rec → output writers (txt → json → md → pdf). Each step
       has a testthat-style unit test with the existing `MunaMadan_full.pdf`
       sample.
5. [ ] **CLI binary.** `clap` sub-commands matching Python exactly.
       Snapshot-test the help output against `paddleknife --help`.
6. [ ] **TUI binary.** Ratatui + crossterm, four-pane layout, mouse
       events, bracketed-paste path parser. Manual test matrix: Windows
       Terminal, conhost, Alacritty, Wezterm.
7. [ ] **Secondary scripts.** Port `surya_injector.py`,
       `surya_to_pdf.py`, `puck.py`, `remove_images.py` as sub-commands
       of the CLI (`paddleknife surya-inject …`, `paddleknife strip-images …`).
8. [ ] **Upstream PR** to `zibo-chen/rust-paddle-ocr` with the backend
       selector, once the fork is stable.
9. [ ] **Documentation.** README with install (`cargo install` path),
       migration table Python↔Rust flags, TUI keybinding cheat sheet,
       license notes (AGPL from MuPDF).

## Resolved Questions

_All open questions were answered during the 2026-04-14 review session:_

- **CLI vs TUI separation:** Resolved → single unified binary
  (`paddleknife.exe`) with mode dispatch. Double-click = TUI; args = CLI.
  Superseded the original "two independent binaries" proposal.
- **Batch mode UX:** Resolved → `paddleknife batch ./dir --tui` opts into
  the TUI pre-populated with the directory. Plain `batch` stays headless.
- **HF Hub cache path:** Resolved → share with Python at
  `~/.cache/huggingface/` during the migration period; revisit if a
  Rust-specific path becomes warranted later.

## Windows-specific behavior (for double-click-launch)

Because double-clicking a `.exe` on Windows spawns a new console window,
the TUI mode must:

- Not require any command-line args to function (all config editable in-TUI).
- Gracefully handle `stdin`/`stdout` being a fresh console (no pipe, no redir).
- Keep its own event loop even when the user isn't touching the mouse
  (so the console doesn't appear frozen).
- Exit cleanly with a visible "Press any key to close" prompt on unhandled
  error paths — otherwise double-click users see the console vanish
  before they can read the message.
- Persist the last-used settings (lang, format, GPU backend) to
  `%APPDATA%\paddleknife\config.toml` so double-click is useful without
  reconfiguring every session.

## What actually shipped

Recorded 2026-09-30, when this ADR was superseded. The decision above is
left as written. Only the workspace layout shipped as decided; the
backend, PDF, concurrency and interface choices did not. The audit
([`docs/AUDIT-2026-09-30.md`](../AUDIT-2026-09-30.md) §3) has the
plan-by-plan history.

| Decision | This ADR | Shipped |
|---|---|---|
| Workspace | Two crates, one binary | As decided: `paddleknife-core` plus the `paddleknife` binary. |
| Inference | MNN via a forked `rust-paddle-ocr` (Vulkan/OpenCL) | oar-ocr on `ort` (ONNX Runtime), with the DirectML execution provider as the default on Windows and CPU as the fallback. This is Action Item 1's fallback. oar-ocr is vendored with patches in `vendor/oar-ocr`. |
| Models | PP-OCRv5 re-converted to MNN | PP-OCRv6 tiers plus PP-OCRv5 script recognizers as ONNX, downloaded on demand (`models/README.md`). |
| PDF | `mupdf-rs` | pdfium (`pdfium-render`) rasterises pages. `lopdf` reads source PDFs, read-only, for page-image passthrough. `pdf-writer` writes the searchable PDF and its glyphless text layer. |
| Concurrency | tokio with `spawn_blocking` | No async runtime. Pages run one after another. |
| Interface | Ratatui TUI first, CLI second | CLI only (`ocr`, `models`). Run with no arguments, it prints usage. There is no TUI. |
| Secondary scripts | Port as CLI sub-commands | Not ported. The Python tool and its helper scripts were retired on 2026-09-30; git history keeps them. The Python evaluation tools in `dev/` remain. |
