# docs/

A map of this folder. Documents marked **living** are kept current with the
code. Everything else is a dated record: useful for *why* things are the way
they are, but not updated when the code moves on.

| Path | Status | What it is |
|---|---|---|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | **living** | Pipeline stages, the geometry and text-layer contracts, page-image embedding, the model registry, GPU resilience, and the invariants and tests that gate them. Start here before changing code. |
| [`OCR_RECIPES.md`](OCR_RECIPES.md) | **living** | Measured command-line settings per document type, and what does *not* help. |
| [`AUDIT-2026-09-30.md`](AUDIT-2026-09-30.md) | latest audit | Whole-project review: code health, build reproducibility, and the ranked roadmap (§8). |
| [`adr/`](adr/) | decisions | Architecture decision records. [`0001`](adr/0001-port-to-rust-tui.md) (Rust port with MNN, MuPDF and a TUI) is **superseded**: none of its technology choices shipped. [`0002`](adr/0002-handwriting-recognition-backends.md) (handwriting backends) is **proposed**. |
| [`plans/`](plans/) | historical | Implementation plans 1–4.5 and Plan 10, as written at the time. Not maintained; unticked checkboxes and code snapshots are expected. |
| [`superpowers/`](superpowers/) | historical | Design specs and a plan written with the superpowers workflow: the text layer (Plan 5), multi-script recognizers (Plan 6), slim PDFs (Plan 8), and the translation roadmap (approved, not started). Not maintained. |
| [`archive/branch-ledger.md`](archive/branch-ledger.md) | ledger | What every non-`main` branch, worktree and stash held, and what it taught. Update it whenever a branch is merged, archived or deleted. |
| [`upstream/`](upstream/) | drafts | Ready-to-file oar-ocr issues found during Plan 10. Issues 1–3 are patched locally in `vendor/oar-ocr`. |

Elsewhere:

- [`../README.md`](../README.md): build, usage and troubleshooting.
- [`../vendor/oar-ocr/PADDLEKNIFE_PATCHES.md`](../vendor/oar-ocr/PADDLEKNIFE_PATCHES.md):
  the local oar-ocr patches and how to drop them.
- [`../vendor/pdfium/README.md`](../vendor/pdfium/README.md): PDFium setup.
- [`../dev/README.md`](../dev/README.md): the Python evaluation tools.
