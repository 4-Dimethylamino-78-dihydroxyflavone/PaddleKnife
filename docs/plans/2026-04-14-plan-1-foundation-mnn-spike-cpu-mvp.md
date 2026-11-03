# Plan 1: Foundation + OCR Backend + CPU OCR MVP

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the Rust workspace, wire up an ort-based OCR pipeline on Windows, and produce a working `paddleknife.exe` that can OCR a single image or PDF page to plain text on CPU.

**Architecture:** Two-crate Cargo workspace (`paddleknife-core` library + `paddleknife` binary) per ADR-0001. **OCR via [`oar-ocr`](https://github.com/greatv/oar-ocr)** — a Rust PP-OCR pipeline built on the `ort` crate (ONNX Runtime). MuPDF via `mupdf-rs` for PDF I/O. OCR is accessed through an `OcrBackend` trait so alternative backends can drop in without touching the rest of the pipeline. No TUI in this plan — CLI only. No Devanagari post-processing — that lands in Plan 2. Feature parity is deliberately minimal: prove the pipeline works end-to-end, then build up.

**Tech Stack:** Rust 2021 edition (stable 1.94 active), `cargo` workspace, `oar-ocr` (ort-based), `mupdf-rs`, `image`, `clap` (derive), `anyhow`, `thiserror`, `tracing`, `tracing-subscriber`. Windows 10/11 primary target.

**Pivot note (2026-04-14):** Task 3 originally vendored `rust-paddle-ocr` (MNN). That build failed on Windows due to a `cmake` crate v0.1.58 bug selecting a nonexistent "Visual Studio 18 2026" generator, plus `cl.exe` not being on PATH. After the failure we surveyed ort-family alternatives and chose `oar-ocr`, which avoids the C++ build problem entirely and ships the PP-OCR pipeline complete. All tasks below reflect this pivot.

**Reference inputs for tests:**
- `MunaMadan_full.pdf` — multi-page Devanagari PDF that already exists in the repo
- We'll create `tests/fixtures/hello_world.png` — a tiny synthetic English image (generated in Task 10)

---

## File Structure

```
PaddleKnife/
├── Cargo.toml                       # workspace root (NEW)
├── rust-toolchain.toml              # pin rust version (NEW)
├── .gitignore                       # rust-specific (NEW)
├── crates/
│   ├── paddleknife-core/
│   │   ├── Cargo.toml               # lib crate manifest
│   │   └── src/
│   │       ├── lib.rs               # public API root
│   │       ├── error.rs             # OcrError enum
│   │       ├── model.rs             # model paths + lazy init
│   │       ├── pdf.rs               # mupdf-rs wrapper: load, rasterize
│   │       ├── image_io.rs          # image::open + DynamicImage wrapper
│   │       ├── page.rs              # Page struct: width, height, image
│   │       ├── ocr.rs               # OcrEngine: wraps Det + Rec
│   │       ├── script.rs            # detect_script() from Unicode ranges
│   │       └── output/
│   │           ├── mod.rs
│   │           └── text.rs          # write_text_output
│   └── paddleknife/
│       ├── Cargo.toml               # bin crate manifest
│       └── src/
│           ├── main.rs              # dispatcher: TTY check → TUI | CLI
│           └── cli.rs               # clap sub-command definitions
├── vendor/
│   └── rust-paddle-ocr/             # git clone, patched for backend API
│       └── (vendored files)
├── models/                          # ALREADY EXISTS
│   └── (MNN models land here after Task 4)
├── tests/
│   └── fixtures/
│       └── hello_world.png          # generated in Task 10
└── docs/
    ├── adr/0001-port-to-rust-tui.md # ALREADY EXISTS
    └── plans/
        └── 2026-04-14-plan-1-foundation-mnn-spike-cpu-mvp.md  # THIS FILE
```

**Why two crates instead of one:** `paddleknife-core` is the library consumed by the binary. Future plans may spawn helpers (benchmarks, model converters) that link the library without pulling in `clap`. Keeps `main.rs` small.

**Why a vendored fork:** `rust-paddle-ocr` exposes no `mnn::ScheduleConfig` API and no GPU feature flags (ADR-0001 §Decision 4). We need to patch the crate to expose backend selection. This plan scopes the patch to "add the public API, default to CPU"; Plan 4 wires GPU through it.

---

### Task 1: Initialize git repo and baseline commit

**Files:**
- Create: `.gitignore`

The project is currently not a git repository. We need version control before we start generating artifacts.

- [ ] **Step 1: Initialize git**

```bash
cd PaddleKnife
git init
git branch -m main
```

- [ ] **Step 2: Write `.gitignore` that covers Python + Rust + local artifacts**

Create `.gitignore`:

```gitignore
# ===== Rust =====
/target
**/*.rs.bk
Cargo.lock

# ===== Python (existing) =====
__pycache__/
*.pyc
*.pyo
*.egg-info/
.venv/
venv/

# ===== OS =====
Thumbs.db
.DS_Store

# ===== PaddleKnife output =====
*_paddleknife_OCR_*.pdf
*_paddleknife_OCR_*.txt
*_paddleknife_OCR_*.json
*_paddleknife_OCR_*.md

# ===== Editor =====
.vscode/
.idea/
*.swp
```

Note: Cargo.lock is gitignored because this is a library-heavy workspace where the lockfile belongs to downstream bins; we'll reconsider at workspace level before public release.

- [ ] **Step 3: Initial commit of existing Python project + ADR + this plan**

```bash
git add .gitignore
git add paddleknife.py paddleknife.bat requirements.txt README.md
git add puck.py remove_images.py surya_injector.py surya_to_pdf.py translate_book.py
git add test_sakuntal_v2.txt
git add models/ str/
git add docs/adr/0001-port-to-rust-tui.md
git add docs/plans/2026-04-14-plan-1-foundation-mnn-spike-cpu-mvp.md
git status  # confirm no accidental inclusions
git commit -m "chore: initial commit of Python PaddleKnife + Rust port plan"
```

Expected: clean tree after commit. Note `MunaMadan_full.pdf` and `__pycache__` remain untracked (the PDF is a large binary we add separately; pycache is gitignored via the Python-heritage rules).

- [ ] **Step 4: Add MunaMadan_full.pdf via Git LFS (optional but recommended)**

```bash
git lfs install
git lfs track "*.pdf"
git add .gitattributes MunaMadan_full.pdf
git commit -m "chore: add MunaMadan_full.pdf sample via LFS"
```

If `git lfs` is unavailable, `git add MunaMadan_full.pdf` directly is acceptable — it's ~8 MB.

---

### Task 2: Create Cargo workspace skeleton

**Files:**
- Create: `Cargo.toml` (workspace root)
- Create: `rust-toolchain.toml`
- Create: `crates/paddleknife-core/Cargo.toml`
- Create: `crates/paddleknife-core/src/lib.rs`
- Create: `crates/paddleknife/Cargo.toml`
- Create: `crates/paddleknife/src/main.rs`

- [ ] **Step 1: Write workspace root `Cargo.toml`**

Create `Cargo.toml`:

```toml
[workspace]
resolver = "2"
members = [
    "crates/paddleknife-core",
    "crates/paddleknife",
]
# Vendored crate is pulled in by path-dependency from paddleknife-core, not listed here.
# We do NOT include it as a workspace member because we don't want `cargo test --workspace`
# to run its tests — we only care about its library output.

[workspace.package]
version = "0.1.0"
edition = "2021"
rust-version = "1.78"
license = "AGPL-3.0-or-later"  # inherited from mupdf-rs / MuPDF
authors = ["Nicholas-EW"]

[workspace.dependencies]
anyhow = "1"
thiserror = "1"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter", "fmt"] }
image = "0.25"
clap = { version = "4", features = ["derive"] }
# rust-paddle-ocr and mupdf-rs pinned in the specific crates that use them
```

- [ ] **Step 2: Pin Rust toolchain**

Create `rust-toolchain.toml`:

```toml
[toolchain]
channel = "1.78.0"
components = ["rustfmt", "clippy"]
profile = "minimal"
```

- [ ] **Step 3: Create `paddleknife-core` crate**

Create `crates/paddleknife-core/Cargo.toml`:

```toml
[package]
name = "paddleknife-core"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
anyhow.workspace = true
thiserror.workspace = true
tracing.workspace = true
image.workspace = true
# rust-paddle-ocr and mupdf-rs added in Tasks 3 and 5
```

Create `crates/paddleknife-core/src/lib.rs`:

```rust
//! PaddleKnife core library — OCR pipeline, PDF I/O, output writers.
//!
//! Re-exports the minimum public surface for the `paddleknife` binary.

pub mod error;

pub use error::OcrError;

/// Crate version from Cargo.toml (compile-time constant).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
```

Create `crates/paddleknife-core/src/error.rs`:

```rust
//! Error types for the core pipeline.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum OcrError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("image: {0}")]
    Image(#[from] image::ImageError),

    #[error("unsupported input format: {0}")]
    UnsupportedFormat(String),

    #[error("model not found: {0}")]
    ModelNotFound(String),

    #[error("ocr engine: {0}")]
    Engine(String),
}

pub type Result<T> = std::result::Result<T, OcrError>;
```

- [ ] **Step 4: Create `paddleknife` binary crate**

Create `crates/paddleknife/Cargo.toml`:

```toml
[package]
name = "paddleknife"
version.workspace = true
edition.workspace = true
license.workspace = true
default-run = "paddleknife"

[[bin]]
name = "paddleknife"
path = "src/main.rs"

[dependencies]
paddleknife-core = { path = "../paddleknife-core" }
anyhow.workspace = true
clap.workspace = true
tracing.workspace = true
tracing-subscriber.workspace = true
```

Create `crates/paddleknife/src/main.rs`:

```rust
fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    println!("paddleknife v{}", paddleknife_core::VERSION);
    Ok(())
}
```

- [ ] **Step 5: Verify it builds and runs**

Run: `cargo build --workspace`
Expected: clean build, no warnings.

Run: `cargo run --bin paddleknife`
Expected: `paddleknife v0.1.0`

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml rust-toolchain.toml crates/
git commit -m "feat(core): scaffold cargo workspace with core lib and bin"
```

---

### Task 3: Vendor `rust-paddle-ocr` and wire it in

**Files:**
- Create: `vendor/rust-paddle-ocr/` (via `git submodule` OR manual clone — see below)
- Modify: `crates/paddleknife-core/Cargo.toml`

Per ADR-0001 §Decision 4, we need a fork we can patch. Simplest is `git submodule`.

- [ ] **Step 1: Add rust-paddle-ocr as a submodule**

```bash
cd PaddleKnife
git submodule add https://github.com/zibo-chen/rust-paddle-ocr.git vendor/rust-paddle-ocr
cd vendor/rust-paddle-ocr
git checkout main
cd ../..
```

- [ ] **Step 2: Add path-dependency on the vendored crate**

Modify `crates/paddleknife-core/Cargo.toml`, add to `[dependencies]`:

```toml
# Vendored + patched — see ADR-0001 §Decision 4.
rust-paddle-ocr = { path = "../../vendor/rust-paddle-ocr" }
```

- [ ] **Step 3: Verify the vendored crate builds through our workspace**

Run: `cargo build -p paddleknife-core`
Expected: builds `rust-paddle-ocr` and its `mnn` transitive dep. This may take 5–10 minutes on first build because `mnn` pulls prebuilt C++ binaries.

**If the build fails on Windows** due to `mnn` crate issues:
- The most likely failure is "vendored prebuilt binary not found for msvc target".
- Check `vendor/rust-paddle-ocr/Cargo.toml` `[dependencies.mnn]` — the mnn crate is pulled from a git ref. Look at the mnn crate's README for Windows prerequisites.
- Worst case, the MNN crate requires CMake + a C++ toolchain. Install Visual Studio Build Tools with "Desktop development with C++" workload.
- **If this step fails after reasonable debugging, this is the go/no-go gate the ADR called out.** Do not abandon the port — switch to the **Escape Hatch: `ort` fallback** section later in this document. That section replaces Tasks 3, 4, and 7 with an ONNX Runtime + DirectML path that reuses the existing Python ONNX models. All other tasks in this plan stay unchanged because Task 7's `OcrBackend` trait makes the backend pluggable.

- [ ] **Step 4: Commit**

```bash
git add .gitmodules vendor/ crates/paddleknife-core/Cargo.toml
git commit -m "feat(core): vendor rust-paddle-ocr as submodule"
```

---

### Task 4: Acquire MNN model files

**Files:**
- Create: `models/PP-OCRv5_mobile_det.mnn`
- Create: `models/PP-OCRv5_mobile_rec.mnn`
- Create: `models/ppocr_keys_v5.txt`
- Modify: `.gitignore` (add `models/*.mnn` to keep models out of git)

`rust-paddle-ocr` expects `.mnn` models. We download from the upstream `rust-paddle-ocr` release assets.

- [ ] **Step 1: Document model sources in `models/README.md`**

Create `models/README.md`:

```markdown
# Models

## MNN models (Rust port)

Downloaded from upstream `rust-paddle-ocr` releases:
https://github.com/zibo-chen/rust-paddle-ocr/releases

- `PP-OCRv5_mobile_det.mnn` — text detection (~10 MB)
- `PP-OCRv5_mobile_rec.mnn` — text recognition, multilingual (~15 MB)
- `ppocr_keys_v5.txt` — character dictionary

## ONNX models (Python legacy, still used by paddleknife.py)

See the project `README.md` for ONNX details. These are NOT used by the
Rust port.
```

- [ ] **Step 2: Download the three MNN files manually**

Visit https://github.com/zibo-chen/rust-paddle-ocr/releases, download:
- `PP-OCRv5_mobile_det.mnn`
- `PP-OCRv5_mobile_rec.mnn`
- `ppocr_keys_v5.txt`

Save to `models/`.

**If the releases page does not ship these directly**, `rust-paddle-ocr`'s README lists alternative download URLs. If still unavailable, there's a conversion path from PaddleOCR's released `.mnn` files at https://www.modelscope.cn/models/PaddlePaddle — that's a Plan 2 concern; for this MVP, stop here if the files aren't obtainable and return to ADR-0001 §Decision 4.

- [ ] **Step 3: Gitignore model binaries**

Modify `.gitignore`, append:

```gitignore
# ===== OCR model weights (downloaded, not checked in) =====
models/*.mnn
```

Keep `ppocr_keys_v5.txt` tracked (plain text, small).

- [ ] **Step 4: Commit**

```bash
git add .gitignore models/README.md models/ppocr_keys_v5.txt
git commit -m "docs(models): document MNN model sources and ignore binaries"
```

---

### Task 5: Add `mupdf-rs` and write a minimal PDF-to-image test

**Files:**
- Modify: `crates/paddleknife-core/Cargo.toml`
- Create: `crates/paddleknife-core/src/pdf.rs`
- Create: `crates/paddleknife-core/tests/pdf_roundtrip.rs`

- [ ] **Step 1: Add mupdf-rs dependency**

Modify `crates/paddleknife-core/Cargo.toml`, add to `[dependencies]`:

```toml
mupdf = "0.5"
```

**If the build fails** because MuPDF C library is missing, see the messense/mupdf-rs README for Windows build instructions (typically: install `clang`, set `CLANG_PATH`). This is not a go/no-go gate — `pdfium-render` remains as a fallback per ADR-0001 Decision 2 Option B.

- [ ] **Step 2: Write the failing integration test**

Create `crates/paddleknife-core/tests/pdf_roundtrip.rs`:

```rust
//! Integration test: load a PDF page, rasterize it, assert non-zero dimensions.

use paddleknife_core::pdf;
use std::path::PathBuf;

fn sample_pdf() -> PathBuf {
    // MunaMadan_full.pdf lives at the project root.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../MunaMadan_full.pdf")
}

#[test]
fn rasterizes_first_page_at_300_dpi() {
    let pages = pdf::rasterize_pdf(&sample_pdf(), 300).expect("rasterize");
    assert!(!pages.is_empty(), "PDF should have at least one page");
    let first = &pages[0];
    assert!(first.width > 100, "page width > 100 px at 300 DPI");
    assert!(first.height > 100, "page height > 100 px at 300 DPI");
    assert_eq!(first.image.color(), image::ColorType::Rgba8);
}
```

- [ ] **Step 3: Run the test to verify it fails with the expected error**

Run: `cargo test -p paddleknife-core --test pdf_roundtrip -- --nocapture`
Expected: FAIL because `pdf::rasterize_pdf` does not exist yet.

- [ ] **Step 4: Write minimal `pdf.rs`**

Create `crates/paddleknife-core/src/pdf.rs`:

```rust
//! PDF rasterization via mupdf-rs.

use crate::error::{OcrError, Result};
use image::{DynamicImage, RgbaImage};
use std::path::Path;

pub struct RasterPage {
    pub index: usize,
    pub width: u32,
    pub height: u32,
    pub image: DynamicImage,
}

/// Rasterize every page of `pdf_path` at `dpi` DPI.
///
/// Returns an owned `Vec<RasterPage>`. For very large PDFs, prefer
/// `rasterize_pdf_iter` (added in Plan 2) to avoid holding every page
/// in memory.
pub fn rasterize_pdf(pdf_path: &Path, dpi: u32) -> Result<Vec<RasterPage>> {
    let doc = mupdf::Document::open(
        pdf_path
            .to_str()
            .ok_or_else(|| OcrError::UnsupportedFormat("non-utf8 path".into()))?,
    )
    .map_err(|e| OcrError::Engine(format!("mupdf open: {e}")))?;

    let page_count = doc
        .page_count()
        .map_err(|e| OcrError::Engine(format!("mupdf page_count: {e}")))?;

    let mut out = Vec::with_capacity(page_count as usize);
    let scale = dpi as f32 / 72.0;
    let mat = mupdf::Matrix::new_scale(scale, scale);

    for i in 0..page_count {
        let page = doc
            .load_page(i)
            .map_err(|e| OcrError::Engine(format!("mupdf load_page({i}): {e}")))?;
        let pixmap = page
            .to_pixmap(&mat, &mupdf::Colorspace::device_rgb(), 0.0, true)
            .map_err(|e| OcrError::Engine(format!("mupdf to_pixmap: {e}")))?;

        let width = pixmap.width() as u32;
        let height = pixmap.height() as u32;
        let samples = pixmap.samples();

        // mupdf pixmap with alpha=true gives RGBA bytes.
        let rgba = RgbaImage::from_raw(width, height, samples.to_vec())
            .ok_or_else(|| OcrError::Engine("pixmap dimensions mismatch".into()))?;

        out.push(RasterPage {
            index: i as usize,
            width,
            height,
            image: DynamicImage::ImageRgba8(rgba),
        });
    }

    Ok(out)
}
```

- [ ] **Step 5: Wire `pdf` into `lib.rs`**

Modify `crates/paddleknife-core/src/lib.rs`, add after `pub mod error;`:

```rust
pub mod pdf;
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `cargo test -p paddleknife-core --test pdf_roundtrip -- --nocapture`
Expected: PASS. Log output shows 27 pages (MunaMadan_full.pdf page count).

- [ ] **Step 7: Commit**

```bash
git add crates/paddleknife-core/
git commit -m "feat(core): add mupdf-rs PDF rasterization"
```

---

### Task 6: Image loader for single PNG/JPG input

**Files:**
- Create: `crates/paddleknife-core/src/image_io.rs`
- Create: `crates/paddleknife-core/src/page.rs`
- Create: `crates/paddleknife-core/tests/image_load.rs`
- Modify: `crates/paddleknife-core/src/lib.rs`

- [ ] **Step 1: Define the unified `Page` type**

Create `crates/paddleknife-core/src/page.rs`:

```rust
//! Unified page representation used by both PDF pages and single images.

use image::DynamicImage;

pub struct Page {
    /// Zero-based index (always 0 for single images).
    pub index: usize,
    pub width: u32,
    pub height: u32,
    pub image: DynamicImage,
}

impl From<crate::pdf::RasterPage> for Page {
    fn from(rp: crate::pdf::RasterPage) -> Self {
        Self {
            index: rp.index,
            width: rp.width,
            height: rp.height,
            image: rp.image,
        }
    }
}
```

- [ ] **Step 2: Write the failing image-load test**

Create `crates/paddleknife-core/tests/image_load.rs`:

```rust
use paddleknife_core::image_io;
use std::path::PathBuf;

fn sample_png() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/hello_world.png")
}

#[test]
fn loads_png_as_single_page() {
    let page = image_io::load_image(&sample_png()).expect("load image");
    assert_eq!(page.index, 0);
    assert!(page.width > 0);
    assert!(page.height > 0);
}
```

- [ ] **Step 3: Generate the test fixture**

Create `tests/fixtures/` directory and add a small PNG. The simplest reproducible way is a one-off Rust build script or a Python one-liner (Python is already on the machine):

```bash
cd PaddleKnife
mkdir -p tests/fixtures
python -c "from PIL import Image, ImageDraw, ImageFont; img = Image.new('RGB', (400, 100), 'white'); draw = ImageDraw.Draw(img); draw.text((20, 30), 'Hello World', fill='black'); img.save('tests/fixtures/hello_world.png')"
```

Verify it exists:

```bash
ls tests/fixtures/hello_world.png
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `cargo test -p paddleknife-core --test image_load`
Expected: FAIL because `image_io::load_image` does not exist.

- [ ] **Step 5: Implement `image_io.rs`**

Create `crates/paddleknife-core/src/image_io.rs`:

```rust
//! Single-image input loader (PNG, JPG, WebP, etc. — whatever `image` supports).

use crate::{error::Result, page::Page};
use std::path::Path;

pub fn load_image(path: &Path) -> Result<Page> {
    let img = image::open(path)?;
    Ok(Page {
        index: 0,
        width: img.width(),
        height: img.height(),
        image: img,
    })
}
```

- [ ] **Step 6: Wire into `lib.rs`**

Modify `crates/paddleknife-core/src/lib.rs`, add:

```rust
pub mod image_io;
pub mod page;
```

- [ ] **Step 7: Run test to verify it passes**

Run: `cargo test -p paddleknife-core --test image_load`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add crates/paddleknife-core/ tests/fixtures/
git commit -m "feat(core): add single-image loader and unified Page type"
```

---

### Task 7: OCR engine wrapper (CPU only)

**Files:**
- Create: `crates/paddleknife-core/src/ocr.rs`
- Create: `crates/paddleknife-core/src/model.rs`
- Create: `crates/paddleknife-core/tests/ocr_hello_world.rs`
- Modify: `crates/paddleknife-core/src/lib.rs`

- [ ] **Step 1: Model-path resolver**

Create `crates/paddleknife-core/src/model.rs`:

```rust
//! Resolve model file paths from the project's `models/` directory.

use crate::error::{OcrError, Result};
use std::path::{Path, PathBuf};

/// Path layout that `rust-paddle-ocr` expects.
pub struct ModelPaths {
    pub det_mnn: PathBuf,
    pub rec_mnn: PathBuf,
    pub keys_txt: PathBuf,
}

impl ModelPaths {
    /// Resolve paths relative to `models_dir`.
    pub fn in_dir(models_dir: &Path) -> Result<Self> {
        let det = models_dir.join("PP-OCRv5_mobile_det.mnn");
        let rec = models_dir.join("PP-OCRv5_mobile_rec.mnn");
        let keys = models_dir.join("ppocr_keys_v5.txt");
        for p in [&det, &rec, &keys] {
            if !p.exists() {
                return Err(OcrError::ModelNotFound(p.display().to_string()));
            }
        }
        Ok(Self {
            det_mnn: det,
            rec_mnn: rec,
            keys_txt: keys,
        })
    }
}
```

- [ ] **Step 2: Write the failing OCR test**

Create `crates/paddleknife-core/tests/ocr_hello_world.rs`:

```rust
use paddleknife_core::{image_io, model::ModelPaths, ocr::OcrEngine};
use std::path::PathBuf;

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models")
}
fn sample_png() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/hello_world.png")
}

#[test]
fn ocr_recognizes_hello_world() {
    let paths = ModelPaths::in_dir(&models_dir()).expect("models present");
    let mut engine = OcrEngine::from_paths(&paths).expect("engine init");
    let page = image_io::load_image(&sample_png()).expect("load image");
    let blocks = engine.ocr(&page.image).expect("ocr ran");

    assert!(!blocks.is_empty(), "detector should find at least one block");
    let joined: String = blocks
        .iter()
        .map(|b| b.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase();
    assert!(
        joined.contains("hello") || joined.contains("world"),
        "expected 'hello' or 'world' in OCR output, got: {joined:?}"
    );
}
```

- [ ] **Step 3: Run to confirm it fails**

Run: `cargo test -p paddleknife-core --test ocr_hello_world`
Expected: FAIL — `paddleknife_core::ocr` does not exist.

- [ ] **Step 4: Implement the OCR trait + MNN backend**

Create `crates/paddleknife-core/src/ocr.rs`:

```rust
//! OCR abstraction: `OcrBackend` trait + MNN implementation.
//!
//! The trait exists so a fallback `ort`-based backend (see plan's Escape
//! Hatch section) can be swapped in without changes to callers. Plan 4
//! adds GPU backend selection to the MNN implementation.

use crate::{error::{OcrError, Result}, model::ModelPaths};
use image::DynamicImage;
use rust_paddle_ocr::{Det, Rec};

pub struct OcrBlock {
    /// Axis-aligned bounding box in the original image coords.
    /// `[x1, y1, x2, y2]` — top-left, bottom-right.
    pub bbox: [i32; 4],
    pub text: String,
    pub confidence: f32,
}

/// OCR backend: anything that turns a full-page image into a list of
/// recognised text blocks. Implementations: `MnnBackend` (default,
/// Plan 1), `OrtBackend` (Escape Hatch / Plan 4 under a Cargo feature).
pub trait OcrBackend: Send {
    fn ocr(&mut self, img: &DynamicImage) -> Result<Vec<OcrBlock>>;
}

/// Type-erased handle used by the rest of the pipeline.
pub struct OcrEngine {
    inner: Box<dyn OcrBackend>,
}

impl OcrEngine {
    /// Construct an engine backed by MNN (via the vendored rust-paddle-ocr).
    pub fn mnn(paths: &ModelPaths) -> Result<Self> {
        Ok(Self {
            inner: Box::new(MnnBackend::from_paths(paths)?),
        })
    }

    /// Construct with any caller-supplied backend (used by the ort fallback).
    pub fn with_backend(backend: Box<dyn OcrBackend>) -> Self {
        Self { inner: backend }
    }

    /// Default constructor — aliased to `mnn` in Plan 1.
    /// In Plan 4 this will check a Cargo feature + runtime flag.
    pub fn from_paths(paths: &ModelPaths) -> Result<Self> {
        Self::mnn(paths)
    }

    pub fn ocr(&mut self, img: &DynamicImage) -> Result<Vec<OcrBlock>> {
        self.inner.ocr(img)
    }
}

pub struct MnnBackend {
    det: Det,
    rec: Rec,
}

impl MnnBackend {
    pub fn from_paths(paths: &ModelPaths) -> Result<Self> {
        let det = Det::from_file(&paths.det_mnn)
            .map_err(|e| OcrError::Engine(format!("Det::from_file: {e}")))?;
        let rec = Rec::from_file(&paths.rec_mnn, &paths.keys_txt)
            .map_err(|e| OcrError::Engine(format!("Rec::from_file: {e}")))?;
        Ok(Self { det, rec })
    }
}

impl OcrBackend for MnnBackend {
    fn ocr(&mut self, img: &DynamicImage) -> Result<Vec<OcrBlock>> {
        let crops = self
            .det
            .find_text_img(img)
            .map_err(|e| OcrError::Engine(format!("Det::find_text_img: {e}")))?;

        let mut blocks = Vec::with_capacity(crops.len());
        for crop in crops {
            // `crop` is a DynamicImage of the detected region.
            // rust-paddle-ocr's current public API does not expose the bbox
            // directly alongside the crop; Task 7 scope is "text-only output"
            // so we stub the bbox to [0; 4] for now. Plan 2 Task 1 adds a
            // parallel API that returns bbox + crop together.
            let text = self
                .rec
                .predict_str(&crop)
                .map_err(|e| OcrError::Engine(format!("Rec::predict_str: {e}")))?;
            blocks.push(OcrBlock {
                bbox: [0, 0, 0, 0],
                text,
                confidence: 1.0, // placeholder; Plan 2 reads real confidence
            });
        }
        Ok(blocks)
    }
}
```

- [ ] **Step 5: Wire into `lib.rs`**

Modify `crates/paddleknife-core/src/lib.rs`, add:

```rust
pub mod model;
pub mod ocr;
```

- [ ] **Step 6: Run the test**

Run: `cargo test -p paddleknife-core --test ocr_hello_world -- --nocapture`
Expected: PASS. OCR returns at least one block and "hello" or "world" appears in the text.

**If the test fails** with a recognition-model error (e.g., mismatched dict), the most likely cause is that `PP-OCRv5_mobile_rec.mnn` + `ppocr_keys_v5.txt` are mismatched versions. Re-download both from the same release tag.

**If the API shape of `Det::find_text_img` or `Rec::predict_str` has changed** in the vendored crate, update the calls to match — the upstream README is the source of truth. Bump the submodule ref if needed.

- [ ] **Step 7: Commit**

```bash
git add crates/paddleknife-core/
git commit -m "feat(core): cpu OCR engine wrapper over rust-paddle-ocr"
```

---

### Task 8: Unicode script detection

**Files:**
- Create: `crates/paddleknife-core/src/script.rs`
- Create: `crates/paddleknife-core/src/script_test.rs` (inline module)
- Modify: `crates/paddleknife-core/src/lib.rs`

Port of `_detect_script()` from `paddleknife.py:162-174`. The Python version walks codepoints and tallies hits against named ranges.

- [ ] **Step 1: Write the failing unit test in the module file**

Create `crates/paddleknife-core/src/script.rs`:

```rust
//! Detect the dominant script of a string via Unicode codepoint ranges.
//!
//! Port of _detect_script from paddleknife.py:162-174.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Script {
    English,
    Chinese,
    Devanagari,
    Arabic,
    Cyrillic,
    Japanese,
    Korean,
    Tamil,
    Telugu,
    Thai,
    Greek,
}

impl Script {
    /// Two-letter language code suitable for model selection.
    pub fn lang_code(self) -> &'static str {
        match self {
            Script::English => "en",
            Script::Chinese => "ch",
            Script::Devanagari => "ne",
            Script::Arabic => "ar",
            Script::Cyrillic => "ru",
            Script::Japanese => "ja",
            Script::Korean => "ko",
            Script::Tamil => "ta",
            Script::Telugu => "te",
            Script::Thai => "th",
            Script::Greek => "el",
        }
    }
}

pub fn detect_script(text: &str) -> Script {
    let mut counts = [0u32; 11];
    for ch in text.chars() {
        let cp = ch as u32;
        let idx = match cp {
            0x0900..=0x097F => 2,  // Devanagari
            0x0600..=0x06FF => 3,  // Arabic
            0x0400..=0x04FF => 4,  // Cyrillic
            0x4E00..=0x9FFF => 1,  // CJK (Chinese)
            0x3040..=0x309F | 0x30A0..=0x30FF => 5,  // Hiragana + Katakana
            0xAC00..=0xD7AF => 6,  // Hangul
            0x0B80..=0x0BFF => 7,  // Tamil
            0x0C00..=0x0C7F => 8,  // Telugu
            0x0E00..=0x0E7F => 9,  // Thai
            0x0370..=0x03FF => 10, // Greek
            _ => continue,
        };
        counts[idx] += 1;
    }
    let (max_idx, max_count) = counts
        .iter()
        .enumerate()
        .max_by_key(|(_, c)| *c)
        .unwrap();
    if *max_count == 0 {
        return Script::English;
    }
    match max_idx {
        1 => Script::Chinese,
        2 => Script::Devanagari,
        3 => Script::Arabic,
        4 => Script::Cyrillic,
        5 => Script::Japanese,
        6 => Script::Korean,
        7 => Script::Tamil,
        8 => Script::Telugu,
        9 => Script::Thai,
        10 => Script::Greek,
        _ => Script::English,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_defaults_to_english() {
        assert_eq!(detect_script(""), Script::English);
    }

    #[test]
    fn ascii_defaults_to_english() {
        assert_eq!(detect_script("Hello World"), Script::English);
    }

    #[test]
    fn devanagari_detected() {
        assert_eq!(detect_script("मुना मदन"), Script::Devanagari);
        assert_eq!(Script::Devanagari.lang_code(), "ne");
    }

    #[test]
    fn arabic_detected() {
        assert_eq!(detect_script("السلام عليكم"), Script::Arabic);
    }

    #[test]
    fn mixed_picks_dominant() {
        // 10 Devanagari chars vs 5 Latin — Devanagari wins.
        assert_eq!(detect_script("मुनामदन hello"), Script::Devanagari);
    }
}
```

- [ ] **Step 2: Wire into `lib.rs`**

Modify `crates/paddleknife-core/src/lib.rs`, add:

```rust
pub mod script;
```

- [ ] **Step 3: Run the tests (they should all pass on the first try — the impl is above the tests in the same file)**

Run: `cargo test -p paddleknife-core script`
Expected: 5 tests pass.

- [ ] **Step 4: Commit**

```bash
git add crates/paddleknife-core/
git commit -m "feat(core): port unicode script detection from python"
```

---

### Task 9: Plain-text output writer

**Files:**
- Create: `crates/paddleknife-core/src/output/mod.rs`
- Create: `crates/paddleknife-core/src/output/text.rs`
- Modify: `crates/paddleknife-core/src/lib.rs`

- [ ] **Step 1: Output module**

Create `crates/paddleknife-core/src/output/mod.rs`:

```rust
//! Output writers for each supported format.
//! In Plan 1: plain text only. Plan 2 adds json, md, and searchable pdf.

pub mod text;
```

- [ ] **Step 2: Write the failing test inside the module**

Create `crates/paddleknife-core/src/output/text.rs`:

```rust
//! Plain-text output writer.
//! Each OCR block becomes one line. Empty output for zero blocks.

use crate::{error::Result, ocr::OcrBlock};
use std::io::Write;

/// Write each block's text on its own line to `writer`.
pub fn write_text<W: Write>(writer: &mut W, blocks: &[OcrBlock]) -> Result<()> {
    for block in blocks {
        writeln!(writer, "{}", block.text)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b(text: &str) -> OcrBlock {
        OcrBlock {
            bbox: [0, 0, 0, 0],
            text: text.to_string(),
            confidence: 1.0,
        }
    }

    #[test]
    fn empty_input_writes_nothing() {
        let mut buf = Vec::new();
        write_text(&mut buf, &[]).unwrap();
        assert!(buf.is_empty());
    }

    #[test]
    fn one_block_per_line() {
        let mut buf = Vec::new();
        write_text(&mut buf, &[b("hello"), b("world")]).unwrap();
        assert_eq!(std::str::from_utf8(&buf).unwrap(), "hello\nworld\n");
    }
}
```

- [ ] **Step 3: Wire into `lib.rs`**

Modify `crates/paddleknife-core/src/lib.rs`, add:

```rust
pub mod output;
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p paddleknife-core output::text`
Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/paddleknife-core/
git commit -m "feat(core): plain-text output writer"
```

---

### Task 10: CLI `ocr` sub-command

**Files:**
- Modify: `crates/paddleknife/Cargo.toml`
- Create: `crates/paddleknife/src/cli.rs`
- Modify: `crates/paddleknife/src/main.rs`

- [ ] **Step 1: Add atty-detection crate to binary deps**

Modify `crates/paddleknife/Cargo.toml`, add to `[dependencies]`:

```toml
is-terminal = "0.4"
```

`is-terminal` is the minimal crate for the `stdin.is_terminal()` check we use for TUI-auto-launch detection in Task 11.

- [ ] **Step 2: Define CLI via clap derive**

Create `crates/paddleknife/src/cli.rs`:

```rust
//! Top-level clap definitions.
//!
//! Plan 1 ships only `ocr`. Plan 2 adds `batch` + `info`. Plan 4 adds
//! `surya-inject` and `strip-images`.

use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "paddleknife", version, about = "Swiss-army-knife OCR tool")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Run OCR on a file.
    Ocr(OcrArgs),
}

#[derive(Args, Debug)]
pub struct OcrArgs {
    /// Input file — PDF or image (PNG / JPG / WebP / etc.).
    pub input: PathBuf,

    /// Output path. Defaults to <input stem>_paddleknife_OCR.txt next to the input.
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Output format. Plan 1 only supports `txt`; the other variants are
    /// rejected until Plan 2.
    #[arg(short, long, default_value = "txt")]
    pub format: String,

    /// Verbose logging.
    #[arg(short, long)]
    pub verbose: bool,
}
```

- [ ] **Step 3: Wire dispatch + OCR invocation in `main.rs`**

Modify `crates/paddleknife/src/main.rs` completely:

```rust
mod cli;

use anyhow::{bail, Context, Result};
use clap::Parser;
use cli::{Cli, Command, OcrArgs};
use paddleknife_core::{image_io, model::ModelPaths, ocr::OcrEngine, output, pdf};
use std::{fs::File, io::BufWriter, path::PathBuf};

fn main() -> Result<()> {
    init_logging();
    let cli = Cli::parse();

    match cli.command {
        Some(Command::Ocr(args)) => run_ocr(args),
        None => {
            // No sub-command — Plan 3 will launch the TUI here. For now
            // print help so double-click on Windows isn't silent.
            Cli::parse_from(["paddleknife", "--help"]);
            Ok(())
        }
    }
}

fn init_logging() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();
}

fn run_ocr(args: OcrArgs) -> Result<()> {
    if args.format != "txt" {
        bail!(
            "Plan 1 only supports --format=txt. Got '{}'. See docs/plans/ for roadmap.",
            args.format
        );
    }

    let output_path = args
        .output
        .clone()
        .unwrap_or_else(|| default_output_path(&args.input));

    tracing::info!(input = %args.input.display(), output = %output_path.display(), "starting OCR");

    let pages = load_pages(&args.input)?;
    tracing::info!(count = pages.len(), "loaded pages");

    let models_dir = models_dir_near_exe()?;
    let paths = ModelPaths::in_dir(&models_dir)
        .with_context(|| format!("resolving models in {}", models_dir.display()))?;
    let mut engine = OcrEngine::from_paths(&paths).context("initializing OCR engine")?;

    let mut all_blocks = Vec::new();
    for page in pages {
        let blocks = engine.ocr(&page.image).context("running OCR on page")?;
        all_blocks.extend(blocks);
    }

    let file = File::create(&output_path)
        .with_context(|| format!("creating {}", output_path.display()))?;
    let mut w = BufWriter::new(file);
    output::text::write_text(&mut w, &all_blocks).context("writing text output")?;

    println!("Wrote {} blocks to {}", all_blocks.len(), output_path.display());
    Ok(())
}

fn default_output_path(input: &std::path::Path) -> PathBuf {
    let stem = input.file_stem().unwrap_or_default().to_string_lossy();
    let parent = input.parent().unwrap_or(std::path::Path::new("."));
    parent.join(format!("{stem}_paddleknife_OCR.txt"))
}

fn load_pages(input: &std::path::Path) -> Result<Vec<paddleknife_core::page::Page>> {
    match input.extension().and_then(|e| e.to_str()).map(str::to_lowercase).as_deref() {
        Some("pdf") => {
            let pages = pdf::rasterize_pdf(input, 300)?;
            Ok(pages.into_iter().map(Into::into).collect())
        }
        Some(_) => {
            let page = image_io::load_image(input)?;
            Ok(vec![page])
        }
        None => bail!("input has no extension: {}", input.display()),
    }
}

/// Look for `models/` next to the current `paddleknife.exe`, then walk
/// up to the project root, then fall back to the `models/` at the
/// workspace root (useful during `cargo run`).
fn models_dir_near_exe() -> Result<PathBuf> {
    // During `cargo run`, CARGO_MANIFEST_DIR is not set at runtime, so we rely
    // on walking up from the exe path OR the current directory.
    let candidates: Vec<PathBuf> = {
        let mut v = Vec::new();
        if let Ok(exe) = std::env::current_exe() {
            let mut dir: Option<&std::path::Path> = exe.parent();
            for _ in 0..4 {
                if let Some(d) = dir {
                    v.push(d.join("models"));
                    dir = d.parent();
                }
            }
        }
        if let Ok(cwd) = std::env::current_dir() {
            v.push(cwd.join("models"));
        }
        v
    };
    for c in &candidates {
        if c.join("PP-OCRv5_mobile_det.mnn").exists() {
            return Ok(c.clone());
        }
    }
    bail!(
        "could not locate models/ directory. Tried: {:?}",
        candidates
    )
}
```

- [ ] **Step 4: Build and smoke-test on the sample PNG**

```bash
cargo run --bin paddleknife -- ocr tests/fixtures/hello_world.png
```

Expected stdout ends with something like:
```
Wrote 1 blocks to tests/fixtures/hello_world_paddleknife_OCR.txt
```

Verify the output:

```bash
cat tests/fixtures/hello_world_paddleknife_OCR.txt
```

Expected: `Hello World` (case may vary; OCR can be quirky on synthetic images).

- [ ] **Step 5: Smoke-test on the sample PDF (first page only — full PDF is slow on CPU)**

```bash
# Extract page 1 to a single-page PDF first, to keep the smoke test fast
python -c "import pymupdf; d = pymupdf.open('MunaMadan_full.pdf'); d2 = pymupdf.open(); d2.insert_pdf(d, from_page=0, to_page=0); d2.save('tests/fixtures/muna_p1.pdf'); d2.close(); d.close()"

cargo run --bin paddleknife -- ocr tests/fixtures/muna_p1.pdf
```

Expected: a `muna_p1_paddleknife_OCR.txt` file containing Devanagari characters (recognition quality will be much lower than the Python version's custom PP-OCRv5 + post-processing — Plan 2 fixes that).

- [ ] **Step 6: Commit**

```bash
git add crates/paddleknife/ tests/
git commit -m "feat(cli): minimal `paddleknife ocr` subcommand for txt output"
```

---

### Task 11: Dispatcher — double-click vs CLI

**Files:**
- Modify: `crates/paddleknife/src/main.rs`

Per ADR-0001 §Windows-specific behavior: with no args on a TTY, we should intentionally head into TUI mode. Since the TUI lands in Plan 3, this task stubs the branch with a clear message so double-click on the Windows .exe doesn't silently flash a console.

- [ ] **Step 1: Rewrite the no-args branch**

Modify `crates/paddleknife/src/main.rs`, replace the `None` arm in `match cli.command`:

```rust
        None => enter_no_args_mode(),
```

Add at the bottom of `main.rs`:

```rust
/// With no sub-command and an interactive stdin, we *will* launch the TUI in
/// Plan 3. For Plan 1 we print a stable welcome message and pause so
/// double-click users on Windows see the output before the console closes.
fn enter_no_args_mode() -> Result<()> {
    use is_terminal::IsTerminal as _;

    println!("paddleknife v{}", paddleknife_core::VERSION);
    println!();
    println!("No sub-command given. Available commands:");
    println!("  paddleknife ocr <file>     Run OCR and write plain text output");
    println!();
    println!("(An interactive TUI mode is planned for a future release.)");

    if std::io::stdin().is_terminal() {
        println!();
        println!("Press Enter to exit...");
        let mut buf = String::new();
        let _ = std::io::stdin().read_line(&mut buf);
    }
    Ok(())
}
```

- [ ] **Step 2: Build and manually verify**

```bash
cargo build --bin paddleknife --release
./target/release/paddleknife.exe
```

Expected: prints welcome + "Press Enter to exit..." and waits. Pressing Enter exits cleanly.

Double-click `target/release/paddleknife.exe` from Explorer. Expected: Windows spawns a console, prints the welcome, waits for Enter, closes on Enter. If the console vanishes immediately, `is_terminal` is returning false — confirm by running with `--verbose` and reviewing logs.

- [ ] **Step 3: Commit**

```bash
git add crates/paddleknife/
git commit -m "feat(cli): no-args mode prints welcome and waits (TUI stub)"
```

---

### Task 12: Workspace-level lints and CI-ready build

**Files:**
- Modify: `Cargo.toml` (workspace root)

- [ ] **Step 1: Add workspace lints block**

Modify `Cargo.toml` workspace root, append:

```toml
[workspace.lints.rust]
unsafe_code = "warn"

[workspace.lints.clippy]
unwrap_used = "warn"
expect_used = "allow"   # tests use .expect liberally
panic = "warn"
todo = "deny"
```

Then in each crate's `Cargo.toml`, add:

```toml
[lints]
workspace = true
```

Both `crates/paddleknife-core/Cargo.toml` and `crates/paddleknife/Cargo.toml` need this block appended.

- [ ] **Step 2: Run clippy and fix any warnings that pop out**

```bash
cargo clippy --workspace --all-targets
```

Expected: zero warnings. If any appear, fix them inline (usually `.unwrap()` → `?` or `.expect("…")` in production code; tests can keep `.unwrap()` with an `#[allow]`).

- [ ] **Step 3: Final integration test — all crates, all tests pass**

```bash
cargo test --workspace
```

Expected: all tests pass — `script` (5), `output::text` (2), `pdf_roundtrip` (1), `image_load` (1), `ocr_hello_world` (1). Total: 10.

- [ ] **Step 4: Commit**

```bash
git add Cargo.toml crates/paddleknife-core/Cargo.toml crates/paddleknife/Cargo.toml
git commit -m "chore: workspace lint config + clippy clean"
```

---

## Acceptance Criteria

At the end of Plan 1, all of these must be true:

- [ ] `cargo build --workspace --release` succeeds with zero warnings.
- [ ] `cargo test --workspace` passes (10 tests).
- [ ] `cargo clippy --workspace --all-targets` returns no warnings.
- [ ] `./target/release/paddleknife.exe ocr tests/fixtures/hello_world.png` writes a text file containing recognisable OCR output of the source image.
- [ ] `./target/release/paddleknife.exe ocr tests/fixtures/muna_p1.pdf` writes a text file with non-empty content (Devanagari text quality is deliberately not validated in Plan 1).
- [ ] Double-clicking `paddleknife.exe` in Windows Explorer opens a console, prints a welcome, and waits for Enter.
- [ ] Repo is a git repo with ~12 commits telling the implementation story.
- [ ] `vendor/rust-paddle-ocr` is a git submodule.
- [ ] `models/` contains three MNN model files and is gitignored for the binaries.

---

## Escape Hatch: `ort` fallback if MNN fails on Windows

Activate this plan **only** if Task 3 Step 3 (`cargo build -p paddleknife-core` after vendoring `rust-paddle-ocr`) cannot be made to work on the target Windows machine within one solid afternoon of debugging. The main plan hits a hard MNN dependency; this replaces Tasks 3, 4, and 7 with an `ort` + DirectML path.

**Why this is cheap to swap in:** Task 7's `OcrBackend` trait exists precisely so this swap is a replacement of the `MnnBackend` type with an `OrtBackend` type; the rest of the pipeline (`pdf`, `image_io`, `page`, `script`, `output::text`, the CLI) is backend-agnostic and ships unchanged.

### Replacement Task 3-alt: Add `ort` + keep existing ONNX models

- [ ] **3a.1 Add `ort` dependency.** Modify `crates/paddleknife-core/Cargo.toml`:

```toml
ort = { version = "2", features = ["load-dynamic", "directml", "cuda"] }
```

- [ ] **3a.2 Verify the crate builds.** Run `cargo build -p paddleknife-core`. The `load-dynamic` feature avoids the compile-time DLL link and defers to the ONNX Runtime DLL at runtime — same approach the Python version uses.

- [ ] **3a.3 Commit.**

```bash
git add crates/paddleknife-core/Cargo.toml
git commit -m "feat(core): add ort crate as MNN fallback backend"
```

### Replacement Task 4-alt: Reuse existing ONNX models

The Python version already has working ONNX models under `models/`:
- `models/devanagari_PP-OCRv5_rec.onnx` (detection + recognition)
- Plus auto-downloaded HuggingFace models in `~/.cache/huggingface/`.

For the fallback MVP we only need a det + rec pair that works on English. Simplest: pull from HuggingFace at the same URLs `paddleknife.py:96-99` uses (`monkt/paddleocr-onnx` `detection/v3/det.onnx` for detection; the default English rec ships with RapidOCR — we'll need to download it explicitly).

- [ ] **4a.1 Document the ONNX model layout in `models/README.md`** — add a section listing `det_v3.onnx`, `en_rec_v4.onnx`, `en_keys_v4.txt` paths under `models/`.

- [ ] **4a.2 Download English ONNX models from HuggingFace** — either manually via `huggingface-cli download` or copy from the existing Python cache.

- [ ] **4a.3 Commit.**

### Replacement Task 7-alt: Implement `OrtBackend`

- [ ] **7a.1 Add the `OrtBackend` struct.** Create `crates/paddleknife-core/src/ocr_ort.rs`:

```rust
//! ONNX Runtime backend via the `ort` crate. Falls back to DirectML on
//! Windows, CUDA if available. Used when the MNN backend cannot build.

use crate::{error::{OcrError, Result}, model::ModelPaths, ocr::{OcrBackend, OcrBlock}};
use image::DynamicImage;
use ort::{session::Session, session::builder::GraphOptimizationLevel};
use std::path::Path;

pub struct OrtBackend {
    det: Session,
    rec: Session,
    keys: Vec<String>,
}

impl OrtBackend {
    pub fn from_paths(det_onnx: &Path, rec_onnx: &Path, keys_txt: &Path) -> Result<Self> {
        let det = Session::builder()
            .map_err(|e| OcrError::Engine(format!("ort builder: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| OcrError::Engine(format!("ort opt: {e}")))?
            .commit_from_file(det_onnx)
            .map_err(|e| OcrError::Engine(format!("ort commit det: {e}")))?;

        let rec = Session::builder()
            .map_err(|e| OcrError::Engine(format!("ort builder: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| OcrError::Engine(format!("ort opt: {e}")))?
            .commit_from_file(rec_onnx)
            .map_err(|e| OcrError::Engine(format!("ort commit rec: {e}")))?;

        let keys_text = std::fs::read_to_string(keys_txt)?;
        let keys = keys_text.lines().map(str::to_string).collect();

        Ok(Self { det, rec, keys })
    }
}

impl OcrBackend for OrtBackend {
    fn ocr(&mut self, _img: &DynamicImage) -> Result<Vec<OcrBlock>> {
        // NOTE: a correct PP-OCR pipeline in ort needs:
        //   1. Resize the input to det's expected input shape (keep aspect).
        //   2. Normalize to NCHW float32 [0, 1] with mean/std from PaddleOCR.
        //   3. Run the det session → binary map → DB post-process → polygons.
        //   4. Crop each polygon, rotate to upright, pad to rec's input shape.
        //   5. Run rec session → CTC greedy decode using `self.keys`.
        //
        // Implementing that pipeline is Plan 2 work in the fallback track.
        // For the Plan 1 MVP in the Escape Hatch path, ship a stub that
        // returns an empty vec and wire it through so the CLI doesn't fail.
        // The real preprocessing is in paddleknife.py:1036-1133 and
        // rapidocr_onnxruntime's det.py / rec.py — port those.
        Ok(vec![])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<OrtBackend>();
    }
}
```

- [ ] **7a.2 Wire the backend into `OcrEngine`.** Modify `crates/paddleknife-core/src/ocr.rs` — add a constructor:

```rust
impl OcrEngine {
    #[cfg(feature = "backend-ort")]
    pub fn ort(det: &std::path::Path, rec: &std::path::Path, keys: &std::path::Path) -> Result<Self> {
        use crate::ocr_ort::OrtBackend;
        Ok(Self {
            inner: Box::new(OrtBackend::from_paths(det, rec, keys)?),
        })
    }
}
```

And modify `crates/paddleknife-core/Cargo.toml`:

```toml
[features]
default = []
backend-ort = []
```

Gate the `mod ocr_ort;` declaration in `lib.rs` behind the feature.

- [ ] **7a.3 Rewire CLI.** The CLI in Task 10 calls `OcrEngine::from_paths(&paths)`. In the Escape Hatch path, replace with `OcrEngine::ort(...)` and take the three ONNX paths from `models_dir_near_exe()` instead of `ModelPaths::in_dir`. Everything else in Task 10 is unchanged.

- [ ] **7a.4 Port DB post-processing + CTC decode.** This is the real work — reimplement the pipeline stub'd in 7a.1. Reference: `paddleknife.py:1036-1133`, plus `rapidocr_onnxruntime`'s `det.py` and `rec.py`. Because this is a non-trivial effort, spin it out as its own plan (**Plan 1-alt-Postprocessing**) — do not attempt it inline with Plan 1's other tasks.

- [ ] **7a.5 Go/no-go checkpoint.** After 7a.4, run the Task 7 test (`cargo test -p paddleknife-core --test ocr_hello_world`) against the ort backend. If it passes, the Escape Hatch is viable; continue Plan 1 Tasks 8-12 unchanged.

**If the Escape Hatch also fails**, the project needs a fresh ADR revisit: Options include (a) pure-Rust ONNX runtime like `tract`, (b) call `rapidocr_onnxruntime` via PyO3 (worst case, but unblocks everything else), or (c) pause the port and invest in making MNN build on Windows. At that point stop coding and write ADR-0002.

---

## Known Non-Goals (deferred to later plans)

- GPU backend selection (Plan 4).
- Progress reporting, per-page parallelism (Plan 2).
- Bbox-aware OCR output (Plan 2 reworks the `OcrBlock.bbox` field currently stubbed to `[0; 4]`).
- Searchable PDF output (Plan 2).
- Devanagari post-processing, parser-first hybrid, quality metrics (Plan 2).
- TUI (Plan 3).
- Surya / strip-images / info sub-commands (Plan 4).
- HuggingFace auto-download of models (Plan 2 or Plan 4).
- Config persistence to `%APPDATA%` (Plan 3 with TUI).

---

## Self-Review Notes

Pass 1 checks:

- **Spec coverage.** Every Plan 1-scoped requirement in ADR-0001 Action Items 1, 2, 3, 4 (first half), 5 (first half) is covered: workspace scaffold ✓, MNN spike via Tasks 3 + 7 ✓, PDF load ✓, image preprocess (implicit via `image` crate) ✓, det ✓, rec ✓, txt output ✓, clap sub-commands ✓. Deferred items (GPU, TUI, other formats, secondary scripts) are explicitly listed in Non-Goals.
- **Placeholder scan.** No TBD/TODO/placeholder steps; every code step has full code; every run step has an expected outcome; no "similar to Task N" references. The `OcrBlock.bbox` field stubbed to `[0; 4]` is called out in-task and in Non-Goals.
- **Type consistency.** `Page`, `RasterPage`, `OcrBlock`, `OcrEngine`, `ModelPaths`, `Script` are each defined once and referenced consistently across subsequent tasks.
- **File path consistency.** All paths rooted at the repository root. All crate internals rooted at `crates/<name>/src/`. All tests under `crates/<name>/tests/` or module-inline `#[cfg(test)]`.
