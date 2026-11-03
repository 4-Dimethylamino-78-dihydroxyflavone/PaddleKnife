# Plan 8: Slim, Faithful Searchable PDFs — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the ~19× output-size blowup: copy original scan-image bytes through verbatim when a source-PDF page is a plain single-image scan, and make the re-encode fallback grayscale-aware, DPI-capped, and quality-tunable.

**Architecture:** A new read-only `source_pdf` module (lopdf) probes each source page for passthrough eligibility and an effective-DPI hint. The searchable-PDF writer's `EncodedPage` gains a `PageImage` enum (Passthrough original bytes | re-encoded JPEG), and the XObject emission matches on it (`pdf-writer` stays the only PDF author). The CLI grows `--pdf-image`, `--jpeg-quality`, `--embed-dpi`.

**Tech Stack:** Rust; `lopdf 0.41` (NEW, read-only), `pdf-writer 0.15` (`Filter::CcittFaxDecode` + typed `decode_parms()` verified present), `image 0.25` (L8 JPEG encode verified), existing test stack (`tempfile`, `serial_test`).

**Spec:** `docs/superpowers/specs/2026-06-13-universal-pdf-local-translation-design.md` (Phase 1)

> **AMENDMENT NOTICE (2026-06-13, post-Task-1 code review):** Task 1 landed
> (`a6a4122` + arity/composition hardening `4fcf055`). Code review of the
> *plan's* Task 2/3 code found two false-ACCEPT vectors; **Amendment A at the
> bottom of this file supersedes the affected Task 2/3 blocks.** Read it
> before implementing Task 2 or Task 3.

**Worktree notes:** Run everything from the worktree root. `models/` and `vendor/pdfium/` are already present here (needed by pdfium-touching tests). Default cargo features (directml + download-binaries) are correct — plain `cargo test -p paddleknife-core` works.

---

## File structure

| File | Status | Responsibility |
|---|---|---|
| `crates/paddleknife-core/src/source_pdf.rs` | create | Read-only source-PDF inspection: passthrough probe + effective-DPI hint. lopdf lives ONLY here. |
| `crates/paddleknife-core/src/lib.rs` | modify | `pub mod source_pdf;` |
| `crates/paddleknife-core/src/output/searchable_pdf/page_image.rs` | create | `PageImage` enum (what bytes get embedded per page). |
| `crates/paddleknife-core/src/output/searchable_pdf/mod.rs` | modify | `EncodedPage` restructure, `EmbedOptions`, `encode_page_opts` (gray detect + DPI cap), XObject emission match. |
| `crates/paddleknife-core/Cargo.toml` | modify | Add `lopdf`. |
| `crates/paddleknife-core/tests/passthrough_probe.rs` | create | Probe accept/reject matrix against lopdf-built fixture PDFs. |
| `crates/paddleknife-core/tests/passthrough_roundtrip.rs` | create | Writer emits passthrough bytes byte-identically; pdfium renders it; size bound. |
| `crates/paddleknife/src/cli.rs` | modify | `--pdf-image`, `--jpeg-quality`, `--embed-dpi`, `PdfImageArg`. |
| `crates/paddleknife/src/main.rs` | modify | Open `SourcePdf` once, probe per page, plumb `EmbedOptions`. |
| `eval_extract_parity.py` | create | pypdf-vs-pdfminer extraction parity check (workspace root). |
| `README.md` | modify | Document the three new flags. |

---

### Task 1: lopdf dependency + `source_pdf` types + content-matrix analysis

**Files:**
- Modify: `crates/paddleknife-core/Cargo.toml` (after the `ureq = "3"` line)
- Create: `crates/paddleknife-core/src/source_pdf.rs`
- Modify: `crates/paddleknife-core/src/lib.rs` (module list)

- [ ] **Step 1: Add the dependency**

In `crates/paddleknife-core/Cargo.toml`, directly under the `ureq = "3"` line, add:

```toml
# Read-only inspection of *source* PDFs for page-image passthrough
# (Plan 8). lopdf never writes — pdf-writer remains the sole author.
lopdf = "0.41"
```

- [ ] **Step 2: Register the module**

In `crates/paddleknife-core/src/lib.rs`, add alongside the existing `pub mod pdf;` line:

```rust
pub mod source_pdf;
```

- [ ] **Step 3: Create `source_pdf.rs` with types + the pure content analyzer and its failing tests**

Create `crates/paddleknife-core/src/source_pdf.rs`:

```rust
//! Read-only inspection of the *source* PDF for the searchable-PDF
//! writer: can a page's original compressed image be embedded verbatim
//! (passthrough), and what is the page's effective scan DPI?
//!
//! A page qualifies for passthrough when it is a plain scan page:
//! exactly one image XObject, drawn axis-aligned covering ≥95 % of the
//! MediaBox, no rotation, no CropBox surprises, and a self-contained
//! encoding (DCT / raw Flate / CCITT with copyable DecodeParms,
//! DeviceGray or DeviceRGB). Anything else falls back to re-encoding —
//! the probe is deliberately strict because the fallback is cheap.
//!
//! lopdf is used strictly read-only here; `pdf-writer` remains the only
//! component that authors PDF bytes.

use crate::error::{OcrError, Result};
use lopdf::content::{Content, Operation};
use lopdf::{Dictionary, Document, Object, ObjectId, Stream};
use std::path::Path;

/// CCITTFaxDecode parameters copied verbatim from the source image's
/// `/DecodeParms` so the bytes decode identically in the output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CcittParms {
    pub k: i64,
    pub columns: i64,
    pub rows: Option<i64>,
    pub black_is_1: bool,
    pub encoded_byte_align: bool,
    pub end_of_block: bool,
}

/// Encodings we can carry through without decoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PtFilter {
    Dct,
    Flate,
    Ccitt(CcittParms),
}

/// A source page image that qualified for verbatim embedding.
#[derive(Debug, Clone)]
pub struct PassthroughImage {
    /// Raw (still-encoded) stream bytes, exactly as stored in the source.
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    /// DeviceGray (true) vs DeviceRGB (false).
    pub gray: bool,
    pub bits_per_component: u8,
    pub filter: PtFilter,
}

/// Probe result for one page. `Default` = nothing usable (re-encode,
/// no DPI cap).
#[derive(Debug, Default)]
pub struct PageProbe {
    pub passthrough: Option<PassthroughImage>,
    /// Resolution the page's scan actually has, dots/inch — drives the
    /// re-encode DPI cap even when byte-passthrough is rejected (e.g.
    /// JBIG2). `None` when page geometry didn't parse as a plain scan.
    pub effective_dpi: Option<f32>,
}

/// One full-page image draw found in a content stream.
#[derive(Debug, PartialEq)]
struct ImageDraw {
    name: Vec<u8>,
    ctm: [f32; 6],
}

const ID: [f32; 6] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

/// Affine multiply, row-vector convention: `mat_mul(a, b)` maps a point
/// first through `a`, then through `b` (PDF `cm` concatenation order).
fn mat_mul(a: [f32; 6], b: [f32; 6]) -> [f32; 6] {
    [
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
        a[4] * b[0] + a[5] * b[2] + b[4],
        a[4] * b[1] + a[5] * b[3] + b[5],
    ]
}

/// Numeric operand (PDF integers and reals both appear in matrices).
fn num(o: &Object) -> Option<f32> {
    o.as_float()
        .ok()
        .or_else(|| o.as_i64().ok().map(|i| i as f32))
}

/// Walk a content stream with a strict operator whitelist (`q Q cm Do`).
/// Returns the single image draw and its accumulated CTM, or `None` the
/// moment anything else appears — text, vector art, a second image,
/// graphics-state changes all disqualify the page.
fn analyze_content(ops: &[Operation]) -> Option<ImageDraw> {
    let mut ctm = ID;
    let mut stack: Vec<[f32; 6]> = Vec::new();
    let mut draw: Option<ImageDraw> = None;
    for o in ops {
        match o.operator.as_str() {
            "q" => stack.push(ctm),
            "Q" => ctm = stack.pop()?,
            "cm" => {
                if o.operands.len() != 6 {
                    return None;
                }
                let mut m = [0f32; 6];
                for (i, v) in o.operands.iter().enumerate() {
                    m[i] = num(v)?;
                }
                ctm = mat_mul(m, ctm);
            }
            "Do" => {
                if draw.is_some() {
                    return None; // a second draw — not a plain scan page
                }
                let name = o.operands.first()?.as_name().ok()?.to_vec();
                draw = Some(ImageDraw { name, ctm });
            }
            _ => return None,
        }
    }
    draw
}

#[cfg(test)]
mod tests {
    use super::*;

    fn op(s: &str, v: Vec<Object>) -> Operation {
        Operation::new(s, v)
    }

    fn cm(a: f32, b: f32, c: f32, d: f32, e: f32, f: f32) -> Operation {
        op(
            "cm",
            vec![a.into(), b.into(), c.into(), d.into(), e.into(), f.into()],
        )
    }

    fn do_im(name: &[u8]) -> Operation {
        op("Do", vec![Object::Name(name.to_vec())])
    }

    #[test]
    fn analyze_plain_scan_page() {
        let ops = vec![
            op("q", vec![]),
            cm(612.0, 0.0, 0.0, 792.0, 0.0, 0.0),
            do_im(b"Im0"),
            op("Q", vec![]),
        ];
        let d = analyze_content(&ops).expect("plain scan accepted");
        assert_eq!(d.name, b"Im0".to_vec());
        assert!((d.ctm[0] - 612.0).abs() < 0.01);
        assert!((d.ctm[3] - 792.0).abs() < 0.01);
    }

    #[test]
    fn analyze_accepts_integer_matrix_and_no_q() {
        // Some scanners emit `cm` with integers and skip q/Q entirely.
        let ops = vec![
            op(
                "cm",
                vec![
                    Object::Integer(612),
                    Object::Integer(0),
                    Object::Integer(0),
                    Object::Integer(792),
                    Object::Integer(0),
                    Object::Integer(0),
                ],
            ),
            do_im(b"Im1"),
        ];
        let d = analyze_content(&ops).expect("integer cm accepted");
        assert!((d.ctm[3] - 792.0).abs() < 0.01);
    }

    #[test]
    fn analyze_composes_nested_cm() {
        // 2× scale, then 306×396 unit square → drawn 612×792.
        let ops = vec![
            op("q", vec![]),
            cm(2.0, 0.0, 0.0, 2.0, 0.0, 0.0),
            op("q", vec![]),
            cm(306.0, 0.0, 0.0, 396.0, 0.0, 0.0),
            do_im(b"Im0"),
            op("Q", vec![]),
            op("Q", vec![]),
        ];
        let d = analyze_content(&ops).expect("nested cm accepted");
        assert!((d.ctm[0] - 612.0).abs() < 0.01, "got {:?}", d.ctm);
        assert!((d.ctm[3] - 792.0).abs() < 0.01, "got {:?}", d.ctm);
    }

    #[test]
    fn analyze_rejects_text_and_vector_ops() {
        let with_text = vec![
            cm(612.0, 0.0, 0.0, 792.0, 0.0, 0.0),
            do_im(b"Im0"),
            op("BT", vec![]),
            op("ET", vec![]),
        ];
        assert!(analyze_content(&with_text).is_none());
        let with_rect = vec![
            op(
                "re",
                vec![0.into(), 0.into(), 10.into(), 10.into()],
            ),
            op("f", vec![]),
        ];
        assert!(analyze_content(&with_rect).is_none());
    }

    #[test]
    fn analyze_rejects_two_draws_and_unbalanced_q() {
        let two = vec![
            cm(612.0, 0.0, 0.0, 792.0, 0.0, 0.0),
            do_im(b"Im0"),
            do_im(b"Im1"),
        ];
        assert!(analyze_content(&two).is_none());
        let unbalanced = vec![op("Q", vec![]), do_im(b"Im0")];
        assert!(analyze_content(&unbalanced).is_none());
    }

    #[test]
    fn analyze_rejects_pageless_content() {
        assert!(analyze_content(&[]).is_none());
    }
}
```

Note: `f32.into()` produces `Object::Real`, `Object::Integer` covers the integer-matrix case; `num()` accepts both.

- [ ] **Step 4: Run the new tests — they must pass (pure functions, no I/O)**

Run: `cargo test -p paddleknife-core --lib source_pdf`
Expected: `test result: ok. 6 passed` (module compiles; warnings about unused `Document`/`Stream`/`OcrError` imports are fine for now — they're used in Task 2).

If the unused-import warnings are denied by workspace lints, temporarily prefix with `#[allow(unused_imports)]` on the `use` lines and remove it in Task 2.

- [ ] **Step 5: Commit**

```bash
git add crates/paddleknife-core/Cargo.toml crates/paddleknife-core/src/lib.rs crates/paddleknife-core/src/source_pdf.rs
git commit -m "feat(source-pdf): lopdf dep + passthrough types + content-stream analyzer"
```

(`Cargo.lock` is gitignored in this repo — do not stage it.)

---

### Task 2: `SourcePdf::open` / `probe_page` — full probe logic

**Files:**
- Modify: `crates/paddleknife-core/src/source_pdf.rs` (append below `analyze_content`, above `#[cfg(test)]`)

- [ ] **Step 1: Append the `SourcePdf` implementation**

```rust
/// A source PDF opened for read-only inspection.
pub struct SourcePdf {
    doc: Document,
    /// 0-indexed page object ids in document order (pdfium index i ↔
    /// lopdf page number i+1).
    pages: Vec<ObjectId>,
}

impl SourcePdf {
    pub fn open(path: &Path) -> Result<Self> {
        let doc = Document::load(path)
            .map_err(|e| OcrError::Engine(format!("lopdf open {}: {e}", path.display())))?;
        if doc.is_encrypted() {
            return Err(OcrError::Engine("source PDF is encrypted".into()));
        }
        let pages: Vec<ObjectId> = doc.get_pages().into_values().collect();
        Ok(Self { doc, pages })
    }

    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// Probe page `idx` (0-based). Never fails: any anomaly returns the
    /// default probe (re-encode, no DPI hint).
    pub fn probe_page(&self, idx: usize) -> PageProbe {
        self.probe_inner(idx).unwrap_or_default()
    }

    /// Follow one level of indirection if `o` is a reference.
    fn resolve<'a>(&'a self, o: &'a Object) -> &'a Object {
        match o.as_reference() {
            Ok(id) => self.doc.get_object(id).unwrap_or(o),
            Err(_) => o,
        }
    }

    /// Inheritable page attribute (`MediaBox`, `Rotate`, `Resources`,
    /// `CropBox`): check the page dict, then walk up the `Parent` chain.
    fn inherited<'a>(&'a self, page: ObjectId, key: &[u8]) -> Option<&'a Object> {
        let mut id = page;
        for _ in 0..16 {
            let dict = self.doc.get_dictionary(id).ok()?;
            if let Ok(v) = dict.get(key) {
                return Some(self.resolve(v));
            }
            id = dict.get(b"Parent").ok()?.as_reference().ok()?;
        }
        None
    }

    fn rect(&self, o: &Object) -> Option<[f32; 4]> {
        let a = o.as_array().ok()?;
        if a.len() != 4 {
            return None;
        }
        let mut r = [0f32; 4];
        for (i, v) in a.iter().enumerate() {
            r[i] = num(self.resolve(v))?;
        }
        Some(r)
    }

    fn probe_inner(&self, idx: usize) -> Option<PageProbe> {
        let page_id = *self.pages.get(idx)?;

        // Rotated pages never qualify: the OCR raster (and therefore the
        // text layer) is post-rotation, the stored image is pre-rotation.
        if let Some(r) = self
            .inherited(page_id, b"Rotate")
            .and_then(|o| o.as_i64().ok())
        {
            if r.rem_euclid(360) != 0 {
                return None;
            }
        }

        let mb = self.rect(self.inherited(page_id, b"MediaBox")?)?;
        let (mb_w, mb_h) = (mb[2] - mb[0], mb[3] - mb[1]);
        if mb_w < 1.0 || mb_h < 1.0 {
            return None;
        }
        // A CropBox that crops anything means pdfium rendered a sub-area
        // of the stored image — passthrough would show the uncropped scan.
        if let Some(cb) = self
            .inherited(page_id, b"CropBox")
            .and_then(|o| self.rect(o))
        {
            if mb.iter().zip(cb.iter()).any(|(m, c)| (m - c).abs() > 1.0) {
                return None;
            }
        }

        // Exactly one XObject, and it must be an image (a Form XObject
        // could draw anything, including more images).
        let res = self.inherited(page_id, b"Resources")?.as_dict().ok()?;
        let xobjs = self.resolve(res.get(b"XObject").ok()?).as_dict().ok()?;
        let mut images: Vec<(Vec<u8>, &Stream)> = Vec::new();
        for (name, val) in xobjs.iter() {
            let stream = self.resolve(val).as_stream().ok()?;
            let sub = stream.dict.get(b"Subtype").ok()?.as_name().ok()?;
            if sub != b"Image" {
                return None;
            }
            images.push((name.clone().into_bytes(), stream));
        }
        if images.len() != 1 {
            return None;
        }
        let (img_name, stream) = (&images[0].0, images[0].1);

        // Content stream: single axis-aligned draw of that image,
        // covering ≥95 % of the MediaBox area.
        let data = self.doc.get_page_content(page_id).ok()?;
        let draw = analyze_content(&Content::decode(&data).ok()?.operations)?;
        if &draw.name != img_name {
            return None;
        }
        let [a, b, c, d, _, _] = draw.ctm;
        if b.abs() > 0.5 || c.abs() > 0.5 {
            return None; // rotated/skewed placement
        }
        let (dw, dh) = (a.abs(), d.abs());
        if dw * dh < 0.95 * mb_w * mb_h {
            return None;
        }

        let iw = self
            .resolve(stream.dict.get(b"Width").ok()?)
            .as_i64()
            .ok()
            .filter(|v| *v > 0)? as u32;
        let ih = self
            .resolve(stream.dict.get(b"Height").ok()?)
            .as_i64()
            .ok()
            .filter(|v| *v > 0)? as u32;

        // The geometry parsed as a plain scan, so the DPI hint is valid
        // even when the byte-level extraction below says no.
        let dpi = (iw as f32 / (dw / 72.0)).max(ih as f32 / (dh / 72.0));
        Some(PageProbe {
            passthrough: self.extract_image(stream, iw, ih),
            effective_dpi: Some(dpi),
        })
    }

    /// Byte-level qualification: self-contained encoding only.
    fn extract_image(&self, s: &Stream, width: u32, height: u32) -> Option<PassthroughImage> {
        let d = &s.dict;
        for k in [b"SMask".as_slice(), b"Mask", b"Decode"] {
            if d.get(k).is_ok() {
                return None;
            }
        }
        if d.get(b"ImageMask")
            .ok()
            .and_then(|o| o.as_bool().ok())
            .unwrap_or(false)
        {
            return None;
        }
        let bpc = d.get(b"BitsPerComponent").ok()?.as_i64().ok()?;
        let gray = match self.resolve(d.get(b"ColorSpace").ok()?).as_name().ok()? {
            b"DeviceGray" => true,
            b"DeviceRGB" => false,
            _ => return None, // Indexed/ICC/Separation → re-encode
        };
        let filter_obj = self.resolve(d.get(b"Filter").ok()?);
        let fname: &[u8] = match filter_obj {
            Object::Array(arr) if arr.len() == 1 => self.resolve(&arr[0]).as_name().ok()?,
            o => o.as_name().ok()?,
        };
        let filter = match fname {
            b"DCTDecode" => {
                if bpc != 8 {
                    return None;
                }
                PtFilter::Dct
            }
            b"FlateDecode" => {
                // Predictors et al. would need DecodeParms plumbing —
                // raw Flate only in v1.
                if d.get(b"DecodeParms").is_ok() || d.get(b"DecodeParams").is_ok() {
                    return None;
                }
                if bpc != 8 && bpc != 1 {
                    return None;
                }
                PtFilter::Flate
            }
            b"CCITTFaxDecode" => {
                if bpc != 1 || !gray {
                    return None;
                }
                PtFilter::Ccitt(self.ccitt_parms(d)?)
            }
            _ => return None, // JBIG2, JPX, … → re-encode (DPI hint kept)
        };
        Some(PassthroughImage {
            data: s.content.clone(),
            width,
            height,
            gray,
            bits_per_component: bpc as u8,
            filter,
        })
    }

    /// Copy CCITT DecodeParms. Reject any key we would not re-emit —
    /// an uncopied parameter (EndOfLine, DamagedRowsBeforeError …) could
    /// change how the bytes decode.
    fn ccitt_parms(&self, d: &Dictionary) -> Option<CcittParms> {
        const KNOWN: [&[u8]; 6] = [
            b"K",
            b"Columns",
            b"Rows",
            b"BlackIs1",
            b"EncodedByteAlign",
            b"EndOfBlock",
        ];
        let dp = match d.get(b"DecodeParms") {
            Ok(o) => self.resolve(o).as_dict().ok()?,
            Err(_) => {
                // No DecodeParms at all → spec defaults.
                return Some(CcittParms {
                    k: 0,
                    columns: 1728,
                    rows: None,
                    black_is_1: false,
                    encoded_byte_align: false,
                    end_of_block: true,
                });
            }
        };
        if dp.iter().any(|(k, _)| !KNOWN.contains(&k.as_bytes())) {
            return None;
        }
        let geti = |key: &[u8]| dp.get(key).ok().and_then(|o| o.as_i64().ok());
        let getb = |key: &[u8]| dp.get(key).ok().and_then(|o| o.as_bool().ok());
        Some(CcittParms {
            k: geti(b"K").unwrap_or(0),
            columns: geti(b"Columns").unwrap_or(1728),
            rows: geti(b"Rows"),
            black_is_1: getb(b"BlackIs1").unwrap_or(false),
            encoded_byte_align: getb(b"EncodedByteAlign").unwrap_or(false),
            end_of_block: getb(b"EndOfBlock").unwrap_or(true),
        })
    }
}
```

API notes for the implementer (verified against lopdf 0.41 docs):
`Object::as_name() -> Result<&[u8]>`, `as_float() -> Result<f32>`,
`as_i64`, `as_dict`, `as_array`, `as_stream`, `as_reference`, `as_bool`;
`Document::{load, is_encrypted, get_pages, get_page_content, get_object, get_dictionary}`.
`Dictionary::iter()` yields `(&String, &Object)` in 0.41 — hence
`name.clone().into_bytes()` / `k.as_bytes()` above. If the compiler says
keys are `&Vec<u8>` instead, replace those two call sites with
`name.clone()` and `k.as_slice()` respectively — that is the only
version-sensitive seam, isolated to these two lines.

- [ ] **Step 2: Build**

Run: `cargo build -p paddleknife-core`
Expected: clean compile (all Task-1 imports now used).

- [ ] **Step 3: Commit**

```bash
git add crates/paddleknife-core/src/source_pdf.rs
git commit -m "feat(source-pdf): SourcePdf probe — geometry, colorspace, filter qualification"
```

---

### Task 3: Probe accept/reject matrix (fixture PDFs built with lopdf)

**Files:**
- Create: `crates/paddleknife-core/tests/passthrough_probe.rs`

- [ ] **Step 1: Write the fixture builder + tests**

Create `crates/paddleknife-core/tests/passthrough_probe.rs`:

```rust
//! Probe qualification matrix: fixture PDFs are built programmatically
//! with lopdf (one knob per rejection reason), then probed via the
//! public SourcePdf API. No pdfium needed — these run anywhere.

use lopdf::content::{Content, Operation};
use lopdf::{dictionary, Dictionary, Document, Object, Stream};
use paddleknife_core::source_pdf::{PtFilter, SourcePdf};
use std::path::Path;

/// Knobs for one fixture page. Defaults = the qualifying happy path:
/// a 2×3-inch page fully covered by one RGB JPEG.
struct Fx {
    img_bytes: Vec<u8>,
    img_w: i64,
    img_h: i64,
    filter: &'static str,
    colorspace: &'static str,
    bpc: i64,
    page_w: f32,
    page_h: f32,
    /// Linear coverage fraction (drawn size = cover × page size).
    cover: f32,
    rotate: Option<i64>,
    crop: Option<[f32; 4]>,
    second_image: bool,
    smask: bool,
    extra_text_op: bool,
    decode_parms: Option<Dictionary>,
}

impl Default for Fx {
    fn default() -> Self {
        Self {
            img_bytes: tiny_jpeg(128, 96, false),
            img_w: 128,
            img_h: 96,
            filter: "DCTDecode",
            colorspace: "DeviceRGB",
            bpc: 8,
            page_w: 144.0,
            page_h: 216.0,
            cover: 1.0,
            rotate: None,
            crop: None,
            second_image: false,
            smask: false,
            extra_text_op: false,
            decode_parms: None,
        }
    }
}

fn tiny_jpeg(w: u32, h: u32, gray: bool) -> Vec<u8> {
    let mut out = Vec::new();
    let enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 70);
    if gray {
        image::GrayImage::from_fn(w, h, |x, y| image::Luma([(x * 3 + y * 5) as u8]))
            .write_with_encoder(enc)
            .unwrap();
    } else {
        image::RgbImage::from_fn(w, h, |x, y| {
            image::Rgb([(x * 2) as u8, (y * 2) as u8, 40])
        })
        .write_with_encoder(enc)
        .unwrap();
    }
    out
}

fn save_fx(fx: &Fx, path: &Path) {
    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();

    let mut img_dict = dictionary! {
        "Type" => "XObject",
        "Subtype" => "Image",
        "Width" => fx.img_w,
        "Height" => fx.img_h,
        "ColorSpace" => fx.colorspace,
        "BitsPerComponent" => fx.bpc,
        "Filter" => fx.filter,
    };
    if let Some(dp) = &fx.decode_parms {
        img_dict.set("DecodeParms", dp.clone());
    }
    if fx.smask {
        // Probe must reject on the key's presence alone — a dangling
        // reference is fine.
        img_dict.set("SMask", Object::Reference((9999, 0)));
    }
    let img_id =
        doc.add_object(Stream::new(img_dict, fx.img_bytes.clone()).with_compress(false));

    let mut xobjects = dictionary! { "Im0" => img_id };
    if fx.second_image {
        let img2 = dictionary! {
            "Type" => "XObject", "Subtype" => "Image",
            "Width" => 8i64, "Height" => 8i64,
            "ColorSpace" => "DeviceRGB", "BitsPerComponent" => 8i64,
            "Filter" => "DCTDecode",
        };
        let img2_id =
            doc.add_object(Stream::new(img2, tiny_jpeg(8, 8, false)).with_compress(false));
        xobjects.set("Im1", img2_id);
    }

    let dw = fx.page_w * fx.cover;
    let dh = fx.page_h * fx.cover;
    let mut ops = vec![
        Operation::new("q", vec![]),
        Operation::new(
            "cm",
            vec![
                dw.into(),
                0.into(),
                0.into(),
                dh.into(),
                0.into(),
                0.into(),
            ],
        ),
        Operation::new("Do", vec![Object::Name(b"Im0".to_vec())]),
        Operation::new("Q", vec![]),
    ];
    if fx.extra_text_op {
        ops.push(Operation::new("BT", vec![]));
        ops.push(Operation::new("ET", vec![]));
    }
    let content = Content { operations: ops };
    let content_id = doc.add_object(
        Stream::new(dictionary! {}, content.encode().unwrap()).with_compress(false),
    );

    let mut page = dictionary! {
        "Type" => "Page",
        "Parent" => pages_id,
        "MediaBox" => vec![
            0.into(), 0.into(), fx.page_w.into(), fx.page_h.into(),
        ],
        "Contents" => content_id,
        "Resources" => dictionary! { "XObject" => xobjects },
    };
    if let Some(r) = fx.rotate {
        page.set("Rotate", r);
    }
    if let Some(cb) = fx.crop {
        page.set(
            "CropBox",
            vec![
                Object::from(cb[0]),
                Object::from(cb[1]),
                Object::from(cb[2]),
                Object::from(cb[3]),
            ],
        );
    }
    let page_id = doc.add_object(page);

    doc.objects.insert(
        pages_id,
        Object::Dictionary(dictionary! {
            "Type" => "Pages",
            "Kids" => vec![page_id.into()],
            "Count" => 1,
        }),
    );
    let catalog_id = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages_id });
    doc.trailer.set("Root", catalog_id);
    doc.save(path).unwrap();
}

fn probe(fx: &Fx) -> paddleknife_core::source_pdf::PageProbe {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("fx.pdf");
    save_fx(fx, &p);
    let src = SourcePdf::open(&p).expect("open fixture");
    assert_eq!(src.page_count(), 1);
    src.probe_page(0)
}

#[test]
fn accepts_full_page_rgb_jpeg_byte_identical() {
    let fx = Fx::default();
    let want = fx.img_bytes.clone();
    let pr = probe(&fx);
    let pt = pr.passthrough.expect("qualifies");
    assert_eq!(pt.data, want, "bytes must be copied verbatim");
    assert!(!pt.gray);
    assert_eq!((pt.width, pt.height), (128, 96));
    assert_eq!(pt.filter, PtFilter::Dct);
    // 128 px over 2 in (144 pt) → 64 dpi; 96 px over 3 in → 32 dpi.
    let dpi = pr.effective_dpi.expect("hint");
    assert!((dpi - 64.0).abs() < 1.0, "dpi {dpi}");
}

#[test]
fn accepts_gray_jpeg() {
    let fx = Fx {
        img_bytes: tiny_jpeg(128, 96, true),
        colorspace: "DeviceGray",
        ..Fx::default()
    };
    let pt = probe(&fx).passthrough.expect("gray qualifies");
    assert!(pt.gray);
}

#[test]
fn accepts_ccitt_and_copies_parms() {
    let fx = Fx {
        img_bytes: b"\x00\x01fake-ccitt-bytes".to_vec(),
        filter: "CCITTFaxDecode",
        colorspace: "DeviceGray",
        bpc: 1,
        decode_parms: Some(dictionary! {
            "K" => -1i64, "Columns" => 1728i64, "BlackIs1" => true,
        }),
        ..Fx::default()
    };
    let pt = probe(&fx).passthrough.expect("ccitt qualifies");
    match pt.filter {
        PtFilter::Ccitt(c) => {
            assert_eq!(c.k, -1);
            assert_eq!(c.columns, 1728);
            assert!(c.black_is_1);
            assert!(c.end_of_block, "spec default");
            assert_eq!(c.rows, None);
        }
        other => panic!("expected ccitt, got {other:?}"),
    }
}

#[test]
fn accepts_raw_flate() {
    let fx = Fx {
        img_bytes: b"raw-flate-payload".to_vec(),
        filter: "FlateDecode",
        colorspace: "DeviceGray",
        ..Fx::default()
    };
    assert_eq!(
        probe(&fx).passthrough.expect("flate qualifies").filter,
        PtFilter::Flate
    );
}

#[test]
fn rejects_unknown_ccitt_parm_keys() {
    let fx = Fx {
        filter: "CCITTFaxDecode",
        colorspace: "DeviceGray",
        bpc: 1,
        decode_parms: Some(dictionary! { "K" => -1i64, "EndOfLine" => true }),
        ..Fx::default()
    };
    assert!(probe(&fx).passthrough.is_none());
}

#[test]
fn rejects_flate_with_decode_parms() {
    let fx = Fx {
        filter: "FlateDecode",
        colorspace: "DeviceGray",
        decode_parms: Some(dictionary! { "Predictor" => 15i64, "Columns" => 128i64 }),
        ..Fx::default()
    };
    assert!(probe(&fx).passthrough.is_none());
}

#[test]
fn rejects_partial_coverage_but_keeps_no_hint() {
    // 80 % linear coverage → 64 % area, under the 95 % area gate; the
    // geometry didn't parse as a *full-page* scan so no hint either.
    let pr = probe(&Fx { cover: 0.8, ..Fx::default() });
    assert!(pr.passthrough.is_none());
    assert!(pr.effective_dpi.is_none());
}

#[test]
fn rejects_rotated_page() {
    let pr = probe(&Fx { rotate: Some(90), ..Fx::default() });
    assert!(pr.passthrough.is_none());
}

#[test]
fn accepts_rotate_zero_and_360() {
    assert!(probe(&Fx { rotate: Some(0), ..Fx::default() })
        .passthrough
        .is_some());
    assert!(probe(&Fx { rotate: Some(360), ..Fx::default() })
        .passthrough
        .is_some());
}

#[test]
fn rejects_cropbox_smaller_than_mediabox() {
    let pr = probe(&Fx {
        crop: Some([10.0, 10.0, 100.0, 100.0]),
        ..Fx::default()
    });
    assert!(pr.passthrough.is_none());
}

#[test]
fn accepts_cropbox_equal_to_mediabox() {
    let pr = probe(&Fx {
        crop: Some([0.0, 0.0, 144.0, 216.0]),
        ..Fx::default()
    });
    assert!(pr.passthrough.is_some());
}

#[test]
fn rejects_second_image() {
    assert!(probe(&Fx { second_image: true, ..Fx::default() })
        .passthrough
        .is_none());
}

#[test]
fn rejects_extra_content_ops() {
    assert!(probe(&Fx { extra_text_op: true, ..Fx::default() })
        .passthrough
        .is_none());
}

#[test]
fn rejects_smask() {
    assert!(probe(&Fx { smask: true, ..Fx::default() }).passthrough.is_none());
}

#[test]
fn unsupported_filter_still_yields_dpi_hint() {
    let fx = Fx {
        img_bytes: b"jbig2-payload".to_vec(),
        filter: "JBIG2Decode",
        colorspace: "DeviceGray",
        bpc: 1,
        ..Fx::default()
    };
    let pr = probe(&fx);
    assert!(pr.passthrough.is_none());
    let dpi = pr.effective_dpi.expect("geometry parsed → hint survives");
    assert!((dpi - 64.0).abs() < 1.0, "dpi {dpi}");
}

#[test]
fn rejects_indexed_colorspace_but_keeps_hint() {
    // ColorSpace as an array (Indexed) → as_name fails → byte-reject,
    // geometry hint survives.
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("fx.pdf");
    let fx = Fx::default();
    save_fx(&fx, &p);
    // Patch the saved fixture: replace the ColorSpace name with an array.
    let mut doc = Document::load(&p).unwrap();
    let img_id = {
        let pages = doc.get_pages();
        let page_id = *pages.values().next().unwrap();
        let res = doc
            .get_dictionary(page_id)
            .unwrap()
            .get(b"Resources")
            .unwrap()
            .as_dict()
            .unwrap();
        res.get(b"XObject")
            .unwrap()
            .as_dict()
            .unwrap()
            .get(b"Im0")
            .unwrap()
            .as_reference()
            .unwrap()
    };
    if let Ok(Object::Stream(s)) = doc.get_object_mut(img_id) {
        s.dict.set(
            "ColorSpace",
            vec![Object::from("Indexed"), Object::from("DeviceRGB")],
        );
    }
    doc.save(&p).unwrap();

    let pr = SourcePdf::open(&p).unwrap().probe_page(0);
    assert!(pr.passthrough.is_none());
    assert!(pr.effective_dpi.is_some());
}
```

- [ ] **Step 2: Run the probe tests**

Run: `cargo test -p paddleknife-core --test passthrough_probe`
Expected: `test result: ok. 16 passed`

Two known version-sensitive seams if anything fails to compile:
(a) the `Dictionary::iter()` key type note from Task 2; (b)
`doc.get_object_mut(img_id)` returns `Result<&mut Object>` — match as
written. `Object::from("…")` makes a Name, which is what the Indexed
array test wants.

- [ ] **Step 3: Commit**

```bash
git add crates/paddleknife-core/tests/passthrough_probe.rs
git commit -m "test(source-pdf): probe accept/reject matrix over lopdf-built fixtures"
```

---

### Task 4: `PageImage` + `EncodedPage` restructure + writer emission

**Files:**
- Create: `crates/paddleknife-core/src/output/searchable_pdf/page_image.rs`
- Modify: `crates/paddleknife-core/src/output/searchable_pdf/mod.rs` (struct `EncodedPage` ~line 70-89, image XObject emission ~line 185-193, `encode_jpeg` ~line 485)
- Create: `crates/paddleknife-core/tests/passthrough_roundtrip.rs`

- [ ] **Step 1: Create `page_image.rs`**

```rust
//! What gets embedded as a page's image XObject: the source scan's
//! original compressed bytes (passthrough — zero quality/size cost) or
//! a JPEG (re)encoded from the OCR raster.

use crate::source_pdf::PassthroughImage;

#[derive(Debug, Clone)]
pub enum PageImage {
    /// Original bytes copied verbatim from the source PDF.
    Passthrough(PassthroughImage),
    /// JPEG re-encoded from the raster. `gray` selects DeviceGray.
    Jpeg {
        data: Vec<u8>,
        width: u32,
        height: u32,
        gray: bool,
    },
}

impl PageImage {
    /// Pixel dimensions of the *embedded* image (NOT the raster).
    pub fn dims(&self) -> (u32, u32) {
        match self {
            Self::Passthrough(p) => (p.width, p.height),
            Self::Jpeg { width, height, .. } => (*width, *height),
        }
    }

    pub fn data(&self) -> &[u8] {
        match self {
            Self::Passthrough(p) => &p.data,
            Self::Jpeg { data, .. } => data,
        }
    }
}
```

- [ ] **Step 2: Restructure `EncodedPage` and the encode fn in `mod.rs`**

Add to the module header area of `mod.rs`:

```rust
mod page_image;
pub use page_image::PageImage;

use crate::source_pdf::{PageProbe, PtFilter};
```

Replace the `EncodedPage` struct (keep its doc comment about JPEG memory
economics, then):

```rust
pub struct EncodedPage {
    /// Raster dimensions in pixels — the MediaBox basis (page size =
    /// raster px × 72/dpi). NOT necessarily the embedded image's size:
    /// passthrough and DPI-capped images carry their own dims in
    /// [`PageImage`]; the content stream stretches them to the page.
    pub width: u32,
    pub height: u32,
    pub image: PageImage,
}
```

Replace `encode_page` and `encode_jpeg` (full replacement; `EmbedOptions`
and the gray/DPI logic land in Tasks 5–6, so this step is the minimal
type migration — same RGB q85 behavior, new shape):

```rust
/// Encode one raster page to its embeddable form (RGB JPEG, q85).
pub fn encode_page(raster: &RasterPage) -> Result<EncodedPage> {
    let rgb = raster.image.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let data = jpeg_bytes(rgb.as_raw(), w, h, image::ExtendedColorType::Rgb8, 85)?;
    Ok(EncodedPage {
        width: raster.width,
        height: raster.height,
        image: PageImage::Jpeg { data, width: w, height: h, gray: false },
    })
}

fn jpeg_bytes(
    raw: &[u8],
    w: u32,
    h: u32,
    color: image::ExtendedColorType,
    quality: u8,
) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, quality);
    enc.encode(raw, w, h, color)
        .map_err(|e| OcrError::Engine(format!("jpeg encode: {e}")))?;
    Ok(out)
}
```

(Delete the old `encode_jpeg`.)

Replace the image-XObject emission block inside
`write_searchable_pdf_encoded` (currently the `// Raster as JPEG
XObject.` braces):

```rust
        // Page image XObject — original source bytes (passthrough) or
        // our (re)encoded JPEG. The content stream stretches whatever
        // pixel grid this is to the full page, so embedded dims are
        // independent of the raster dims that set the MediaBox.
        {
            let (iw, ih) = enc.image.dims();
            let mut img = pdf.image_xobject(*image_id, enc.image.data());
            img.width(iw as i32);
            img.height(ih as i32);
            match &enc.image {
                PageImage::Jpeg { gray, .. } => {
                    if *gray {
                        img.color_space().device_gray();
                    } else {
                        img.color_space().device_rgb();
                    }
                    img.bits_per_component(8);
                    img.filter(Filter::DctDecode);
                }
                PageImage::Passthrough(p) => {
                    if p.gray {
                        img.color_space().device_gray();
                    } else {
                        img.color_space().device_rgb();
                    }
                    img.bits_per_component(p.bits_per_component as i32);
                    match &p.filter {
                        PtFilter::Dct => {
                            img.filter(Filter::DctDecode);
                        }
                        PtFilter::Flate => {
                            img.filter(Filter::FlateDecode);
                        }
                        PtFilter::Ccitt(c) => {
                            img.filter(Filter::CcittFaxDecode);
                            let mut dp = img.decode_parms();
                            dp.k(c.k as i32);
                            dp.columns(c.columns as i32);
                            if let Some(rows) = c.rows {
                                dp.rows(rows as i32);
                            }
                            if c.black_is_1 {
                                dp.black_is_1(true);
                            }
                            if c.encoded_byte_align {
                                dp.encoded_byte_align(true);
                            }
                            if !c.end_of_block {
                                dp.end_of_block(false);
                            }
                        }
                    }
                }
            }
        }
```

(`decode_parms()` reaches `ImageXObject` through its `Deref` to
`pdf_writer::writers::Stream` — verified present in pdf-writer 0.15.)

- [ ] **Step 3: Build + run the existing suite (the migration must be invisible)**

Run: `cargo test -p paddleknife-core`
Expected: all existing tests pass (`alignment_metrics`, `searchable_pdf_roundtrip`, `devanagari_pdf_unicode`, `all_formats`, …) — `write_searchable_pdf`'s signature is unchanged and `encode_page` still produces RGB q85.

- [ ] **Step 4: Write the passthrough writer roundtrip test**

Create `crates/paddleknife-core/tests/passthrough_roundtrip.rs`:

```rust
//! Writer-side passthrough: a PassthroughImage handed to the writer
//! must come out byte-identical in the output XObject, render in
//! pdfium, and keep the output near input size.

use image::RgbImage;
use lopdf::{Document, Object};
use paddleknife_core::layout::{DocPage, PageMetrics};
use paddleknife_core::output::searchable_pdf::{
    write_searchable_pdf_encoded, EncodedPage, PageImage,
};
use paddleknife_core::pdf::RasterPage;
use paddleknife_core::source_pdf::{PassthroughImage, PtFilter};
use pdfium_render::prelude::*;
use serial_test::serial;

fn jpeg_fixture(w: u32, h: u32) -> Vec<u8> {
    let mut out = Vec::new();
    let enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 70);
    RgbImage::from_fn(w, h, |x, y| image::Rgb([(x % 256) as u8, (y % 256) as u8, 90]))
        .write_with_encoder(enc)
        .unwrap();
    out
}

/// 600×900 raster @300 dpi → 144×216 pt page; embedded image is a
/// different (smaller) pixel grid, as passthrough usually is.
fn encoded_passthrough(jpeg: Vec<u8>) -> (EncodedPage, DocPage) {
    let enc = EncodedPage {
        width: 600,
        height: 900,
        image: PageImage::Passthrough(PassthroughImage {
            data: jpeg,
            width: 200,
            height: 300,
            gray: false,
            bits_per_component: 8,
            filter: PtFilter::Dct,
        }),
    };
    let page = DocPage {
        index: 0,
        size: (600, 900),
        lines: vec![],
        metrics: PageMetrics::from_blocks(&[]),
    };
    (enc, page)
}

#[test]
fn passthrough_bytes_survive_verbatim_and_size_stays_bounded() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("pt.pdf");
    let jpeg = jpeg_fixture(200, 300);
    let src_len = jpeg.len();
    let (enc, page) = encoded_passthrough(jpeg.clone());
    write_searchable_pdf_encoded(&out, &[page], vec![enc], 300).expect("writer");

    // Reopen with lopdf and find the image XObject stream.
    let doc = Document::load(&out).expect("reopen output");
    let mut found = None;
    for (_, obj) in doc.objects.iter() {
        if let Object::Stream(s) = obj {
            let is_image = s
                .dict
                .get(b"Subtype")
                .ok()
                .and_then(|o| o.as_name().ok())
                .map(|n| n == b"Image")
                .unwrap_or(false);
            if is_image {
                found = Some(s.clone());
            }
        }
    }
    let s = found.expect("output has an image XObject");
    assert_eq!(s.content, jpeg, "bytes must be byte-identical");
    assert_eq!(
        s.dict.get(b"Width").unwrap().as_i64().unwrap(),
        200,
        "XObject dims are the image's, not the raster's"
    );
    assert_eq!(
        s.dict
            .get(b"ColorSpace")
            .unwrap()
            .as_name()
            .unwrap(),
        b"DeviceRGB"
    );
    assert_eq!(s.dict.get(b"Filter").unwrap().as_name().unwrap(), b"DCTDecode");

    // Size bound: output = image + text-layer scaffolding (empty page
    // here, so the bound is tight).
    let out_len = std::fs::metadata(&out).unwrap().len() as usize;
    assert!(
        out_len <= src_len + 60_000,
        "output {out_len} vs source image {src_len}"
    );
}

#[test]
#[serial]
fn passthrough_output_renders_in_pdfium() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("pt_render.pdf");
    let (enc, page) = encoded_passthrough(jpeg_fixture(200, 300));
    write_searchable_pdf_encoded(&out, &[page], vec![enc], 300).expect("writer");

    let pdfium = paddleknife_core::pdf::bind_pdfium().expect("pdfium");
    let doc = pdfium
        .load_pdf_from_file(out.to_str().unwrap(), None)
        .expect("open output");
    let pg = doc.pages().get(0).expect("page 0");
    let bmp = pg
        .render_with_config(&PdfRenderConfig::new().set_target_width(200))
        .expect("render");
    let img = bmp.as_image().to_rgb8();
    assert!(
        img.pixels().any(|p| p.0 != [255, 255, 255]),
        "rendered page must not be blank"
    );
}
```

- [ ] **Step 5: Run the roundtrip tests**

Run: `cargo test -p paddleknife-core --test passthrough_roundtrip`
Expected: `test result: ok. 2 passed` (second test needs `vendor/pdfium` — present in this worktree).

If `DocPage`/`PageMetrics` field shapes differ from the above, copy the
construction pattern used at the top of
`crates/paddleknife-core/tests/searchable_pdf_roundtrip.rs` — that file
builds the same structs against the real definitions.

- [ ] **Step 6: Commit**

```bash
git add crates/paddleknife-core/src/output/searchable_pdf/page_image.rs crates/paddleknife-core/src/output/searchable_pdf/mod.rs crates/paddleknife-core/tests/passthrough_roundtrip.rs
git commit -m "feat(pdf): PageImage passthrough emission — original scan bytes survive verbatim"
```

---

### Task 5: Grayscale auto-detect in the re-encode path

**Files:**
- Modify: `crates/paddleknife-core/src/output/searchable_pdf/mod.rs` (`encode_page`, new helpers + in-module tests)

- [ ] **Step 1: Write failing tests** (append to the existing `#[cfg(test)] mod tests` in `mod.rs`, or create one if absent)

```rust
#[cfg(test)]
mod gray_tests {
    use super::*;
    use crate::pdf::RasterPage;
    use image::{DynamicImage, RgbImage};

    fn raster_of(img: RgbImage) -> RasterPage {
        RasterPage {
            index: 0,
            width: img.width(),
            height: img.height(),
            image: DynamicImage::ImageRgb8(img),
        }
    }

    #[test]
    fn near_neutral_page_encodes_gray() {
        // Scan-like: dark gray text pixels on off-white, channels within ±2.
        let img = RgbImage::from_fn(64, 64, |x, _| {
            if x % 7 == 0 {
                image::Rgb([40, 41, 40])
            } else {
                image::Rgb([245, 244, 245])
            }
        });
        let enc = encode_page(&raster_of(img)).unwrap();
        match enc.image {
            PageImage::Jpeg { gray, .. } => assert!(gray, "neutral page must be gray"),
            other => panic!("expected jpeg, got {other:?}"),
        }
    }

    #[test]
    fn colorful_page_stays_rgb() {
        let img = RgbImage::from_fn(64, 64, |x, y| {
            image::Rgb([(x * 4) as u8, 0, (y * 4) as u8])
        });
        let enc = encode_page(&raster_of(img)).unwrap();
        match enc.image {
            PageImage::Jpeg { gray, .. } => assert!(!gray, "colorful page must stay RGB"),
            other => panic!("expected jpeg, got {other:?}"),
        }
    }

    #[test]
    fn gray_jpeg_is_smaller_than_rgb_jpeg() {
        let img = RgbImage::from_fn(256, 256, |x, y| {
            let v = ((x * 7 + y * 13) % 251) as u8;
            image::Rgb([v, v, v])
        });
        let raster = raster_of(img.clone());
        let gray_len = encode_page(&raster).unwrap().image.data().len();
        let rgb_len =
            jpeg_bytes(img.as_raw(), 256, 256, image::ExtendedColorType::Rgb8, 85)
                .unwrap()
                .len();
        assert!(
            gray_len < rgb_len,
            "gray {gray_len} should beat rgb {rgb_len}"
        );
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p paddleknife-core --lib gray_tests`
Expected: FAIL — `near_neutral_page_encodes_gray` panics with "neutral page must be gray" (encode_page still hardcodes `gray: false`).

- [ ] **Step 3: Implement gray detection inside `encode_page`**

Replace `encode_page` from Task 4 with:

```rust
/// Encode one raster page to its embeddable form: grayscale JPEG when
/// the page is effectively neutral (typical for B&W book scans —
/// roughly 2–3× smaller than RGB), RGB JPEG otherwise. Quality 85.
pub fn encode_page(raster: &RasterPage) -> Result<EncodedPage> {
    let rgb = raster.image.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let image = if is_effectively_gray(&rgb) {
        let gray = image::DynamicImage::ImageRgb8(rgb).to_luma8();
        PageImage::Jpeg {
            data: jpeg_bytes(gray.as_raw(), w, h, image::ExtendedColorType::L8, 85)?,
            width: w,
            height: h,
            gray: true,
        }
    } else {
        PageImage::Jpeg {
            data: jpeg_bytes(rgb.as_raw(), w, h, image::ExtendedColorType::Rgb8, 85)?,
            width: w,
            height: h,
            gray: false,
        }
    };
    Ok(EncodedPage { width: raster.width, height: raster.height, image })
}

/// True when ≥99.5 % of sampled pixels have a max channel spread ≤ 8 —
/// i.e. the page carries no real color information. Samples ~100k
/// pixels regardless of page size.
fn is_effectively_gray(rgb: &image::RgbImage) -> bool {
    let total = (rgb.width() as usize * rgb.height() as usize).max(1);
    let step = (total / 100_000).max(1);
    let mut seen = 0usize;
    let mut colorful = 0usize;
    for (i, p) in rgb.pixels().enumerate() {
        if i % step != 0 {
            continue;
        }
        seen += 1;
        let [r, g, b] = p.0;
        let mx = r.max(g).max(b);
        let mn = r.min(g).min(b);
        if mx - mn > 8 {
            colorful += 1;
        }
    }
    // colorful/seen < 0.5 %
    colorful * 200 < seen.max(1)
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p paddleknife-core --lib gray_tests`
Expected: `3 passed`.

Then the full crate (gray output must not disturb rendering-based gates):
Run: `cargo test -p paddleknife-core`
Expected: all pass — `alignment_metrics` renders via pdfium, which decodes grayscale JPEG identically.

- [ ] **Step 5: Commit**

```bash
git add crates/paddleknife-core/src/output/searchable_pdf/mod.rs
git commit -m "feat(pdf): grayscale auto-detect — neutral scans embed as DeviceGray JPEG"
```

---

### Task 6: `EmbedOptions` + `encode_page_opts` (passthrough switch, quality, DPI cap)

**Files:**
- Modify: `crates/paddleknife-core/src/output/searchable_pdf/mod.rs`

- [ ] **Step 1: Write failing tests** (append to `gray_tests`' sibling level, a new `#[cfg(test)] mod embed_tests` in `mod.rs`)

```rust
#[cfg(test)]
mod embed_tests {
    use super::*;
    use crate::pdf::RasterPage;
    use crate::source_pdf::{PageProbe, PassthroughImage, PtFilter};
    use image::{DynamicImage, RgbImage};

    fn raster(w: u32, h: u32) -> RasterPage {
        RasterPage {
            index: 0,
            width: w,
            height: h,
            image: DynamicImage::ImageRgb8(RgbImage::from_fn(w, h, |x, y| {
                image::Rgb([(x % 250) as u8, (y % 250) as u8, 10])
            })),
        }
    }

    fn probe_with(pt: Option<PassthroughImage>, dpi: Option<f32>) -> PageProbe {
        PageProbe { passthrough: pt, effective_dpi: dpi }
    }

    fn fake_pt() -> PassthroughImage {
        PassthroughImage {
            data: b"original-bytes".to_vec(),
            width: 100,
            height: 150,
            gray: true,
            bits_per_component: 8,
            filter: PtFilter::Dct,
        }
    }

    #[test]
    fn passthrough_wins_when_allowed() {
        let enc = encode_page_opts(
            &raster(600, 900),
            Some(&probe_with(Some(fake_pt()), Some(64.0))),
            &EmbedOptions::default(),
            300,
        )
        .unwrap();
        match enc.image {
            PageImage::Passthrough(p) => assert_eq!(p.data, b"original-bytes".to_vec()),
            other => panic!("expected passthrough, got {other:?}"),
        }
        // MediaBox basis stays the raster.
        assert_eq!((enc.width, enc.height), (600, 900));
    }

    #[test]
    fn reencode_mode_ignores_passthrough() {
        let opts = EmbedOptions { allow_passthrough: false, ..EmbedOptions::default() };
        let enc = encode_page_opts(
            &raster(600, 900),
            Some(&probe_with(Some(fake_pt()), None)),
            &opts,
            300,
        )
        .unwrap();
        assert!(matches!(enc.image, PageImage::Jpeg { .. }));
    }

    #[test]
    fn dpi_hint_downscales_embedded_image() {
        // 1000×1500 raster at 300 dpi, source scan only 100 dpi →
        // embedded image 333×500; MediaBox basis unchanged.
        let enc = encode_page_opts(
            &raster(1000, 1500),
            Some(&probe_with(None, Some(100.0))),
            &EmbedOptions::default(),
            300,
        )
        .unwrap();
        assert_eq!((enc.width, enc.height), (1000, 1500));
        let (w, h) = enc.image.dims();
        assert_eq!((w, h), (333, 500));
    }

    #[test]
    fn embed_dpi_override_beats_hint() {
        let opts = EmbedOptions { embed_dpi_override: Some(150), ..EmbedOptions::default() };
        let enc = encode_page_opts(
            &raster(1000, 1500),
            Some(&probe_with(None, Some(72.0))),
            &opts,
            300,
        )
        .unwrap();
        assert_eq!(enc.image.dims(), (500, 750));
    }

    #[test]
    fn near_full_dpi_hint_skips_downscale() {
        // 290 dpi vs 300 ocr → within 5 %, not worth a resample.
        let enc = encode_page_opts(
            &raster(600, 900),
            Some(&probe_with(None, Some(290.0))),
            &EmbedOptions::default(),
            300,
        )
        .unwrap();
        assert_eq!(enc.image.dims(), (600, 900));
    }

    #[test]
    fn zero_ocr_dpi_means_no_scaling() {
        // Image inputs: raster pixels ARE the native resolution.
        let enc = encode_page_opts(
            &raster(640, 480),
            None,
            &EmbedOptions { embed_dpi_override: Some(72), ..EmbedOptions::default() },
            0,
        )
        .unwrap();
        assert_eq!(enc.image.dims(), (640, 480));
    }

    #[test]
    fn quality_knob_changes_output_size() {
        let r = raster(256, 256);
        let q30 = encode_page_opts(
            &r,
            None,
            &EmbedOptions { jpeg_quality: 30, ..EmbedOptions::default() },
            300,
        )
        .unwrap();
        let q95 = encode_page_opts(
            &r,
            None,
            &EmbedOptions { jpeg_quality: 95, ..EmbedOptions::default() },
            300,
        )
        .unwrap();
        assert!(q30.image.data().len() < q95.image.data().len());
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p paddleknife-core --lib embed_tests`
Expected: FAIL to compile — `EmbedOptions` / `encode_page_opts` don't exist yet.

- [ ] **Step 3: Implement**

In `mod.rs`, replace `encode_page` (again — it becomes a thin wrapper)
and add the options machinery:

```rust
/// Options governing how page images are embedded.
#[derive(Debug, Clone)]
pub struct EmbedOptions {
    /// Copy original source-scan bytes through when a page qualifies.
    pub allow_passthrough: bool,
    /// JPEG quality (1–100) for re-encoded pages.
    pub jpeg_quality: u8,
    /// Hard cap on embedded-image DPI (overrides the probe's hint).
    pub embed_dpi_override: Option<u32>,
}

impl Default for EmbedOptions {
    fn default() -> Self {
        Self { allow_passthrough: true, jpeg_quality: 85, embed_dpi_override: None }
    }
}

/// Encode one raster page with default options (no probe → re-encode;
/// grayscale auto-detect still applies).
pub fn encode_page(raster: &RasterPage) -> Result<EncodedPage> {
    encode_page_opts(raster, None, &EmbedOptions::default(), 0)
}

/// Full embedding decision for one page.
///
/// `ocr_dpi` is the rasterization DPI of `raster` (pass 0 for image
/// inputs, whose raster pixels are already the native resolution — DPI
/// capping is then skipped). The DPI cap never upsamples: it only
/// shrinks the *embedded* image when the source scan demonstrably has
/// less resolution than the OCR raster.
pub fn encode_page_opts(
    raster: &RasterPage,
    probe: Option<&PageProbe>,
    opts: &EmbedOptions,
    ocr_dpi: u32,
) -> Result<EncodedPage> {
    if opts.allow_passthrough {
        if let Some(p) = probe.and_then(|p| p.passthrough.as_ref()) {
            return Ok(EncodedPage {
                width: raster.width,
                height: raster.height,
                image: PageImage::Passthrough(p.clone()),
            });
        }
    }

    // Re-encode path. Pick the target embed resolution.
    let target_dpi = opts
        .embed_dpi_override
        .map(|d| d as f32)
        .or_else(|| probe.and_then(|p| p.effective_dpi));
    let scaled;
    let src: &image::DynamicImage = match target_dpi {
        Some(t) if ocr_dpi > 0 && t < ocr_dpi as f32 * 0.95 => {
            let t = t.clamp(50.0, ocr_dpi as f32);
            let w = ((raster.width as f32) * t / ocr_dpi as f32).round().max(1.0) as u32;
            let h = ((raster.height as f32) * t / ocr_dpi as f32).round().max(1.0) as u32;
            scaled = raster
                .image
                .resize_exact(w, h, image::imageops::FilterType::Lanczos3);
            &scaled
        }
        _ => &raster.image,
    };

    let rgb = src.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let image = if is_effectively_gray(&rgb) {
        let gray = image::DynamicImage::ImageRgb8(rgb).to_luma8();
        PageImage::Jpeg {
            data: jpeg_bytes(
                gray.as_raw(),
                w,
                h,
                image::ExtendedColorType::L8,
                opts.jpeg_quality,
            )?,
            width: w,
            height: h,
            gray: true,
        }
    } else {
        PageImage::Jpeg {
            data: jpeg_bytes(
                rgb.as_raw(),
                w,
                h,
                image::ExtendedColorType::Rgb8,
                opts.jpeg_quality,
            )?,
            width: w,
            height: h,
            gray: false,
        }
    };
    Ok(EncodedPage { width: raster.width, height: raster.height, image })
}
```

(The Task-5 `encode_page` body is absorbed into `encode_page_opts`; the
standalone gray-encode code added there is deleted with it. `is_effectively_gray`
and `jpeg_bytes` stay.)

- [ ] **Step 4: Run the new and full tests**

Run: `cargo test -p paddleknife-core --lib embed_tests`
Expected: `7 passed`.
Run: `cargo test -p paddleknife-core`
Expected: everything green.

- [ ] **Step 5: Commit**

```bash
git add crates/paddleknife-core/src/output/searchable_pdf/mod.rs
git commit -m "feat(pdf): EmbedOptions — passthrough switch, quality knob, source-DPI cap"
```

---

### Task 7: CLI flags + main.rs plumbing

**Files:**
- Modify: `crates/paddleknife/src/cli.rs` (OcrArgs + new enum)
- Modify: `crates/paddleknife/src/main.rs` (`run_ocr`, `ocr_all_pages`)

- [ ] **Step 1: Add the flags to `OcrArgs`** (after `detect_line_orientation`, before `verbose`)

```rust
    /// How the page image is embedded in searchable-PDF output.
    /// `auto` copies the original scan bytes through verbatim when a
    /// page is a plain single-image scan (output stays ≈ input size);
    /// `reencode` always re-encodes from the OCR raster.
    #[arg(long, value_enum, default_value_t = PdfImageArg::Auto)]
    pub pdf_image: PdfImageArg,

    /// JPEG quality (1-100) for re-encoded page images in searchable-PDF
    /// output.
    #[arg(long, value_name = "Q", default_value_t = 85,
          value_parser = clap::value_parser!(u8).range(1..=100))]
    pub jpeg_quality: u8,

    /// Cap the embedded page-image resolution (DPI) in searchable-PDF
    /// output. Default: the source page's own effective DPI, so scans
    /// are never stored above their true resolution. OCR accuracy is
    /// unaffected — recognition always runs at full rasterization DPI.
    #[arg(long, value_name = "DPI")]
    pub embed_dpi: Option<u32>,
```

And the enum next to `FormatArg`:

```rust
/// Page-image embedding strategy for searchable-PDF output.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum PdfImageArg {
    /// Passthrough original scan bytes when safe, else re-encode.
    Auto,
    /// Always re-encode from the OCR raster.
    Reencode,
}
```

- [ ] **Step 2: Plumb `EmbedOptions` through `main.rs`**

In `run_ocr`, right before the `ocr_all_pages` call, build the options:

```rust
    let embed = output::searchable_pdf::EmbedOptions {
        allow_passthrough: args.pdf_image == cli::PdfImageArg::Auto,
        jpeg_quality: args.jpeg_quality,
        embed_dpi_override: args.embed_dpi,
    };
```

Change the call to pass it:

```rust
    let (doc_pages, rasters_for_pdf) = ocr_all_pages(
        &args.input,
        &mut engine,
        args.format == FormatArg::Pdf,
        script,
        args.format,
        &embed,
    )?;
```

Change `ocr_all_pages`'s signature:

```rust
fn ocr_all_pages(
    input: &std::path::Path,
    engine: &mut OcrEngine,
    keep_rasters: bool,
    script: RecognizerScript,
    format: FormatArg,
    embed: &paddleknife_core::output::searchable_pdf::EmbedOptions,
) -> Result<(Vec<DocPage>, Vec<EncodedPage>)> {
```

In the `Some("pdf")` branch, before the `pdf::for_each_page` call, open
the source PDF once (failure is never fatal — it just disables
passthrough and the DPI hint):

```rust
            // Passthrough probe: read-only view of the source PDF. Any
            // open failure (encrypted, malformed) downgrades gracefully
            // to the re-encode path.
            let source = if keep_rasters && embed.allow_passthrough {
                match paddleknife_core::source_pdf::SourcePdf::open(input) {
                    Ok(s) => Some(s),
                    Err(e) => {
                        tracing::warn!(error = %e,
                            "source-PDF inspection unavailable; pages will be re-encoded");
                        None
                    }
                }
            } else {
                None
            };
```

Replace the existing `if keep_rasters { encoded.push(...) }` inside the
closure with:

```rust
                if keep_rasters {
                    let probe = source.as_ref().map(|s| s.probe_page(raster.index));
                    encoded.push(output::searchable_pdf::encode_page_opts(
                        &raster,
                        probe.as_ref(),
                        embed,
                        DEFAULT_DPI,
                    )?);
                }
```

In the single-image branch, replace the `encode_page` call with (note
`ocr_dpi = 0`: image inputs keep native resolution):

```rust
            let encoded = if keep_rasters {
                vec![output::searchable_pdf::encode_page_opts(&raster, None, embed, 0)
                    .context("encoding page raster")?]
            } else {
                Vec::new()
            };
```

- [ ] **Step 3: Build, clippy, full workspace test**

Run: `cargo clippy --workspace --all-targets`
Expected: no new warnings.
Run: `cargo test --workspace`
Expected: all green.

- [ ] **Step 4: Flag smoke test (no OCR run needed)**

Run: `cargo run -p paddleknife -- ocr --help`
Expected: help text lists `--pdf-image <PDF_IMAGE>` with `auto`/`reencode`, `--jpeg-quality <Q>`, `--embed-dpi <DPI>`.

- [ ] **Step 5: Commit**

```bash
git add crates/paddleknife/src/cli.rs crates/paddleknife/src/main.rs
git commit -m "feat(cli): --pdf-image auto|reencode, --jpeg-quality, --embed-dpi"
```

---

### Task 8: Cross-engine parity script + MunaMadan acceptance + docs

**Files:**
- Create: `eval_extract_parity.py` (workspace root, next to `eval_box_accuracy.py`)
- Modify: `README.md` (flags section)

- [ ] **Step 1: Write the parity script**

Create `eval_extract_parity.py`:

```python
"""Cross-engine extraction parity: pypdf vs pdfminer.six.

A universally-parseable text layer must read back identically (modulo
whitespace) through independent PDF parsers. Exit 0 on parity.

Usage: python eval_extract_parity.py <file.pdf>
"""
import re
import sys

from pdfminer.high_level import extract_text
from pypdf import PdfReader


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    reader = PdfReader(path)
    bad = 0
    for i in range(len(reader.pages)):
        a = norm(reader.pages[i].extract_text())
        b = norm(extract_text(path, page_numbers=[i]))
        if a != b:
            bad += 1
            print(f"page {i + 1}: pypdf {len(a)} chars != pdfminer {len(b)} chars")
    print(f"{len(reader.pages)} pages checked, {bad} mismatched")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Build release and run the MunaMadan acceptance**

```bash
cargo build --release -p paddleknife
cargo run --release -p paddleknife -- ocr MunaMadan_full.pdf -f pdf -s nepali -o dev/MunaMadan_slim_OCR.pdf
```

Expected: `Wrote 51 pages (… lines, avg confidence …) to dev/MunaMadan_slim_OCR.pdf`.

- [ ] **Step 3: Verify size + parity + box coverage**

```bash
ls -la MunaMadan_full.pdf dev/MunaMadan_slim_OCR.pdf
python eval_extract_parity.py dev/MunaMadan_slim_OCR.pdf
python eval_box_accuracy.py dev/MunaMadan_slim_OCR.pdf
```

Expected:
- output size **< 2,500,000 bytes** (source is 1,829,621; passthrough adds only the text layer — if it's tens of MB, passthrough silently fell back: debug `SourcePdf::probe_page` against page 0 before proceeding);
- parity: `51 pages checked, 0 mismatched`;
- box accuracy in line with the Plan-6 baseline for this file.

- [ ] **Step 4: Spot-check in a real viewer**

Open `dev/MunaMadan_slim_OCR.pdf` in a pdf.js viewer (Firefox) — pages render at original scan quality, Devanagari selects/searches.

- [ ] **Step 5: Document the flags in README.md**

In the README's `ocr` usage/options section, add:

```markdown
- `--pdf-image <auto|reencode>` — searchable-PDF page images: `auto`
  (default) copies the original scan's compressed bytes through
  verbatim when a page is a plain single-image scan, keeping output ≈
  input size; `reencode` always re-encodes from the OCR raster.
- `--jpeg-quality <Q>` — JPEG quality (1–100, default 85) for
  re-encoded page images. Grayscale scans are auto-detected and stored
  as DeviceGray.
- `--embed-dpi <DPI>` — cap the embedded page-image resolution. By
  default the source page's own effective DPI is the cap, so scans are
  never stored above their true resolution. OCR always runs at full
  rasterization DPI.
```

- [ ] **Step 6: Final full suite + commit**

```bash
cargo test --workspace
cargo clippy --workspace --all-targets
git add eval_extract_parity.py README.md
git commit -m "feat(pdf): Plan 8 close — slim faithful PDFs verified on Muna Madan (34 MB -> ~2 MB)"
```

(If the measured size differs, put the real number in the commit message.)

---

## Amendment A (2026-06-13, post-Task-1 code review) — SUPERSEDES parts of Tasks 2 & 3

Code review of the plan's own Task 2 code found two false-ACCEPT vectors.
Both violate the module's prime directive (false ACCEPT corrupts visual
output; false REJECT only costs bytes). Apply these replacements.

### A.1 — Task 2: per-axis/position/sign CTM gate (replaces the area gate)

The planned gate `let [a, b, c, d, _, _] = draw.ctm; … a.abs()*d.abs() ≥
0.95·area` ignores translation and sign: a full-size draw shifted half
off-page, a mirrored draw (negative `a`/`d` — the writer would re-draw it
unmirrored), or a 2×-width draw all pass. In `probe_inner`, REPLACE the
block from `let [a, b, c, d, _, _] = draw.ctm;` through the
`dw * dh < 0.95 …` check with:

```rust
        let [a, b, c, d, e, f] = draw.ctm;
        // Axis-aligned, unflipped, full-page: per-axis size within 3 %
        // of the MediaBox, positioned at its lower-left corner. No
        // abs() — a negative a/d is a mirrored/flipped draw, which
        // passthrough would silently un-mirror.
        if b.abs() > 0.5 || c.abs() > 0.5 {
            return None;
        }
        if a <= 0.0 || d <= 0.0 {
            return None;
        }
        let (tol_w, tol_h) = (0.03 * mb_w, 0.03 * mb_h);
        if (a - mb_w).abs() > tol_w || (d - mb_h).abs() > tol_h {
            return None;
        }
        if (e - mb[0]).abs() > tol_w || (f - mb[1]).abs() > tol_h {
            return None;
        }
        let (dw, dh) = (a, d);
```

Also update the module doc line `//! exactly one image XObject, drawn
axis-aligned covering ≥95 % of the` / `//! MediaBox, …` to read:

```rust
//! exactly one image XObject, drawn axis-aligned and unflipped,
//! matching the MediaBox within 3 % per axis (size and position),
//! no rotation, no CropBox surprises, and a self-contained
```

### A.2 — Task 2: strict content pre-lexer (lopdf's parser truncates silently)

Verified in lopdf 0.41 source: `Content::decode` is `many0(operation)` +
a helper that **discards unconsumed input** — a stream like
`q … /Im0 Do Q <unparseable token> <more drawing ops>` decodes Ok with
only the benign prefix, blinding the whitelist while pdfium happily
renders the suffix. Defense: before parsing, require the raw bytes to
consist ONLY of benign tokens. Add to `source_pdf.rs` (free function,
above `SourcePdf`):

```rust
/// lopdf's content parser stops at the first token it can't parse and
/// silently returns the prefix, so a malformed-but-renderable stream
/// could hide drawing operators from the whitelist. Gate: the raw
/// bytes must consist ONLY of benign tokens — plain numbers, plain
/// /Names, and the four whitelisted operators — before lopdf parses
/// them. Anything else (strings, arrays, dicts, comments, hex, inline
/// images, name escapes) rejects the page.
fn content_tokens_are_plain(data: &[u8]) -> bool {
    fn is_ws(b: u8) -> bool {
        matches!(b, b'\0' | b'\t' | b'\n' | b'\x0c' | b'\r' | b' ')
    }
    fn is_delim_or_ws(b: u8) -> bool {
        is_ws(b)
            || matches!(
                b,
                b'/' | b'(' | b')' | b'<' | b'>' | b'[' | b']' | b'{' | b'}' | b'%'
            )
    }
    let n = data.len();
    let mut i = 0;
    while i < n {
        let b = data[i];
        if is_ws(b) {
            i += 1;
        } else if b == b'/' {
            // Name: plain alphanumerics plus _.+- only (no #xx escapes).
            i += 1;
            let start = i;
            while i < n && !is_delim_or_ws(data[i]) {
                if !(data[i].is_ascii_alphanumeric()
                    || matches!(data[i], b'_' | b'.' | b'-' | b'+'))
                {
                    return false;
                }
                i += 1;
            }
            if i == start {
                return false; // bare slash
            }
        } else if b == b'+' || b == b'-' || b == b'.' || b.is_ascii_digit() {
            // Number: optional sign, digits, at most one dot.
            if data[i] == b'+' || data[i] == b'-' {
                i += 1;
            }
            let (mut digits, mut dots) = (0u32, 0u32);
            while i < n && !is_delim_or_ws(data[i]) {
                match data[i] {
                    b'0'..=b'9' => digits += 1,
                    b'.' => dots += 1,
                    _ => return false,
                }
                i += 1;
            }
            if digits == 0 || dots > 1 {
                return false;
            }
        } else if b.is_ascii_alphabetic() {
            // Operator keyword: whitelist only.
            let start = i;
            while i < n && !is_delim_or_ws(data[i]) {
                i += 1;
            }
            if !matches!(&data[start..i], b"q" | b"Q" | b"cm" | b"Do") {
                return false;
            }
        } else {
            return false;
        }
    }
    true
}
```

And in `probe_inner`, REPLACE:

```rust
        let data = self.doc.get_page_content(page_id).ok()?;
        let draw = analyze_content(&Content::decode(&data).ok()?.operations)?;
```

with:

```rust
        let data = self.doc.get_page_content(page_id).ok()?;
        if !content_tokens_are_plain(&data) {
            return None;
        }
        let draw = analyze_content(&Content::decode(&data).ok()?.operations)?;
```

Append to the existing `mod tests` in `source_pdf.rs`:

```rust
    #[test]
    fn lexer_accepts_plain_scan_tokens() {
        assert!(content_tokens_are_plain(b"q 612 0 0 792 0 0 cm /Im0 Do Q"));
        assert!(content_tokens_are_plain(b"  .5 -0.25 612. 0 0 792 cm\n/I_m-0.a Do "));
        assert!(content_tokens_are_plain(b"")); // empty is benign (no draw → probe rejects later)
    }

    #[test]
    fn lexer_rejects_everything_else() {
        assert!(!content_tokens_are_plain(b"q 1 0 0 1 0 0 cm /Im0 Do Q % comment"));
        assert!(!content_tokens_are_plain(b"(string) Tj"));
        assert!(!content_tokens_are_plain(b"q /Im0 Do Q BI /W 8 ID xxxx EI"));
        assert!(!content_tokens_are_plain(b"/Im#41 Do")); // name escape
        assert!(!content_tokens_are_plain(b"<</K 1>> gs"));
        assert!(!content_tokens_are_plain(b"612 0.0.1 cm")); // two dots
        assert!(!content_tokens_are_plain(b"q \x80 Q")); // raw garbage byte
        assert!(!content_tokens_are_plain(b"0 0 100 100 re f"));
    }
```

### A.3 — Task 3: four extra reject fixtures

Add knobs to `Fx`: `offset: Option<(f32, f32)>` (adds e/f to the `cm`),
`flip_vertical: bool` (emits `cm dw 0 0 -dh 0 dh` — the mirrored form),
`raw_content_suffix: Option<&'static [u8]>` (bytes appended verbatim to
the encoded content stream). In `save_fx`: build the `cm` operands from
`offset`/`flip_vertical`; after `content.encode().unwrap()`, append the
suffix bytes if set, then wrap in the stream. Defaults: `None`, `false`,
`None`. Add tests:

```rust
#[test]
fn rejects_translated_draw() {
    let pr = probe(&Fx { offset: Some((-72.0, 0.0)), ..Fx::default() });
    assert!(pr.passthrough.is_none());
}

#[test]
fn rejects_flipped_draw() {
    let pr = probe(&Fx { flip_vertical: true, ..Fx::default() });
    assert!(pr.passthrough.is_none());
}

#[test]
fn rejects_oversized_draw() {
    // 2× page on both axes: old area gate would have passed this.
    let pr = probe(&Fx { cover: 2.0, ..Fx::default() });
    assert!(pr.passthrough.is_none());
}

#[test]
fn rejects_garbage_suffix_in_content() {
    // Valid scan prefix + a token lopdf cannot parse + drawing ops:
    // lopdf's truncating parser would only surface the prefix — the
    // pre-lexer must reject the whole page.
    let pr = probe(&Fx {
        raw_content_suffix: Some(b" \x80 0 0 100 100 re f"),
        ..Fx::default()
    });
    assert!(pr.passthrough.is_none());
}
```

Expected test count for `--test passthrough_probe` becomes **20 passed**
(16 planned + these 4).

### A.4 — bookkeeping

Task 1's commit command originally said to stage `Cargo.lock`; it is
gitignored in this repo (already corrected inline). Tasks 1 executed as
`a6a4122` + review hardening `4fcf055` (strict operator arity in
`analyze_content`, non-commuting composition test — keep both when
editing this file in later tasks).

---

## Amendment B (2026-06-13, post-Task-2 reviews) — Task 3 fixtures, Task 6 belt, Task 7 gate

Task 2 review (two rounds, adversarial fixtures) closed seven false-ACCEPT
vectors directly in `source_pdf.rs` (commits `b4f2aec`, `f8fb9aa`): annotation
reject, multi-stream `\n` joining, lexer narrowed strictly inside lopdf's
parser (no NUL/FF, ≤12-digit numbers), non-finite CTM guard, Rotate via
`num()` + parent-chain poison (`parent_chain_is_safe`, cap 48 strictly inside
`inherited()`'s 64), `/OC`+`/Interpolate`+DCT-DecodeParms rejects, 1 %
position tolerance. Consequences for later tasks:

### B.1 — Task 3: five more reject fixtures (expected count becomes 25)

New `Fx` knobs: `annots: bool` (adds one Square annotation object to the
page), `content_streams: Option<Vec<&'static [u8]>>` (page `/Contents`
becomes an ARRAY of separate raw streams), `raw_content: Option<&'static
[u8]>` (single raw stream replacing the built ops). Builder logic:

```rust
// cm operands honor offset/flip (Amendment A.3 knobs):
let (a_val, d_val, e_val, f_val) = if fx.flip_vertical {
    (dw, -dh, 0.0, dh)
} else {
    let (ox, oy) = fx.offset.unwrap_or((0.0, 0.0));
    (dw, dh, ox, oy)
};
// content: explicit streams > raw_content > built ops (+ optional suffix)
let content_ids: Vec<ObjectId> = if let Some(streams) = &fx.content_streams {
    streams
        .iter()
        .map(|b| doc.add_object(Stream::new(dictionary! {}, b.to_vec()).with_compress(false)))
        .collect()
} else {
    let bytes = if let Some(raw) = fx.raw_content {
        raw.to_vec()
    } else {
        let mut b = content.encode().unwrap();
        if let Some(sfx) = fx.raw_content_suffix {
            b.extend_from_slice(sfx);
        }
        b
    };
    vec![doc.add_object(Stream::new(dictionary! {}, bytes).with_compress(false))]
};
if fx.annots {
    let annot = doc.add_object(dictionary! {
        "Type" => "Annot", "Subtype" => "Square",
        "Rect" => vec![0.into(), 0.into(), 10.into(), 10.into()],
    });
    page.set("Annots", vec![Object::Reference(annot)]);
}
// Contents: single ref or array of refs
if content_ids.len() == 1 {
    page.set("Contents", content_ids[0]);
} else {
    page.set(
        "Contents",
        content_ids.iter().map(|id| Object::Reference(*id)).collect::<Vec<_>>(),
    );
}
```

The five tests (all expect `passthrough.is_none()`):

```rust
#[test]
fn rejects_annotated_page() {
    // pdfium renders annotations into the raster; passthrough would
    // drop them (stamps, signature widgets).
    let pr = probe(&Fx { annots: true, ..Fx::default() });
    assert!(pr.passthrough.is_none());
}

#[test]
fn rejects_token_split_across_streams() {
    // pdfium puts whitespace between /Contents streams; gluing them
    // raw would weld "21"+"6" into one 216. With \n-joining the cm
    // gets 8 operands and the analyzer rejects.
    let pr = probe(&Fx {
        content_streams: Some(vec![b"q 144 0 0 21", b"6 0 0 cm /Im0 Do Q"]),
        ..Fx::default()
    });
    assert!(pr.passthrough.is_none());
}

#[test]
fn rejects_nan_matrix() {
    // Four ~1e12 scales overflow f32 to inf, then a zero matrix makes
    // NaN — reject-on-true gates are blind to NaN without the guard.
    let pr = probe(&Fx {
        raw_content: Some(
            b"999999999999 0 0 999999999999 0 0 cm \
              999999999999 0 0 999999999999 0 0 cm \
              999999999999 0 0 999999999999 0 0 cm \
              999999999999 0 0 999999999999 0 0 cm \
              0 0 0 0 0 0 cm /Im0 Do",
        ),
        ..Fx::default()
    });
    assert!(pr.passthrough.is_none());
}

#[test]
fn rejects_huge_integer_lexer_cap() {
    // >12-digit numeral: lopdf's i64 parser would halt and silently
    // truncate, hiding the trailing draw — the lexer rejects first.
    let pr = probe(&Fx {
        raw_content: Some(
            b"q 144 0 0 216 0 0 cm /Im0 Do Q 12345678901234567890 0 0 9 0 0 cm /Im0 Do",
        ),
        ..Fx::default()
    });
    assert!(pr.passthrough.is_none());
}

#[test]
fn rejects_formfeed_hidden_draw() {
    // Form feed is whitespace to pdfium but halts lopdf's parser —
    // everything after it would be invisible to the whitelist.
    let pr = probe(&Fx {
        raw_content: Some(b"q 144 0 0 216 0 0 cm /Im0 Do Q\x0c9 0 0 9 0 0 cm /Im0 Do"),
        ..Fx::default()
    });
    assert!(pr.passthrough.is_none());
}
```

### B.2 — Task 6: aspect-ratio belt in the passthrough arm

Review issue: lopdf↔pdfium page-index skew (typeless page-tree nodes) or any
residual box divergence would pair the wrong image with a raster. Belt in
`encode_page_opts`'s passthrough arm — replace the plain `if let Some(p) = …
return passthrough` with:

```rust
    if opts.allow_passthrough {
        if let Some(p) = probe.and_then(|p| p.passthrough.as_ref()) {
            // Belt against residual probe/pdfium divergence (page order,
            // box weirdness): the source image must match the raster's
            // aspect within 4 %. Legit anisotropic-pixel scans (some
            // faxes) false-reject here — they just re-encode.
            let raster_aspect = raster.width as f32 / raster.height.max(1) as f32;
            let img_aspect = p.width as f32 / p.height.max(1) as f32;
            if (raster_aspect - img_aspect).abs() <= 0.04 * raster_aspect {
                return Ok(EncodedPage {
                    width: raster.width,
                    height: raster.height,
                    image: PageImage::Passthrough(p.clone()),
                });
            }
        }
    }
```

Extra `embed_tests` case:

```rust
    #[test]
    fn passthrough_belt_rejects_aspect_mismatch() {
        // Raster 600×900 (0.667) vs image 300×150 (2.0): belt trips,
        // page re-encodes instead of embedding the wrong image.
        let mut pt = fake_pt();
        pt.width = 300;
        pt.height = 150;
        let enc = encode_page_opts(
            &raster(600, 900),
            Some(&probe_with(Some(pt), None)),
            &EmbedOptions::default(),
            300,
        )
        .unwrap();
        assert!(matches!(enc.image, PageImage::Jpeg { .. }));
    }
```

(Task 6's `embed_tests` count becomes 8.)

### B.3 — Task 7: page-count equality gate

After constructing `source` in `ocr_all_pages` (PDF branch), add:

```rust
            // Index-skew belt: if lopdf and pdfium disagree on page
            // count, the probe could pair the wrong source page with a
            // raster — disable passthrough for the whole document.
            let source = source.filter(|s| s.page_count() == total);
```

### B.4 — worktree provisioning note

`vendor/pdfium/bin/pdfium.dll` and the `models/*.onnx` files were copied into
this worktree (controller action, untracked) — the full integration suite now
runs here: expect `104 passed, 1 ignored` before Task 3, with the ignored
test being the pre-existing `muna_madan_searchable_pdf_has_extractable_nepali`.

---

## Amendment C (2026-06-13, post-Task-7 end-to-end) — DPI hint decoupled from passthrough gate

Task 7's first end-to-end run on `MunaMadan_full.pdf` exposed that the feature
was **inert on real scans**: 0/51 pages passed through and output was ~27 MB.
Root cause (a genuine Task 2/6 design gap, not a Task 7 wiring bug):

- Real scanners bake a **sub-degree deskew rotation** into each page's image
  draw CTM (observed b≈-2.6, c≈4.1 → ~0.3°). The strict axis-alignment gate
  (`b.abs() > 0.5 || c.abs() > 0.5`) **correctly** rejects byte-passthrough —
  the writer redraws axis-aligned and cannot reproduce the rotation, so a
  verbatim copy would misalign the invisible text layer (OCR'd from pdfium's
  deskewed raster) against the still-skewed image.
- BUT `effective_dpi` was computed only *after* that gate, so a rejected page
  also forfeited its DPI hint → the re-encode fell back to the full 300-DPI
  OCR raster instead of the scan's true ~96 DPI → ~10× size blowup.

**Fix (`2d00888`):** in `probe_inner`, compute `effective_dpi` independently
of passthrough eligibility, from **rotation-robust drawn-size magnitudes**
(`sqrt(a²+b²)`, `sqrt(c²+d²)`), gated only on the image being substantially the
page (½–1½× per axis). Skew/offset/flip now zero out `passthrough` only — they
no longer early-return the default probe — so a skewed scan returns
`PageProbe { passthrough: None, effective_dpi: Some(~96) }` and re-encodes at
its true resolution (small, faithful, **and** correctly aligned). The strict
passthrough gate is unchanged; byte-passthrough still requires a genuinely
axis-aligned full-page draw.

**Explicitly NOT done:** widening the passthrough skew tolerance. That would
drop the baked-in rotation and misalign the text layer. Achieving verbatim
passthrough for deskewed scans would require carrying the draw CTM into
`PassthroughImage` and re-applying it in the writer's content stream — a
worthwhile **future** optimization (it would get MunaMadan to ~1.9 MB instead
of the re-encode's larger-but-still-slim result), out of scope for Phase 1.

Test impact (`passthrough_probe.rs`, now 27 tests): `rejects_partial_coverage_but_keeps_no_hint`
became `partial_coverage_rejects_passthrough_but_keeps_hint` (partial coverage
fails the ±3 % size gate but keeps a DPI hint); added
`skewed_full_page_keeps_hint_but_rejects_passthrough` + an `Fx.skew` knob.

**Task 8 consequence:** acceptance MUST assert on output **size**, not just text
extractability — this regression would have shipped silently otherwise. The
realistic target is re-encode-at-source-DPI grayscale (well under the 34 MB
status quo and under ~5 MB), not the 1.9 MB a full passthrough would yield.

---

## Plan self-review (done at authoring time)

- **Spec coverage:** 1a passthrough criteria (single image ≥95 %, DCT/CCITT/Flate, DeviceGray/RGB, lopdf read-only, fallback-on-anomaly) → Tasks 1–4; rotation/CropBox safety (not in spec, found during planning — raster is post-rotation) → Tasks 2–3; 1b grayscale → Task 5; DPI cap incl. image-input exemption → Task 6; quality knob + CLI → Tasks 6–7; size/parity/alignment testing → Tasks 4, 7, 8.
- **Placeholders:** none; every step carries code or exact commands.
- **Type consistency:** `PtFilter`/`CcittParms`/`PassthroughImage`/`PageProbe` (Tasks 1–2) are consumed with identical shapes in Tasks 3, 4, 6; `EncodedPage { width, height, image }` consistent across Tasks 4–7; `encode_page_opts(raster, probe, opts, ocr_dpi)` signature identical in Tasks 6 and 7.
- **Known seams called out inline:** lopdf `Dictionary::iter()` key type (two isolated call sites), `DocPage` construction pattern (pointer to existing test).
