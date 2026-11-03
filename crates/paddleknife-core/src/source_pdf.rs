//! Read-only inspection of the *source* PDF for the searchable-PDF
//! writer: can a page's original compressed image be embedded verbatim
//! (passthrough), and what is the page's effective scan DPI?
//!
//! A page qualifies for passthrough when it is a plain scan page:
//! exactly one image XObject, drawn axis-aligned and unflipped,
//! matching the MediaBox within 3 % per axis (size and position),
//! no rotation, no CropBox surprises, and a self-contained
//! encoding (DCT / raw Flate / CCITT with copyable DecodeParms,
//! DeviceGray or DeviceRGB). Anything else falls back to re-encoding —
//! the probe is deliberately strict because the fallback is cheap.
//!
//! The fallback is only cheap **while the DPI hint survives**: a
//! re-encode with no hint embeds the raster at the full OCR DPI, which
//! for a 96-DPI screenshot rasterized at 300 is 9.8× the source's pixels
//! (a 923 KB document came out at 4.2 MB). So the probe runs as two
//! stages — a geometry stage that feeds the hint, and the strict
//! passthrough gates on top — and a passthrough-only rejection (rotation,
//! CropBox, skew, encoding, a link or hidden annotation) never discards
//! the hint. A *visible* annotation does, on purpose: pdfium draws it
//! over the scan at the full render DPI, so the scan's resolution no
//! longer bounds what the raster holds. For the same reason the hint
//! follows an image's `/SMask` or `/Mask` when that is the sharper of the
//! two ([`MaskScale`]).
//!
//! The source is untrusted input, and the probe runs on every page before
//! any cheap gate, so its work is bounded up front: a content stream is
//! size-capped before a byte is inflated and again while inflating
//! (`MAX_CONTENT_RAW_BYTES`, `MAX_CONTENT_DECODED_BYTES`).
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
    /// JBIG2, a baked-in deskew, a rotated page). `None` when the page
    /// can't be established as a single image substantially covering the
    /// box (½–1½× per axis), and when a visible annotation is drawn over
    /// it (the raster then holds more than the scan). Measured from the
    /// image's own pixel grid *and* any `/SMask` or `/Mask` it carries
    /// (see [`MaskScale`]).
    pub effective_dpi: Option<f32>,
    /// What masks do to the resolution of the page's images, over every
    /// image XObject in its resources. Only the writer's pdfium-side
    /// backstop (`RasterPage::native_dpi`) needs it: pdfium measures an
    /// image by its own pixel grid, and a mask is not a page object it
    /// could measure. `effective_dpi` already accounts for it.
    pub masks: MaskScale,
}

/// How much finer than their own pixel grid a page's images render.
///
/// An image's `/SMask` or explicit `/Mask` is a separate image of its own
/// size, and pdfium draws it at the full render resolution. MRC
/// compressors store a low-resolution colour image under a
/// full-resolution mask that carries the glyph edges, so capping the
/// embed at the colour image's resolution would blur exactly the text.
/// A colour-key `/Mask` (an array) adds no resolution and is ignored.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum MaskScale {
    /// No mask renders finer than its image.
    #[default]
    Unmasked,
    /// The finest mask renders this many times finer than its image, on
    /// its finer axis (> 1).
    Finer(f32),
    /// A mask whose resolution can't be established: an unreadable mask
    /// size, or an `ExtGState` soft-mask group, whose content (vectors
    /// included) pdfium renders at full resolution.
    Unknown,
}

impl MaskScale {
    /// `ratio` = mask pixels ÷ image pixels along one axis.
    fn of_ratio(ratio: f32) -> Self {
        if ratio > 1.0 {
            Self::Finer(ratio)
        } else {
            Self::Unmasked
        }
    }

    /// The finer of two scales; `Unknown` wins.
    fn max(self, other: Self) -> Self {
        match (self, other) {
            (Self::Unknown, _) | (_, Self::Unknown) => Self::Unknown,
            (Self::Finer(a), Self::Finer(b)) => Self::Finer(a.max(b)),
            (Self::Finer(a), Self::Unmasked) | (Self::Unmasked, Self::Finer(a)) => Self::Finer(a),
            (Self::Unmasked, Self::Unmasked) => Self::Unmasked,
        }
    }

    /// Correct a resolution measured from image pixel grids alone.
    /// `Unknown` drops it: no measurement beats an underestimate, which
    /// would blur what the mask draws.
    pub fn apply(self, dpi: Option<f32>) -> Option<f32> {
        match self {
            Self::Unmasked => dpi,
            Self::Finer(f) => dpi.map(|d| d * f),
            Self::Unknown => None,
        }
    }
}

/// One full-page image draw found in a content stream.
#[derive(Debug, PartialEq)]
struct ImageDraw {
    name: Vec<u8>,
    ctm: [f32; 6],
}

/// Why a probe stage declined. Only ever logged — the probe's callers
/// act on the `Option`s in [`PageProbe`], never on the wording.
type Decline = &'static str;

/// The page's single image draw, measured: everything known once the
/// geometry stage passes. `ctm` is in unrotated user space (the MediaBox's
/// space), so none of this depends on /Rotate or the CropBox.
struct ImageGeom<'a> {
    mb: [f32; 4],
    ctm: [f32; 6],
    iw: u32,
    ih: u32,
    stream: &'a Stream,
}

impl ImageGeom<'_> {
    /// The DPI hint (which caps the re-encode resolution) is INDEPENDENT
    /// of byte-passthrough eligibility: a page can be rotated, carry a
    /// link, or be too skewed or offset to copy verbatim yet still tell us
    /// its scan resolution, letting the re-encode store it at that resolution
    /// instead of the full OCR DPI. Use the rotation-robust drawn-size
    /// magnitudes (|x-basis|, |y-basis|) so the sub-degree deskew baked
    /// into most real scans' CTMs doesn't perturb the estimate. Gate only
    /// on the image being substantially the page (½–1½× per axis), which
    /// excludes thumbnails and oversized/garbage draws.
    fn effective_dpi(&self) -> Option<f32> {
        let [a, b, c, d, _, _] = self.ctm;
        let (mb_w, mb_h) = (self.mb[2] - self.mb[0], self.mb[3] - self.mb[1]);
        let dmw = (a * a + b * b).sqrt();
        let dmh = (c * c + d * d).sqrt();
        let covers =
            dmw >= 0.5 * mb_w && dmw <= 1.5 * mb_w && dmh >= 0.5 * mb_h && dmh <= 1.5 * mb_h;
        covers.then(|| (self.iw as f32 / (dmw / 72.0)).max(self.ih as f32 / (dmh / 72.0)))
    }
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
            "q" => {
                if !o.operands.is_empty() {
                    return None;
                }
                stack.push(ctm)
            }
            "Q" => {
                if !o.operands.is_empty() {
                    return None;
                }
                ctm = stack.pop()?
            }
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
                if o.operands.len() != 1 {
                    return None;
                }
                let name = o.operands[0].as_name().ok()?.to_vec();
                draw = Some(ImageDraw { name, ctm });
            }
            _ => return None,
        }
    }
    draw
}

/// lopdf's content parser stops at the first token it can't parse and
/// silently returns the prefix, so a malformed-but-renderable stream
/// could hide drawing operators from the whitelist. Gate: the raw
/// bytes must consist ONLY of benign tokens — plain numbers, plain
/// /Names, and the four whitelisted operators — before lopdf parses
/// them. Anything else (strings, arrays, dicts, comments, hex, inline
/// images, name escapes) rejects the page.
///
/// Invariant: every stream this function accepts must be fully parseable
/// to completion by lopdf's content parser — the lexer stays strictly
/// NARROWER than lopdf, never wider.
fn content_tokens_are_plain(data: &[u8]) -> bool {
    fn is_ws(b: u8) -> bool {
        // NUL (\0) and form-feed (\x0c) are intentionally excluded:
        // lopdf's parser halts at those bytes and silently truncates the
        // op list, so they must fall through to the reject branch below.
        matches!(b, b'\t' | b'\n' | b'\r' | b' ')
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
            // Limit the INTEGER part to 12 digits: lopdf's parser overflows
            // on >i64 integers and silently truncates — huge numbers must
            // be rejected. The fraction is deliberately uncapped: lopdf
            // tries `real` before `integer`, and reals go through
            // f32::from_str, which cannot fail on digit count. Producers
            // that print page sizes at full f64 precision are common
            // (`951.6310461192352 0 0 749.9062617173895 0 0 cm` — 96 DPI
            // after a PNG pHYs round trip); capping total digits rejected
            // those pages for no safety gain.
            if data[i] == b'+' || data[i] == b'-' {
                i += 1;
            }
            let (mut int_digits, mut frac_digits, mut dots) = (0u32, 0u32, 0u32);
            while i < n && !is_delim_or_ws(data[i]) {
                match data[i] {
                    b'0'..=b'9' if dots == 0 => int_digits += 1,
                    b'0'..=b'9' => frac_digits += 1,
                    b'.' => dots += 1,
                    _ => return false,
                }
                i += 1;
            }
            if int_digits + frac_digits == 0 || dots > 1 || int_digits > 12 {
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

/// Raw (still-encoded) content bytes a page may have, summed over its
/// `/Contents` streams, before the probe declines it unread. A
/// single-image page's content is `q W 0 0 H X Y cm /Im0 Do Q` — under
/// 100 bytes, and under 1 KiB even with full-precision operands and a few
/// nested `q`/`cm` pairs — so anything this big is not a plain scan page,
/// and the probe never inflates it to find out.
const MAX_CONTENT_RAW_BYTES: usize = 64 * 1024;

/// Decoded content bytes a page may have, summed over its `/Contents`
/// streams. lopdf's `Stream::decompressed_content` inflates without any
/// bound (a 64 KiB zlib bomb is ~66 MB, on every page), so the probe
/// decodes through [`bounded_content`], which stops reading here.
const MAX_CONTENT_DECODED_BYTES: usize = 256 * 1024;

/// Longest `/Filter` chain a content stream may declare. Real producers
/// use at most two (`[/ASCII85Decode /FlateDecode]`); each step's output
/// is bounded, but a thousand-step chain would still be a thousand steps.
const MAX_CONTENT_FILTERS: usize = 2;

const CONTENT_TOO_LARGE: Decline = "content stream too large for a single-image page";
const CONTENT_TOO_LARGE_DECODED: Decline = "content stream decodes past the size cap";

/// Annotation flag bits (PDF 32000-1:2008 §12.5.3, bits 2 and 6).
/// Either one keeps an annotation off screen, and pdfium honors both
/// when rendering for display.
const ANNOT_FLAG_HIDDEN: i64 = 1 << 1;
const ANNOT_FLAG_NO_VIEW: i64 = 1 << 5;

/// Decode one content stream, producing at most `limit` bytes.
///
/// Applies the filters lopdf's `decompressed_content` would, minus the
/// ones whose output can't be bounded cheaply: no filter, `FlateDecode`
/// (inflated through a byte limit), and `ASCII85Decode` (at most 4 output
/// bytes per input byte, so the raw cap already bounds it). LZW (weezl
/// decodes with no output limit), predictors (`/DecodeParms`) and every
/// other filter decline — plain scan pages don't use them, and a decline
/// costs only the DPI hint, which pdfium's `RasterPage::native_dpi` still
/// backs up.
fn bounded_content(stream: &Stream, limit: usize) -> std::result::Result<Vec<u8>, Decline> {
    if stream
        .dict
        .get(b"DecodeParms")
        .is_ok_and(|p| !matches!(p, Object::Null))
    {
        return Err("content stream has /DecodeParms");
    }
    let filters = match stream.dict.get(b"Filter") {
        Err(_) | Ok(Object::Null) => Vec::new(),
        Ok(_) => stream
            .filters()
            .map_err(|_| "content stream /Filter is malformed")?,
    };
    if filters.len() > MAX_CONTENT_FILTERS {
        return Err("content stream filter chain is too long");
    }
    let mut data = stream.content.clone();
    for filter in filters {
        data = match filter {
            b"FlateDecode" => inflate_bounded(&data, limit)?,
            b"ASCII85Decode" => {
                let mut dict = Dictionary::new();
                dict.set("Filter", Object::Name(b"ASCII85Decode".to_vec()));
                Stream::new(dict, data)
                    .decompressed_content()
                    .map_err(|_| "content stream is not valid ASCII85")?
            }
            _ => return Err("content stream filter is not Flate or ASCII85"),
        };
    }
    if data.len() > limit {
        return Err(CONTENT_TOO_LARGE_DECODED);
    }
    Ok(data)
}

/// zlib-inflate `input`, refusing to produce more than `limit` bytes.
///
/// Like lopdf (and pdfium), a stream that breaks off mid-way keeps the
/// bytes decoded so far, and a zlib header over raw deflate with a bad
/// checksum falls back to raw deflate — so the probe reads what it read
/// through lopdf before, only bounded.
fn inflate_bounded(input: &[u8], limit: usize) -> std::result::Result<Vec<u8>, Decline> {
    use flate2::read::{DeflateDecoder, ZlibDecoder};
    use std::io::Read as _;
    // Read one byte past the limit: reaching it means "too big" — never a
    // silent truncation that could hide operators from the lexer.
    let cap = (limit as u64).saturating_add(1);
    let mut out = Vec::new();
    if ZlibDecoder::new(input)
        .take(cap)
        .read_to_end(&mut out)
        .is_err()
        && out.is_empty()
        && input.len() > 2
    {
        let _ = DeflateDecoder::new(&input[2..])
            .take(cap)
            .read_to_end(&mut out);
    }
    if out.len() > limit {
        return Err(CONTENT_TOO_LARGE_DECODED);
    }
    Ok(out)
}

/// A page box every downstream gate can reason about: finite corners,
/// finite extents, at least 1 pt per axis. The gates reject on `>`
/// comparisons, which NaN always fails — and inf − inf is NaN — so a box
/// with a non-finite corner would otherwise pass every size and offset
/// check (a `[1e40 …]` box went through to passthrough with a misaligned
/// text layer). Finite corners can still overflow to an infinite extent
/// (`[-3e38 0 3e38 792]`), hence the extent check.
fn is_usable_box(r: [f32; 4]) -> bool {
    let (w, h) = (r[2] - r[0], r[3] - r[1]);
    r.iter().all(|v| v.is_finite()) && w.is_finite() && h.is_finite() && w >= 1.0 && h >= 1.0
}

/// Largest image `/Width` or `/Height` pdfium decodes (its
/// `kMaxImageDimension`). A larger image renders as nothing, so a
/// dimension past it describes nothing the raster shows, and every value
/// up to it fits the `i32` the searchable-PDF writer emits.
const MAX_IMAGE_DIMENSION: u32 = 0x1_FFFF;

/// An image `/Width` or `/Height` as a pixel count. A plain `as u32`
/// would wrap 2³² + 128 to 128 and let a lying header pass as an
/// ordinary 128-pixel image; anything past [`MAX_IMAGE_DIMENSION`] would
/// reach the writer's `i32` fields negative.
fn pixel_dim(v: Option<i64>) -> std::result::Result<u32, Decline> {
    let v = v.ok_or("image /Width or /Height missing")?;
    if v <= 0 {
        return Err("image /Width or /Height not positive");
    }
    u32::try_from(v)
        .ok()
        .filter(|&d| d <= MAX_IMAGE_DIMENSION)
        .ok_or("image /Width or /Height exceeds what pdfium renders")
}

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

    /// Test-only: wrap an in-memory document.
    #[cfg(test)]
    fn from_document(doc: Document) -> Self {
        let pages: Vec<ObjectId> = doc.get_pages().into_values().collect();
        Self { doc, pages }
    }

    /// True when the page's Parent chain is fully visible to
    /// `inherited()`: every link is a reference, no cycles, and the
    /// chain is strictly shorter than the visibility cap — anything
    /// deeper (or weirder) could hide a Rotate/CropBox that pdfium
    /// would honor, so the caller must reject the page.
    ///
    /// Invariant: this cap (48) stays strictly below `inherited()`'s
    /// walk cap (64), so poison always fires before visibility runs
    /// out. Real documents nest a handful of levels at most.
    fn parent_chain_is_safe(&self, page_id: ObjectId) -> bool {
        let mut depth = 0;
        let mut cursor = page_id;
        loop {
            let parent = match self
                .doc
                .get_dictionary(cursor)
                .ok()
                .and_then(|d| d.get(b"Parent").ok())
            {
                Some(p) => p,
                None => return true, // chain ended in visible range
            };
            // pdfium follows direct-dict Parents; inherited() cannot —
            // poison rather than mistake "invisible" for "absent".
            let Ok(p) = parent.as_reference() else {
                return false;
            };
            depth += 1;
            if depth > 48 {
                return false; // over-deep or cyclic
            }
            cursor = p;
        }
    }

    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// Probe page `idx` (0-based). Never fails: any anomaly degrades to
    /// "re-encode", and — only if the page's image draw can't be
    /// established at all — "no DPI hint".
    ///
    /// The two answers are decided by separate stages on purpose. The
    /// geometry stage ([`Self::single_image_draw`]) asks "is this page one
    /// image, and how big is it drawn?", which is all the DPI hint needs.
    /// The passthrough stage ([`Self::passthrough_gates`] + byte-level
    /// checks) asks the much stricter "is a verbatim copy pixel-identical
    /// to what pdfium rendered?". A page that fails the second question
    /// still has a perfectly well-defined scan resolution, and losing it
    /// is expensive: the re-encode path would embed the full OCR-DPI
    /// raster (9.8× the pixels of a 96-DPI screenshot at 300 DPI).
    ///
    /// A visible annotation vetoes the hint (see
    /// [`Self::has_visible_annotations`]): pdfium draws stamps, signatures
    /// and filled form fields over the scan at the full render DPI, so the
    /// scan's resolution no longer bounds what the raster holds, and
    /// capping the embed at it would blur them.
    ///
    /// A mask sharper than its image raises the hint to the mask's
    /// resolution, and one whose resolution can't be read vetoes it (see
    /// [`MaskScale`]). [`PageProbe::masks`] carries the page-wide answer
    /// even when the geometry stage declines, for the writer's pdfium-side
    /// backstop.
    ///
    /// Every decline is logged at `debug` with its reason
    /// (`RUST_LOG=paddleknife_core::source_pdf=debug`).
    pub fn probe_page(&self, idx: usize) -> PageProbe {
        let page = idx + 1; // 1-based in logs, matching the progress bar
        let Some(&page_id) = self.pages.get(idx) else {
            tracing::debug!(page, "probe: page index out of range");
            return PageProbe::default();
        };
        let masks = self.page_mask_scale(page_id);
        let draw = match self.single_image_draw(page_id) {
            Ok(d) => d,
            Err(why) => {
                tracing::debug!(page, why, ?masks, "probe: no passthrough, no DPI hint");
                return PageProbe {
                    masks,
                    ..PageProbe::default()
                };
            }
        };
        let mut effective_dpi = draw.effective_dpi();
        if effective_dpi.is_none() {
            tracing::debug!(
                page,
                "probe: image does not substantially cover the page; no DPI hint"
            );
        } else if self.has_visible_annotations(page_id) {
            tracing::debug!(
                page,
                "probe: visible annotations are drawn over the scan; no DPI hint"
            );
            effective_dpi = None;
        } else {
            // Only the drawn image's own masks: the content whitelist has no
            // `gs`, so no ExtGState soft mask can be in effect here.
            let image_masks = self.image_mask_scale(draw.stream);
            effective_dpi = image_masks.apply(effective_dpi);
            if effective_dpi.is_none() {
                tracing::debug!(page, "probe: image mask of unknown resolution; no DPI hint");
            }
        }
        let passthrough = match self
            .passthrough_gates(page_id, &draw)
            .and_then(|()| self.extract_image(draw.stream, draw.iw, draw.ih))
        {
            Ok(p) => Some(p),
            Err(why) => {
                tracing::debug!(page, why, ?effective_dpi, "probe: passthrough declined");
                None
            }
        };
        PageProbe {
            passthrough,
            effective_dpi,
            masks,
        }
    }

    /// `/Width` × `/Height` of an image (or mask) stream, bounded by
    /// [`pixel_dim`].
    fn image_dims(&self, s: &Stream) -> std::result::Result<(u32, u32), Decline> {
        let dim = |key: &[u8]| {
            pixel_dim(
                s.dict
                    .get(key)
                    .ok()
                    .and_then(|o| self.resolve(o).as_i64().ok()),
            )
        };
        Ok((dim(b"Width")?, dim(b"Height")?))
    }

    /// How much finer than its own pixel grid `image` renders, through its
    /// `/SMask` or stream `/Mask` (both are considered when both are set).
    fn image_mask_scale(&self, image: &Stream) -> MaskScale {
        let mut scale = MaskScale::Unmasked;
        for key in [b"SMask".as_slice(), b"Mask"] {
            let Some(mask) = image
                .dict
                .get(key)
                .ok()
                .and_then(|o| self.resolve(o).as_stream().ok())
            else {
                continue; // absent, or a colour-key array
            };
            let (Ok((iw, ih)), Ok((mw, mh))) = (self.image_dims(image), self.image_dims(mask))
            else {
                return MaskScale::Unknown;
            };
            let ratio = (mw as f32 / iw as f32).max(mh as f32 / ih as f32);
            scale = scale.max(MaskScale::of_ratio(ratio));
        }
        scale
    }

    /// [`Self::image_mask_scale`] over every image XObject in the page's
    /// resources, plus the soft-mask groups its `ExtGState`s can set.
    /// Form XObjects are not descended into: pdfium's native DPI already
    /// refuses a page that draws one.
    fn page_mask_scale(&self, page_id: ObjectId) -> MaskScale {
        let Some(resources) = self
            .inherited(page_id, b"Resources")
            .and_then(|r| r.as_dict().ok())
        else {
            return MaskScale::Unmasked;
        };
        let sub_dict = |key: &[u8]| {
            resources
                .get(key)
                .ok()
                .and_then(|o| self.resolve(o).as_dict().ok())
        };
        let soft_mask_group = sub_dict(b"ExtGState").is_some_and(|states| {
            states.iter().any(|(_, gs)| {
                self.resolve(gs)
                    .as_dict()
                    .ok()
                    .and_then(|gs| gs.get(b"SMask").ok())
                    .is_some_and(|m| self.resolve(m).as_dict().is_ok()) // `/None` is a name
            })
        });
        if soft_mask_group {
            return MaskScale::Unknown;
        }
        sub_dict(b"XObject")
            .into_iter()
            .flat_map(Dictionary::iter)
            .filter_map(|(_, x)| self.resolve(x).as_stream().ok())
            .filter(|s| {
                s.dict
                    .get(b"Subtype")
                    .and_then(Object::as_name)
                    .is_ok_and(|n| n == b"Image")
            })
            .fold(MaskScale::Unmasked, |acc, image| {
                acc.max(self.image_mask_scale(image))
            })
    }

    /// Does pdfium draw any of this page's annotations into the raster?
    ///
    /// `/Link` and `/Popup` never paint scan-like content (a link has no
    /// appearance to speak of; a popup only opens on demand), and the
    /// `/F` Hidden and NoView bits keep any annotation off screen.
    /// Everything else — stamps, signature widgets, filled form fields,
    /// ink, sticky-note icons, and entries with no `/Subtype` whose
    /// appearance pdfium may still draw — counts. `/Annots` is not
    /// inheritable; non-dictionary entries are skipped, as pdfium skips
    /// them.
    ///
    /// Stricter than this for passthrough: [`Self::passthrough_gates`]
    /// rejects *any* annotation, since a verbatim copy drops them all.
    fn has_visible_annotations(&self, page_id: ObjectId) -> bool {
        let Some(annots) = self
            .doc
            .get_dictionary(page_id)
            .ok()
            .and_then(|d| d.get(b"Annots").ok())
        else {
            return false;
        };
        let Ok(annots) = self.resolve(annots).as_array() else {
            return false;
        };
        annots.iter().any(|a| {
            let Ok(annot) = self.resolve(a).as_dict() else {
                return false;
            };
            let subtype = annot
                .get(b"Subtype")
                .ok()
                .and_then(|s| self.resolve(s).as_name().ok());
            if matches!(subtype, Some(b"Link" | b"Popup")) {
                return false;
            }
            let flags = annot
                .get(b"F")
                .ok()
                .and_then(|f| num(self.resolve(f)))
                .map_or(0, |f| f as i64);
            (flags & (ANNOT_FLAG_HIDDEN | ANNOT_FLAG_NO_VIEW)) == 0
        })
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
        for _ in 0..64 {
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

    /// Geometry stage: establish that the page is a single image draw and
    /// measure it. This is everything the DPI hint depends on — and
    /// nothing else. Gates that only matter for a verbatim byte copy
    /// (rotation, CropBox, annotations, axis alignment, encoding) live in
    /// [`Self::passthrough_gates`] / [`Self::extract_image`] so that
    /// failing them cannot take the hint down too.
    ///
    /// `inherited()` stops at the first Parent link it cannot follow, so
    /// on a chain [`Self::parent_chain_is_safe`] would poison it yields
    /// `None` (→ no geometry) rather than a farther ancestor's value —
    /// the hint needs no separate chain check.
    ///
    /// Content streams are size-capped before and while they are decoded
    /// (`MAX_CONTENT_RAW_BYTES`, `MAX_CONTENT_DECODED_BYTES`): this stage
    /// runs on every page of an untrusted file.
    fn single_image_draw(&self, page_id: ObjectId) -> std::result::Result<ImageGeom<'_>, Decline> {
        let mb = self
            .inherited(page_id, b"MediaBox")
            .and_then(|o| self.rect(o))
            .ok_or("MediaBox missing or malformed")?;
        if !is_usable_box(mb) {
            return Err("MediaBox is non-finite or degenerate");
        }

        // Exactly one XObject, and it must be an image (a Form XObject
        // could draw anything, including more images).
        let xobjs = self
            .inherited(page_id, b"Resources")
            .and_then(|r| r.as_dict().ok())
            .and_then(|r| r.get(b"XObject").ok())
            .and_then(|x| self.resolve(x).as_dict().ok())
            .ok_or("no /Resources /XObject dictionary")?;
        let mut images: Vec<(Vec<u8>, &Stream)> = Vec::new();
        for (name, val) in xobjs.iter() {
            let stream = self
                .resolve(val)
                .as_stream()
                .map_err(|_| "XObject is not a stream")?;
            let is_image = stream
                .dict
                .get(b"Subtype")
                .and_then(Object::as_name)
                .is_ok_and(|s| s == b"Image");
            if !is_image {
                return Err("non-image XObject in page resources");
            }
            images.push((name.clone(), stream));
        }
        if images.len() != 1 {
            return Err("page resources do not hold exactly one image");
        }
        let (img_name, stream) = (&images[0].0, images[0].1);

        // Content stream: pre-lex (lopdf's parser silently truncates at
        // unparseable tokens), then a single draw of that image.
        // pdfium inserts a whitespace byte between multiple /Contents
        // streams; lopdf's get_page_content glues them raw, which can
        // weld two tokens across the boundary into one. Join with \n
        // ourselves so the probe lexes what pdfium executes.
        //
        // Every byte here is attacker-controlled: bound the raw size
        // before inflating anything, and the decoded size while inflating.
        let mut data = Vec::new();
        let mut raw_len = 0usize;
        for (i, sid) in self.doc.get_page_contents(page_id).into_iter().enumerate() {
            let contents = self
                .doc
                .get_object(sid)
                .and_then(Object::as_stream)
                .map_err(|_| "content stream unreadable")?;
            raw_len = raw_len.saturating_add(contents.content.len());
            if raw_len > MAX_CONTENT_RAW_BYTES {
                return Err(CONTENT_TOO_LARGE);
            }
            if i > 0 {
                data.push(b'\n');
            }
            let budget = MAX_CONTENT_DECODED_BYTES
                .checked_sub(data.len())
                .ok_or(CONTENT_TOO_LARGE_DECODED)?;
            data.extend_from_slice(&bounded_content(contents, budget)?);
        }
        if !content_tokens_are_plain(&data) {
            return Err("content stream has tokens beyond `q Q cm Do`, numbers and names");
        }
        let ops = Content::decode(&data).map_err(|_| "content stream does not parse")?;
        let draw = analyze_content(&ops.operations).ok_or("content is not a single image draw")?;
        if &draw.name != img_name {
            return Err("drawn name is not the page's image");
        }
        // Reject-on-true gates downstream are blind to NaN (every
        // comparison is false) — refuse non-finite matrices outright.
        if !draw.ctm.iter().all(|v| v.is_finite()) {
            return Err("non-finite image matrix");
        }

        let (iw, ih) = self.image_dims(stream)?;

        Ok(ImageGeom {
            mb,
            ctm: draw.ctm,
            iw,
            ih,
            stream,
        })
    }

    /// Passthrough-only page gates: conditions under which a verbatim copy
    /// of the stored image would NOT reproduce what pdfium rendered. None
    /// of them says anything about the scan's resolution.
    fn passthrough_gates(
        &self,
        page_id: ObjectId,
        draw: &ImageGeom<'_>,
    ) -> std::result::Result<(), Decline> {
        // inherited() walks the Parent chain; anything it could fail to
        // see (a Rotate or CropBox pdfium would honor) must reject the
        // page, not pass silently.
        if !self.parent_chain_is_safe(page_id) {
            return Err("page-tree Parent chain not fully visible");
        }

        // Rotated pages never qualify: the OCR raster (and therefore the
        // text layer) is post-rotation, the stored image is pre-rotation.
        // pdfium coerces Real-typed /Rotate too, so parse numerically;
        // anything not ≡ 0 (mod 360) — including fractional — rejects.
        if let Some(o) = self.inherited(page_id, b"Rotate") {
            let r = num(o).ok_or("/Rotate is not a number")?;
            if r.rem_euclid(360.0) != 0.0 {
                return Err("page is rotated");
            }
        }

        // A CropBox that crops anything means pdfium rendered a sub-area
        // of the stored image — passthrough would show the uncropped scan.
        let mb = draw.mb;
        if let Some(cb) = self
            .inherited(page_id, b"CropBox")
            .and_then(|o| self.rect(o))
        {
            // Same NaN blindness as the MediaBox: refuse outright rather
            // than trust the `>` comparison below.
            if !is_usable_box(cb) {
                return Err("CropBox is non-finite or degenerate");
            }
            if mb.iter().zip(cb.iter()).any(|(m, c)| (m - c).abs() > 1.0) {
                return Err("CropBox crops the MediaBox");
            }
        }

        // Annotations render into pdfium's raster (FPDF_ANNOT is on in
        // our render config) but would vanish from a passthrough page —
        // stamps and signature widgets are common on scanned mail, so
        // any annotation disqualifies. /Annots is not inheritable.
        if let Some(a) = self
            .doc
            .get_dictionary(page_id)
            .ok()
            .and_then(|d| d.get(b"Annots").ok())
        {
            match self.resolve(a) {
                Object::Array(arr) if arr.is_empty() => {}
                _ => return Err("page has annotations"),
            }
        }

        // The writer re-draws the image axis-aligned at the page box, so
        // the source draw must already be axis-aligned (b≈c≈0 — there is
        // no skew for us to drop), unflipped (a,d>0), the right size
        // (±3 %), and at the origin (±1 %). Looser would misalign the
        // invisible text layer (OCR'd from the deskewed raster) against a
        // verbatim, still-skewed image.
        let [a, b, c, d, e, f] = draw.ctm;
        let (mb_w, mb_h) = (mb[2] - mb[0], mb[3] - mb[1]);
        if b.abs() > 0.5 || c.abs() > 0.5 {
            return Err("image draw is skewed or rotated");
        }
        if a <= 0.0 || d <= 0.0 {
            return Err("image draw is flipped");
        }
        if (a - mb_w).abs() > 0.03 * mb_w || (d - mb_h).abs() > 0.03 * mb_h {
            return Err("image draw does not match the page size (±3 %)");
        }
        if (e - mb[0]).abs() > 0.01 * mb_w || (f - mb[1]).abs() > 0.01 * mb_h {
            return Err("image draw is offset from the page origin (±1 %)");
        }
        Ok(())
    }

    /// Byte-level qualification: self-contained encoding only.
    fn extract_image(
        &self,
        s: &Stream,
        width: u32,
        height: u32,
    ) -> std::result::Result<PassthroughImage, Decline> {
        let d = &s.dict;
        for k in [
            b"SMask".as_slice(),
            b"Mask",
            b"Decode",
            b"OC",
            b"Interpolate",
        ] {
            if d.get(k).is_ok() {
                return Err("image has SMask/Mask/Decode/OC/Interpolate");
            }
        }
        if d.get(b"ImageMask")
            .ok()
            .and_then(|o| o.as_bool().ok())
            .unwrap_or(false)
        {
            return Err("image is a stencil mask");
        }
        let bpc = d
            .get(b"BitsPerComponent")
            .and_then(Object::as_i64)
            .map_err(|_| "image /BitsPerComponent missing")?;
        let gray = match d
            .get(b"ColorSpace")
            .ok()
            .and_then(|o| self.resolve(o).as_name().ok())
        {
            Some(b"DeviceGray") => true,
            Some(b"DeviceRGB") => false,
            // Indexed/ICC/Separation → re-encode
            _ => return Err("image colorspace is not DeviceGray/DeviceRGB"),
        };
        let fname: &[u8] = d
            .get(b"Filter")
            .ok()
            .and_then(|o| match self.resolve(o) {
                Object::Array(arr) if arr.len() == 1 => self.resolve(&arr[0]).as_name().ok(),
                o => o.as_name().ok(),
            })
            .ok_or("image /Filter missing or chained")?;
        let filter = match fname {
            b"DCTDecode" => {
                if bpc != 8 {
                    return Err("DCT image is not 8 bpc");
                }
                // /ColorTransform in DecodeParms changes how the JPEG
                // decodes — reject rather than plumb it through.
                if d.get(b"DecodeParms").is_ok() || d.get(b"DecodeParams").is_ok() {
                    return Err("DCT image has DecodeParms");
                }
                PtFilter::Dct
            }
            b"FlateDecode" => {
                // Predictors et al. would need DecodeParms plumbing —
                // raw Flate only in v1.
                if d.get(b"DecodeParms").is_ok() || d.get(b"DecodeParams").is_ok() {
                    return Err("Flate image has DecodeParms (predictor)");
                }
                if bpc != 8 && bpc != 1 {
                    return Err("Flate image is not 1 or 8 bpc");
                }
                PtFilter::Flate
            }
            b"CCITTFaxDecode" => {
                if bpc != 1 || !gray {
                    return Err("CCITT image is not 1-bpc gray");
                }
                PtFilter::Ccitt(
                    self.ccitt_parms(d)
                        .ok_or("CCITT DecodeParms has keys we would not re-emit")?,
                )
            }
            // JBIG2, JPX, … → re-encode (DPI hint kept)
            _ => return Err("image filter is not DCT/Flate/CCITT"),
        };
        Ok(PassthroughImage {
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
        if dp.iter().any(|(k, _)| !KNOWN.contains(&k.as_slice())) {
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
            op("re", vec![0.into(), 0.into(), 10.into(), 10.into()]),
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

    #[test]
    fn analyze_composition_order_scales_inner_translation() {
        // Outer 2× scale, inner (10, 20) translation: PDF applies the
        // most recent cm to points first, so the translation is scaled
        // by the outer matrix — e/f must come out (20, 40), not (10, 20).
        let ops = vec![
            cm(2.0, 0.0, 0.0, 2.0, 0.0, 0.0),
            cm(1.0, 0.0, 0.0, 1.0, 10.0, 20.0),
            do_im(b"Im0"),
        ];
        let d = analyze_content(&ops).expect("accepted");
        assert_eq!(d.ctm, [2.0, 0.0, 0.0, 2.0, 20.0, 40.0]);
    }

    #[test]
    fn analyze_rejects_malformed_operand_counts() {
        // Two names before Do: a stack-based interpreter would draw the
        // *last* one — strictness demands rejection, not guessing.
        let two_names = vec![
            cm(612.0, 0.0, 0.0, 792.0, 0.0, 0.0),
            op(
                "Do",
                vec![Object::Name(b"ImA".to_vec()), Object::Name(b"ImB".to_vec())],
            ),
        ];
        assert!(analyze_content(&two_names).is_none());
        // q with stray operand.
        let q_with_operand = vec![
            op("q", vec![Object::Integer(5)]),
            cm(612.0, 0.0, 0.0, 792.0, 0.0, 0.0),
            do_im(b"Im0"),
        ];
        assert!(analyze_content(&q_with_operand).is_none());
    }

    #[test]
    fn lexer_accepts_plain_scan_tokens() {
        assert!(content_tokens_are_plain(b"q 612 0 0 792 0 0 cm /Im0 Do Q"));
        assert!(content_tokens_are_plain(
            b"  .5 -0.25 612. 0 0 792 cm\n/I_m-0.a Do "
        ));
        assert!(content_tokens_are_plain(b"")); // empty is benign (no draw -> probe rejects later)
    }

    #[test]
    fn lexer_rejects_everything_else() {
        assert!(!content_tokens_are_plain(
            b"q 1 0 0 1 0 0 cm /Im0 Do Q % comment"
        ));
        assert!(!content_tokens_are_plain(b"(string) Tj"));
        assert!(!content_tokens_are_plain(b"q /Im0 Do Q BI /W 8 ID xxxx EI"));
        assert!(!content_tokens_are_plain(b"/Im#41 Do")); // name escape
        assert!(!content_tokens_are_plain(b"<</K 1>> gs"));
        assert!(!content_tokens_are_plain(b"612 0.0.1 cm")); // two dots
        assert!(!content_tokens_are_plain(b"q \x80 Q")); // raw garbage byte
        assert!(!content_tokens_are_plain(b"0 0 100 100 re f"));
    }

    /// Build a Document whose single page sits under `extra_parents`
    /// chained Pages nodes (page -> P1 -> ... -> Pn). Returns it with
    /// the page's ObjectId. Bare dicts only — parent_chain_is_safe
    /// never looks at anything else.
    fn doc_with_parent_chain(extra_parents: usize) -> (Document, ObjectId) {
        use lopdf::dictionary;
        let mut doc = Document::with_version("1.5");
        let mut ancestors: Vec<ObjectId> = Vec::new();
        for _ in 0..extra_parents {
            ancestors.push(doc.add_object(dictionary! { "Type" => "Pages" }));
        }
        // Link each node to the next one up; topmost has no Parent.
        for i in 0..ancestors.len().saturating_sub(1) {
            let next = ancestors[i + 1];
            if let Ok(Object::Dictionary(d)) = doc.get_object_mut(ancestors[i]) {
                d.set("Parent", next);
            }
        }
        let mut page = dictionary! { "Type" => "Page" };
        if let Some(first) = ancestors.first() {
            page.set("Parent", *first);
        }
        let page_id = doc.add_object(page);
        (doc, page_id)
    }

    #[test]
    fn parent_chain_poison_boundaries() {
        // 48 edges: still safe (strictly under inherited()'s 64 cap).
        let (doc, page) = doc_with_parent_chain(48);
        assert!(SourcePdf::from_document(doc).parent_chain_is_safe(page));
        // 49 edges: poisoned.
        let (doc, page) = doc_with_parent_chain(49);
        assert!(!SourcePdf::from_document(doc).parent_chain_is_safe(page));
    }

    #[test]
    fn parent_chain_poisons_direct_dict_and_cycles() {
        use lopdf::dictionary;
        // Direct-dict Parent: pdfium follows it, inherited() can't.
        let mut doc = Document::with_version("1.5");
        let page_id = doc.add_object(dictionary! {
            "Type" => "Page",
            "Parent" => dictionary! { "Type" => "Pages", "Rotate" => 90i64 },
        });
        assert!(!SourcePdf::from_document(doc).parent_chain_is_safe(page_id));
        // Cycle: A -> B -> A must terminate and poison.
        let mut doc = Document::with_version("1.5");
        let a = doc.new_object_id();
        let b = doc.add_object(dictionary! { "Type" => "Pages", "Parent" => a });
        doc.objects.insert(
            a,
            Object::Dictionary(dictionary! { "Type" => "Pages", "Parent" => b }),
        );
        let page_id = doc.add_object(dictionary! { "Type" => "Page", "Parent" => a });
        assert!(!SourcePdf::from_document(doc).parent_chain_is_safe(page_id));
    }

    #[test]
    fn lexer_is_narrower_than_lopdf() {
        // lopdf's parser halts at NUL/form-feed and overflows on huge
        // integers — silently truncating the op list. The lexer must
        // reject anything lopdf cannot parse to completion.
        assert!(!content_tokens_are_plain(b"q Q\x0cDo"));
        assert!(!content_tokens_are_plain(b"q\x00Q"));
        assert!(!content_tokens_are_plain(
            b"12345678901234567890 0 0 1 0 0 cm"
        ));
        assert!(content_tokens_are_plain(b"123456789012 0 0 1 0 0 cm")); // 12 digits ok
    }

    #[test]
    fn lexer_caps_the_integer_part_not_the_fraction() {
        // Full-f64-precision page sizes (17 significant digits) are what
        // real producers emit for a 96.012-DPI screenshot. The old cap
        // counted ALL digits and rejected these pages outright.
        let long_fraction: &[u8] =
            b"q\n951.6310461192352 0 0 749.9062617173895 0 0 cm\n/Im0 Do\nQ\n";
        assert!(content_tokens_are_plain(long_fraction));
        assert!(content_tokens_are_plain(
            b".00000000000000000001 0 0 1 0 0 cm"
        ));

        // The invariant the lexer protects: whatever it accepts, lopdf
        // must parse to completion. Reals go through f32::from_str, so the
        // fraction length cannot truncate the op list.
        let ops = Content::decode(long_fraction)
            .expect("lopdf parses")
            .operations;
        assert_eq!(ops.len(), 4, "q cm Do Q — nothing dropped: {ops:?}");
        let d = analyze_content(&ops).expect("single draw");
        assert!((d.ctm[0] - 951.631).abs() < 0.01 && (d.ctm[3] - 749.906).abs() < 0.01);

        // The integer part stays capped whether or not a dot follows.
        assert!(!content_tokens_are_plain(b"1234567890123 0 0 1 0 0 cm"));
        assert!(!content_tokens_are_plain(b"1234567890123.5 0 0 1 0 0 cm"));
        assert!(!content_tokens_are_plain(b". 0 0 1 0 0 cm")); // bare dot
    }

    // ---- untrusted-input hardening -----------------------------------

    const DRAW: &[u8] = b"q 144 0 0 216 0 0 cm /Im0 Do Q";

    fn zlib(data: &[u8]) -> Vec<u8> {
        use flate2::{write::ZlibEncoder, Compression};
        use std::io::Write as _;
        let mut enc = ZlibEncoder::new(Vec::new(), Compression::default());
        enc.write_all(data).expect("deflate");
        enc.finish().expect("deflate finish")
    }

    fn flate_stream(data: &[u8]) -> Stream {
        let dict = lopdf::dictionary! { "Filter" => "FlateDecode" };
        Stream::new(dict, zlib(data))
    }

    fn ints(v: [i64; 4]) -> Vec<Object> {
        v.into_iter().map(Object::Integer).collect()
    }

    /// In-memory scan page: one 128×96 image drawn by `content` on
    /// `media_box`. Only the geometry stage and page gates look at it,
    /// so the image bytes are a placeholder.
    fn scan_page(media_box: Vec<Object>, content: Stream) -> (Document, ObjectId) {
        use lopdf::dictionary;
        let mut doc = Document::with_version("1.5");
        let img = doc.add_object(Stream::new(
            dictionary! {
                "Type" => "XObject", "Subtype" => "Image",
                "Width" => 128i64, "Height" => 96i64,
                "ColorSpace" => "DeviceGray", "BitsPerComponent" => 8i64,
                "Filter" => "DCTDecode",
            },
            b"placeholder".to_vec(),
        ));
        let content_id = doc.add_object(content);
        let page_id = doc.add_object(dictionary! {
            "Type" => "Page",
            "MediaBox" => media_box,
            "Resources" => dictionary! { "XObject" => dictionary! { "Im0" => img } },
            "Contents" => content_id,
        });
        (doc, page_id)
    }

    fn geometry_decline(media_box: Vec<Object>, content: Stream) -> Option<Decline> {
        let (doc, page) = scan_page(media_box, content);
        SourcePdf::from_document(doc).single_image_draw(page).err()
    }

    #[test]
    fn usable_box_refuses_non_finite_and_degenerate_boxes() {
        assert!(is_usable_box([0.0, 0.0, 612.0, 792.0]));
        assert!(is_usable_box([-10.0, -10.0, 602.0, 782.0]));
        for bad in [
            [0.0, 0.0, f32::INFINITY, 792.0],
            // inf − inf = NaN: every reject-on-`>` gate would pass it.
            [f32::INFINITY, 0.0, f32::INFINITY, 792.0],
            [0.0, f32::NAN, 612.0, 792.0],
            // Finite corners, infinite width.
            [-3e38, 0.0, 3e38, 792.0],
            [0.0, 0.0, 0.5, 792.0],
            [612.0, 792.0, 0.0, 0.0], // inverted
        ] {
            assert!(!is_usable_box(bad), "{bad:?} must be refused");
        }
    }

    #[test]
    fn non_finite_media_box_declines_the_geometry_stage() {
        let inf = || Object::Real(f32::INFINITY);
        let real = Object::Real;
        for mb in [
            vec![0.into(), 0.into(), inf(), 216.into()],
            vec![inf(), 0.into(), inf(), 216.into()],
            vec![real(-3e38), 0.into(), real(3e38), 216.into()],
        ] {
            assert_eq!(
                geometry_decline(mb, Stream::new(Dictionary::new(), DRAW.to_vec())),
                Some("MediaBox is non-finite or degenerate")
            );
        }
        // Control: the same page on a sane box measures.
        let content = Stream::new(Dictionary::new(), DRAW.to_vec());
        assert_eq!(geometry_decline(ints([0, 0, 144, 216]), content), None);
    }

    #[test]
    fn non_finite_crop_box_declines_passthrough() {
        let (mut doc, page) = scan_page(
            ints([0, 0, 144, 216]),
            Stream::new(Dictionary::new(), DRAW.to_vec()),
        );
        if let Ok(Object::Dictionary(d)) = doc.get_object_mut(page) {
            d.set(
                "CropBox",
                vec![0.into(), 0.into(), Object::Real(f32::INFINITY), 216.into()],
            );
        }
        let src = SourcePdf::from_document(doc);
        let geom = src.single_image_draw(page).expect("geometry is fine");
        assert_eq!(
            src.passthrough_gates(page, &geom),
            Err("CropBox is non-finite or degenerate")
        );
    }

    #[test]
    fn pixel_dims_decline_instead_of_wrapping() {
        assert_eq!(pixel_dim(Some(128)), Ok(128));
        assert_eq!(
            pixel_dim(Some(i64::from(MAX_IMAGE_DIMENSION))),
            Ok(MAX_IMAGE_DIMENSION)
        );
        // pdfium renders nothing past its limit — and 2.4e9 would reach the
        // writer's `i32` /Width as -1894967296.
        assert!(pixel_dim(Some(i64::from(MAX_IMAGE_DIMENSION) + 1)).is_err());
        assert!(pixel_dim(Some(2_400_000_000)).is_err());
        assert!(pixel_dim(Some(i64::from(u32::MAX))).is_err());
        // `as u32` would make this 128 and pass it as an ordinary image.
        assert!(pixel_dim(Some((1i64 << 32) + 128)).is_err());
        assert!(pixel_dim(Some(0)).is_err());
        assert!(pixel_dim(Some(-5)).is_err());
        assert!(pixel_dim(None).is_err());
    }

    #[test]
    fn bounded_content_decodes_what_lopdf_did() {
        let plain = Stream::new(Dictionary::new(), DRAW.to_vec());
        assert_eq!(bounded_content(&plain, 1024).expect("plain"), DRAW);
        let flate = flate_stream(DRAW);
        assert_eq!(bounded_content(&flate, 1024).expect("flate"), DRAW);
        // ASCII85 `z` is four zero bytes: A85 output is at most 4× its
        // input, so the raw cap bounds it without a limit of its own.
        let a85 = Stream::new(
            lopdf::dictionary! { "Filter" => "ASCII85Decode" },
            b"z~>".to_vec(),
        );
        assert_eq!(bounded_content(&a85, 1024).expect("a85"), vec![0u8; 4]);
    }

    #[test]
    fn bounded_content_stops_at_the_limit() {
        // 4 MiB of spaces deflates to a few KiB — tiny on disk, and
        // lopdf's decompressed_content would inflate every byte of it.
        let bomb = flate_stream(&vec![b' '; 4 << 20]);
        assert!(bomb.content.len() < MAX_CONTENT_RAW_BYTES);
        assert_eq!(
            bounded_content(&bomb, MAX_CONTENT_DECODED_BYTES),
            Err(CONTENT_TOO_LARGE_DECODED)
        );
        // Exactly at the limit is fine; one byte over is not.
        let hundred = flate_stream(&[b' '; 100]);
        assert!(bounded_content(&hundred, 100).is_ok());
        assert_eq!(
            bounded_content(&hundred, 99),
            Err(CONTENT_TOO_LARGE_DECODED)
        );
    }

    #[test]
    fn bounded_content_declines_unbounded_filters_and_predictors() {
        let lzw = Stream::new(
            lopdf::dictionary! { "Filter" => "LZWDecode" },
            b"any bytes".to_vec(),
        );
        assert!(bounded_content(&lzw, 1024).is_err());
        let mut predictor = flate_stream(DRAW);
        predictor
            .dict
            .set("DecodeParms", lopdf::dictionary! { "Predictor" => 12i64 });
        assert!(bounded_content(&predictor, 1024).is_err());
        let mut chain = flate_stream(DRAW);
        chain.dict.set(
            "Filter",
            vec![Object::from("FlateDecode"); MAX_CONTENT_FILTERS + 1],
        );
        assert!(bounded_content(&chain, 1024).is_err());
    }

    #[test]
    fn content_size_caps_decline_the_geometry_stage() {
        let mb = || ints([0, 0, 144, 216]);
        // Raw cap: whitespace-padded, otherwise a valid single draw — it is
        // refused before anything is decoded or lexed.
        let mut padded = DRAW.to_vec();
        padded.resize(MAX_CONTENT_RAW_BYTES + 1, b' ');
        let padded = Stream::new(Dictionary::new(), padded);
        assert_eq!(geometry_decline(mb(), padded), Some(CONTENT_TOO_LARGE));
        // Decoded cap: small on disk, over the cap once inflated.
        let mut inflated = DRAW.to_vec();
        inflated.resize(MAX_CONTENT_DECODED_BYTES + 1, b' ');
        assert_eq!(
            geometry_decline(mb(), flate_stream(&inflated)),
            Some(CONTENT_TOO_LARGE_DECODED)
        );
        // Control: under both caps, compressed, the draw measures.
        assert_eq!(geometry_decline(mb(), flate_stream(DRAW)), None);
    }

    fn annot(subtype: Option<&str>, flags: Option<i64>) -> Object {
        let mut d = lopdf::dictionary! { "Type" => "Annot" };
        if let Some(s) = subtype {
            d.set("Subtype", Object::Name(s.as_bytes().to_vec()));
        }
        if let Some(f) = flags {
            d.set("F", f);
        }
        Object::Dictionary(d)
    }

    fn visible(annots: Vec<Object>) -> bool {
        let mut doc = Document::with_version("1.5");
        let page = doc.add_object(lopdf::dictionary! { "Type" => "Page", "Annots" => annots });
        SourcePdf::from_document(doc).has_visible_annotations(page)
    }

    #[test]
    fn only_visible_drawing_annotations_count() {
        assert!(!visible(vec![]));
        // Drawn over the scan: stamps, widgets, shapes — and entries with
        // no /Subtype, whose appearance pdfium may still draw.
        assert!(visible(vec![annot(Some("Square"), None)]));
        assert!(visible(vec![annot(Some("Stamp"), Some(4))])); // Print only
        assert!(visible(vec![annot(None, None)]));
        let link_and_widget = vec![annot(Some("Link"), None), annot(Some("Widget"), None)];
        assert!(visible(link_and_widget));
        // Never paint scan-like content.
        assert!(!visible(vec![annot(Some("Link"), None)]));
        assert!(!visible(vec![annot(Some("Popup"), None)]));
        // Hidden (bit 2) or NoView (bit 6): off screen.
        assert!(!visible(vec![annot(Some("Square"), Some(2))]));
        assert!(!visible(vec![annot(Some("Square"), Some(32 | 4))]));
        // Not a dictionary: pdfium can't draw it either.
        assert!(!visible(vec![Object::Integer(7)]));

        // No /Annots at all, and /Annots behind a reference.
        let mut doc = Document::with_version("1.5");
        let bare = doc.add_object(lopdf::dictionary! { "Type" => "Page" });
        let stamp = doc.add_object(annot(Some("Stamp"), None));
        let list = doc.add_object(vec![Object::Reference(stamp)]);
        let referenced = doc.add_object(lopdf::dictionary! { "Type" => "Page", "Annots" => list });
        let src = SourcePdf::from_document(doc);
        assert!(!src.has_visible_annotations(bare));
        assert!(src.has_visible_annotations(referenced));
    }

    // ---- image masks ---------------------------------------------------

    /// Hook `page` into a page tree so [`SourcePdf::probe_page`] finds it.
    fn paginate(doc: &mut Document, page: ObjectId) {
        use lopdf::dictionary;
        let pages = doc.add_object(dictionary! {
            "Type" => "Pages", "Kids" => vec![page.into()], "Count" => 1,
        });
        if let Ok(Object::Dictionary(d)) = doc.get_object_mut(page) {
            d.set("Parent", pages);
        }
        let catalog = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages });
        doc.trailer.set("Root", catalog);
    }

    fn mask_stream(width: Option<i64>, height: i64) -> Stream {
        use lopdf::dictionary;
        let mut dict = dictionary! {
            "Type" => "XObject", "Subtype" => "Image", "Height" => height,
            "ColorSpace" => "DeviceGray", "BitsPerComponent" => 8i64,
        };
        if let Some(w) = width {
            dict.set("Width", w);
        }
        Stream::new(dict, b"placeholder".to_vec())
    }

    /// Store `mask` (a stream goes in by reference) as `key` of `image`.
    fn set_mask(doc: &mut Document, image: ObjectId, key: &str, mask: Object) {
        let mask = match mask {
            Object::Stream(s) => Object::Reference(doc.add_object(s)),
            other => other,
        };
        if let Ok(Object::Stream(s)) = doc.get_object_mut(image) {
            s.dict.set(key, mask);
        }
    }

    /// The paginated [`scan_page`] (128×96 px drawn over 2×3 in: 64 DPI)
    /// and its image's id.
    fn probe_fixture() -> (Document, ObjectId, ObjectId) {
        let (mut doc, page) = scan_page(
            ints([0, 0, 144, 216]),
            Stream::new(Dictionary::new(), DRAW.to_vec()),
        );
        let image = doc
            .get_dictionary(page)
            .and_then(|p| p.get(b"Resources"))
            .and_then(Object::as_dict)
            .and_then(|r| r.get(b"XObject"))
            .and_then(Object::as_dict)
            .and_then(|x| x.get(b"Im0"))
            .and_then(Object::as_reference)
            .expect("scan_page's image reference");
        paginate(&mut doc, page);
        (doc, page, image)
    }

    fn probe_with_mask(key: &str, mask: Object) -> PageProbe {
        let (mut doc, _, image) = probe_fixture();
        set_mask(&mut doc, image, key, mask);
        SourcePdf::from_document(doc).probe_page(0)
    }

    #[test]
    fn a_sharper_mask_raises_the_dpi_hint_to_its_own_resolution() {
        let (doc, _, _) = probe_fixture();
        let plain = SourcePdf::from_document(doc).probe_page(0);
        assert_eq!(plain.effective_dpi, Some(64.0), "control");
        assert_eq!(plain.masks, MaskScale::Unmasked);

        // MRC: a 3× soft mask carries the glyph edges the raster shows.
        let mrc = probe_with_mask("SMask", Object::Stream(mask_stream(Some(384), 288)));
        assert_eq!(mrc.effective_dpi, Some(192.0));
        assert_eq!(mrc.masks, MaskScale::Finer(3.0));
        let explicit = probe_with_mask("Mask", Object::Stream(mask_stream(Some(256), 192)));
        assert_eq!(explicit.effective_dpi, Some(128.0));
        assert_eq!(explicit.masks, MaskScale::Finer(2.0));

        // A screenshot's same-size alpha channel keeps the hint as it was —
        // dropping it would bring back the 9.8× re-encode.
        let alpha = probe_with_mask("SMask", Object::Stream(mask_stream(Some(128), 96)));
        assert_eq!(alpha.effective_dpi, Some(64.0));
        assert_eq!(alpha.masks, MaskScale::Unmasked);
        // A colour-key mask is an array: no pixels of its own.
        let key = probe_with_mask("Mask", Object::Array(vec![0.into(), 10.into()]));
        assert_eq!(key.effective_dpi, Some(64.0));
        assert_eq!(key.masks, MaskScale::Unmasked);
    }

    #[test]
    fn a_mask_of_unknown_size_vetoes_the_hint() {
        let probe = probe_with_mask("SMask", Object::Stream(mask_stream(None, 288)));
        assert_eq!(probe.effective_dpi, None);
        assert_eq!(probe.masks, MaskScale::Unknown);
    }

    #[test]
    fn page_masks_survive_a_geometry_decline() {
        // Two images: the probe can't measure the page, but the writer's
        // pdfium backstop still needs to know one of them has a 3× mask.
        let (mut doc, page, image) = probe_fixture();
        set_mask(
            &mut doc,
            image,
            "SMask",
            Object::Stream(mask_stream(Some(384), 288)),
        );
        let second = doc.add_object(mask_stream(Some(10), 10));
        if let Ok(Object::Dictionary(p)) = doc.get_object_mut(page) {
            if let Ok(xobjects) = p
                .get_mut(b"Resources")
                .and_then(Object::as_dict_mut)
                .and_then(|r| r.get_mut(b"XObject"))
                .and_then(Object::as_dict_mut)
            {
                xobjects.set("Im1", second);
            }
        }
        let probe = SourcePdf::from_document(doc).probe_page(0);
        assert_eq!(probe.effective_dpi, None, "two images: no hint");
        assert_eq!(probe.masks, MaskScale::Finer(3.0));
    }

    #[test]
    fn soft_mask_groups_make_the_page_resolution_unknown() {
        let with_state = |smask: Object| {
            let (mut doc, page, _) = probe_fixture();
            if let Ok(Object::Dictionary(p)) = doc.get_object_mut(page) {
                if let Ok(resources) = p.get_mut(b"Resources").and_then(Object::as_dict_mut) {
                    resources.set(
                        "ExtGState",
                        lopdf::dictionary! { "GS0" => lopdf::dictionary! { "SMask" => smask } },
                    );
                }
            }
            SourcePdf::from_document(doc).probe_page(0)
        };
        let group = lopdf::dictionary! { "S" => "Luminosity" };
        let masked = with_state(Object::Dictionary(group));
        assert_eq!(masked.masks, MaskScale::Unknown);
        // The drawn image itself is unmasked and the whitelist has no `gs`:
        // the probe's own hint stands.
        assert_eq!(masked.effective_dpi, Some(64.0));
        let none = with_state(Object::Name(b"None".to_vec()));
        assert_eq!(none.masks, MaskScale::Unmasked);
    }

    #[test]
    fn mask_scale_applies_to_a_measured_resolution() {
        assert_eq!(MaskScale::Unmasked.apply(Some(96.0)), Some(96.0));
        assert_eq!(MaskScale::Finer(3.0).apply(Some(96.0)), Some(288.0));
        assert_eq!(MaskScale::Finer(3.0).apply(None), None);
        assert_eq!(MaskScale::Unknown.apply(Some(96.0)), None);
        assert_eq!(MaskScale::of_ratio(0.5), MaskScale::Unmasked);
        assert_eq!(
            MaskScale::Finer(2.0).max(MaskScale::Finer(3.0)),
            MaskScale::Finer(3.0)
        );
        assert_eq!(
            MaskScale::Finer(2.0).max(MaskScale::Unknown),
            MaskScale::Unknown
        );
    }
}
