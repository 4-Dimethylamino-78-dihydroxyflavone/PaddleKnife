//! Searchable PDF writer — raster + character-correct invisible text layer.
//!
//! Plan 5 ground-up rewrite. The text layer is hand-authored with
//! `pdf-writer` (pdfium's object API cannot express per-glyph
//! advances), using the glyphless CID font from [`glyphless`]:
//!
//! - One text object (`BT..ET`) per [`TextLine`].
//! - The line's **baseline** (start, direction, height) is fitted
//!   through its char-span quads — continuous tilt and RTL come out of
//!   the geometry, no script knowledge needed. The text matrix `Tm`
//!   carries rotation + translation; font size = line height.
//! - Every grapheme cluster is one CID shown at its **measured**
//!   position: `Td` walks the pen to the cluster's detected start, and
//!   the CID's `/W` entry is exactly the detected advance (one CID per
//!   distinct cluster × width bucket). There is no `Tz`: pdf.js
//!   mis-scales per-instance horizontal scaling, while `/W` is the one
//!   mechanism every viewer's text layer honors identically (see
//!   [`glyphless`]). Character-correct by construction — no font
//!   metrics, no shaping, no per-script constants.
//! - Explicit space CIDs between words of spaced scripts span the
//!   measured inter-word gap, so extractors get real word breaks
//!   (CJK/Thai neighbors get none, matching the line-join policy).
//!
//! **Coordinates:** OCR quads are raster pixels (top-left origin,
//! Y down). PDF is points (bottom-left origin, Y up). The single
//! transform is `pt = px × 72/dpi` with `y_pdf = H_pts − y_px·s`, where
//! `dpi` is the page's own render DPI ([`EncodedPage::dpi`]).
//!
//! Each page's image is one XObject: the source scan's original bytes
//! when the page qualifies for passthrough, else a JPEG re-encoded from
//! the raster, stored no sharper than the source's measured resolution
//! (see [`encode_page_opts`]).

mod glyphless;
mod page_image;
pub use page_image::PageImage;

use crate::{
    error::{OcrError, Result},
    geometry::{self, Quad},
    layout::{DocPage, TextLine},
    ocr::OcrBlock,
    pdf::RasterPage,
    script::is_unspaced_script,
    source_pdf::{MaskScale, PageProbe, PtFilter},
};
use flate2::{write::ZlibEncoder, Compression};
use glyphless::CidAllocator;
use pdf_writer::{
    types::{CidFontType, FontFlags, SystemInfo, TextRenderingMode},
    Content, Filter, Name, Pdf, Rect, Ref, Str,
};
use std::io::Write as _;
use std::path::Path;
use unicode_segmentation::UnicodeSegmentation;

/// Minimum space width (fraction of line height) when detector boxes
/// overlap or abut: extractors need a non-degenerate space advance to
/// emit a word break.
const MIN_SPACE_FRAC: f32 = 0.2;

/// One positioned cluster, ready for content-stream emission.
struct PlacedSpan {
    cid: u16,
    /// Pen start along the baseline, points. The advance (selection
    /// width) is encoded in the CID's /W entry, not stored here.
    t0: f32,
}

/// One line's worth of text operators.
struct PlacedLine {
    /// Text matrix: unit rotation + translation to the baseline start.
    tm: [f32; 6],
    /// Font size in points (= line band height).
    size: f32,
    spans: Vec<PlacedSpan>,
}

/// A page raster pre-encoded as the JPEG that will be embedded.
///
/// Long documents should collect these instead of `RasterPage`s: a
/// 300-dpi A4 RGB bitmap is ~25 MB while its JPEG is typically well
/// under 1 MB, so a 1000-page book stays under ~1 GB instead of
/// needing tens of GB of RAM before the writer runs.
pub struct EncodedPage {
    /// Raster dimensions in pixels — the MediaBox basis (page size =
    /// raster px × 72/[`Self::dpi`]). NOT necessarily the embedded image's
    /// size: passthrough and DPI-capped images carry their own dims in
    /// [`PageImage`]; the content stream stretches them to the page.
    pub width: u32,
    pub height: u32,
    /// Pixels per inch of `width` × `height`: the raster's own
    /// [`RasterPage::dpi`], carried here so the writer maps each page's
    /// MediaBox and text layer through the scale its OCR geometry was
    /// measured at (lower than the document's DPI on a page that hit the
    /// rasterizer's pixel budget). 0 = use the writer's document DPI.
    pub dpi: u32,
    pub image: PageImage,
}

/// Options governing how page images are embedded.
#[derive(Debug, Clone)]
pub struct EmbedOptions {
    /// Copy original source-scan bytes through when a page qualifies.
    pub allow_passthrough: bool,
    /// JPEG quality (1–100) for re-encoded pages.
    pub jpeg_quality: u8,
    /// Ceiling on the embedded image's DPI (`--embed-dpi`). It lowers the
    /// measured source resolution, never raises it; a page with no
    /// measurement is stored at this DPI. See [`encode_page_opts`].
    pub embed_dpi_override: Option<u32>,
}

impl Default for EmbedOptions {
    fn default() -> Self {
        Self {
            allow_passthrough: true,
            jpeg_quality: 85,
            embed_dpi_override: None,
        }
    }
}

/// Encode one raster page with default options (no probe → re-encode;
/// grayscale auto-detect still applies).
pub fn encode_page(raster: &RasterPage) -> Result<EncodedPage> {
    encode_page_opts(raster, None, &EmbedOptions::default(), 0)
}

/// The resolution to store a re-encoded page at, before the
/// never-upsample check. `None` = no cap (embed the raster as rendered).
///
/// The source's resolution comes from the lopdf probe (`hint`) and from
/// pdfium's own view of an image-only page (`native`). Both measure the
/// same image and normally agree; when both exist the larger wins,
/// because disagreement means one of them read another page (lopdf and
/// pdfium can order a malformed page tree differently) or missed an
/// image, and the larger estimate never throws real pixels away.
/// `native` alone is the backstop for pages the strict probe can't
/// analyze (clip paths, `gs`, a source lopdf can't open): without it
/// those pages embed the raster at the full OCR DPI — many times the
/// pixels the source image ever had, for no added information.
///
/// `--embed-dpi` (`cap`) is a ceiling on top: it lowers the estimate,
/// never raises it — `--embed-dpi 300` on a 96-DPI screenshot must not
/// bring back the 9.8× blow-up. With no estimate, the cap is the target.
fn embed_target_dpi(cap: Option<u32>, hint: Option<f32>, native: Option<f32>) -> Option<f32> {
    let source = match (hint, native) {
        (Some(h), Some(n)) => Some(h.max(n)),
        (h, n) => h.or(n),
    };
    match (cap.map(|c| c as f32), source) {
        (Some(c), Some(s)) => Some(c.min(s)),
        (c, s) => c.or(s),
    }
}

/// Full embedding decision for one page.
///
/// `ocr_dpi` is 0 for image inputs, whose raster pixels are already the
/// native resolution — DPI capping is then skipped. Otherwise it is the
/// DPI the page was rasterized at, and the raster's own
/// [`RasterPage::dpi`] takes precedence when set: a page that hit the
/// rasterizer's pixel budget was rendered below the requested DPI.
///
/// The DPI cap never upsamples: it only shrinks the *embedded* image when
/// the source scan demonstrably has less resolution than the OCR raster.
/// The OCR raster is rendered at a fixed DPI whatever the source holds, so
/// an uncapped re-encode of a low-resolution page inflates it (96 → 300
/// DPI is 9.8× the pixels); `probe.effective_dpi` and `raster.native_dpi`
/// exist to make sure a page with an identifiable source image never
/// takes that path (`embed_target_dpi` has how they combine with
/// `opts.embed_dpi_override`). `raster.native_dpi` only sees the images'
/// own pixel grids, so `probe.masks` corrects it for masks sharper than
/// their images (MRC scans), or drops it when their resolution is unknown.
pub fn encode_page_opts(
    raster: &RasterPage,
    probe: Option<&PageProbe>,
    opts: &EmbedOptions,
    ocr_dpi: u32,
) -> Result<EncodedPage> {
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
                    dpi: raster.dpi,
                    image: PageImage::Passthrough(p.clone()),
                });
            }
        }
    }

    // The raster's own DPI is what its pixels mean. Scaling a page that
    // was rendered below the requested DPI (pixel budget) by the requested
    // DPI would shrink its image twice.
    let ocr_dpi = if ocr_dpi > 0 && raster.dpi > 0 {
        raster.dpi
    } else {
        ocr_dpi
    };

    // Re-encode path: store the page no sharper than its source. pdfium
    // measures each image by its own pixel grid; a sharper /SMask or
    // /Mask is only visible to the probe, which scales (or vetoes) it.
    let target_dpi = embed_target_dpi(
        opts.embed_dpi_override,
        probe.and_then(|p| p.effective_dpi),
        probe
            .map_or(MaskScale::Unmasked, |p| p.masks)
            .apply(raster.native_dpi),
    );
    let scaled;
    let src: &image::DynamicImage = match target_dpi {
        Some(t) if ocr_dpi > 0 && t < ocr_dpi as f32 * 0.95 => {
            // Floor the embed DPI at 50 to avoid absurdly tiny images,
            // but never let the floor exceed the ceiling (a future
            // sub-50 --dpi would make clamp(50, ocr_dpi) panic).
            let floor = 50.0_f32.min(ocr_dpi as f32);
            let t = t.clamp(floor, ocr_dpi as f32);
            let w = ((raster.width as f32) * t / ocr_dpi as f32)
                .round()
                .max(1.0) as u32;
            let h = ((raster.height as f32) * t / ocr_dpi as f32)
                .round()
                .max(1.0) as u32;
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
    Ok(EncodedPage {
        width: raster.width,
        height: raster.height,
        dpi: raster.dpi,
        image,
    })
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

/// Write `doc_pages` + `raster_pages` (same length, matching indices)
/// as a searchable PDF at `output`. Each page maps pixels to points
/// through its own [`RasterPage::dpi`]; `dpi` is the fallback for pages
/// that leave it 0 (hand-built rasters).
///
/// Convenience wrapper that JPEG-encodes every raster up front; for
/// long documents prefer encoding per page during OCR and calling
/// [`write_searchable_pdf_encoded`].
pub fn write_searchable_pdf(
    output: &Path,
    doc_pages: &[DocPage],
    raster_pages: Vec<RasterPage>,
    dpi: u32,
) -> Result<()> {
    let pages = raster_pages
        .iter()
        .map(encode_page)
        .collect::<Result<Vec<_>>>()?;
    write_searchable_pdf_encoded(output, doc_pages, pages, dpi)
}

/// Write `doc_pages` + pre-encoded `pages` (same length, matching
/// indices) as a searchable PDF at `output`. Each page maps pixels to
/// points through its own [`EncodedPage::dpi`]; `dpi` is the fallback
/// for pages that leave it 0.
pub fn write_searchable_pdf_encoded(
    output: &Path,
    doc_pages: &[DocPage],
    pages: Vec<EncodedPage>,
    dpi: u32,
) -> Result<()> {
    if doc_pages.len() != pages.len() {
        return Err(OcrError::Engine(format!(
            "searchable_pdf: page count mismatch (doc={}, raster={})",
            doc_pages.len(),
            pages.len()
        )));
    }
    // Per page: a page rendered below `dpi` (the rasterizer's pixel
    // budget) has its OCR geometry in its own, coarser pixels.
    let pt_per_px = |enc: &EncodedPage| {
        let page_dpi = if enc.dpi > 0 { enc.dpi } else { dpi };
        72.0_f32 / page_dpi as f32
    };

    // Pass 1: place every line, allocating CIDs as clusters appear.
    let mut alloc = CidAllocator::new();
    let mut placed: Vec<Vec<PlacedLine>> = Vec::with_capacity(doc_pages.len());
    for (doc_page, enc) in doc_pages.iter().zip(pages.iter()) {
        let scale = pt_per_px(enc);
        let page_h_pts = enc.height as f32 * scale;
        let mut lines = Vec::with_capacity(doc_page.lines.len());
        for line in &doc_page.lines {
            lines.extend(place_line(line, &mut alloc, scale, page_h_pts)?);
        }
        placed.push(lines);
    }

    // Pass 2: assemble the document.
    let mut pdf = Pdf::new();
    let mut next_ref = 0i32;
    let mut alloc_ref = || {
        next_ref += 1;
        Ref::new(next_ref)
    };

    let catalog_id = alloc_ref();
    let page_tree_id = alloc_ref();
    let type0_id = alloc_ref();
    let cid_font_id = alloc_ref();
    let descriptor_id = alloc_ref();
    let font_file_id = alloc_ref();
    let cid_gid_id = alloc_ref();
    let to_unicode_id = alloc_ref();
    let page_refs: Vec<(Ref, Ref, Ref)> = doc_pages
        .iter()
        .map(|_| (alloc_ref(), alloc_ref(), alloc_ref()))
        .collect();

    pdf.catalog(catalog_id).pages(page_tree_id);
    pdf.pages(page_tree_id)
        .kids(page_refs.iter().map(|(p, _, _)| *p))
        .count(page_refs.len() as i32);

    write_font_objects(
        &mut pdf,
        &alloc,
        type0_id,
        cid_font_id,
        descriptor_id,
        font_file_id,
        cid_gid_id,
        to_unicode_id,
    )?;

    for ((enc, lines), (page_id, content_id, image_id)) in
        pages.into_iter().zip(placed).zip(page_refs.iter())
    {
        let scale = pt_per_px(&enc);
        let w_pts = enc.width as f32 * scale;
        let h_pts = enc.height as f32 * scale;

        // Page image XObject — original source bytes (passthrough) or
        // our (re)encoded JPEG. The content stream stretches whatever
        // pixel grid this is to the full page, so embedded dims are
        // independent of the raster dims that set the MediaBox.
        {
            // `i32`, as PDF integers are: `as i32` would write a dimension
            // past i32::MAX as a negative /Width.
            let (iw, ih) = enc.image.dims();
            let dim = |v: u32| {
                i32::try_from(v).map_err(|_| {
                    OcrError::Engine(format!(
                        "searchable_pdf: page image dimension {v} exceeds the PDF integer range"
                    ))
                })
            };
            let (iw, ih) = (dim(iw)?, dim(ih)?);
            let mut img = pdf.image_xobject(*image_id, enc.image.data());
            img.width(iw);
            img.height(ih);
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

        // Content: full-page image + invisible text.
        let mut content = Content::new();
        content
            .save_state()
            .transform([w_pts, 0.0, 0.0, h_pts, 0.0, 0.0])
            .x_object(Name(b"Im0"))
            .restore_state();
        for line in &lines {
            emit_line(&mut content, line);
        }
        let compressed = flate(&content.finish())?;
        pdf.stream(*content_id, &compressed)
            .filter(Filter::FlateDecode);

        let mut page = pdf.page(*page_id);
        page.parent(page_tree_id)
            .media_box(Rect::new(0.0, 0.0, w_pts, h_pts))
            .contents(*content_id);
        let mut resources = page.resources();
        resources.x_objects().pair(Name(b"Im0"), *image_id);
        resources.fonts().pair(Name(b"F1"), type0_id);
    }

    std::fs::write(output, pdf.finish())
        .map_err(|e| OcrError::Engine(format!("write pdf: {e}")))?;
    Ok(())
}

/// Cluster spans for one block: measured `chars` when present, else a
/// uniform synthesis across the block quad (legacy JSON input).
fn block_spans(block: &OcrBlock) -> Vec<(String, Quad)> {
    if !block.chars.is_empty() {
        return block
            .chars
            .iter()
            .map(|c| (c.text.clone(), c.quad))
            .collect();
    }
    let quad = block.quad_or_bbox();
    let graphemes: Vec<&str> = block.text.graphemes(true).collect();
    let n = graphemes.len().max(1) as f32;
    graphemes
        .iter()
        .enumerate()
        .map(|(i, g)| {
            (
                (*g).to_string(),
                geometry::quad_slice(&quad, i as f32 / n, (i + 1) as f32 / n),
            )
        })
        .collect()
}

/// Fit the line's baseline, project every cluster onto it, and insert
/// measured spaces between words of spaced scripts. Returns `None` for
/// lines with no placeable content.
fn place_line(
    line: &TextLine,
    alloc: &mut CidAllocator,
    pt_per_px: f32,
    page_h_pts: f32,
) -> Result<Vec<PlacedLine>> {
    // Cluster spans per block, in reading order.
    let per_block: Vec<Vec<(String, Quad)>> = line
        .blocks
        .iter()
        .map(block_spans)
        .filter(|s| !s.is_empty())
        .collect();
    let all_quads: Vec<Quad> = per_block
        .iter()
        .flat_map(|spans| spans.iter().map(|(_, q)| *q))
        .collect();
    let Some(baseline) = geometry::fit_baseline(&all_quads) else {
        return Ok(Vec::new());
    };

    // Nonlinear chain (arched/curved text, e.g. display type)? A single
    // straight baseline would flatten it — place every cluster in its
    // own frame instead, so boxes follow the curve at cluster
    // granularity. Straightness = spread of cluster-center
    // perpendicular offsets relative to band height; a straight line's
    // centers all sit mid-band.
    if !is_straight_chain(&all_quads, &baseline) {
        let mut out = Vec::with_capacity(all_quads.len());
        for (cluster, quad) in per_block.iter().flatten() {
            let Some(b) = geometry::fit_baseline(std::slice::from_ref(quad)) else {
                continue;
            };
            let e = b.start[0] * pt_per_px;
            let (a_, b_) = (b.dir[0], -b.dir[1]);
            // Text-space "up" is the +90° CCW rotation of the writing
            // direction, i.e. (-b_, a_); its y-component is a_.
            let (size, f) = size_clamped_to_page(
                (b.height * pt_per_px).max(1.0),
                page_h_pts - b.start[1] * pt_per_px,
                a_,
                page_h_pts,
            );
            out.push(PlacedLine {
                tm: [a_, b_, -b_, a_, e, f],
                size,
                spans: vec![PlacedSpan {
                    cid: alloc.cid_for(cluster, 1000.0 * (b.len * pt_per_px).max(0.01) / size)?,
                    t0: 0.0,
                }],
            });
        }
        return Ok(out);
    }
    // Clamp before the span loop: each CID's `/W` is derived from `size`,
    // so the two must agree or advances would be scaled wrongly.
    let (size, clamped_f) = size_clamped_to_page(
        (baseline.height * pt_per_px).max(1.0),
        page_h_pts - baseline.start[1] * pt_per_px,
        baseline.dir[0],
        page_h_pts,
    );

    // Project clusters onto the baseline; track per-block t-intervals
    // for the inter-word spaces.
    let mut spans: Vec<PlacedSpan> = Vec::new();
    let mut prev_block: Option<([f32; 2], char)> = None; // (t interval px, last char)
    for block_spans in &per_block {
        let block_first_char = block_spans
            .first()
            .and_then(|(s, _)| s.chars().next())
            .unwrap_or(' ');
        let mut block_t = [f32::MAX, f32::MIN];
        let mut pending: Vec<PlacedSpan> = Vec::new();
        for (cluster, quad) in block_spans {
            let mut t0 = f32::MAX;
            let mut t1 = f32::MIN;
            for corner in quad {
                let t = baseline.project(*corner);
                t0 = t0.min(t);
                t1 = t1.max(t);
            }
            block_t = [block_t[0].min(t0), block_t[1].max(t1)];
            let width = ((t1 - t0) * pt_per_px).max(0.01);
            pending.push(PlacedSpan {
                cid: alloc.cid_for(cluster, 1000.0 * width / size)?,
                t0: t0 * pt_per_px,
            });
        }
        // Space between this block and the previous one, when both
        // boundary characters belong to spaced scripts (mirrors the
        // line-join policy, so PDF text == TextLine.text). The space
        // sits in the geometric gap between the two blocks' baseline
        // intervals — which works regardless of whether logical order
        // runs left-to-right or (RTL) right-to-left.
        if let Some((prev_t, prev_char)) = prev_block {
            if !(is_unspaced_script(prev_char) && is_unspaced_script(block_first_char)) {
                let gap_lo = prev_t[1].min(block_t[1]);
                let gap_hi = prev_t[0].max(block_t[0]);
                let min_px = MIN_SPACE_FRAC * baseline.height;
                let width_px = (gap_hi - gap_lo).max(min_px);
                let width = width_px * pt_per_px;
                spans.push(PlacedSpan {
                    cid: alloc.cid_for(" ", 1000.0 * width / size)?,
                    t0: gap_lo * pt_per_px,
                });
            }
        }
        let block_last_char = block_spans
            .last()
            .and_then(|(s, _)| s.chars().next_back())
            .unwrap_or(' ');
        prev_block = Some((block_t, block_last_char));
        spans.extend(pending);
    }
    if spans.is_empty() {
        return Ok(Vec::new());
    }

    // Baseline → PDF space: scale to points, flip Y. The text-space
    // x-axis is the writing direction; its y-axis is the +90° CCW
    // rotation (PDF coords), i.e. "up" relative to the line.
    let e = baseline.start[0] * pt_per_px;
    let f = clamped_f;
    let a = baseline.dir[0];
    let b = -baseline.dir[1];
    let tm = [a, b, -b, a, e, f];
    Ok(vec![PlacedLine { tm, size, spans }])
}

/// Trim a font size so the glyph's selection rectangle stays on the page.
///
/// The glyphless font's ascent is one full em and its descent ≈ 0, so a
/// glyph placed with its pen at y = `f` claims the band `f .. f + size`
/// along the line's "up" direction, whose PDF y-component is `up_y`
/// (1.0 upright, 0.0 for a 90° rotation, -1.0 upside-down). DBNet's
/// unclip step routinely pushes a band past the raster edge, and a
/// selection rectangle that leaves the page is never right — viewers clip
/// it and hit-testing near the margin misbehaves.
///
/// Only the overhang is trimmed. The pen stays exactly where the detector
/// put it, and cluster advances are unaffected: each CID's `/W` is
/// `1000 × width / size`, so the rendered advance is `size × W/1000 =
/// width` and `size` cancels out.
fn size_clamped_to_page(size: f32, f: f32, up_y: f32, page_h_pts: f32) -> (f32, f32) {
    // The glyph band runs from the pen to pen + up_y * size. Either end
    // can be the top depending on the line's rotation.
    let far = f + up_y * size;
    let (lo, hi) = if far >= f { (f, far) } else { (far, f) };

    let clipped_lo = lo.max(0.0);
    let clipped_hi = hi.min(page_h_pts);
    if clipped_hi - clipped_lo < 1.0 {
        // Degenerate after clipping (band almost entirely off-page):
        // leave it alone rather than emit a 0-height selection box.
        return (size, f);
    }
    if clipped_lo <= lo + 1e-6 && clipped_hi >= hi - 1e-6 {
        return (size, f); // already on-page — the common case, untouched
    }

    // Rebuild (size, pen) from the clipped interval. up_y is ±1 for
    // upright/upside-down and 0 for a 90° rotation (handled above, since
    // then far == f and the band has no vertical extent to clip).
    let new_size = ((clipped_hi - clipped_lo) / up_y.abs().max(1e-6)).max(1.0);
    let new_f = if up_y >= 0.0 { clipped_lo } else { clipped_hi };
    (new_size, new_f)
}

/// Maximum spread of cluster-center perpendicular offsets, as a
/// fraction of band height, for one straight baseline to represent the
/// chain. Straight lines measure ~0 (centers sit mid-band); arched
/// text measures large. Generous enough that detector jitter and
/// ascender/descender variation never trip it.
const MAX_STRAIGHT_DEVIATION: f32 = 0.5;

fn is_straight_chain(quads: &[Quad], baseline: &geometry::Baseline) -> bool {
    if quads.len() < 3 {
        return true; // 1-2 clusters: a line through them is exact.
    }
    // Perpendicular direction matching fit_baseline's convention
    // (downward in image space).
    let (mut px, mut py) = (-baseline.dir[1], baseline.dir[0]);
    if py < 0.0 {
        px = -px;
        py = -py;
    }
    let mut lo = f32::MAX;
    let mut hi = f32::MIN;
    for q in quads {
        let c = geometry::center(q);
        let perp = (c[0] - baseline.start[0]) * px + (c[1] - baseline.start[1]) * py;
        lo = lo.min(perp);
        hi = hi.max(perp);
    }
    (hi - lo) <= MAX_STRAIGHT_DEVIATION * baseline.height
}

/// Emit one line's `BT..ET`. `Td` is relative to the previous `Td`
/// (the line matrix chains), so each span's offset is the delta of
/// baseline positions. Each CID's advance is already its measured
/// width (advance = size × W/1000, with W from the font's `/W` array),
/// so nothing here scales it.
fn emit_line(content: &mut Content, line: &PlacedLine) {
    content.begin_text();
    content.set_font(Name(b"F1"), line.size);
    content.set_text_rendering_mode(TextRenderingMode::Invisible);
    content.set_text_matrix(line.tm);
    let mut pen = 0.0f32;
    for span in &line.spans {
        content.next_line(span.t0 - pen, 0.0);
        pen = span.t0;
        // No Tz: each CID's advance is its measured width via /W —
        // the one mechanism every viewer's text layer scales
        // identically (see glyphless.rs on the pdf.js Tz bug).
        content.show(Str(&span.cid.to_be_bytes()));
    }
    content.end_text();
}

/// Glyphless Type0 font graph: Type0 → CIDFontType2 → descriptor →
/// embedded TTF, plus CIDToGIDMap (all → glyph 0) and ToUnicode.
#[allow(clippy::too_many_arguments)]
fn write_font_objects(
    pdf: &mut Pdf,
    alloc: &CidAllocator,
    type0_id: Ref,
    cid_font_id: Ref,
    descriptor_id: Ref,
    font_file_id: Ref,
    cid_gid_id: Ref,
    to_unicode_id: Ref,
) -> Result<()> {
    pdf.type0_font(type0_id)
        .base_font(Name(b"GlyphLessFont"))
        .encoding_predefined(Name(b"Identity-H"))
        .descendant_font(cid_font_id)
        .to_unicode(to_unicode_id);

    {
        let mut cid_font = pdf.cid_font(cid_font_id);
        cid_font
            .subtype(CidFontType::Type2)
            .base_font(Name(b"GlyphLessFont"))
            .system_info(SystemInfo {
                registry: pdf_writer::Str(b"Adobe"),
                ordering: pdf_writer::Str(b"Identity"),
                supplement: 0,
            })
            .font_descriptor(descriptor_id)
            .default_width(glyphless::DEFAULT_WIDTH as f32);
        // Per-CID measured advances: /W [1 [w1 w2 …]] (CIDs are
        // allocated consecutively from 1).
        let w = alloc.widths();
        if !w.is_empty() {
            cid_font.widths().consecutive(1, w.iter().copied());
        }
        cid_font.pair(Name(b"CIDToGIDMap"), cid_gid_id);
    }

    pdf.font_descriptor(descriptor_id)
        .name(Name(b"GlyphLessFont"))
        .flags(FontFlags::SYMBOLIC)
        .bbox(Rect::new(0.0, 0.0, 1000.0, 1000.0))
        .italic_angle(0.0)
        .ascent(1000.0)
        .descent(0.0)
        .cap_height(800.0)
        .stem_v(80.0)
        .font_file2(font_file_id);

    let compressed = flate(glyphless::FONT)?;
    {
        let mut stream = pdf.stream(font_file_id, &compressed);
        stream.filter(Filter::FlateDecode);
        stream.pair(Name(b"Length1"), glyphless::FONT.len() as i32);
    }

    let cid_gid = flate(&alloc.cid_to_gid_map())?;
    pdf.stream(cid_gid_id, &cid_gid).filter(Filter::FlateDecode);

    let cmap = flate(&alloc.to_unicode_cmap())?;
    pdf.stream(to_unicode_id, &cmap).filter(Filter::FlateDecode);
    Ok(())
}

fn jpeg_bytes(
    raw: &[u8],
    w: u32,
    h: u32,
    color: image::ExtendedColorType,
    quality: u8,
) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, quality);
    enc.encode(raw, w, h, color)
        .map_err(|e| OcrError::Engine(format!("jpeg encode: {e}")))?;
    Ok(out)
}

fn flate(data: &[u8]) -> Result<Vec<u8>> {
    let mut enc = ZlibEncoder::new(Vec::new(), Compression::default());
    enc.write_all(data)
        .map_err(|e| OcrError::Engine(format!("flate: {e}")))?;
    enc.finish()
        .map_err(|e| OcrError::Engine(format!("flate finish: {e}")))
}

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
            dpi: 300,
            native_dpi: None,
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
        let enc = encode_page(&raster_of(img)).expect("encode");
        match enc.image {
            PageImage::Jpeg { gray, .. } => assert!(gray, "neutral page must be gray"),
            other => unreachable!("expected jpeg, got {other:?}"),
        }
    }

    #[test]
    fn colorful_page_stays_rgb() {
        let img = RgbImage::from_fn(64, 64, |x, y| image::Rgb([(x * 4) as u8, 0, (y * 4) as u8]));
        let enc = encode_page(&raster_of(img)).expect("encode");
        match enc.image {
            PageImage::Jpeg { gray, .. } => assert!(!gray, "colorful page must stay RGB"),
            other => unreachable!("expected jpeg, got {other:?}"),
        }
    }

    #[test]
    fn gray_jpeg_is_smaller_than_rgb_jpeg() {
        let img = RgbImage::from_fn(256, 256, |x, y| {
            let v = ((x * 7 + y * 13) % 251) as u8;
            image::Rgb([v, v, v])
        });
        let raster = raster_of(img.clone());
        let gray_len = encode_page(&raster).expect("encode").image.data().len();
        let rgb_len = jpeg_bytes(img.as_raw(), 256, 256, image::ExtendedColorType::Rgb8, 85)
            .expect("rgb encode")
            .len();
        assert!(
            gray_len < rgb_len,
            "gray {gray_len} should beat rgb {rgb_len}"
        );
    }
}

#[cfg(test)]
mod embed_tests {
    use super::*;
    use crate::pdf::RasterPage;
    use crate::source_pdf::{MaskScale, PageProbe, PassthroughImage, PtFilter};
    use image::{DynamicImage, RgbImage};

    fn raster(w: u32, h: u32) -> RasterPage {
        RasterPage {
            index: 0,
            width: w,
            height: h,
            image: DynamicImage::ImageRgb8(RgbImage::from_fn(w, h, |x, y| {
                image::Rgb([(x % 250) as u8, (y % 250) as u8, 10])
            })),
            dpi: 300,
            native_dpi: None,
        }
    }

    fn probe_with(pt: Option<PassthroughImage>, dpi: Option<f32>) -> PageProbe {
        PageProbe {
            passthrough: pt,
            effective_dpi: dpi,
            masks: MaskScale::Unmasked,
        }
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
        .expect("encode");
        match enc.image {
            PageImage::Passthrough(p) => assert_eq!(p.data, b"original-bytes".to_vec()),
            other => unreachable!("expected passthrough, got {other:?}"),
        }
        assert_eq!((enc.width, enc.height), (600, 900));
    }

    #[test]
    fn reencode_mode_ignores_passthrough() {
        let opts = EmbedOptions {
            allow_passthrough: false,
            ..EmbedOptions::default()
        };
        let enc = encode_page_opts(
            &raster(600, 900),
            Some(&probe_with(Some(fake_pt()), None)),
            &opts,
            300,
        )
        .expect("encode");
        assert!(matches!(enc.image, PageImage::Jpeg { .. }));
    }

    #[test]
    fn dpi_hint_downscales_embedded_image() {
        let enc = encode_page_opts(
            &raster(1000, 1500),
            Some(&probe_with(None, Some(100.0))),
            &EmbedOptions::default(),
            300,
        )
        .expect("encode");
        assert_eq!((enc.width, enc.height), (1000, 1500));
        assert_eq!(enc.image.dims(), (333, 500));
    }

    #[test]
    fn embed_dpi_is_a_ceiling_not_an_override() {
        let opts = EmbedOptions {
            embed_dpi_override: Some(150),
            ..EmbedOptions::default()
        };
        let encode = |hint: Option<f32>| {
            encode_page_opts(
                &raster(1000, 1500),
                Some(&probe_with(None, hint)),
                &opts,
                300,
            )
            .expect("encode")
        };
        // A 72-DPI source stays at 72: `--embed-dpi 150` must not upsample
        // it back toward the OCR raster (the 9.8× blow-up it used to allow).
        assert_eq!(encode(Some(72.0)).image.dims(), (240, 360));
        // A sharper source is lowered to the ceiling…
        assert_eq!(encode(Some(200.0)).image.dims(), (500, 750));
        // …and with no measurement at all, the ceiling is the target.
        assert_eq!(encode(None).image.dims(), (500, 750));
    }

    #[test]
    fn embed_target_dpi_policy() {
        // (cap, probe hint, pdfium native) → target
        let cases = [
            (None, None, None, None),
            (None, Some(96.0), None, Some(96.0)),
            (None, None, Some(96.0), Some(96.0)),
            // Both measured: the larger wins (a page-order mismatch or a
            // missed image must not cost real pixels).
            (None, Some(96.0), Some(150.0), Some(150.0)),
            (None, Some(150.0), Some(96.0), Some(150.0)),
            // The cap only ever lowers.
            (Some(120), Some(96.0), Some(150.0), Some(120.0)),
            (Some(300), Some(96.0), None, Some(96.0)),
            (Some(300), None, Some(96.0), Some(96.0)),
            (Some(150), None, None, Some(150.0)),
        ];
        for (cap, hint, native, want) in cases {
            assert_eq!(
                embed_target_dpi(cap, hint, native),
                want,
                "cap {cap:?}, hint {hint:?}, native {native:?}"
            );
        }
    }

    fn raster_with_native_dpi(w: u32, h: u32, native_dpi: Option<f32>) -> RasterPage {
        RasterPage {
            native_dpi,
            ..raster(w, h)
        }
    }

    #[test]
    fn native_dpi_caps_the_embed_when_the_probe_has_no_hint() {
        // The worst case this guards: probe came back empty (or there is
        // no probe at all — lopdf couldn't open the source), so without a
        // backstop the full 300-DPI raster would be embedded.
        for probe in [None, Some(probe_with(None, None))] {
            let enc = encode_page_opts(
                &raster_with_native_dpi(1000, 1500, Some(96.0)),
                probe.as_ref(),
                &EmbedOptions::default(),
                300,
            )
            .expect("encode");
            assert_eq!((enc.width, enc.height), (1000, 1500), "MediaBox basis");
            assert_eq!(enc.image.dims(), (320, 480));
        }
    }

    #[test]
    fn hint_and_native_dpi_take_the_larger_and_the_cap_lowers_it() {
        let r = raster_with_native_dpi(1000, 1500, Some(96.0));
        let with_hint = |hint: f32| {
            encode_page_opts(
                &r,
                Some(&probe_with(None, Some(hint))),
                &EmbedOptions::default(),
                300,
            )
            .expect("encode")
        };
        assert_eq!(with_hint(150.0).image.dims(), (500, 750), "hint is larger");
        assert_eq!(with_hint(64.0).image.dims(), (320, 480), "native is larger");

        // `--embed-dpi 200` over a 96-DPI source: the cap is a ceiling, so
        // the source's 96 stands (it used to override, embedding at 200).
        let opts = EmbedOptions {
            embed_dpi_override: Some(200),
            ..EmbedOptions::default()
        };
        let enc = encode_page_opts(&r, None, &opts, 300).expect("encode");
        assert_eq!(enc.image.dims(), (320, 480), "cap never raises");
    }

    #[test]
    fn a_page_rendered_below_the_requested_dpi_scales_by_its_own() {
        // The rasterizer clamped this page to 150 DPI (pixel budget) though
        // 300 was requested. A 100-DPI source must embed at 100/150 of the
        // raster — scaling by 100/300 would shrink it twice.
        let r = RasterPage {
            dpi: 150,
            ..raster(1000, 1500)
        };
        let enc = encode_page_opts(
            &r,
            Some(&probe_with(None, Some(100.0))),
            &EmbedOptions::default(),
            300,
        )
        .expect("encode");
        assert_eq!(enc.image.dims(), (667, 1000));
        assert_eq!((enc.width, enc.height, enc.dpi), (1000, 1500, 150));
    }

    #[test]
    fn writer_maps_each_page_through_its_own_dpi() {
        let jpeg =
            jpeg_bytes(&[128u8; 16 * 16], 16, 16, image::ExtendedColorType::L8, 80).expect("jpeg");
        let page = |dpi: u32| EncodedPage {
            width: 600,
            height: 900,
            dpi,
            image: PageImage::Jpeg {
                data: jpeg.clone(),
                width: 16,
                height: 16,
                gray: true,
            },
        };
        let doc_page = |index: usize| DocPage {
            index,
            size: (600, 900),
            lines: Vec::new(),
            metrics: Default::default(),
        };
        let dir = tempfile::tempdir().expect("tempdir");
        let out = dir.path().join("per_page_dpi.pdf");
        // Page 1 was clamped to 150 DPI; page 2 leaves its DPI unset and
        // falls back to the document's 300.
        write_searchable_pdf_encoded(
            &out,
            &[doc_page(0), doc_page(1)],
            vec![page(150), page(0)],
            300,
        )
        .expect("write");

        let doc = lopdf::Document::load(&out).expect("reopen");
        let boxes: Vec<Vec<f32>> = doc
            .get_pages()
            .values()
            .map(|id| {
                doc.get_dictionary(*id)
                    .and_then(|d| d.get(b"MediaBox"))
                    .and_then(lopdf::Object::as_array)
                    .expect("MediaBox")
                    .iter()
                    .map(|v| v.as_float().expect("number"))
                    .collect()
            })
            .collect();
        // 600×900 px at 150 DPI is 288×432 pt; at 300 DPI, 144×216 pt.
        let want = [[0.0, 0.0, 288.0, 432.0], [0.0, 0.0, 144.0, 216.0]];
        assert_eq!(boxes.len(), want.len());
        for (got, want) in boxes.iter().zip(want) {
            assert!(
                got.iter().zip(want).all(|(g, w)| (g - w).abs() < 0.01),
                "MediaBox {got:?}, want {want:?}"
            );
        }
    }

    #[test]
    fn native_dpi_never_upsamples() {
        // A page whose images are sharper than the OCR raster: the cap is
        // a ceiling, so the raster is embedded as-is.
        let enc = encode_page_opts(
            &raster_with_native_dpi(600, 900, Some(600.0)),
            None,
            &EmbedOptions::default(),
            300,
        )
        .expect("encode");
        assert_eq!(enc.image.dims(), (600, 900));
    }

    #[test]
    fn masks_sharper_than_their_images_lift_the_native_cap() {
        // An MRC page pdfium measures at 96 DPI (its colour image), whose
        // masks the probe found finer than that.
        let r = raster_with_native_dpi(1000, 1500, Some(96.0));
        let with_masks = |masks: MaskScale| {
            let probe = PageProbe {
                masks,
                ..PageProbe::default()
            };
            encode_page_opts(&r, Some(&probe), &EmbedOptions::default(), 300)
                .expect("encode")
                .image
                .dims()
        };
        assert_eq!(with_masks(MaskScale::Unmasked), (320, 480), "control");
        assert_eq!(with_masks(MaskScale::Finer(2.0)), (640, 960));
        // 3× is 288 DPI: within 5 % of the raster, which is kept whole.
        assert_eq!(with_masks(MaskScale::Finer(3.0)), (1000, 1500));
        assert_eq!(with_masks(MaskScale::Unknown), (1000, 1500), "no cap");
    }

    #[test]
    fn near_full_dpi_hint_skips_downscale() {
        let enc = encode_page_opts(
            &raster(600, 900),
            Some(&probe_with(None, Some(290.0))),
            &EmbedOptions::default(),
            300,
        )
        .expect("encode");
        assert_eq!(enc.image.dims(), (600, 900));
    }

    #[test]
    fn zero_ocr_dpi_means_no_scaling() {
        let enc = encode_page_opts(
            &raster(640, 480),
            None,
            &EmbedOptions {
                embed_dpi_override: Some(72),
                ..EmbedOptions::default()
            },
            0,
        )
        .expect("encode");
        assert_eq!(enc.image.dims(), (640, 480));
    }

    #[test]
    fn quality_knob_changes_output_size() {
        let r = raster(256, 256);
        let q30 = encode_page_opts(
            &r,
            None,
            &EmbedOptions {
                jpeg_quality: 30,
                ..EmbedOptions::default()
            },
            300,
        )
        .expect("encode");
        let q95 = encode_page_opts(
            &r,
            None,
            &EmbedOptions {
                jpeg_quality: 95,
                ..EmbedOptions::default()
            },
            300,
        )
        .expect("encode");
        assert!(q30.image.data().len() < q95.image.data().len());
    }

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
        .expect("encode");
        assert!(matches!(enc.image, PageImage::Jpeg { .. }));
    }

    #[test]
    fn sub_50_ocr_dpi_does_not_panic() {
        // Pathological low ocr_dpi: clamp floor must not exceed ceiling.
        // hint 10 < 40*0.95 so the downscale branch runs; floor collapses
        // to 40 (=ocr_dpi), t stays 10, embed ~ raster*10/40.
        let enc = encode_page_opts(
            &RasterPage {
                dpi: 40,
                ..raster(800, 600)
            },
            Some(&probe_with(None, Some(10.0))),
            &EmbedOptions::default(),
            40,
        )
        .expect("encode");
        let (w, h) = enc.image.dims();
        assert!(w <= 800 && h <= 600, "must not upscale: {w}x{h}");
        assert!(w >= 1 && h >= 1);
    }

    // ---- selection rectangles stay on the page ----------------------

    /// A band whose unclipped height would run past the top of the page
    /// gets trimmed to the margin, not emitted overhanging. Observed on a
    /// handwriting scan where the first line's box reached y = 796.8 on a
    /// 792 pt page.
    #[test]
    fn oversized_band_is_trimmed_to_the_page_top() {
        // Pen 40 pt below the top, band wants 100 pt of upward extent.
        let (size, f) = size_clamped_to_page(100.0, 752.0, 1.0, 792.0);
        assert!((size - 40.0).abs() < 1e-3, "got {size}");
        assert!((f - 752.0).abs() < 1e-3, "pen must not move for a top trim");
        assert!(f + size <= 792.0 + 1e-3, "box must end on the page");
    }

    /// An upside-down line extends downward; it must stay above y = 0.
    #[test]
    fn upside_down_band_is_trimmed_to_the_page_bottom() {
        let (size, f) = size_clamped_to_page(100.0, 30.0, -1.0, 792.0);
        assert!((size - 30.0).abs() < 1e-3, "got {size}");
        assert!((f - 30.0).abs() < 1e-3, "pen stays at the band top");
    }

    /// An upright band whose pen sits BELOW the page bottom — the detector
    /// unclipped past the raster edge. Both ends must come back on-page:
    /// the pen moves up to 0 and the height loses exactly the overhang.
    #[test]
    fn band_below_the_page_bottom_is_lifted_onto_the_page() {
        let (size, f) = size_clamped_to_page(100.0, -20.0, 1.0, 792.0);
        assert!(
            (f - 0.0).abs() < 1e-3,
            "pen lifted to the page edge, got {f}"
        );
        assert!((size - 80.0).abs() < 1e-3, "overhang trimmed, got {size}");
        assert!(f >= -1e-3 && f + size <= 792.0 + 1e-3);
    }

    /// A band that already fits is returned untouched — no silent shrink
    /// of ordinary lines.
    #[test]
    fn fitting_band_is_left_alone() {
        assert_eq!(size_clamped_to_page(40.0, 100.0, 1.0, 792.0), (40.0, 100.0));
        assert_eq!(
            size_clamped_to_page(40.0, 400.0, -1.0, 792.0),
            (40.0, 400.0)
        );
    }

    /// A 90°-rotated line has no vertical extent from `size`, so the
    /// vertical clamp must not touch it.
    #[test]
    fn rotated_line_is_not_clamped_vertically() {
        let (size, f) = size_clamped_to_page(100.0, 790.0, 0.0, 792.0);
        assert!((size - 100.0).abs() < 1e-3, "got {size}");
        assert!((f - 790.0).abs() < 1e-3);
    }
}
