//! PDF rasterization via pdfium-render.
//!
//! Loads pdfium (`pdfium.dll`, `libpdfium.so` or `libpdfium.dylib`) from
//! (in priority order):
//!   1. `PDFIUM_LIB_PATH` environment variable (directory containing it)
//!   2. The directory of the currently running executable (production layout)
//!   3. `<workspace root>/vendor/pdfium/bin/` (development layout)

use crate::error::{OcrError, Result};
use image::DynamicImage;
use pdfium_render::prelude::*;
use std::path::{Path, PathBuf};

pub struct RasterPage {
    pub index: usize,
    pub width: u32,
    pub height: u32,
    pub image: DynamicImage,
    /// Pixels per inch of page space in `image`: the scale everything that
    /// maps this raster's pixel geometry back to the page (OCR quads, the
    /// searchable-PDF MediaBox and text layer, the embed-DPI cap) must use.
    /// For PDF pages it is the DPI pdfium actually rendered at — the
    /// requested one, or lower when that would break the pixel budget
    /// (see [`render_dpi_within_budget`]). 0 = unknown (a hand-built
    /// raster): the writer then falls back to its document-wide DPI.
    pub dpi: u32,
    /// Resolution the page's own content actually has, dots/inch, when
    /// pdfium sees an image-only page (see [`image_only_page_dpi`]).
    /// The raster above is rendered at a fixed OCR DPI regardless, so
    /// for a 96-DPI screenshot it holds 9.8× the source's pixels and no
    /// extra information — this is the searchable-PDF writer's last-
    /// resort cap when the source-PDF probe yields no hint. `None` for
    /// image inputs, for pages with any vector/text content or a visible
    /// annotation, and when [`RasterOptions::native_dpi`] is off. Blind to
    /// image masks: the writer scales it by `PageProbe::masks`.
    pub native_dpi: Option<f32>,
}

/// What the rasterizer computes beyond the bitmap.
#[derive(Debug, Clone, Copy)]
pub struct RasterOptions {
    /// Fill [`RasterPage::native_dpi`]. Only the searchable-PDF writer
    /// reads it, and it costs two pdfium image-metadata fetches per image
    /// on image-only pages (each one loads the image, which for JPX is a
    /// full decode), so text-only outputs should turn it off.
    pub native_dpi: bool,
}

impl Default for RasterOptions {
    fn default() -> Self {
        Self { native_dpi: true }
    }
}

/// Most pixels one page may rasterize to. A0 at 300 DPI is 139 MP —
/// 558 MB as RGBA before OCR makes its working copies — and an untrusted
/// PDF can declare a far larger page (the spec's 14 400 pt a side is not
/// enforced by anything). 100 MP leaves every format up to A1 at 300 DPI
/// (70 MP) untouched and keeps one page's RGBA bitmap under 400 MB.
pub const MAX_RENDER_PIXELS: u64 = 100_000_000;

/// Longest raster side, in pixels. Bounds a banner-shaped page that the
/// area cap alone lets through: 200 × 1 in at 300 DPI is only 18 MP, but
/// 60 000 px long.
pub const MAX_RENDER_SIDE: u32 = 20_000;

/// Pixel size pdfium is asked for at `dpi`. Shared by the budget check and
/// the render so the two can never round differently.
fn raster_size(width_pt: f32, height_pt: f32, dpi: u32) -> (i32, i32) {
    let scale = dpi as f32 / 72.0;
    (
        (width_pt * scale).round() as i32,
        (height_pt * scale).round() as i32,
    )
}

/// Render DPI for a `width_pt` × `height_pt` page (PDF points) that keeps
/// its raster within [`MAX_RENDER_PIXELS`] and [`MAX_RENDER_SIDE`]:
/// `requested` when it already fits, else the highest whole DPI that does.
///
/// `None` when no DPI of at least 1 fits (a side over ~20 000 inches — no
/// real document) or the size is not finite: the caller refuses the page
/// rather than ask pdfium for an unbounded bitmap.
pub fn render_dpi_within_budget(width_pt: f32, height_pt: f32, requested: u32) -> Option<u32> {
    if !width_pt.is_finite() || !height_pt.is_finite() {
        return None;
    }
    let fits = |dpi: u32| {
        let (w, h) = raster_size(width_pt, height_pt, dpi);
        // A negative size renders nothing; count it as zero pixels.
        let w = u64::try_from(w).unwrap_or(0);
        let h = u64::try_from(h).unwrap_or(0);
        let side = u64::from(MAX_RENDER_SIDE);
        w <= side && h <= side && w * h <= MAX_RENDER_PIXELS
    };
    if fits(requested) {
        return Some(requested);
    }
    // Continuous bound first, then step down past the renderer's rounding
    // (a step or two at most).
    let w_in = f64::from(width_pt.max(0.0)) / 72.0;
    let h_in = f64::from(height_pt.max(0.0)) / 72.0;
    let by_side = f64::from(MAX_RENDER_SIDE) / w_in.max(h_in);
    let by_area = (MAX_RENDER_PIXELS as f64 / (w_in * h_in)).sqrt();
    let mut dpi = by_side.min(by_area).min(f64::from(requested)).floor() as u32;
    while dpi >= 1 {
        if fits(dpi) {
            return Some(dpi);
        }
        dpi -= 1;
    }
    None
}

/// Highest effective resolution among the page's images, if the page is
/// made of images and nothing else.
///
/// Text, paths, shadings and form XObjects have no native resolution —
/// they genuinely benefit from the full OCR DPI — so a single one of them
/// means "no cap". The same goes for a visible annotation (see
/// [`has_visible_annotations`]): pdfium draws it into the raster, but it
/// is not a page object. With images only, the page carries no more
/// information than its sharpest image, so embedding above that
/// resolution is pure upsampling. Taking the max over images (and over
/// both axes) errs toward keeping pixels: a small high-DPI logo raises
/// the cap, never lowers it.
///
/// One blind spot: an image's `/SMask` or `/Mask` is not a page object,
/// and pdfium-render 0.9 has no accessor for it, so this measures each
/// image's own pixel grid only. An MRC scan's full-resolution mask over a
/// low-resolution colour image would be undercounted; the writer corrects
/// for that with the source probe's `PageProbe::masks`.
///
/// This reads pdfium's object model — the same parse that produced the
/// raster — so unlike the lopdf probe it is indifferent to content-stream
/// syntax (clip paths, `gs`, odd number formats) and to files lopdf
/// cannot open at all.
fn image_only_page_dpi(page: &PdfPage) -> Option<f32> {
    if has_visible_annotations(page) {
        return None;
    }
    let objects = page.objects();
    // Classify every object before fetching any image metadata: the
    // fetches are the expensive part, and one text or path object anywhere
    // on the page makes all of them moot.
    let vetoed = objects
        .iter()
        .any(|object| object.as_image_object().is_none() && !is_invisible_on_the_raster(&object));
    if vetoed {
        return None; // visible non-image content → no cap
    }
    let mut best: Option<f32> = None;
    for object in objects.iter() {
        let Some(image) = object.as_image_object() else {
            continue;
        };
        let dpi = image_dpi(image)?;
        best = Some(best.map_or(dpi, |b| b.max(dpi)));
    }
    best
}

/// One image object's resolution: the sharper axis.
///
/// Two metadata fetches, where one would do: pdfium-render 0.9 hands the
/// image metadata out a field at a time — every accessor re-runs
/// `FPDFImageObj_GetImageMetadata`, which loads the image's DIB source —
/// and its single-fetch `get_raw_metadata` is crate-private. Getting both
/// DPIs from one fetch needs an upstream accessor or an `unsafe` FFI
/// call, which this crate does not make.
fn image_dpi(image: &PdfPageImageObject) -> Option<f32> {
    let dpi = image.horizontal_dpi().ok()?.max(image.vertical_dpi().ok()?);
    (dpi.is_finite() && dpi > 0.0).then_some(dpi)
}

/// Does pdfium draw any of this page's annotations into the raster?
///
/// Mirrors `SourcePdf::has_visible_annotations` on pdfium's side:
/// `/Link` and `/Popup` never paint scan-like content, and the Hidden and
/// NoView flags keep any annotation off screen (pdfium honors both when
/// rendering for display). Everything else — stamps, signature widgets,
/// filled form fields, ink, sticky-note icons — counts. An entry pdfium
/// can't open (not a dictionary) is one it can't draw either, so it is
/// skipped; indexing directly keeps one such entry from hiding the rest,
/// as pdfium-render's iterator would (it stops at the first error).
fn has_visible_annotations(page: &PdfPage) -> bool {
    let annotations = page.annotations();
    annotations.as_range().any(|i| {
        annotations.get(i).is_ok_and(|annotation| {
            !matches!(
                annotation.annotation_type(),
                PdfPageAnnotationType::Link | PdfPageAnnotationType::Popup
            ) && !annotation.is_hidden()
                && !annotation.is_printable_but_not_viewable()
        })
    })
}

/// Policy hook for [`image_only_page_dpi`]: does this non-image object
/// leave the raster untouched, so it shouldn't veto the resolution cap?
///
/// The conservative answer — and today's behavior — is "no, never": any
/// non-image object means vector content that deserves the full OCR DPI.
/// That is always safe (worst case: a bigger file), but it means a scan
/// that ALREADY carries an invisible OCR text layer (render mode 3 — the
/// output of Acrobat, ABBYY, ocrmypdf, or a previous paddleknife run) is
/// never capped, because its text objects veto it.
///
// TODO(policy): decide whether invisible text should be transparent here,
// e.g. `object.as_text_object()` with `render_mode()` ==
// `PdfPageTextRenderMode::Invisible`. Trade-off: slimmer re-OCR of
// already-OCR'd scans vs. trusting pdfium's render-mode report on files
// where "invisible" text is sometimes abused for watermarks/clipping.
fn is_invisible_on_the_raster(_object: &PdfPageObject) -> bool {
    false
}

/// Resolve the directory holding `pdfium.dll` (or the platform equivalent:
/// `libpdfium.so`, `libpdfium.dylib`).
fn find_pdfium_lib_dir() -> Option<PathBuf> {
    let lib_name = Pdfium::pdfium_platform_library_name();
    // 1) Explicit env override
    if let Ok(p) = std::env::var("PDFIUM_LIB_PATH") {
        let path = PathBuf::from(p);
        if path.exists() {
            return Some(path);
        }
    }
    // 2) Alongside the running exe
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            if dir.join(&lib_name).exists() {
                return Some(dir.to_path_buf());
            }
        }
    }
    // 3) Dev layout: <root>/vendor/pdfium/bin/, searched upward from the
    //    running exe first (<root>/target/release/paddleknife.exe), then
    //    from this crate's compile-time manifest dir. The exe comes first
    //    because CARGO_MANIFEST_DIR is frozen at build time: with a target
    //    dir shared across git worktrees it names whichever worktree built
    //    last, and worktrees lack the gitignored vendor/pdfium/. Walking a
    //    worktree's ancestors still reaches the main checkout, since
    //    worktrees live under <main>/.claude/worktrees/.
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf));
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let starts = exe_dir
        .iter()
        .map(PathBuf::as_path)
        .chain([manifest_dir.as_path()]);
    find_vendored_pdfium(starts, Path::new(&lib_name))
}

/// First `<ancestor>/vendor/pdfium/bin` holding `lib_name`, trying every
/// ancestor of each start directory (itself included) in order.
fn find_vendored_pdfium<'a>(
    starts: impl IntoIterator<Item = &'a Path>,
    lib_name: &Path,
) -> Option<PathBuf> {
    starts
        .into_iter()
        .flat_map(Path::ancestors)
        .map(|dir| dir.join("vendor").join("pdfium").join("bin"))
        .find(|bin| bin.join(lib_name).is_file())
}

/// Bind to pdfium, idempotently. The first caller loads the DLL and stores
/// the bindings in a process-wide slot; later callers reuse them. Multiple
/// rebind attempts otherwise fail with `PdfiumLibraryBindingsAlreadyInitialized`
/// (we see this in integration tests, which exercise several pdfium paths
/// in the same process).
///
/// Public so downstream callers (other crates, integration tests) that
/// want to use pdfium-render's own APIs directly can share the same
/// DLL-resolution logic — set `PDFIUM_LIB_PATH`, place `pdfium.dll` next
/// to the exe, or use the `vendor/pdfium/bin/` dev-layout.
pub fn bind_pdfium() -> Result<Pdfium> {
    match Pdfium::bind_to_library(Pdfium::pdfium_platform_library_name_at_path(
        &find_pdfium_lib_dir().ok_or_else(|| {
            OcrError::Engine(
                "pdfium.dll not found. Set PDFIUM_LIB_PATH, place it next to the exe, \
                 or use the dev layout under vendor/pdfium/bin/."
                    .into(),
            )
        })?,
    )) {
        Ok(bindings) => Ok(Pdfium::new(bindings)),
        // Already bound earlier in this process — reuse the cached bindings.
        // `Pdfium` is technically a unit struct acting as a facade over global
        // bindings, so clippy flags `default()`. pdfium-render documents
        // `Pdfium::default()` as the idiomatic constructor for the facade,
        // so suppress the lint locally rather than spelling it `Pdfium {}`.
        #[allow(clippy::default_constructed_unit_structs)]
        Err(PdfiumError::PdfiumLibraryBindingsAlreadyInitialized) => Ok(Pdfium::default()),
        Err(e) => Err(OcrError::Engine(format!("pdfium bind: {e}"))),
    }
}

/// Rasterize every page of `pdf_path` at `dpi` DPI.
///
/// Returns an owned `Vec<RasterPage>`. Kept for callers that need
/// random access (e.g. the searchable-PDF writer, which interleaves
/// original rasters with OCR text overlays). For straight-through
/// OCR, prefer `for_each_page` — peak memory drops to one page's
/// bitmap because each `RasterPage` is dropped before the next one
/// is rendered.
pub fn rasterize_pdf(pdf_path: &Path, dpi: u32) -> Result<Vec<RasterPage>> {
    let mut out = Vec::new();
    for_each_page(pdf_path, dpi, |page| {
        out.push(page);
        Ok(())
    })?;
    Ok(out)
}

/// Number of pages in `pdf_path`. Cheap (no rasterization) — used by
/// the CLI to size its progress bar before streaming pages.
pub fn page_count(pdf_path: &Path) -> Result<usize> {
    let pdf_str = pdf_path
        .to_str()
        .ok_or_else(|| OcrError::Engine(format!("non-utf8 path: {}", pdf_path.display())))?;
    let pdfium = bind_pdfium()?;
    let doc = pdfium
        .load_pdf_from_file(pdf_str, None)
        .map_err(|e| OcrError::Engine(format!("pdfium open {pdf_str}: {e}")))?;
    Ok(doc.pages().len() as usize)
}

/// Rasterize just one page (0-indexed). Cheap shortcut for samplers
/// like the auto-script-detect pass that don't need the whole document.
pub fn rasterize_page(pdf_path: &Path, page_idx: usize, dpi: u32) -> Result<RasterPage> {
    rasterize_page_with(pdf_path, page_idx, dpi, RasterOptions::default())
}

/// [`rasterize_page`] with explicit [`RasterOptions`].
pub fn rasterize_page_with(
    pdf_path: &Path,
    page_idx: usize,
    dpi: u32,
    options: RasterOptions,
) -> Result<RasterPage> {
    let pdf_str = pdf_path
        .to_str()
        .ok_or_else(|| OcrError::UnsupportedFormat("non-utf8 path".into()))?;
    let pdfium = bind_pdfium()?;
    let doc = pdfium
        .load_pdf_from_file(pdf_str, None)
        .map_err(|e| OcrError::Engine(format!("pdfium load: {e}")))?;
    let pages = doc.pages();
    let page = pages
        .get((page_idx as u16).into())
        .map_err(|e| OcrError::Engine(format!("pdfium page {page_idx}: {e}")))?;
    render_page(&page, page_idx, dpi, options)
}

/// Streaming rasterizer: calls `f` once per page, dropping each
/// `RasterPage` after `f` returns. `f` gets ownership of the page
/// so it can shuttle the image into an OCR pass without a clone.
///
/// Use this for pipelines that produce format-agnostic outputs
/// (txt/json/md), where no step needs every bitmap simultaneously.
/// Computes [`RasterPage::native_dpi`]; [`for_each_page_with`] can skip it.
pub fn for_each_page<F>(pdf_path: &Path, dpi: u32, f: F) -> Result<()>
where
    F: FnMut(RasterPage) -> Result<()>,
{
    for_each_page_with(pdf_path, dpi, RasterOptions::default(), f)
}

/// [`for_each_page`] with explicit [`RasterOptions`].
pub fn for_each_page_with<F>(
    pdf_path: &Path,
    dpi: u32,
    options: RasterOptions,
    mut f: F,
) -> Result<()>
where
    F: FnMut(RasterPage) -> Result<()>,
{
    let pdf_str = pdf_path
        .to_str()
        .ok_or_else(|| OcrError::UnsupportedFormat("non-utf8 path".into()))?;

    let pdfium = bind_pdfium()?;
    let doc = pdfium
        .load_pdf_from_file(pdf_str, None)
        .map_err(|e| OcrError::Engine(format!("pdfium load: {e}")))?;

    for (idx, page) in doc.pages().iter().enumerate() {
        f(render_page(&page, idx, dpi, options)?)?;
        // The bitmap and the `RasterPage` both drop at the end of this
        // iteration, before the next page renders.
    }
    Ok(())
}

/// Render one page at `dpi`, or lower when `dpi` would break the pixel
/// budget ([`render_dpi_within_budget`]; warned about). The DPI actually
/// used is recorded as [`RasterPage::dpi`]: OCR geometry comes out in this
/// raster's pixels, so everything that maps it back to the page must use
/// that DPI, not the requested one.
fn render_page(page: &PdfPage, idx: usize, dpi: u32, opts: RasterOptions) -> Result<RasterPage> {
    let (width_pt, height_pt) = (page.width().value, page.height().value);
    let Some(page_dpi) = render_dpi_within_budget(width_pt, height_pt, dpi) else {
        return Err(OcrError::Engine(format!(
            "pdfium page {idx}: {width_pt}×{height_pt} pt is too large to rasterize"
        )));
    };
    if page_dpi < dpi {
        tracing::warn!(
            page = idx + 1,
            requested_dpi = dpi,
            dpi = page_dpi,
            width_pt,
            height_pt,
            "page exceeds the rasterization pixel budget; rendering it at a lower DPI"
        );
    }
    let (target_w, target_h) = raster_size(width_pt, height_pt, page_dpi);
    let config = PdfRenderConfig::new()
        .set_target_width(target_w)
        .set_maximum_height(target_h);
    let bitmap = page
        .render_with_config(&config)
        .map_err(|e| OcrError::Engine(format!("pdfium render page {idx}: {e}")))?;
    let dyn_img = bitmap
        .as_image()
        .map_err(|e| OcrError::Engine(format!("pdfium as_image page {idx}: {e}")))?;
    let rgba = dyn_img.to_rgba8();
    let (width, height) = rgba.dimensions();
    Ok(RasterPage {
        index: idx,
        width,
        height,
        image: DynamicImage::ImageRgba8(rgba),
        dpi: page_dpi,
        native_dpi: if opts.native_dpi {
            image_only_page_dpi(page)
        } else {
            None
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const A4: (f32, f32) = (595.0, 842.0);

    #[test]
    fn vendored_pdfium_is_found_from_a_shared_target_dir_or_a_worktree() {
        // The failing layout: the exe sits in the main checkout's shared
        // target/release, but was built from a worktree that has no
        // vendor/pdfium/ of its own.
        let root = tempfile::tempdir().expect("tempdir");
        let main = root.path();
        let bin = main.join("vendor").join("pdfium").join("bin");
        std::fs::create_dir_all(&bin).expect("mkdir bin");
        std::fs::write(bin.join("pdfium.dll"), b"").expect("write lib");
        let exe_dir = main.join("target").join("release");
        let worktree_crate = main.join(".claude/worktrees/wt/crates/paddleknife-core");
        std::fs::create_dir_all(&exe_dir).expect("mkdir exe dir");
        std::fs::create_dir_all(&worktree_crate).expect("mkdir worktree");
        let lib = Path::new("pdfium.dll");

        let found = |starts: &[&Path]| find_vendored_pdfium(starts.iter().copied(), lib);
        assert_eq!(found(&[&exe_dir, &worktree_crate]), Some(bin.clone()));
        // An exe copied elsewhere still finds it through the worktree's
        // ancestors, which climb out of .claude/worktrees/ into main.
        let elsewhere = tempfile::tempdir().expect("tempdir");
        assert_eq!(found(&[elsewhere.path(), &worktree_crate]), Some(bin));
        // A bin dir without the library is not a match.
        assert_eq!(found(&[elsewhere.path()]), None);
    }

    #[test]
    fn ordinary_pages_render_at_the_requested_dpi() {
        // A4 at 300 DPI is 8.7 MP; A1 (1684 × 2384 pt) is 70 MP.
        assert_eq!(render_dpi_within_budget(A4.0, A4.1, 300), Some(300));
        assert_eq!(render_dpi_within_budget(1684.0, 2384.0, 300), Some(300));
        assert_eq!(render_dpi_within_budget(A4.1, A4.0, 600), Some(600));
    }

    #[test]
    fn an_oversized_page_is_clamped_to_the_pixel_budget() {
        // A0 (2384 × 3370 pt) at 300 DPI is 139 MP.
        let (w, h) = (2384.0, 3370.0);
        let dpi = render_dpi_within_budget(w, h, 300).expect("A0 renders");
        assert!(dpi < 300, "must clamp, got {dpi}");
        let (pw, ph) = raster_size(w, h, dpi);
        assert!(
            pw as u64 * ph as u64 <= MAX_RENDER_PIXELS,
            "{pw}×{ph} over budget"
        );
        // …and it is the highest whole DPI that fits, not an arbitrary cut.
        let (nw, nh) = raster_size(w, h, dpi + 1);
        assert!(nw as u64 * nh as u64 > MAX_RENDER_PIXELS);
    }

    #[test]
    fn a_banner_page_is_clamped_by_its_longest_side() {
        // 200 × 1 in: 18 MP at 300 DPI (under the area cap) but 60 000 px long.
        assert_eq!(render_dpi_within_budget(14_400.0, 72.0, 300), Some(100));
        assert_eq!(raster_size(14_400.0, 72.0, 100), (20_000, 100));
    }

    #[test]
    fn a_page_no_dpi_can_fit_is_refused() {
        // A side over 20 000 inches exceeds MAX_RENDER_SIDE even at 1 DPI.
        assert_eq!(render_dpi_within_budget(1.5e6, 72.0, 300), None);
        assert_eq!(render_dpi_within_budget(f32::INFINITY, 72.0, 300), None);
        assert_eq!(render_dpi_within_budget(f32::NAN, 72.0, 300), None);
    }

    #[test]
    fn degenerate_sizes_are_left_to_the_renderer() {
        // Nothing to bound: zero or negative sizes render no pixels.
        assert_eq!(render_dpi_within_budget(0.0, 0.0, 300), Some(300));
        assert_eq!(render_dpi_within_budget(-10.0, 842.0, 300), Some(300));
    }
}
