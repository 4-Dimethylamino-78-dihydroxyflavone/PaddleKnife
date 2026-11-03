//! Last-resort embed cap: `RasterPage::native_dpi`, pdfium's own view of
//! an image-only page.
//!
//! The source-PDF probe (lopdf) is deliberately strict — its whitelist
//! is `q Q cm Do`. Plenty of ordinary producers wrap the page image in a
//! clip path or set a graphics state first (Quartz, "Print to PDF"), and
//! the probe rightly refuses to analyze those. Before this fallback such
//! a page embedded the raster at the full OCR DPI: for a 96-DPI
//! screenshot rasterized at 300, 9.8× the source's pixels.
//!
//! A visible annotation vetoes the cap, as it does the probe's hint:
//! pdfium draws it into the raster, but it is not a page object, so the
//! image's resolution says nothing about it. An image mask isn't a page
//! object either; the probe's `masks` corrects the cap for one that is
//! sharper than its image.

use image::RgbImage;
use lopdf::{dictionary, Dictionary, Document, Object, Stream};
use paddleknife_core::output::searchable_pdf::{encode_page_opts, EmbedOptions, PageImage};
use paddleknife_core::pdf::{self, RasterPage};
use paddleknife_core::source_pdf::{MaskScale, SourcePdf};
use serial_test::serial;
use std::path::Path;

const IMG_W: u32 = 317;
const IMG_H: u32 = 239;
/// 317×239 px at 96 DPI.
const PAGE_W: f32 = 237.75;
const PAGE_H: f32 = 179.25;

fn jpeg(w: u32, h: u32) -> Vec<u8> {
    let mut out = Vec::new();
    let enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 80);
    RgbImage::from_fn(w, h, |x, y| {
        image::Rgb([(x % 256) as u8, (y % 256) as u8, 90])
    })
    .write_with_encoder(enc)
    .expect("encode fixture jpeg");
    out
}

/// One-page PDF: a 317×239 RGB JPEG filling a 96-DPI page, drawn by
/// `content` (which decides whether the lopdf probe can analyze it).
fn save_screenshot_pdf(path: &Path, content: &[u8]) {
    save_annotated_screenshot_pdf(path, content, Vec::new());
}

/// [`save_screenshot_pdf`] with `annots` as the page's /Annots.
fn save_annotated_screenshot_pdf(path: &Path, content: &[u8], annots: Vec<Dictionary>) {
    save_fixture_pdf(path, content, annots, None);
}

/// [`save_screenshot_pdf`] whose image carries a soft mask `scale` times
/// its own resolution per axis.
fn save_masked_screenshot_pdf(path: &Path, content: &[u8], scale: u32) {
    save_fixture_pdf(path, content, Vec::new(), Some(scale));
}

fn save_fixture_pdf(path: &Path, content: &[u8], annots: Vec<Dictionary>, smask: Option<u32>) {
    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();
    let mut image = dictionary! {
        "Type" => "XObject", "Subtype" => "Image",
        "Width" => IMG_W as i64, "Height" => IMG_H as i64,
        "ColorSpace" => "DeviceRGB", "BitsPerComponent" => 8i64,
        "Filter" => "DCTDecode",
    };
    if let Some(scale) = smask {
        // An opaque disc: a hard edge only the mask's own grid resolves.
        let (w, h) = (IMG_W * scale, IMG_H * scale);
        let (cx, cy, r) = (w as f32 / 2.0, h as f32 / 2.0, h as f32 * 0.4);
        let alpha: Vec<u8> = (0..h)
            .flat_map(|y| (0..w).map(move |x| (x, y)))
            .map(|(x, y)| {
                let d = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt();
                if d <= r {
                    255
                } else {
                    0
                }
            })
            .collect();
        let mask_id = doc.add_object(
            Stream::new(
                dictionary! {
                    "Type" => "XObject", "Subtype" => "Image",
                    "Width" => w as i64, "Height" => h as i64,
                    "ColorSpace" => "DeviceGray", "BitsPerComponent" => 8i64,
                },
                alpha,
            )
            .with_compression(false),
        );
        image.set("SMask", mask_id);
    }
    let img_id = doc.add_object(Stream::new(image, jpeg(IMG_W, IMG_H)).with_compression(false));
    let content_id =
        doc.add_object(Stream::new(dictionary! {}, content.to_vec()).with_compression(false));
    let mut page = dictionary! {
        "Type" => "Page",
        "Parent" => pages_id,
        "MediaBox" => vec![0.into(), 0.into(), PAGE_W.into(), PAGE_H.into()],
        "Resources" => dictionary! { "XObject" => dictionary! { "Im0" => img_id } },
        "Contents" => content_id,
    };
    if !annots.is_empty() {
        let refs: Vec<Object> = annots
            .into_iter()
            .map(|a| Object::Reference(doc.add_object(a)))
            .collect();
        page.set("Annots", refs);
    }
    let page_id = doc.add_object(page);
    doc.objects.insert(
        pages_id,
        Object::Dictionary(dictionary! {
            "Type" => "Pages", "Kids" => vec![page_id.into()], "Count" => 1,
        }),
    );
    let catalog_id = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages_id });
    doc.trailer.set("Root", catalog_id);
    doc.save(path).expect("save fixture pdf");
}

fn rasterize_only_page(path: &Path) -> RasterPage {
    let mut pages = pdf::rasterize_pdf(path, 300).expect("rasterize");
    assert_eq!(pages.len(), 1);
    pages.remove(0)
}

/// Quartz / Print-to-PDF shape: clip to the page, then draw the image.
/// The lopdf probe can't analyze it, so pdfium's native DPI is the only
/// thing that can cap the page.
const CLIPPED: &[u8] = b"q 0 0 237.75 179.25 re W n q 237.75 0 0 179.25 0 0 cm /Im0 Do Q Q";

fn annotation(subtype: &str, flags: Option<i64>) -> Dictionary {
    let mut d = dictionary! {
        "Type" => "Annot",
        "Subtype" => subtype,
        "Rect" => vec![20.into(), 20.into(), 120.into(), 80.into()],
    };
    if let Some(f) = flags {
        d.set("F", f);
    }
    d
}

#[test]
#[serial(pdfium)]
fn clip_wrapped_screenshot_is_capped_by_pdfiums_view_of_the_page() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("clipped.pdf");
    save_screenshot_pdf(&path, CLIPPED);

    // The strict probe can't vouch for this page — correctly: `re W n`
    // is outside its whitelist. No passthrough, and no hint either.
    let probe = SourcePdf::open(&path).expect("lopdf open").probe_page(0);
    assert!(probe.passthrough.is_none());
    assert!(probe.effective_dpi.is_none());

    // pdfium sees one image object and nothing else (a clip path is
    // graphics state, not a page object), and reports its resolution.
    let raster = rasterize_only_page(&path);
    assert_eq!((raster.width, raster.height), (991, 747), "300-DPI raster");
    let native = raster.native_dpi.expect("image-only page has a native DPI");
    assert!((native - 96.0).abs() < 0.5, "native dpi {native}");

    // …so the re-encode stores the page at the source's resolution, not
    // the OCR raster's.
    let enc =
        encode_page_opts(&raster, Some(&probe), &EmbedOptions::default(), 300).expect("encode");
    assert_eq!((enc.width, enc.height), (991, 747), "MediaBox basis kept");
    assert!(matches!(enc.image, PageImage::Jpeg { .. }));
    let (w, h) = enc.image.dims();
    let (embedded, source) = (w as u64 * h as u64, (IMG_W * IMG_H) as u64);
    assert!(
        embedded <= source * 102 / 100,
        "embedded {w}×{h} = {embedded} px exceeds the source's {source} px"
    );
    assert!(embedded >= source * 95 / 100, "over-shrunk to {w}×{h}");
}

#[test]
#[serial(pdfium)]
fn vector_content_means_no_native_dpi() {
    // The same screenshot with a filled rectangle drawn over it. Vector
    // art has no native resolution — it is genuinely sharper at 300 DPI —
    // so the page must NOT be capped to the image's 96.
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("with_vector.pdf");
    save_screenshot_pdf(
        &path,
        b"q 237.75 0 0 179.25 0 0 cm /Im0 Do Q 0.2 0.4 0.8 rg 20 20 80 40 re f",
    );
    let raster = rasterize_only_page(&path);
    assert_eq!(raster.native_dpi, None);

    let enc = encode_page_opts(&raster, None, &EmbedOptions::default(), 300).expect("encode");
    assert_eq!(enc.image.dims(), (raster.width, raster.height));
}

#[test]
#[serial(pdfium)]
fn visible_annotation_vetoes_the_native_dpi() {
    // A stamp over the screenshot is rendered at the full 300 DPI. It is
    // not a page object, so without the veto the page would be capped at
    // the image's 96 DPI and the stamp blurred with it.
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("stamped.pdf");
    save_annotated_screenshot_pdf(&path, CLIPPED, vec![annotation("Square", Some(4))]);
    let raster = rasterize_only_page(&path);
    assert_eq!(raster.native_dpi, None);

    let enc = encode_page_opts(&raster, None, &EmbedOptions::default(), 300).expect("encode");
    assert_eq!(enc.image.dims(), (raster.width, raster.height), "uncapped");
}

#[test]
#[serial(pdfium)]
fn links_and_hidden_annotations_keep_the_native_dpi() {
    let dir = tempfile::tempdir().expect("tempdir");
    for (name, annot) in [
        ("link", annotation("Link", None)),
        ("hidden", annotation("Square", Some(2))),
        ("noview", annotation("Square", Some(32))),
    ] {
        let path = dir.path().join(format!("{name}.pdf"));
        save_annotated_screenshot_pdf(&path, CLIPPED, vec![annot]);
        let native = rasterize_only_page(&path)
            .native_dpi
            .expect("still an image-only page");
        assert!((native - 96.0).abs() < 0.5, "{name}: native dpi {native}");
    }
}

#[test]
#[serial(pdfium)]
fn a_sharper_soft_mask_lifts_the_native_cap() {
    // MRC shape: a 96-DPI colour image under a 3× (288-DPI) soft mask,
    // behind a clip the strict probe refuses. pdfium measures the colour
    // image alone; the mask's edges are in the raster at full resolution,
    // and capping at 96 would blur them.
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("mrc.pdf");
    save_masked_screenshot_pdf(&path, CLIPPED, 3);

    let probe = SourcePdf::open(&path).expect("lopdf open").probe_page(0);
    assert!(
        probe.effective_dpi.is_none(),
        "clip: the probe can't measure it"
    );
    assert_eq!(probe.masks, MaskScale::Finer(3.0));

    let raster = rasterize_only_page(&path);
    let native = raster
        .native_dpi
        .expect("still an image-only page to pdfium");
    assert!((native - 96.0).abs() < 0.5, "native dpi {native}");

    // 96 × 3 = 288 DPI: within 5 % of the 300-DPI raster, kept whole.
    let enc =
        encode_page_opts(&raster, Some(&probe), &EmbedOptions::default(), 300).expect("encode");
    assert_eq!(enc.image.dims(), (raster.width, raster.height), "uncapped");
}
