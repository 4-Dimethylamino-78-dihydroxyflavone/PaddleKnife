//! Rasterizer bounds for untrusted input (2026-09-30 audit).
//!
//! A PDF can declare any page size, and pdfium would happily allocate the
//! bitmap: an A0 page at 300 DPI is 558 MB as RGBA before OCR copies it.
//! The rasterizer renders such a page at a lower DPI instead — and that
//! DPI travels with the raster, so the searchable PDF keeps the page's
//! physical size and its text layer stays aligned with the image.

use lopdf::{dictionary, Document, Object};
use paddleknife_core::layout::DocPage;
use paddleknife_core::output::searchable_pdf::{
    encode_page_opts, write_searchable_pdf_encoded, EmbedOptions,
};
use paddleknife_core::pdf::{self, MAX_RENDER_SIDE};
use serial_test::serial;
use std::path::Path;

/// 200 × 1 in. At 300 DPI that is 60 000 px long, so the longest-side cap
/// clamps it to 100 DPI: 20 000 × 100 px, 8 MB as RGBA — cheap to render.
const BANNER_W: f32 = 14_400.0;
const BANNER_H: f32 = 72.0;

/// One blank page of `w` × `h` points.
fn save_blank_page(path: &Path, w: f32, h: f32) {
    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();
    let page_id = doc.add_object(dictionary! {
        "Type" => "Page",
        "Parent" => pages_id,
        "MediaBox" => vec![0.into(), 0.into(), w.into(), h.into()],
    });
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

#[test]
#[serial(pdfium)]
fn oversized_page_is_rendered_within_budget_at_a_recorded_dpi() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("banner.pdf");
    save_blank_page(&path, BANNER_W, BANNER_H);

    let raster = pdf::rasterize_page(&path, 0, 300).expect("rasterize");
    assert_eq!(raster.dpi, 100, "clamped by the longest-side cap");
    assert_eq!((raster.width, raster.height), (MAX_RENDER_SIDE, 100));

    // The streaming path clamps the same way.
    let mut streamed = Vec::new();
    pdf::for_each_page(&path, 300, |page| {
        streamed.push((page.dpi, page.width, page.height));
        Ok(())
    })
    .expect("stream");
    assert_eq!(streamed, vec![(100, MAX_RENDER_SIDE, 100)]);

    // The recorded DPI reaches the writer: the output page is still
    // 200 × 1 in, not 200 × 1 in scaled by 100/300.
    let enc = encode_page_opts(&raster, None, &EmbedOptions::default(), 300).expect("encode");
    assert_eq!(enc.dpi, 100);
    let page = DocPage {
        index: 0,
        size: (raster.width, raster.height),
        lines: Vec::new(),
        metrics: Default::default(),
    };
    let out = dir.path().join("banner_out.pdf");
    write_searchable_pdf_encoded(&out, &[page], vec![enc], 300).expect("write");

    let doc = Document::load(&out).expect("reopen output");
    let page_id = *doc.get_pages().values().next().expect("one page");
    let mb: Vec<f32> = doc
        .get_dictionary(page_id)
        .and_then(|d| d.get(b"MediaBox"))
        .and_then(Object::as_array)
        .expect("MediaBox")
        .iter()
        .map(|v| v.as_float().expect("number"))
        .collect();
    assert_eq!(mb.len(), 4);
    assert!(
        (mb[2] - BANNER_W).abs() < 0.5 && (mb[3] - BANNER_H).abs() < 0.5,
        "MediaBox {mb:?} should keep the source page's size"
    );
}

#[test]
#[serial(pdfium)]
fn ordinary_page_renders_at_the_requested_dpi() {
    // US Letter at 300 DPI (8.4 MP) is nowhere near the budget.
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("letter.pdf");
    save_blank_page(&path, 612.0, 792.0);
    let raster = pdf::rasterize_page(&path, 0, 300).expect("rasterize");
    assert_eq!(raster.dpi, 300);
    assert_eq!((raster.width, raster.height), (2550, 3300));
}
