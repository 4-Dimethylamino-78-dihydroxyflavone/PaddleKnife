//! Integration test: load a PDF page, rasterize it, assert its dimensions.

use lopdf::{dictionary, Document, Object};
use paddleknife_core::pdf;
use std::path::Path;

/// One blank US Letter page (612 × 792 pt).
fn save_letter_page(path: &Path) {
    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();
    let page_id = doc.add_object(dictionary! {
        "Type" => "Page",
        "Parent" => pages_id,
        "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
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
#[serial_test::serial(pdfium)]
fn rasterizes_first_page_at_300_dpi() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("letter.pdf");
    save_letter_page(&path);

    let pages = pdf::rasterize_pdf(&path, 300).expect("rasterize");
    assert_eq!(pages.len(), 1);
    let first = &pages[0];
    assert_eq!((first.width, first.height), (2550, 3300), "8.5 × 11 in at 300 DPI");
    assert_eq!(first.image.color(), image::ColorType::Rgba8);
}
