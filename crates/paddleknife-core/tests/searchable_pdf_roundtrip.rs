//! End-to-end test: OCR an image → emit searchable PDF → re-open and
//! extract text → assert the OCR'd words survived the text-layer trip.

use paddleknife_core::{
    image_io,
    layout::{band_sort, DocPage},
    model::ModelPaths,
    ocr::OcrEngine,
    output::searchable_pdf::write_searchable_pdf,
    pdf::RasterPage,
};
use pdfium_render::prelude::*;
use std::path::PathBuf;

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models")
}
fn sample_png() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/hello_world.png")
}

#[test]
#[serial_test::serial(pdfium)]
fn searchable_pdf_preserves_ocr_text() {
    // 1. OCR the PNG.
    let paths = ModelPaths::in_dir(&models_dir()).expect("models present");
    let mut engine = OcrEngine::from_paths(&paths).expect("engine init");
    let page = image_io::load_image(&sample_png()).expect("load image");
    let raster = RasterPage {
        index: 0,
        width: page.width,
        height: page.height,
        image: page.image.clone(),
        dpi: 300,
        native_dpi: None,
    };
    let blocks = engine.ocr(&page.image).expect("ocr ran");
    let doc_page = DocPage {
        index: 0,
        size: (page.width, page.height),
        lines: band_sort(blocks),
        metrics: Default::default(),
    };

    // 2. Write searchable PDF to temp location.
    let tmp = tempfile::tempdir().expect("tempdir");
    let out = tmp.path().join("hello_searchable.pdf");
    write_searchable_pdf(&out, &[doc_page], vec![raster], 300).expect("writer");
    assert!(out.exists(), "writer didn't produce a file");
    let file_size = std::fs::metadata(&out).expect("metadata").len();
    assert!(file_size > 0, "searchable PDF is empty");

    // 3. Re-open with pdfium and extract text. The writer already bound
    //    pdfium; Pdfium::default() gives us the cached bindings so we
    //    don't trip `PdfiumLibraryBindingsAlreadyInitialized`.
    let pdfium = paddleknife_core::pdf::bind_pdfium().expect("bind pdfium");
    let doc = pdfium
        .load_pdf_from_file(out.to_str().expect("utf8 path"), None)
        .expect("pdfium open");

    let mut combined = String::new();
    for p in doc.pages().iter() {
        combined.push_str(&p.text().expect("text extraction").all());
        combined.push('\n');
    }
    let lower = combined.to_ascii_lowercase();
    assert!(
        lower.contains("hello") || lower.contains("world"),
        "expected 'hello' or 'world' in extracted text; got: {combined:?}"
    );

    // 4. Convey that the image layer is there too by checking the page
    //    has at least one non-text object (the raster).
    let first_page = doc.pages().first().expect("at least one page");
    let obj_count = first_page.objects().len();
    assert!(
        obj_count >= 2,
        "expected ≥2 objects per page (image + text); got {obj_count}"
    );
}

/// Mismatched page counts must fail loudly.
#[test]
#[serial_test::serial(pdfium)]
fn mismatched_page_counts_error() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let out = tmp.path().join("wont_be_written.pdf");

    let doc_page = DocPage {
        index: 0,
        size: (100, 100),
        lines: vec![],
        metrics: Default::default(),
    };
    // Pass 1 doc_page but 0 rasters.
    let err = write_searchable_pdf(&out, &[doc_page], vec![], 300)
        .expect_err("should reject mismatched counts");
    let msg = format!("{err}");
    assert!(
        msg.contains("page count mismatch") || msg.contains("count"),
        "unexpected error: {msg}"
    );
    assert!(!out.exists(), "nothing should have been written");
}
