//! Integration tests — OCR hello_world.png once, then round-trip through
//! each writer and assert output is well-formed.

use paddleknife_core::{
    image_io,
    layout::{band_sort, DocPage},
    model::ModelPaths,
    ocr::OcrEngine,
    output,
};
use std::path::PathBuf;

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models")
}
fn sample_png() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/hello_world.png")
}

fn ocr_to_docpage() -> DocPage {
    let paths = ModelPaths::in_dir(&models_dir()).expect("models present");
    let mut engine = OcrEngine::from_paths(&paths).expect("engine init");
    let page = image_io::load_image(&sample_png()).expect("load image");
    let blocks = engine.ocr(&page.image).expect("ocr ran");
    let metrics = paddleknife_core::layout::PageMetrics::from_blocks(&blocks);
    DocPage {
        index: 0,
        size: (page.width, page.height),
        lines: band_sort(blocks),
        metrics,
    }
}

#[test]
fn txt_has_page_header_and_ocr_words() {
    let doc = ocr_to_docpage();
    let mut buf = Vec::new();
    output::text::write_text(&mut buf, &[doc]).expect("txt writer");
    let s = String::from_utf8(buf).expect("utf8");
    assert!(
        s.starts_with("--- Page 1 ---"),
        "missing page header: {s:?}"
    );
    let lower = s.to_ascii_lowercase();
    assert!(
        lower.contains("hello") || lower.contains("world"),
        "OCR words missing from txt: {s:?}"
    );
}

#[test]
fn markdown_has_heading_and_ocr_words() {
    let doc = ocr_to_docpage();
    let mut buf = Vec::new();
    output::markdown::write_markdown(&mut buf, &[doc]).expect("md writer");
    let s = String::from_utf8(buf).expect("utf8");
    assert!(s.starts_with("# Page 1"), "missing heading: {s:?}");
    let lower = s.to_ascii_lowercase();
    assert!(
        lower.contains("hello") || lower.contains("world"),
        "OCR words missing from md: {s:?}"
    );
}

#[test]
fn json_parses_and_has_ocr_words() {
    let doc = ocr_to_docpage();
    let mut buf = Vec::new();
    output::json::write_json(&mut buf, &[doc]).expect("json writer");

    // Valid JSON.
    let v: serde_json::Value = serde_json::from_slice(&buf).expect("valid JSON");
    assert!(v.is_array(), "root is not a JSON array");
    let arr = v.as_array().expect("array");
    assert_eq!(arr.len(), 1, "expected one page");

    // "hello" or "world" must appear somewhere in the serialized text.
    let s = String::from_utf8(buf).expect("utf8");
    let lower = s.to_ascii_lowercase();
    assert!(
        lower.contains("hello") || lower.contains("world"),
        "OCR words missing from json: {s:?}"
    );
}

// The searchable-PDF path has its own integration test file
// (tests/searchable_pdf_roundtrip.rs) that binds pdfium; no need to
// duplicate it here.
