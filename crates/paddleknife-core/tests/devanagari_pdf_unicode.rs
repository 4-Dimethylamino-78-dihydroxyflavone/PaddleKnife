//! Devanagari round-trip test for the searchable-PDF writer.
//!
//! Builds a synthetic DocPage carrying real Nepali text, writes a
//! searchable PDF with the Devanagari font, re-opens it, and asserts
//! pdfium's text extractor recovers the Unicode codepoints. This
//! exercises the ToUnicode CMap path — the Plan-2 Helvetica writer
//! would have silently produced garbage or crashed readers here.

use image::{DynamicImage, RgbImage};
use paddleknife_core::{
    layout::{DocPage, TextLine},
    ocr::OcrBlock,
    output::searchable_pdf::write_searchable_pdf,
    pdf::RasterPage,
};

/// A snippet from the Muna Madan dedication verse — all Devanagari
/// codepoints (U+0900..U+097F), the exact failure mode Plan 2 suffered.
const NEPALI_TEXT: &str = "हे मेरा भाइ";

fn synth_raster(width: u32, height: u32) -> RasterPage {
    // Plain white page — we only care about the text layer round-trip.
    let img = RgbImage::from_pixel(width, height, image::Rgb([255, 255, 255]));
    RasterPage {
        index: 0,
        width,
        height,
        image: DynamicImage::ImageRgb8(img),
        dpi: 300,
        native_dpi: None,
    }
}

fn synth_docpage(text: &str, width: u32, height: u32) -> DocPage {
    // One text line at (100,100)..(600,160). Font sized from bbox height.
    let block = OcrBlock {
        bbox: [100, 100, 600, 160],
        text: text.to_string(),
        confidence: 0.99,
        ..Default::default()
    };
    let line = TextLine {
        bbox: [100, 100, 600, 160],
        text: text.to_string(),
        blocks: vec![block],
        ..Default::default()
    };
    DocPage {
        index: 0,
        size: (width, height),
        lines: vec![line],
        metrics: Default::default(),
    }
}

#[test]
#[serial_test::serial(pdfium)]
fn devanagari_text_layer_roundtrips_unicode() {
    let (w, h) = (800_u32, 1000_u32);
    let raster = synth_raster(w, h);
    let doc_page = synth_docpage(NEPALI_TEXT, w, h);

    let tmp = tempfile::tempdir().expect("tempdir");
    let out = tmp.path().join("devanagari_roundtrip.pdf");
    write_searchable_pdf(&out, &[doc_page], vec![raster], 300).expect("writer");

    // Re-open and extract text.
    let pdfium = paddleknife_core::pdf::bind_pdfium().expect("bind pdfium");
    let doc = pdfium
        .load_pdf_from_file(out.to_str().expect("utf8 path"), None)
        .expect("pdfium open");
    let page = doc.pages().first().expect("at least one page");
    let extracted = page.text().expect("text extraction").all();

    assert!(
        extracted.contains(NEPALI_TEXT),
        "expected Devanagari text to round-trip; got: {extracted:?}"
    );
}

#[test]
#[serial_test::serial(pdfium)]
fn latin_text_layer_still_roundtrips_after_switch_to_notosans() {
    // Make sure the Plan-2 Latin path didn't regress when we switched
    // from built-in Helvetica to embedded Noto Sans.
    let (w, h) = (800, 1000);
    let raster = synth_raster(w, h);
    let doc_page = synth_docpage("Hello World", w, h);

    let tmp = tempfile::tempdir().expect("tempdir");
    let out = tmp.path().join("latin_roundtrip.pdf");
    write_searchable_pdf(&out, &[doc_page], vec![raster], 300).expect("writer");

    let pdfium = paddleknife_core::pdf::bind_pdfium().expect("bind pdfium");
    let doc = pdfium
        .load_pdf_from_file(out.to_str().expect("utf8 path"), None)
        .expect("pdfium open");
    let page = doc.pages().first().expect("at least one page");
    let extracted = page.text().expect("text extraction").all();

    let lower = extracted.to_ascii_lowercase();
    assert!(
        lower.contains("hello") && lower.contains("world"),
        "Latin round-trip lost text; got: {extracted:?}"
    );
}
