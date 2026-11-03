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

#[test]
fn ocr_recognizes_hello_world() {
    let paths = ModelPaths::in_dir(&models_dir()).expect("models present");
    let mut engine = OcrEngine::from_paths(&paths).expect("engine init");
    let page = image_io::load_image(&sample_png()).expect("load image");
    let blocks = engine.ocr(&page.image).expect("ocr ran");

    assert!(
        !blocks.is_empty(),
        "detector should find at least one block"
    );
    let lines = band_sort(blocks);
    assert!(!lines.is_empty(), "band_sort should keep at least one line");
    let joined: String = lines
        .iter()
        .map(|l| l.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase();
    assert!(
        joined.contains("hello") || joined.contains("world"),
        "expected 'hello' or 'world' in OCR output, got: {joined:?}"
    );
}

/// End-to-end: OCR → band_sort → DocPage → text writer with page separator.
#[test]
fn writes_page_header_in_text_output() {
    let paths = ModelPaths::in_dir(&models_dir()).expect("models present");
    let mut engine = OcrEngine::from_paths(&paths).expect("engine init");
    let page = image_io::load_image(&sample_png()).expect("load image");
    let blocks = engine.ocr(&page.image).expect("ocr ran");

    let doc = DocPage {
        index: 0,
        size: (page.width, page.height),
        lines: band_sort(blocks),
        metrics: Default::default(),
    };
    let mut buf = Vec::new();
    output::text::write_text(&mut buf, &[doc]).expect("write_text");
    let s = std::str::from_utf8(&buf).expect("utf8 output");
    assert!(
        s.starts_with("--- Page 1 ---\n"),
        "expected page header, got: {s:?}"
    );
}
