//! Real-file Devanagari round-trip — `#[ignore]`d by default so it
//! doesn't run in CI, but invokable manually to spot-check against a
//! freshly-generated searchable PDF from the Muna Madan fixture.
//! That scan is not distributed with the repository (it is gitignored);
//! any Devanagari book scan at the same path works.
//!
//! Run with:
//!   cargo test --test real_pdf_extraction -- --ignored --nocapture

use paddleknife_core::pdf::bind_pdfium;
use std::path::PathBuf;

fn muna_madan_pdf() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test-output/muna_madan_searchable_v2.pdf")
}

#[test]
#[ignore]
#[serial_test::serial(pdfium)]
fn muna_madan_searchable_pdf_has_extractable_nepali() {
    let path = muna_madan_pdf();
    assert!(
        path.exists(),
        "expected searchable PDF at {} — generate with:\n  \
         paddleknife ocr tests/fixtures/MunaMadan_full.pdf --script devanagari --format pdf \
         --output test-output/muna_madan_searchable_v2.pdf",
        path.display()
    );

    // Explicitly bind pdfium via our own idempotent helper — Pdfium::default()
    // assumes the bindings are already loaded, which isn't the case here
    // because this test only reads an existing PDF (it never calls the writer).
    let pdfium = bind_pdfium().expect("bind pdfium");
    let doc = pdfium
        .load_pdf_from_file(path.to_str().expect("utf8 path"), None)
        .expect("open pdf");

    // Scan every page for any Devanagari codepoint (U+0900..U+097F).
    // Success here means:
    //   (a) The text layer is present (not just rasters).
    //   (b) The ToUnicode CMap maps glyphs back to Devanagari codepoints
    //       correctly — i.e. copy-paste will yield real Unicode.
    //   (c) pdfium's extractor doesn't choke on the content stream — if
    //       it did, readers opening the PDF would likely choke too.
    let mut dev_chars = 0usize;
    let mut total_chars = 0usize;
    for p in doc.pages().iter() {
        let text = p.text().expect("extract").all();
        for c in text.chars() {
            total_chars += 1;
            if ('\u{0900}'..='\u{097F}').contains(&c) {
                dev_chars += 1;
            }
        }
    }
    println!(
        "extracted chars: total={total_chars} devanagari={dev_chars} \
         ({:.1}%)",
        100.0 * dev_chars as f32 / total_chars.max(1) as f32
    );
    assert!(
        dev_chars > 100,
        "expected hundreds of Devanagari codepoints; got {dev_chars}. \
         Text layer may still be using a non-Unicode font."
    );
}
