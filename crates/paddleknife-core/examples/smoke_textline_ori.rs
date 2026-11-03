//! Smoke test: build an OAROCR pipeline with the text-line orientation
//! classifier wired in, run it on the hello_world.png fixture, and
//! print whether `orientation_angle` got populated. Used as a one-shot
//! verification that the downloaded ONNX model loads and runs.
//!
//! cargo run --example smoke_textline_ori --release

use oar_ocr::oarocr::OAROCRBuilder;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let det = root.join("models/pp-ocrv5_mobile_det.onnx");
    let rec = root.join("models/pp-ocrv5_mobile_rec.onnx");
    let dict = root.join("models/ppocrv5_dict.txt");
    let textline_ori = root.join("models/pp-lcnet_x1_0_textline_ori.onnx");
    let png = root.join("tests/fixtures/hello_world.png");

    println!("Building OAROCR with text-line orientation classifier…");
    let ocr = OAROCRBuilder::new(&det, &rec, &dict)
        .with_text_line_orientation_classification(&textline_ori)
        .return_word_box(true)
        .build()?;
    println!("Build OK.");

    println!("Loading {png:?}");
    let img = image::open(&png)?.to_rgb8();
    println!("OCR-ing…");
    let results = ocr.predict(vec![img])?;
    println!("Got {} result(s)", results.len());

    let first = results.into_iter().next().expect("one result");
    for (i, region) in first.text_regions.iter().enumerate() {
        let text = region.text.as_deref().unwrap_or("");
        println!(
            "  region #{i}: text={text:?}  orientation_angle={:?}",
            region.orientation_angle,
        );
    }
    println!("Done.");
    Ok(())
}
