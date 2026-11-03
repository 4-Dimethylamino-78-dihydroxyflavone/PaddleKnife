//! Generate a small searchable PDF for manual inspection.
//! cargo run -p paddleknife-core --example debug_writer

use image::DynamicImage;
use paddleknife_core::layout::{DocPage, TextLine};
use paddleknife_core::ocr::{CharSpan, OcrBlock};
use paddleknife_core::output::searchable_pdf::write_searchable_pdf;
use paddleknife_core::pdf::RasterPage;

fn main() {
    let (w, h) = (800u32, 600u32);
    let img = image::RgbImage::from_pixel(w, h, image::Rgb([255, 255, 255]));
    let raster = RasterPage {
        index: 0,
        width: w,
        height: h,
        image: DynamicImage::ImageRgb8(img),
        dpi: 300,
        native_dpi: None,
    };
    // "Hi there" with handcrafted char boxes on one line at y 100..140.
    let mk = |t: &str, x0: f32, x1: f32| CharSpan {
        text: t.to_string(),
        quad: [[x0, 100.0], [x1, 100.0], [x1, 140.0], [x0, 140.0]],
    };
    let b1 = OcrBlock {
        bbox: [100, 100, 205, 140],
        text: "Wide".into(),
        confidence: 0.99,
        chars: vec![
            mk("W", 100.0, 145.0),
            mk("i", 145.0, 155.0),
            mk("d", 155.0, 180.0),
            mk("e", 180.0, 205.0),
        ],
        quad: Some([
            [100.0, 100.0],
            [205.0, 100.0],
            [205.0, 140.0],
            [100.0, 140.0],
        ]),
        ..Default::default()
    };
    let b2 = OcrBlock {
        bbox: [200, 100, 350, 140],
        text: "there".into(),
        confidence: 0.99,
        chars: vec![
            mk("t", 200.0, 230.0),
            mk("h", 230.0, 260.0),
            mk("e", 260.0, 290.0),
            mk("r", 290.0, 320.0),
            mk("e", 320.0, 350.0),
        ],
        quad: Some([
            [200.0, 100.0],
            [350.0, 100.0],
            [350.0, 140.0],
            [200.0, 140.0],
        ]),
        ..Default::default()
    };
    let line = TextLine {
        bbox: [100, 100, 350, 140],
        text: "Hi there".into(),
        blocks: vec![b1, b2],
        ..Default::default()
    };
    let page = DocPage {
        index: 0,
        size: (w, h),
        lines: vec![line],
        metrics: Default::default(),
    };
    std::fs::create_dir_all("test-output").ok();
    write_searchable_pdf(
        std::path::Path::new("test-output/debug_writer.pdf"),
        &[page],
        vec![raster],
        300,
    )
    .expect("write");
    println!("wrote test-output/debug_writer.pdf");
}
