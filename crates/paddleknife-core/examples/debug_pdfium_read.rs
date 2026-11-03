//! Interrogate pdfium about test-output/debug_writer.pdf.
//! cargo run -p paddleknife-core --example debug_pdfium_read

use pdfium_render::prelude::*;

fn main() {
    let pdfium = paddleknife_core::pdf::bind_pdfium().expect("bind");
    let doc = pdfium
        .load_pdf_from_file("test-output/debug_writer.pdf", None)
        .expect("open");
    let page = doc.pages().first().expect("page");
    println!("page objects: {}", page.objects().len());
    for (i, obj) in page.objects().iter().enumerate() {
        println!("  obj {}: {:?}", i, obj.object_type());
    }
    let text = page.text().expect("text page");
    println!("chars: {}", text.chars().len());
    println!("all(): {:?}", text.all());
    for ch in text.chars().iter().take(12) {
        println!(
            "  char {:?} unicode={:?} loose={:?}",
            ch.index(),
            ch.unicode_char(),
            ch.loose_bounds()
        );
    }
}
