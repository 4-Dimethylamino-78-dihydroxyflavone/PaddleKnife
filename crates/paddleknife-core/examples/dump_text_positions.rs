//! Diagnostic: open a searchable PDF produced by the writer, dump each
//! text object's first 10 positions to see what pdfium actually wrote.
//!
//! Run with:
//!   cargo run --example dump_text_positions --release -- <path-to-pdf>
//!
//! Compares against what we *intended* the position to be (the union of
//! the per-word char-box bbox in pixel space, scaled to PDF points).

use paddleknife_core::pdf::bind_pdfium;
use pdfium_render::prelude::*;
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args()
        .nth(1)
        .expect("usage: dump_text_positions <pdf-path>");

    let pdfium = bind_pdfium()?;
    let doc = pdfium.load_pdf_from_file(&path, None)?;

    let pages = doc.pages();
    let page = pages.first().expect("at least one page");

    let page_w = page.width().value;
    let page_h = page.height().value;
    println!("Page 1: {:.1}pt × {:.1}pt", page_w, page_h);
    println!();

    let mut count = 0usize;
    for obj in page.objects().iter() {
        if let PdfPageObject::Text(ref t) = obj {
            let bounds = t.bounds()?;
            println!(
                "obj #{:03}  bounds=[L={:7.2} B={:7.2} R={:7.2} T={:7.2}]  size={:6.2}×{:5.2}  scaleH={:.3} scaleV={:.3}",
                count,
                bounds.left().value,
                bounds.bottom().value,
                bounds.right().value,
                bounds.top().value,
                bounds.width().value,
                bounds.height().value,
                t.get_horizontal_scale(),
                t.get_vertical_scale(),
            );
            count += 1;
            if count >= 25 {
                break;
            }
        }
    }
    println!();
    println!("Showed first {count} text objects (page has more).");
    Ok(())
}
