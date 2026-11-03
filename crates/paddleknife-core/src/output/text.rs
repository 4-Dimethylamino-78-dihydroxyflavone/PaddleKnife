//! Plain-text output writer.
//!
//! Emits `--- Page {n} ---` separators between pages (1-based for humans)
//! and one `TextLine` per output line. A `DocPage` with no lines still
//! emits its header so readers can see the page exists.

use crate::{error::Result, layout::DocPage};
use std::io::Write;

/// Write each page's header + text lines to `writer`.
pub fn write_text<W: Write>(writer: &mut W, pages: &[DocPage]) -> Result<()> {
    for page in pages {
        writeln!(writer, "--- Page {} ---", page.index + 1)?;
        for line in &page.lines {
            writeln!(writer, "{}", line.text)?;
        }
        // Blank line between pages for readability.
        writeln!(writer)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{DocPage, TextLine};
    use crate::ocr::OcrBlock;

    fn line(text: &str) -> TextLine {
        TextLine {
            bbox: [0, 0, 0, 0],
            text: text.to_string(),
            blocks: vec![OcrBlock {
                bbox: [0, 0, 0, 0],
                text: text.to_string(),
                confidence: 1.0,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    #[allow(clippy::unwrap_used)]
    fn empty_pages_writes_nothing() {
        let mut buf = Vec::new();
        write_text(&mut buf, &[]).unwrap();
        assert!(buf.is_empty());
    }

    #[test]
    #[allow(clippy::unwrap_used)]
    fn single_page_with_lines_has_header_and_trailer() {
        let pages = vec![DocPage {
            index: 0,
            size: (100, 100),
            lines: vec![line("hello"), line("world")],
            metrics: Default::default(),
        }];
        let mut buf = Vec::new();
        write_text(&mut buf, &pages).unwrap();
        let s = std::str::from_utf8(&buf).unwrap();
        assert_eq!(s, "--- Page 1 ---\nhello\nworld\n\n");
    }

    #[test]
    #[allow(clippy::unwrap_used)]
    fn multi_page_emits_separator_between_pages() {
        let pages = vec![
            DocPage {
                index: 0,
                size: (100, 100),
                lines: vec![line("p1")],
                metrics: Default::default(),
            },
            DocPage {
                index: 1,
                size: (100, 100),
                lines: vec![line("p2")],
                metrics: Default::default(),
            },
        ];
        let mut buf = Vec::new();
        write_text(&mut buf, &pages).unwrap();
        let s = std::str::from_utf8(&buf).unwrap();
        // Both pages get their own header; blank line between.
        assert!(s.contains("--- Page 1 ---\np1\n\n"));
        assert!(s.contains("--- Page 2 ---\np2\n\n"));
    }

    #[test]
    #[allow(clippy::unwrap_used)]
    fn empty_page_still_emits_header() {
        let pages = vec![DocPage {
            index: 4, // page 5 (1-based)
            size: (100, 100),
            lines: vec![],
            metrics: Default::default(),
        }];
        let mut buf = Vec::new();
        write_text(&mut buf, &pages).unwrap();
        let s = std::str::from_utf8(&buf).unwrap();
        assert_eq!(s, "--- Page 5 ---\n\n");
    }
}
