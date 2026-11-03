//! Markdown output writer.
//!
//! `# Page {n}` heading per page (1-based), one line per TextLine,
//! blank line between pages. Lines are emitted verbatim without
//! escaping — assumes downstream consumers will either render the
//! text as-is or do their own escaping (OCR'd text rarely contains
//! markdown control characters that matter).

use crate::{error::Result, layout::DocPage};
use std::io::Write;

/// Write the pages as Markdown to `writer`.
pub fn write_markdown<W: Write>(writer: &mut W, pages: &[DocPage]) -> Result<()> {
    for (i, page) in pages.iter().enumerate() {
        if i > 0 {
            // Explicit blank line between pages to keep headings separated
            // even if the previous page had no lines.
            writeln!(writer)?;
        }
        writeln!(writer, "# Page {}", page.index + 1)?;
        writeln!(writer)?;
        for line in &page.lines {
            writeln!(writer, "{}", line.text)?;
        }
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
        write_markdown(&mut buf, &[]).unwrap();
        assert!(buf.is_empty());
    }

    #[test]
    #[allow(clippy::unwrap_used)]
    fn single_page_has_heading_and_lines() {
        let pages = vec![DocPage {
            index: 0,
            size: (100, 100),
            lines: vec![line("hello"), line("world")],
            metrics: Default::default(),
        }];
        let mut buf = Vec::new();
        write_markdown(&mut buf, &pages).unwrap();
        let s = std::str::from_utf8(&buf).unwrap();
        assert_eq!(s, "# Page 1\n\nhello\nworld\n");
    }

    #[test]
    #[allow(clippy::unwrap_used)]
    fn multi_page_separates_with_blank_line() {
        let pages = vec![
            DocPage {
                index: 0,
                size: (100, 100),
                lines: vec![line("first")],
                metrics: Default::default(),
            },
            DocPage {
                index: 1,
                size: (100, 100),
                lines: vec![line("second")],
                metrics: Default::default(),
            },
        ];
        let mut buf = Vec::new();
        write_markdown(&mut buf, &pages).unwrap();
        let s = std::str::from_utf8(&buf).unwrap();
        assert!(s.contains("# Page 1\n\nfirst\n\n# Page 2\n\nsecond\n"));
    }

    #[test]
    #[allow(clippy::unwrap_used)]
    fn empty_page_still_emits_heading() {
        let pages = vec![DocPage {
            index: 3,
            size: (100, 100),
            lines: vec![],
            metrics: Default::default(),
        }];
        let mut buf = Vec::new();
        write_markdown(&mut buf, &pages).unwrap();
        let s = std::str::from_utf8(&buf).unwrap();
        assert_eq!(s, "# Page 4\n\n");
    }
}
