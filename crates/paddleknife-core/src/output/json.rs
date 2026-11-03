//! JSON output writer.
//!
//! Emits the full document as pretty-printed JSON via serde_json. The
//! root is a JSON array of page objects — convenient for streaming
//! consumers that want to `jq '.[0].lines[0].text'` or similar.

use crate::{error::OcrError, error::Result, layout::DocPage};
use std::io::Write;

/// Write the pages as pretty-printed JSON to `writer`.
pub fn write_json<W: Write>(writer: &mut W, pages: &[DocPage]) -> Result<()> {
    serde_json::to_writer_pretty(writer, pages)
        .map_err(|e| OcrError::Engine(format!("json serialize: {e}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{DocPage, TextLine};
    use crate::ocr::OcrBlock;

    fn line(text: &str) -> TextLine {
        TextLine {
            bbox: [10, 20, 30, 40],
            text: text.to_string(),
            blocks: vec![OcrBlock {
                bbox: [10, 20, 30, 40],
                text: text.to_string(),
                confidence: 0.98,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    #[allow(clippy::unwrap_used)]
    fn empty_pages_writes_empty_array() {
        let mut buf = Vec::new();
        write_json(&mut buf, &[]).unwrap();
        let s = std::str::from_utf8(&buf).unwrap();
        assert_eq!(s, "[]");
    }

    #[test]
    #[allow(clippy::unwrap_used)]
    fn roundtrip_preserves_data() {
        let pages = vec![DocPage {
            index: 0,
            size: (800, 600),
            lines: vec![line("hello"), line("world")],
            metrics: Default::default(),
        }];

        let mut buf = Vec::new();
        write_json(&mut buf, &pages).unwrap();

        // Parse back and check key fields survive the round trip.
        let parsed: Vec<DocPage> = serde_json::from_slice(&buf).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].index, 0);
        assert_eq!(parsed[0].size, (800, 600));
        assert_eq!(parsed[0].lines.len(), 2);
        assert_eq!(parsed[0].lines[0].text, "hello");
        assert_eq!(parsed[0].lines[0].blocks[0].confidence, 0.98);
    }

    #[test]
    #[allow(clippy::unwrap_used)]
    fn output_is_valid_pretty_json() {
        let pages = vec![DocPage {
            index: 2,
            size: (100, 200),
            lines: vec![line("x")],
            metrics: Default::default(),
        }];
        let mut buf = Vec::new();
        write_json(&mut buf, &pages).unwrap();
        let s = std::str::from_utf8(&buf).unwrap();
        // Pretty-printed output contains indentation.
        assert!(s.contains("\n  "));
        // And it's valid JSON.
        let v: serde_json::Value = serde_json::from_str(s).unwrap();
        assert!(v.is_array());
    }
}
