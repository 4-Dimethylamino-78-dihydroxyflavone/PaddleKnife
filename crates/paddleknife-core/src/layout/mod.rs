//! Layout passes: turn a flat list of detected `OcrBlock`s into reading-
//! ordered lines and pages.
//!
//! The PP-OCR detector emits bboxes in whatever order the head's heatmap
//! components surface them — roughly left-to-right but not line-aware,
//! and multi-column pages come out interleaved. `band_sort` fixes that
//! with a two-pass sort:
//!
//! 1. Group blocks into horizontal "bands" (same visual line) by comparing
//!    y-centers against a dynamic threshold (median block height × 0.5).
//! 2. Within each band, sort by x-min (left-to-right).
//!
//! The output is a `Vec<TextLine>`. A page is `DocPage { index, size, lines }`.

pub mod band_sort;

pub use band_sort::{band_sort, DocPage, PageMetrics, TextLine};
