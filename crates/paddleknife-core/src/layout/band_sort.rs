//! Band-sort: group OCR blocks into reading-ordered lines.

use crate::geometry::{self, Quad};
use crate::ocr::OcrBlock;

/// One visual line of text, reconstructed from one or more detector bboxes.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct TextLine {
    /// Union bbox: min-x, min-y, max-x, max-y across `blocks`.
    pub bbox: [i32; 4],
    /// Text of the blocks in reading order, joined per script policy
    /// (space for spaced scripts, nothing between CJK/Thai neighbors).
    pub text: String,
    /// Orientation angle in degrees, inherited from the first block in
    /// the line. Blocks sharing a visual line share an orientation.
    #[serde(default)]
    pub rotation_deg: f32,
    /// The individual blocks making up this line, in reading order
    /// (left-to-right, or right-to-left for majority-RTL lines).
    pub blocks: Vec<OcrBlock>,
    /// Union quad across `blocks` (axis-aligned envelope for tilted
    /// content). `None` on JSON predating Plan 5.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quad: Option<Quad>,
}

/// A page of text lines, ready for any writer.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DocPage {
    /// 0-based page index within the source document.
    pub index: usize,
    /// Source raster dimensions in pixels (width, height).
    pub size: (u32, u32),
    /// Lines in reading order (top-to-bottom).
    pub lines: Vec<TextLine>,
    /// Page-level confidence summary. Computed from the underlying
    /// `OcrBlock::confidence` values at OCR-time, preserved through
    /// `band_sort` so writers and downstream consumers can flag
    /// low-quality pages without re-scanning block arrays.
    #[serde(default)]
    pub metrics: PageMetrics,
}

/// Aggregate confidence statistics for one page.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct PageMetrics {
    /// Mean OCR confidence across all blocks. 0.0 if the page is empty.
    pub avg_confidence: f32,
    /// Smallest confidence on the page. 1.0 if the page is empty
    /// (so it doesn't trip a low-conf filter).
    pub min_confidence: f32,
    /// Count of lines containing any block with confidence < 0.7.
    pub low_conf_lines: usize,
    /// Total block count — useful for distinguishing empty pages.
    pub total_blocks: usize,
}

impl PageMetrics {
    /// Compute metrics from a slice of blocks. `band_sort` calls this
    /// before re-grouping, so the data reflects the raw OCR output.
    pub fn from_blocks(blocks: &[OcrBlock]) -> Self {
        if blocks.is_empty() {
            return Self {
                avg_confidence: 0.0,
                min_confidence: 1.0,
                low_conf_lines: 0,
                total_blocks: 0,
            };
        }
        let n = blocks.len();
        let sum: f32 = blocks.iter().map(|b| b.confidence).sum();
        let min = blocks
            .iter()
            .map(|b| b.confidence)
            .fold(f32::INFINITY, f32::min);
        let low = blocks.iter().filter(|b| b.confidence < 0.7).count();
        Self {
            avg_confidence: sum / n as f32,
            min_confidence: if min.is_finite() { min } else { 0.0 },
            low_conf_lines: low,
            total_blocks: n,
        }
    }
}

/// Which way a block's text runs, inferred from its detected quad.
///
/// A page can mix both — margin notes and spine text sit at 90° to the
/// body — so this is decided per block, not per page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Axis {
    /// Text advances along +x; lines stack along +y. The common case.
    Horizontal,
    /// Text advances along +y; lines stack along +x (rotated 90°).
    Vertical,
}

/// Infer a block's reading axis from the direction its text advances.
///
/// `ocr::normalize_quad` orders every quad along the reading direction, so
/// the top edge `tl→tr` is the advance vector whatever the page rotation.
/// Comparing that vector's components is exact and needs no aspect-ratio
/// threshold — a lone narrow glyph like "l" still advances horizontally,
/// and a rotated line still advances vertically however short it is.
///
/// Blocks with no quad (legacy JSON) fall back to their bbox, whose
/// advance vector is horizontal by construction — the safe default.
fn block_axis(b: &OcrBlock) -> Axis {
    let q = b.quad_or_bbox();
    let dx = (q[1][0] - q[0][0]).abs();
    let dy = (q[1][1] - q[0][1]).abs();
    if dy > dx {
        Axis::Vertical
    } else {
        Axis::Horizontal
    }
}

impl Axis {
    /// Coordinate that lines stack along — the one we group bands by.
    fn band_centre(self, b: &OcrBlock) -> f32 {
        match self {
            Self::Horizontal => (b.bbox[1] + b.bbox[3]) as f32 * 0.5,
            Self::Vertical => (b.bbox[0] + b.bbox[2]) as f32 * 0.5,
        }
    }
    /// Block extent across the reading direction (its "line height").
    fn line_extent(self, b: &OcrBlock) -> i32 {
        match self {
            Self::Horizontal => b.bbox[3] - b.bbox[1],
            Self::Vertical => b.bbox[2] - b.bbox[0],
        }
    }
    /// Leading edge along the reading direction — the within-band sort key.
    fn order_start(self, b: &OcrBlock) -> i32 {
        match self {
            Self::Horizontal => b.bbox[0],
            Self::Vertical => b.bbox[1],
        }
    }
    /// Trailing edge along the reading direction, for gap detection.
    fn order_end(self, b: &OcrBlock) -> i32 {
        match self {
            Self::Horizontal => b.bbox[2],
            Self::Vertical => b.bbox[3],
        }
    }
}

/// Group `blocks` into reading-ordered lines.
///
/// Blocks are first split by reading axis, because banding coordinates
/// only make sense within one axis: horizontal text bands by y and orders
/// by x, vertical text bands by x and orders by y. Horizontal lines are
/// emitted first, then vertical ones — body text before margin text.
///
/// Within an axis the algorithm is unchanged:
/// 1. Median line extent → band threshold = median * 0.5. Median (not
///    mean) keeps oversized headings from blowing up the threshold.
/// 2. Sort by band coordinate ascending.
/// 3. A block joins the current band if its band coordinate is within
///    `threshold` of the band's running mean; else start a new band.
/// 4. Within each band, sort along the reading direction and split wide
///    gaps into columns.
/// 5. Emit each column as a `TextLine` with joined text and union bbox.
pub fn band_sort(blocks: Vec<OcrBlock>) -> Vec<TextLine> {
    if blocks.is_empty() {
        return Vec::new();
    }
    let (horizontal, vertical): (Vec<OcrBlock>, Vec<OcrBlock>) = blocks
        .into_iter()
        .partition(|b| block_axis(b) == Axis::Horizontal);

    let mut lines = band_sort_axis(horizontal, Axis::Horizontal);
    lines.extend(band_sort_axis(vertical, Axis::Vertical));
    lines
}

/// Band-sort blocks that all share one reading axis.
fn band_sort_axis(blocks: Vec<OcrBlock>, axis: Axis) -> Vec<TextLine> {
    if blocks.is_empty() {
        return Vec::new();
    }

    // Median line extent → band-grouping threshold.
    let mut heights: Vec<i32> = blocks.iter().map(|b| axis.line_extent(b)).collect();
    heights.sort_unstable();
    let median_h = heights[heights.len() / 2].max(1);
    let threshold = (median_h as f32) * 0.5;

    // Annotate each block with its band coordinate, then sort.
    let mut annotated: Vec<(f32, OcrBlock)> = blocks
        .into_iter()
        .map(|b| (axis.band_centre(&b), b))
        .collect();
    annotated.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Walk top-to-bottom, grouping into bands.
    let mut bands: Vec<Vec<(f32, OcrBlock)>> = Vec::new();
    for (yc, block) in annotated {
        let join_existing = bands.last().is_some_and(|band| {
            let band_yc_mean: f32 = band.iter().map(|(y, _)| *y).sum::<f32>() / band.len() as f32;
            (yc - band_yc_mean).abs() <= threshold
        });
        if join_existing {
            if let Some(last) = bands.last_mut() {
                last.push((yc, block));
            }
        } else {
            bands.push(vec![(yc, block)]);
        }
    }

    // Within each band, sort by x-min; compute a BAND-LOCAL gap threshold
    // (NOT page-wide — see the comment below); split wide x-gaps into
    // sub-bands (columns); emit each sub-band as a TextLine.
    //
    // Why per-band: a page that mixes large invoice headers (h≈40) with
    // small receipt text (h≈20) would have a page-median of ~40. The
    // 2.5× threshold = 100 px then fails to split the 60 px column gutter
    // in the receipt section. Using each band's own median heights as the
    // reference means the threshold adapts to the local text size.
    let mut lines = Vec::new();
    for mut band in bands {
        band.sort_by_key(|(_, b)| axis.order_start(b));
        let local_h = band_median_h(&band, axis).max(1.0);
        let gap_threshold = local_h * 2.5;
        let column_groups = split_by_reading_gap(band, gap_threshold, axis);
        for group in column_groups {
            let mut blocks: Vec<OcrBlock> = group.into_iter().map(|(_, b)| b).collect();
            // Majority-RTL lines (Arabic, Hebrew) read right-to-left:
            // reverse the ascending order so `text` and `blocks` are
            // in logical order, not visual-LTR order. Only meaningful for
            // horizontal text — RTL is a horizontal-axis property.
            if axis == Axis::Horizontal && is_majority_rtl(&blocks) {
                blocks.sort_by_key(|b| std::cmp::Reverse(b.bbox[0]));
            }
            let bbox = union_bbox(&blocks);
            let text = join_blocks(&blocks);
            let rotation_deg = blocks.first().map_or(0.0, |b| b.rotation_deg);
            let quad = blocks
                .iter()
                .map(OcrBlock::quad_or_bbox)
                .reduce(|a, b| geometry::union(&a, &b));
            lines.push(TextLine {
                bbox,
                text,
                rotation_deg,
                blocks,
                quad,
            });
        }
    }
    lines
}

/// Join block texts in reading order. A space separates neighbors
/// unless both boundary characters belong to scripts written without
/// inter-word spaces (CJK, kana, Thai, Lao, Khmer, Myanmar) — fixes
/// the "中 国" corruption where detector splits injected spaces.
fn join_blocks(blocks: &[OcrBlock]) -> String {
    let mut out = String::new();
    for b in blocks {
        if !out.is_empty() {
            let prev = out.chars().next_back();
            let next = b.text.chars().next();
            let unspaced =
                prev.is_some_and(is_unspaced_script) && next.is_some_and(is_unspaced_script);
            if !unspaced {
                out.push(' ');
            }
        }
        out.push_str(&b.text);
    }
    out
}

use crate::script::{is_rtl_char, is_unspaced_script};

/// True when the group's strong-direction characters are majority RTL
/// (Arabic, Hebrew, and their extended/presentation blocks).
fn is_majority_rtl(blocks: &[OcrBlock]) -> bool {
    let mut rtl = 0usize;
    let mut ltr = 0usize;
    for c in blocks.iter().flat_map(|b| b.text.chars()) {
        if is_rtl_char(c) {
            rtl += 1;
        } else if c.is_alphabetic() {
            ltr += 1;
        }
    }
    rtl > ltr
}

/// Median line extent within one band, in pixels.
fn band_median_h(band: &[(f32, OcrBlock)], axis: Axis) -> f32 {
    if band.is_empty() {
        return 1.0;
    }
    let mut heights: Vec<i32> = band.iter().map(|(_, b)| axis.line_extent(b)).collect();
    heights.sort_unstable();
    heights[heights.len() / 2].max(1) as f32
}

/// Split a reading-order-sorted band into sub-bands wherever the gap
/// between consecutive blocks exceeds `gap_threshold` points. Returns at
/// least one group; a band with no wide gaps is returned as a single
/// group unchanged.
fn split_by_reading_gap(
    band: Vec<(f32, OcrBlock)>,
    gap_threshold: f32,
    axis: Axis,
) -> Vec<Vec<(f32, OcrBlock)>> {
    let mut groups: Vec<Vec<(f32, OcrBlock)>> = Vec::new();
    let mut current: Vec<(f32, OcrBlock)> = Vec::new();
    let mut prev_end: Option<i32> = None;

    for (bc, block) in band {
        if let Some(pe) = prev_end {
            let gap = (axis.order_start(&block) - pe) as f32;
            if gap > gap_threshold && !current.is_empty() {
                groups.push(std::mem::take(&mut current));
            }
        }
        prev_end = Some(axis.order_end(&block));
        current.push((bc, block));
    }
    if !current.is_empty() {
        groups.push(current);
    }
    groups
}

fn union_bbox(blocks: &[OcrBlock]) -> [i32; 4] {
    let mut out = [i32::MAX, i32::MAX, i32::MIN, i32::MIN];
    for b in blocks {
        out[0] = out[0].min(b.bbox[0]);
        out[1] = out[1].min(b.bbox[1]);
        out[2] = out[2].max(b.bbox[2]);
        out[3] = out[3].max(b.bbox[3]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blk(x1: i32, y1: i32, x2: i32, y2: i32, text: &str) -> OcrBlock {
        OcrBlock {
            bbox: [x1, y1, x2, y2],
            text: text.into(),
            confidence: 1.0,
            ..Default::default()
        }
    }

    /// A block whose text runs top-to-bottom, as `normalize_quad` emits it
    /// for a rotated region: the advance edge `tl->tr` points along +y.
    /// The bbox is the same rectangle, which is exactly why the bbox alone
    /// cannot distinguish this from a tall column of upright text.
    fn rot_blk(x1: i32, y1: i32, x2: i32, y2: i32, text: &str) -> OcrBlock {
        let (fx1, fy1, fx2, fy2) = (x1 as f32, y1 as f32, x2 as f32, y2 as f32);
        OcrBlock {
            bbox: [x1, y1, x2, y2],
            text: text.into(),
            confidence: 1.0,
            // [start-top, end-top, end-bottom, start-bottom] reading down
            // the right edge: tr -> br -> bl -> tl.
            quad: Some([[fx2, fy1], [fx2, fy2], [fx1, fy2], [fx1, fy1]]),
            ..Default::default()
        }
    }

    #[test]
    fn cjk_neighbors_join_without_space() {
        let input = vec![blk(100, 10, 180, 40, "中国"), blk(185, 10, 260, 40, "人民")];
        let out = band_sort(input);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text, "中国人民");
    }

    #[test]
    fn cjk_latin_boundary_keeps_space() {
        let input = vec![blk(100, 10, 180, 40, "中国"), blk(185, 10, 260, 40, "GDP")];
        let out = band_sort(input);
        assert_eq!(out[0].text, "中国 GDP");
    }

    #[test]
    fn majority_rtl_line_reads_right_to_left() {
        // Arabic words: rightmost block is logically first.
        let input = vec![
            blk(100, 10, 180, 40, "العربية"),
            blk(200, 10, 300, 40, "اللغة"),
        ];
        let out = band_sort(input);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text, "اللغة العربية");
        assert_eq!(out[0].blocks[0].bbox[0], 200, "rightmost block first");
    }

    #[test]
    fn line_quad_is_union_of_block_quads() {
        let out = band_sort(vec![
            blk(100, 10, 180, 40, "hello"),
            blk(200, 10, 300, 40, "world"),
        ]);
        let quad = out[0].quad.expect("quad populated");
        assert_eq!(crate::geometry::bbox_i32_of(&quad), [100, 10, 300, 40]);
    }

    #[test]
    fn empty_input_returns_empty() {
        assert!(band_sort(Vec::new()).is_empty());
    }

    #[test]
    fn single_line_same_y_sorts_left_to_right() {
        // Three blocks on one visual line, emitted right-to-left.
        let input = vec![
            blk(200, 10, 280, 40, "world"),
            blk(100, 10, 180, 40, "hello"),
            blk(300, 10, 380, 40, "rust"),
        ];
        let out = band_sort(input);
        assert_eq!(out.len(), 1, "one band expected");
        assert_eq!(out[0].text, "hello world rust");
    }

    #[test]
    fn two_column_page_splits_into_four_lines() {
        // Two receipts side-by-side, two rows each. Block height ≈ 20,
        // gap between columns is 120 px (6× height) > 2.5× threshold,
        // so each row splits into separate left/right TextLines.
        // Reading order: row-by-row top-down, column-by-column left-right.
        let input = vec![
            blk(300, 20, 380, 40, "rightA1"),
            blk(100, 20, 180, 40, "leftA1"),
            blk(300, 80, 380, 100, "rightB1"),
            blk(100, 80, 180, 100, "leftB1"),
        ];
        let out = band_sort(input);
        assert_eq!(out.len(), 4, "expected 2 rows × 2 columns = 4 lines");
        // Row A (top): left then right.
        assert_eq!(out[0].text, "leftA1");
        assert_eq!(out[1].text, "rightA1");
        // Row B (bottom): left then right.
        assert_eq!(out[2].text, "leftB1");
        assert_eq!(out[3].text, "rightB1");
    }

    #[test]
    fn y_jitter_within_threshold_stays_in_band() {
        // Real-world scans jitter ±1-2 px on the same line.
        // Block height = 20, threshold = 10, so ±3 px still joins.
        let input = vec![
            blk(100, 10, 180, 30, "a"),
            blk(200, 12, 280, 32, "b"), // 2 px below
            blk(300, 8, 380, 28, "c"),  // 2 px above
        ];
        let out = band_sort(input);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text, "a b c");
    }

    #[test]
    fn large_y_gap_starts_new_band() {
        let input = vec![
            blk(100, 10, 180, 30, "top"),
            blk(100, 100, 180, 120, "bottom"),
        ];
        let out = band_sort(input);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].text, "top");
        assert_eq!(out[1].text, "bottom");
    }

    #[test]
    fn large_x_gap_splits_band_into_columns() {
        // Two side-by-side receipts on the same row. Block height = 20,
        // column gutter of 800 px = 40× height → far above 2.5× threshold.
        let input = vec![
            blk(100, 10, 300, 30, "leftColA"),
            blk(1100, 10, 1300, 30, "rightColA"),
        ];
        let out = band_sort(input);
        assert_eq!(out.len(), 2, "columns should split into separate lines");
        assert_eq!(out[0].text, "leftColA");
        assert_eq!(out[1].text, "rightColA");
        // Union bboxes must be per-column, not merged.
        assert_eq!(out[0].bbox, [100, 10, 300, 30]);
        assert_eq!(out[1].bbox, [1100, 10, 1300, 30]);
    }

    #[test]
    fn normal_inter_word_gap_stays_merged() {
        // Two words on one line, 30 px gap vs 20 px line height.
        // 30 / 20 = 1.5× threshold 2.5×, so they stay in one band.
        let input = vec![
            blk(100, 10, 180, 30, "hello"),
            blk(210, 10, 290, 30, "world"),
        ];
        let out = band_sort(input);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text, "hello world");
    }

    #[test]
    fn three_column_layout_splits_into_three_lines() {
        // Three columns on one row, each separated by wide gutters.
        let input = vec![
            blk(50, 10, 200, 30, "col1"),
            blk(500, 10, 650, 30, "col2"),
            blk(950, 10, 1100, 30, "col3"),
        ];
        let out = band_sort(input);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].text, "col1");
        assert_eq!(out[1].text, "col2");
        assert_eq!(out[2].text, "col3");
    }

    // ---- rotated (90°) text -----------------------------------------

    /// A rotated column: each word's *bbox* spans most of the page height,
    /// so y-banding would fuse the whole column into one line. Banding by
    /// x instead recovers the column, and ordering by y recovers the words.
    #[test]
    fn rotated_column_bands_by_x_not_y() {
        // Three words stacked downward in one narrow vertical column.
        let input = vec![
            rot_blk(100, 400, 140, 700, "world"),
            rot_blk(100, 50, 140, 380, "hello"),
            rot_blk(100, 720, 140, 990, "again"),
        ];
        let out = band_sort(input);
        assert_eq!(out.len(), 1, "one rotated column = one line");
        assert_eq!(out[0].text, "hello world again");
    }

    /// Two side-by-side rotated columns must not interleave — the exact
    /// failure that turned a rotated medical insert into word salad.
    #[test]
    fn two_rotated_columns_do_not_interleave() {
        let input = vec![
            rot_blk(100, 50, 140, 380, "leftA"),
            rot_blk(600, 50, 640, 380, "rightA"),
            rot_blk(100, 400, 140, 700, "leftB"),
            rot_blk(600, 400, 640, 700, "rightB"),
        ];
        let out = band_sort(input);
        assert_eq!(out.len(), 2, "one line per rotated column");
        assert_eq!(out[0].text, "leftA leftB");
        assert_eq!(out[1].text, "rightA rightB");
    }

    /// Mixed pages keep horizontal body text first, rotated margin text after.
    #[test]
    fn mixed_orientation_page_separates_axes() {
        let input = vec![
            blk(100, 10, 400, 40, "body text here"),
            rot_blk(20, 100, 60, 500, "margin note"),
            blk(100, 60, 400, 90, "more body"),
        ];
        let out = band_sort(input);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].text, "body text here");
        assert_eq!(out[1].text, "more body");
        assert_eq!(out[2].text, "margin note", "rotated text emitted last");
    }

    /// A tall, narrow *single glyph* is ordinary horizontal text, not a
    /// rotated column — "l" and "1" must not derail the line they sit on.
    /// Its advance edge is horizontal, so no aspect-ratio guard is needed.
    #[test]
    fn single_narrow_glyph_stays_horizontal() {
        let input = vec![
            blk(100, 10, 180, 40, "Total"),
            blk(190, 10, 200, 40, "1"), // 10 wide x 30 tall
            blk(210, 10, 290, 40, "item"),
        ];
        let out = band_sort(input);
        assert_eq!(out.len(), 1, "narrow glyph must not split the line");
        assert_eq!(out[0].text, "Total 1 item");
    }

    #[test]
    fn union_bbox_spans_all_blocks_in_line() {
        // Two blocks on one line with a narrow gap (10 px vs 30 px height,
        // 0.33× — well under the 2.5× column-gap threshold).
        let input = vec![blk(100, 10, 180, 40, "a"), blk(190, 15, 270, 35, "b")];
        let out = band_sort(input);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].bbox, [100, 10, 270, 40]);
    }
}
