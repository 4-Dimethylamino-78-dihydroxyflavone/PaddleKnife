//! Lossless recognition geometry: quadrilaterals in raster-pixel space.
//!
//! PP-OCRv5's DBNet detection head emits 4-point polygons; the recognizer's
//! CTC head yields per-character boxes. Both are kept as f32 [`Quad`]s from
//! the adapter all the way to the writers — the axis-aligned `[i32; 4]`
//! bbox on `OcrBlock`/`TextLine` is a *derived* convenience for legacy
//! consumers, never the source of truth.
//!
//! Coordinate space: raster pixels, top-left origin, Y down. The one
//! px→PDF-points transform lives in the searchable-PDF writer.

/// Four `[x, y]` corners, clockwise from the quad's top-left.
pub type Quad = [[f32; 2]; 4];

/// Axis-aligned `[x_min, y_min, x_max, y_max]` bound of `quad`.
pub fn bbox_of(quad: &Quad) -> [f32; 4] {
    let mut b = [f32::MAX, f32::MAX, f32::MIN, f32::MIN];
    for &[x, y] in quad {
        b[0] = b[0].min(x);
        b[1] = b[1].min(y);
        b[2] = b[2].max(x);
        b[3] = b[3].max(y);
    }
    b
}

/// `bbox_of` rounded outward to integer pixels (legacy `[i32; 4]` contract).
///
/// A 1e-3 px epsilon absorbs f32 interpolation noise (0.3 × 100 =
/// 30.0000004 must round to 30, not ceil to 31); a thousandth of a
/// pixel can never matter optically.
pub fn bbox_i32_of(quad: &Quad) -> [i32; 4] {
    const EPS: f32 = 1e-3;
    let b = bbox_of(quad);
    [
        (b[0] + EPS).floor() as i32,
        (b[1] + EPS).floor() as i32,
        (b[2] - EPS).ceil() as i32,
        (b[3] - EPS).ceil() as i32,
    ]
}

/// Build a clockwise-from-top-left rectangle quad from axis-aligned corners.
pub fn quad_from_bbox(bbox: [f32; 4]) -> Quad {
    let [x1, y1, x2, y2] = bbox;
    [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
}

/// Axis-aligned union of two quads.
///
/// Like Plan 2.9's `polygon_union` this degrades tilted input to a
/// rectangle — an over-approximation, never less information than the
/// legacy bbox union. Line *placement* doesn't read the union: the
/// writer fits the baseline through char-span centers instead.
pub fn union(a: &Quad, b: &Quad) -> Quad {
    let ba = bbox_of(a);
    let bb = bbox_of(b);
    quad_from_bbox([
        ba[0].min(bb[0]),
        ba[1].min(bb[1]),
        ba[2].max(bb[2]),
        ba[3].max(bb[3]),
    ])
}

/// The quad covering the `[t0, t1]` slice (0..1, reading direction) of
/// a region quad, by linear interpolation along its top and bottom
/// edges. For an axis-aligned region this is an axis-aligned sub-box;
/// for tilted regions the slice inherits the region's slope.
pub fn quad_slice(region: &Quad, t0: f32, t1: f32) -> Quad {
    let lerp =
        |a: [f32; 2], b: [f32; 2], t: f32| [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
    let [tl, tr, br, bl] = *region;
    [
        lerp(tl, tr, t0),
        lerp(tl, tr, t1),
        lerp(bl, br, t1),
        lerp(bl, br, t0),
    ]
}

/// Center point of a quad (mean of corners).
pub fn center(quad: &Quad) -> [f32; 2] {
    let mut cx = 0.0;
    let mut cy = 0.0;
    for &[x, y] in quad {
        cx += x;
        cy += y;
    }
    [cx * 0.25, cy * 0.25]
}

/// Mean of the two top corners' y (image coords) — the quad's "top" height.
pub fn top_y(quad: &Quad) -> f32 {
    (quad[0][1] + quad[1][1]) * 0.5
}

/// Mean of the two bottom corners' y (image coords).
pub fn bottom_y(quad: &Quad) -> f32 {
    (quad[2][1] + quad[3][1]) * 0.5
}

/// A straight pen path for one text line: start point, unit direction,
/// length, and perpendicular height. Derived by [`fit_baseline`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Baseline {
    /// Pen start, image coords (first char's leading edge, on the
    /// *bottom* edge of the line band).
    pub start: [f32; 2],
    /// Unit direction of the baseline in image coords (y down),
    /// canonicalized to the rightward half-plane (`dir[0] >= 0`):
    /// `[1, 0]` for upright text of either reading direction — RTL
    /// shows up as char positions *decreasing* along the baseline,
    /// not as a flipped frame.
    pub dir: [f32; 2],
    /// Distance from `start` to the trailing edge of the last char.
    pub len: f32,
    /// Perpendicular extent of the line band (cap-to-descender, px).
    pub height: f32,
}

impl Baseline {
    /// Project a point onto the baseline direction, returning the signed
    /// distance from `start` along `dir`.
    pub fn project(&self, p: [f32; 2]) -> f32 {
        (p[0] - self.start[0]) * self.dir[0] + (p[1] - self.start[1]) * self.dir[1]
    }
}

/// Fit a [`Baseline`] through an ordered sequence of char-span quads.
///
/// Direction: from the first span's center to the last span's center
/// (the *writing order* of the recognizer output), **canonicalized to
/// the rightward half-plane**: a leftward fit (RTL lines, where boxes
/// run right→left in logical order) is negated, so the baseline frame
/// is always upright-ish and glyph "up" stays up. RTL comes out as
/// span positions *decreasing* along the baseline — the PDF writer's
/// relative moves handle that without any script knowledge.
/// Single-span lines fall back to the span's own top-edge direction.
///
/// Start/len/height: every span corner is projected on `dir` and its
/// perpendicular; start sits at the minimum along-projection on the
/// *maximum* perpendicular line (the bottom edge, since y grows down
/// and the perpendicular is oriented downward in image space).
pub fn fit_baseline(quads: &[Quad]) -> Option<Baseline> {
    let first = quads.first()?;
    let last = quads.last()?;
    let c0 = center(first);
    let c1 = center(last);
    let (mut dx, mut dy) = (c1[0] - c0[0], c1[1] - c0[1]);
    let norm = (dx * dx + dy * dy).sqrt();
    if norm < 1.0 {
        // Single span (or fully overlapping): use the top edge of the
        // first quad as the writing direction.
        dx = first[1][0] - first[0][0];
        dy = first[1][1] - first[0][1];
        let n2 = (dx * dx + dy * dy).sqrt();
        if n2 < f32::EPSILON {
            return None;
        }
        dx /= n2;
        dy /= n2;
    } else {
        dx /= norm;
        dy /= norm;
    }
    // Canonicalize: leftward writing (RTL) flips to the upright frame.
    // A Tm built from a leftward dir would be a 180° rotation — glyph
    // "up" would point down and selection rects would hang below the
    // baseline.
    if dx < 0.0 {
        dx = -dx;
        dy = -dy;
    }
    // Perpendicular pointing "down" in image space (y grows down), so
    // perp_max is always the line's bottom edge regardless of writing
    // direction. Rotate dir +90° ((x,y) → (-y,x)) and flip if it came
    // out pointing up — for RTL (dir=[-1,0]) the raw rotation is (0,-1).
    let (mut px, mut py) = (-dy, dx);
    if py < 0.0 {
        px = -px;
        py = -py;
    }

    let mut along_min = f32::MAX;
    let mut along_max = f32::MIN;
    let mut perp_min = f32::MAX;
    let mut perp_max = f32::MIN;
    for q in quads {
        for &[x, y] in q {
            let along = (x - c0[0]) * dx + (y - c0[1]) * dy;
            let perp = (x - c0[0]) * px + (y - c0[1]) * py;
            along_min = along_min.min(along);
            along_max = along_max.max(along);
            perp_min = perp_min.min(perp);
            perp_max = perp_max.max(perp);
        }
    }
    Some(Baseline {
        start: [
            c0[0] + along_min * dx + perp_max * px,
            c0[1] + along_min * dy + perp_max * py,
        ],
        dir: [dx, dy],
        len: (along_max - along_min).max(1.0),
        height: (perp_max - perp_min).max(1.0),
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] // tests panic-on-fail by design
mod tests {
    use super::*;

    fn rect(x1: f32, y1: f32, x2: f32, y2: f32) -> Quad {
        quad_from_bbox([x1, y1, x2, y2])
    }

    #[test]
    fn bbox_of_tilted_quad_takes_extremes() {
        let q: Quad = [[20.0, 10.0], [120.0, 10.0], [110.0, 50.0], [10.0, 50.0]];
        assert_eq!(bbox_of(&q), [10.0, 10.0, 120.0, 50.0]);
    }

    #[test]
    fn bbox_i32_rounds_outward() {
        let q = rect(10.4, 19.6, 110.2, 60.9);
        assert_eq!(bbox_i32_of(&q), [10, 19, 111, 61]);
    }

    #[test]
    fn union_is_axis_aligned_envelope() {
        let a = rect(10.0, 10.0, 50.0, 40.0);
        let b = rect(60.0, 15.0, 100.0, 45.0);
        assert_eq!(bbox_of(&union(&a, &b)), [10.0, 10.0, 100.0, 45.0]);
    }

    #[test]
    fn baseline_upright_ltr() {
        // Three chars on one upright line: y 100..140.
        let quads = vec![
            rect(10.0, 100.0, 30.0, 140.0),
            rect(30.0, 100.0, 55.0, 140.0),
            rect(55.0, 100.0, 80.0, 140.0),
        ];
        let b = fit_baseline(&quads).unwrap();
        assert!((b.dir[0] - 1.0).abs() < 1e-3, "dir {:?}", b.dir);
        assert!(b.dir[1].abs() < 1e-3);
        assert!((b.start[0] - 10.0).abs() < 1e-3, "start {:?}", b.start);
        assert!((b.start[1] - 140.0).abs() < 1e-3, "start {:?}", b.start);
        assert!((b.len - 70.0).abs() < 1e-3, "len {}", b.len);
        assert!((b.height - 40.0).abs() < 1e-3, "height {}", b.height);
    }

    #[test]
    fn baseline_rtl_canonicalizes_to_upright_frame() {
        // Logical order right→left (Arabic): first char is rightmost.
        // The fit canonicalizes to a rightward dir; logical span
        // positions then *decrease* along the baseline.
        let quads = vec![
            rect(80.0, 100.0, 100.0, 140.0),
            rect(55.0, 100.0, 80.0, 140.0),
            rect(30.0, 100.0, 55.0, 140.0),
        ];
        let b = fit_baseline(&quads).unwrap();
        assert!((b.dir[0] - 1.0).abs() < 1e-3, "dir {:?}", b.dir);
        // Pen origin at the *left* end, on the bottom edge.
        assert!((b.start[0] - 30.0).abs() < 1e-3, "start {:?}", b.start);
        assert!((b.start[1] - 140.0).abs() < 1e-3, "start {:?}", b.start);
        assert!((b.len - 70.0).abs() < 1e-3);
        // First logical char (rightmost) projects to the far end.
        assert!((b.project([100.0, 120.0]) - 70.0).abs() < 1e-3);
    }

    #[test]
    fn baseline_tilted_follows_slope() {
        // ~11.3° downhill line: each box 4px lower than the previous.
        let quads = vec![
            rect(0.0, 100.0, 20.0, 130.0),
            rect(20.0, 104.0, 40.0, 134.0),
            rect(40.0, 108.0, 60.0, 138.0),
        ];
        let b = fit_baseline(&quads).unwrap();
        let angle = b.dir[1].atan2(b.dir[0]).to_degrees();
        assert!((angle - 11.3).abs() < 1.0, "angle {angle}");
    }

    #[test]
    fn baseline_single_span_uses_top_edge() {
        let quads = vec![rect(10.0, 100.0, 30.0, 140.0)];
        let b = fit_baseline(&quads).unwrap();
        assert!((b.dir[0] - 1.0).abs() < 1e-3);
        assert!((b.len - 20.0).abs() < 1e-3);
        assert!((b.height - 40.0).abs() < 1e-3);
    }

    #[test]
    fn baseline_empty_is_none() {
        assert!(fit_baseline(&[]).is_none());
    }
}
