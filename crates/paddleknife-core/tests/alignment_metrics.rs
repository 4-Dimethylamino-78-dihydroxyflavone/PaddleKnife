//! Quantitative alignment gate for the searchable-PDF text layer.
//!
//! Plans 2.6–2.8 stacked five positioning fixes, each "verified" by
//! eyeballing a PDF, while a matrix-order bug hid underneath. This
//! harness replaces the eyeball: synthetic pages with **known**
//! per-cluster quads per script family go through the writer, come
//! back out through pdfium's text page, and every extracted character
//! rectangle is compared against the quad it was written from.
//!
//! Gate: exact text round-trip; per-char center offset ≤ 15 % of line
//! height; per-char width within 20 % of the detected width (tolerance
//! exists because pdfium reports loose bounds via font metrics).

use image::DynamicImage;
use paddleknife_core::geometry::{self, Quad};
use paddleknife_core::layout::{DocPage, TextLine};
use paddleknife_core::ocr::{CharSpan, OcrBlock};
use paddleknife_core::output::searchable_pdf::write_searchable_pdf;
use paddleknife_core::pdf::RasterPage;
use pdfium_render::prelude::*;

const DPI: u32 = 300;
const PT_PER_PX: f32 = 72.0 / DPI as f32;

/// A synthetic page: white raster + lines built from explicit cluster
/// boxes. Returns the DocPage and the flat list of (cluster, quad)
/// ground truth in reading order (whitespace clusters excluded — the
/// writer synthesizes spaces from gaps).
struct Fixture {
    page: DocPage,
    raster: RasterPage,
    truth: Vec<(String, Quad)>,
    page_h_px: f32,
}

fn fixture(width: u32, height: u32, lines: Vec<Vec<(&str, [f32; 4])>>) -> Fixture {
    let as_quads = lines
        .into_iter()
        .map(|line| {
            line.into_iter()
                .map(|(s, b)| (s, geometry::quad_from_bbox(b)))
                .collect()
        })
        .collect();
    fixture_quads(width, height, as_quads)
}

fn fixture_quads(width: u32, height: u32, lines: Vec<Vec<(&str, Quad)>>) -> Fixture {
    let img = image::RgbImage::from_pixel(width, height, image::Rgb([255, 255, 255]));
    let raster = RasterPage {
        index: 0,
        width,
        height,
        image: DynamicImage::ImageRgb8(img),
        dpi: DPI,
        native_dpi: None,
    };
    let mut truth = Vec::new();
    let mut text_lines = Vec::new();
    for line_spec in &lines {
        // Whitespace-split into blocks, mirroring the OCR adapter.
        let mut blocks: Vec<OcrBlock> = Vec::new();
        let mut current: Vec<(String, Quad)> = Vec::new();
        let flush = |current: &mut Vec<(String, Quad)>, blocks: &mut Vec<OcrBlock>| {
            if current.is_empty() {
                return;
            }
            let quad = current
                .iter()
                .map(|(_, q)| *q)
                .reduce(|a, b| geometry::union(&a, &b))
                .expect("non-empty");
            blocks.push(OcrBlock {
                bbox: geometry::bbox_i32_of(&quad),
                text: current.iter().map(|(s, _)| s.as_str()).collect(),
                confidence: 0.99,
                quad: Some(quad),
                chars: current
                    .iter()
                    .map(|(s, q)| CharSpan {
                        text: s.clone(),
                        quad: *q,
                    })
                    .collect(),
                ..Default::default()
            });
            current.clear();
        };
        for (cluster, quad) in line_spec {
            if cluster.chars().all(char::is_whitespace) {
                flush(&mut current, &mut blocks);
                continue;
            }
            truth.push((cluster.to_string(), *quad));
            current.push((cluster.to_string(), *quad));
        }
        flush(&mut current, &mut blocks);
        let bbox = blocks
            .iter()
            .map(|b| b.bbox)
            .reduce(|a, b| {
                [
                    a[0].min(b[0]),
                    a[1].min(b[1]),
                    a[2].max(b[2]),
                    a[3].max(b[3]),
                ]
            })
            .expect("blocks");
        let quad = blocks
            .iter()
            .map(OcrBlock::quad_or_bbox)
            .reduce(|a, b| geometry::union(&a, &b));
        let text = blocks
            .iter()
            .map(|b| b.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        text_lines.push(TextLine {
            bbox,
            text,
            blocks,
            quad,
            ..Default::default()
        });
    }
    Fixture {
        page: DocPage {
            index: 0,
            size: (width, height),
            lines: text_lines,
            metrics: Default::default(),
        },
        raster,
        truth,
        page_h_px: height as f32,
    }
}

/// Write the fixture, re-open with pdfium, and assert per-cluster
/// geometry. `label` names the script family in failure messages.
fn assert_alignment(label: &str, fx: Fixture) {
    assert_alignment_opts(label, fx, true)
}

/// `check_width = false` for tilted fixtures: pdfium reports the
/// axis-aligned bound of the rotated glyph box, which legitimately
/// differs from the quad's axis-aligned width by the rotation
/// inflation. Centers are rotation-invariant, so they stay asserted.
fn assert_alignment_opts(label: &str, fx: Fixture, check_width: bool) {
    let tmp = tempfile::tempdir().expect("tempdir");
    let out = tmp.path().join(format!("{label}.pdf"));
    write_searchable_pdf(&out, &[fx.page], vec![fx.raster], DPI).expect("writer");

    let pdfium = paddleknife_core::pdf::bind_pdfium().expect("bind pdfium");
    let doc = pdfium
        .load_pdf_from_file(out.to_str().expect("utf8"), None)
        .expect("open");
    let page = doc.pages().first().expect("page");
    let text = page.text().expect("text page");

    // 1. Text round-trip: every truth cluster appears, in order, in the
    //    extracted stream (spaces and combining-mark handling vary by
    //    extractor, so compare the whitespace-stripped concatenation).
    let extracted = text.all();
    let extracted_compact: String = extracted.chars().filter(|c| !c.is_whitespace()).collect();
    let expected_compact: String = fx
        .truth
        .iter()
        .map(|(s, _)| s.as_str())
        .collect::<String>()
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect();
    assert_eq!(
        extracted_compact, expected_compact,
        "[{label}] text round-trip mismatch (raw extraction: {extracted:?})"
    );

    // 2. Geometry: walk extracted chars, grouping by truth cluster (a
    //    multi-codepoint cluster comes back as several chars sharing
    //    one box). Spaces in the extraction are skipped.
    let all_chars = text.chars();
    let chars: Vec<_> = all_chars.iter().collect();
    let mut idx = 0usize;
    for (cluster, quad) in &fx.truth {
        // Expected rect in PDF points (y-flip).
        let bbox = geometry::bbox_of(quad);
        let exp_left = bbox[0] * PT_PER_PX;
        let exp_right = bbox[2] * PT_PER_PX;
        let exp_bottom = (fx.page_h_px - bbox[3]) * PT_PER_PX;
        let exp_top = (fx.page_h_px - bbox[1]) * PT_PER_PX;
        let line_h = exp_top - exp_bottom;
        let tol_center = 0.15 * line_h;
        let tol_width = 0.20 * (exp_right - exp_left).max(0.5 * line_h);

        let n_codepoints = cluster.chars().count();
        let mut got_left = f32::MAX;
        let mut got_right = f32::MIN;
        let mut got_bottom = f32::MAX;
        let mut got_top = f32::MIN;
        let mut matched = 0usize;
        while matched < n_codepoints {
            assert!(
                idx < chars.len(),
                "[{label}] ran out of extracted chars at cluster {cluster:?}"
            );
            let ch = &chars[idx];
            idx += 1;
            let uni = ch.unicode_char().unwrap_or('\u{0}');
            if uni.is_whitespace() {
                continue; // writer-inserted word space
            }
            let r = ch.loose_bounds().expect("bounds");
            got_left = got_left.min(r.left().value);
            got_right = got_right.max(r.right().value);
            got_bottom = got_bottom.min(r.bottom().value);
            got_top = got_top.max(r.top().value);
            matched += 1;
        }

        let got_cx = (got_left + got_right) * 0.5;
        let got_cy = (got_bottom + got_top) * 0.5;
        let exp_cx = (exp_left + exp_right) * 0.5;
        let exp_cy = (exp_bottom + exp_top) * 0.5;
        assert!(
            (got_cx - exp_cx).abs() <= tol_center,
            "[{label}] {cluster:?}: x-center off by {:.2}pt (tol {:.2}): got {:.2}, expected {:.2}",
            (got_cx - exp_cx).abs(),
            tol_center,
            got_cx,
            exp_cx
        );
        assert!(
            (got_cy - exp_cy).abs() <= tol_center,
            "[{label}] {cluster:?}: y-center off by {:.2}pt (tol {:.2}): got {:.2}, expected {:.2}",
            (got_cy - exp_cy).abs(),
            tol_center,
            got_cy,
            exp_cy
        );
        let got_w = got_right - got_left;
        let exp_w = exp_right - exp_left;
        assert!(
            !check_width || (got_w - exp_w).abs() <= tol_width,
            "[{label}] {cluster:?}: width off by {:.2}pt (tol {:.2}): got {:.2}, expected {:.2}",
            (got_w - exp_w).abs(),
            tol_width,
            got_w,
            exp_w
        );
    }
}

/// Latin with deliberately non-uniform char widths (the case the old
/// uniform line-stretch could never get right).
#[test]
#[serial_test::serial(pdfium)]
fn latin_proportional_chars_align() {
    let fx = fixture(
        800,
        400,
        vec![vec![
            ("W", [100.0, 100.0, 145.0, 140.0]),
            ("i", [145.0, 100.0, 155.0, 140.0]),
            ("d", [155.0, 100.0, 180.0, 140.0]),
            ("e", [180.0, 100.0, 205.0, 140.0]),
            (" ", [205.0, 100.0, 215.0, 140.0]),
            ("n", [215.0, 100.0, 240.0, 140.0]),
            ("o", [240.0, 100.0, 268.0, 140.0]),
            ("w", [268.0, 100.0, 305.0, 140.0]),
        ]],
    );
    assert_alignment("latin", fx);
}

/// Devanagari with multi-codepoint grapheme clusters (base + matra,
/// conjuncts) — one cluster = one selection box = several codepoints.
#[test]
#[serial_test::serial(pdfium)]
fn devanagari_clusters_align() {
    let fx = fixture(
        800,
        400,
        vec![vec![
            ("मु", [100.0, 100.0, 135.0, 150.0]),
            ("ना", [135.0, 100.0, 175.0, 150.0]),
            (" ", [175.0, 100.0, 185.0, 150.0]),
            ("म", [185.0, 100.0, 220.0, 150.0]),
            ("द", [220.0, 100.0, 252.0, 150.0]),
            ("न", [252.0, 100.0, 285.0, 150.0]),
        ]],
    );
    assert_alignment("devanagari", fx);
}

/// CJK: no spaces anywhere, per-ideograph boxes.
#[test]
#[serial_test::serial(pdfium)]
fn cjk_chars_align_without_spaces() {
    let fx = fixture(
        800,
        400,
        vec![vec![
            ("中", [100.0, 100.0, 140.0, 140.0]),
            ("华", [140.0, 100.0, 180.0, 140.0]),
            ("人", [180.0, 100.0, 220.0, 140.0]),
            ("民", [220.0, 100.0, 260.0, 140.0]),
        ]],
    );
    let tmp_truth = fx.truth.clone();
    assert_alignment("cjk", fx);
    assert_eq!(tmp_truth.len(), 4);
}

/// Arabic in logical order with boxes running right-to-left. The
/// baseline fit must point the pen leftward; extraction order stays
/// logical.
#[test]
#[serial_test::serial(pdfium)]
fn arabic_rtl_chars_align() {
    // "سلام" — logical order س ل ا م, visually right→left.
    let fx = fixture(
        800,
        400,
        vec![vec![
            ("س", [300.0, 100.0, 340.0, 140.0]),
            ("ل", [270.0, 100.0, 300.0, 140.0]),
            ("ا", [250.0, 100.0, 270.0, 140.0]),
            ("م", [210.0, 100.0, 250.0, 140.0]),
        ]],
    );
    assert_alignment("arabic-rtl", fx);
}

/// A ~5.7° tilted line: char quads are slices of a sheared region
/// (exactly what the OCR adapter produces via `quad_slice`); the
/// fitted baseline must follow the slope, keeping each char box
/// centered on its tilted quad.
#[test]
#[serial_test::serial(pdfium)]
fn tilted_line_follows_baseline() {
    // Sheared region: 150px wide, top edge drops 100→115, height 40.
    let region: Quad = [
        [100.0, 100.0],
        [250.0, 115.0],
        [250.0, 155.0],
        [100.0, 140.0],
    ];
    let labels = ["a", "b", "c", "d", "e"];
    let n = labels.len() as f32;
    let mut fx_lines = Vec::new();
    let mut spec: Vec<(&str, Quad)> = Vec::new();
    for (i, l) in labels.iter().enumerate() {
        let q = geometry::quad_slice(&region, i as f32 / n, (i + 1) as f32 / n);
        spec.push((*l, q));
    }
    fx_lines.push(spec);
    let fx = fixture_quads(800, 400, fx_lines);
    assert_alignment_opts("tilted", fx, false);
}

/// Legacy input (no per-char geometry) must still produce a usable
/// uniform layer rather than failing.
#[test]
#[serial_test::serial(pdfium)]
fn legacy_blocks_without_chars_still_roundtrip() {
    let img = image::RgbImage::from_pixel(800, 400, image::Rgb([255, 255, 255]));
    let raster = RasterPage {
        index: 0,
        width: 800,
        height: 400,
        image: DynamicImage::ImageRgb8(img),
        dpi: DPI,
        native_dpi: None,
    };
    let page = DocPage {
        index: 0,
        size: (800, 400),
        lines: vec![TextLine {
            bbox: [100, 100, 400, 140],
            text: "legacy line".into(),
            blocks: vec![OcrBlock {
                bbox: [100, 100, 400, 140],
                text: "legacy line".into(),
                confidence: 0.9,
                ..Default::default()
            }],
            ..Default::default()
        }],
        metrics: Default::default(),
    };
    let tmp = tempfile::tempdir().expect("tempdir");
    let out = tmp.path().join("legacy.pdf");
    write_searchable_pdf(&out, &[page], vec![raster], DPI).expect("writer");
    let pdfium = paddleknife_core::pdf::bind_pdfium().expect("bind");
    let doc = pdfium
        .load_pdf_from_file(out.to_str().expect("utf8"), None)
        .expect("open");
    let extracted = doc
        .pages()
        .first()
        .expect("page")
        .text()
        .expect("text")
        .all();
    assert!(
        extracted.contains("legacy") && extracted.contains("line"),
        "legacy extraction: {extracted:?}"
    );
}

/// Arched text — clusters along a circular arc, each quad rotated to
/// the local tangent (the "60s display type / curved logo" case). One
/// straight baseline can't represent this; the writer must place each
/// cluster in its own frame. Width checks off (rotated quads inflate
/// the axis-aligned width); centers are rotation-invariant and gated.
#[test]
#[serial_test::serial(pdfium)]
fn arched_text_boxes_follow_the_curve() {
    let labels = ["C", "U", "R", "V", "E", "D"];
    let (cx, cy, r) = (400.0_f32, 600.0_f32, 320.0_f32);
    let (w, h) = (36.0_f32, 44.0_f32);
    let mut spec: Vec<(&str, Quad)> = Vec::new();
    for (i, l) in labels.iter().enumerate() {
        // Sweep -50°..+50° across the top of the circle.
        let theta = (-50.0 + 100.0 * i as f32 / (labels.len() - 1) as f32).to_radians();
        let c = [cx + r * theta.sin(), cy - r * theta.cos()];
        // Tangent (reading direction) and image-space "up" normal.
        let t = [theta.cos(), theta.sin()];
        let u = [t[1], -t[0]];
        let quad: Quad = [
            [
                c[0] - t[0] * w / 2.0 + u[0] * h / 2.0,
                c[1] - t[1] * w / 2.0 + u[1] * h / 2.0,
            ],
            [
                c[0] + t[0] * w / 2.0 + u[0] * h / 2.0,
                c[1] + t[1] * w / 2.0 + u[1] * h / 2.0,
            ],
            [
                c[0] + t[0] * w / 2.0 - u[0] * h / 2.0,
                c[1] + t[1] * w / 2.0 - u[1] * h / 2.0,
            ],
            [
                c[0] - t[0] * w / 2.0 - u[0] * h / 2.0,
                c[1] - t[1] * w / 2.0 - u[1] * h / 2.0,
            ],
        ];
        spec.push((*l, quad));
    }
    let fx = fixture_quads(800, 700, vec![spec]);
    assert_alignment_opts("arched", fx, false);
}
