//! Writer-side passthrough: a PassthroughImage handed to the writer
//! must come out byte-identical in the output XObject, render in
//! pdfium, and keep the output near input size.

use flate2::{write::ZlibEncoder, Compression};
use image::RgbImage;
use lopdf::{Document, Object};
use paddleknife_core::layout::{DocPage, PageMetrics};
use paddleknife_core::output::searchable_pdf::{
    write_searchable_pdf_encoded, EncodedPage, PageImage,
};
use paddleknife_core::source_pdf::{PassthroughImage, PtFilter};
use pdfium_render::prelude::*;
use serial_test::serial;
use std::io::Write as _;

fn zlib(raw: &[u8]) -> Vec<u8> {
    let mut e = ZlibEncoder::new(Vec::new(), Compression::default());
    e.write_all(raw).expect("zlib write");
    e.finish().expect("zlib finish")
}

fn jpeg_fixture(w: u32, h: u32) -> Vec<u8> {
    let mut out = Vec::new();
    let enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 70);
    RgbImage::from_fn(w, h, |x, y| {
        image::Rgb([(x % 256) as u8, (y % 256) as u8, 90])
    })
    .write_with_encoder(enc)
    .expect("encode fixture jpeg");
    out
}

/// 600×900 raster @300 dpi → 144×216 pt page; embedded image is a
/// different (smaller) pixel grid, as passthrough usually is.
fn encoded_passthrough(jpeg: Vec<u8>) -> (EncodedPage, DocPage) {
    let enc = EncodedPage {
        width: 600,
        height: 900,
        dpi: 300,
        image: PageImage::Passthrough(PassthroughImage {
            data: jpeg,
            width: 200,
            height: 300,
            gray: false,
            bits_per_component: 8,
            filter: PtFilter::Dct,
        }),
    };
    let page = DocPage {
        index: 0,
        size: (600, 900),
        lines: vec![],
        metrics: PageMetrics::from_blocks(&[]),
    };
    (enc, page)
}

#[test]
fn passthrough_bytes_survive_verbatim_and_size_stays_bounded() {
    let dir = tempfile::tempdir().expect("tempdir");
    let out = dir.path().join("pt.pdf");
    let jpeg = jpeg_fixture(200, 300);
    let src_len = jpeg.len();
    let (enc, page) = encoded_passthrough(jpeg.clone());
    write_searchable_pdf_encoded(&out, &[page], vec![enc], 300).expect("writer");

    // Reopen with lopdf and find the image XObject stream.
    let doc = Document::load(&out).expect("reopen output");
    let mut found = None;
    for (_, obj) in doc.objects.iter() {
        if let Object::Stream(s) = obj {
            let is_image = s
                .dict
                .get(b"Subtype")
                .ok()
                .and_then(|o| o.as_name().ok())
                .map(|n| n == b"Image")
                .unwrap_or(false);
            if is_image {
                found = Some(s.clone());
            }
        }
    }
    let s = found.expect("output has an image XObject");
    assert_eq!(s.content, jpeg, "bytes must be byte-identical");
    assert_eq!(
        s.dict
            .get(b"Width")
            .expect("width key")
            .as_i64()
            .expect("width int"),
        200,
        "XObject dims are the image's, not the raster's"
    );
    assert_eq!(
        s.dict
            .get(b"ColorSpace")
            .expect("colorspace key")
            .as_name()
            .expect("colorspace name"),
        b"DeviceRGB"
    );
    assert_eq!(
        s.dict
            .get(b"Filter")
            .expect("filter key")
            .as_name()
            .expect("filter name"),
        b"DCTDecode"
    );

    // Size bound: output = image + text-layer scaffolding (empty page
    // here, so the bound is tight).
    let out_len = std::fs::metadata(&out).expect("stat output").len() as usize;
    assert!(
        out_len <= src_len + 60_000,
        "output {out_len} vs source image {src_len}"
    );
}

#[test]
#[serial]
fn passthrough_output_renders_in_pdfium() {
    let dir = tempfile::tempdir().expect("tempdir");
    let out = dir.path().join("pt_render.pdf");
    let (enc, page) = encoded_passthrough(jpeg_fixture(200, 300));
    write_searchable_pdf_encoded(&out, &[page], vec![enc], 300).expect("writer");

    let pdfium = paddleknife_core::pdf::bind_pdfium().expect("pdfium");
    let doc = pdfium
        .load_pdf_from_file(out.to_str().expect("utf8 path"), None)
        .expect("open output");
    let pg = doc.pages().get(0).expect("page 0");
    let bmp = pg
        .render_with_config(&PdfRenderConfig::new().set_target_width(200))
        .expect("render");
    let img = bmp.as_image().expect("as_image").to_rgb8();
    assert!(
        img.pixels().any(|p| p.0 != [255, 255, 255]),
        "rendered page must not be blank"
    );
}

#[test]
#[serial]
fn flate_passthrough_roundtrips_and_renders_known_gray() {
    const W: u32 = 120;
    const H: u32 = 180;
    const GRAY: u8 = 0x40;

    // Build a flat luma plane and zlib-compress it.
    let raw = vec![GRAY; (W * H) as usize];
    let comp = zlib(&raw);

    let enc = EncodedPage {
        width: 360,
        height: 540,
        dpi: 300,
        image: PageImage::Passthrough(PassthroughImage {
            data: comp.clone(),
            width: W,
            height: H,
            gray: true,
            bits_per_component: 8,
            filter: PtFilter::Flate,
        }),
    };
    let page = DocPage {
        index: 0,
        size: (360, 540),
        lines: vec![],
        metrics: PageMetrics::from_blocks(&[]),
    };

    let dir = tempfile::tempdir().expect("tempdir");
    let out = dir.path().join("flate_pt.pdf");
    write_searchable_pdf_encoded(&out, &[page], vec![enc], 300).expect("writer");

    // ── lopdf assertions ──────────────────────────────────────────────
    let doc = Document::load(&out).expect("reopen output");
    let mut found = None;
    for (_, obj) in doc.objects.iter() {
        if let Object::Stream(s) = obj {
            let is_image = s
                .dict
                .get(b"Subtype")
                .ok()
                .and_then(|o| o.as_name().ok())
                .map(|n| n == b"Image")
                .unwrap_or(false);
            if is_image {
                found = Some(s.clone());
            }
        }
    }
    let s = found.expect("output has an image XObject");

    assert_eq!(s.content, comp, "Flate bytes must be byte-identical");
    assert_eq!(
        s.dict
            .get(b"Filter")
            .expect("filter key")
            .as_name()
            .expect("filter name"),
        b"FlateDecode",
        "/Filter must be FlateDecode"
    );
    assert_eq!(
        s.dict
            .get(b"ColorSpace")
            .expect("colorspace key")
            .as_name()
            .expect("colorspace name"),
        b"DeviceGray",
        "/ColorSpace must be DeviceGray"
    );
    assert_eq!(
        s.dict
            .get(b"Width")
            .expect("width key")
            .as_i64()
            .expect("width int"),
        W as i64,
        "/Width must match image width"
    );
    assert_eq!(
        s.dict
            .get(b"Height")
            .expect("height key")
            .as_i64()
            .expect("height int"),
        H as i64,
        "/Height must match image height"
    );
    assert_eq!(
        s.dict
            .get(b"BitsPerComponent")
            .expect("bpc key")
            .as_i64()
            .expect("bpc int"),
        8,
        "/BitsPerComponent must be 8"
    );
    assert!(
        s.dict.get(b"DecodeParms").is_err(),
        "/DecodeParms must not be present for plain FlateDecode"
    );

    // ── pdfium render + center-pixel oracle ──────────────────────────
    let pdfium = paddleknife_core::pdf::bind_pdfium().expect("pdfium");
    let doc = pdfium
        .load_pdf_from_file(out.to_str().expect("utf8 path"), None)
        .expect("open output");
    let pg = doc.pages().get(0).expect("page 0");
    let bmp = pg
        .render_with_config(&PdfRenderConfig::new().set_target_width(W as i32))
        .expect("render");
    let img = bmp.as_image().expect("as_image").to_rgb8();

    // Sample the center pixel; for a flat DeviceGray FlateDecode image
    // with no JPEG quantisation the value should be very close to GRAY.
    let cx = img.width() / 2;
    let cy = img.height() / 2;
    let p = img.get_pixel(cx, cy);
    let [r, g, b] = p.0;
    assert!(
        (r as i32 - GRAY as i32).abs() <= 20
            && (g as i32 - GRAY as i32).abs() <= 20
            && (b as i32 - GRAY as i32).abs() <= 20,
        "center pixel RGB ({r},{g},{b}) must be within ±20 of GRAY {GRAY}"
    );
}
