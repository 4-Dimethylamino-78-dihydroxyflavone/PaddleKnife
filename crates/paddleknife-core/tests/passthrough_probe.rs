//! Probe qualification matrix: fixture PDFs are built programmatically
//! with lopdf (one knob per rejection reason), then probed via the
//! public SourcePdf API. No pdfium needed — these run anywhere.

use image::{DynamicImage, RgbImage};
use lopdf::content::{Content, Operation};
use lopdf::{dictionary, Dictionary, Document, Object, ObjectId, Stream};
use paddleknife_core::output::searchable_pdf::{encode_page_opts, EmbedOptions, PageImage};
use paddleknife_core::pdf::RasterPage;
use paddleknife_core::source_pdf::{PtFilter, SourcePdf};
use std::path::Path;

/// Knobs for one fixture page. Defaults = the qualifying happy path:
/// a 2×3-inch page fully covered by one RGB JPEG.
struct Fx {
    img_bytes: Vec<u8>,
    img_w: i64,
    img_h: i64,
    filter: &'static str,
    colorspace: &'static str,
    bpc: i64,
    page_w: f32,
    page_h: f32,
    /// Linear coverage fraction (drawn size = cover × page size).
    cover: f32,
    rotate: Option<i64>,
    crop: Option<[f32; 4]>,
    second_image: bool,
    smask: bool,
    extra_text_op: bool,
    decode_parms: Option<Dictionary>,
    /// Offset (e, f) added to the `cm` translation.
    offset: Option<(f32, f32)>,
    /// When true, emit a vertically-flipped draw (`cm dw 0 0 -dh 0 dh`).
    flip_vertical: bool,
    /// Bytes appended verbatim to the encoded content stream.
    raw_content_suffix: Option<&'static [u8]>,
    /// When Some, add this annotation to the page.
    annot: Option<Annot>,
    /// When Some, page /Contents becomes an array of these raw streams.
    content_streams: Option<Vec<&'static [u8]>>,
    /// When Some, a single raw stream replacing the built ops.
    raw_content: Option<&'static [u8]>,
    /// Off-diagonal `cm` terms (b, c) — a baked-in skew like real scans'
    /// sub-degree deskew. Composes with offset on the b,c slots only.
    skew: Option<(f32, f32)>,
    /// Spaces appended to the single content stream: whitespace the lexer
    /// accepts, so only the content-size caps can reject the page.
    content_pad: usize,
    /// Flate-compress the single content stream.
    flate_content: bool,
}

/// One page annotation: its /Subtype and, when Some, its /F flags.
#[derive(Clone, Copy)]
struct Annot {
    subtype: &'static str,
    flags: Option<i64>,
}

/// Drawn over the scan by pdfium (no /F: visible).
const SQUARE: Annot = Annot {
    subtype: "Square",
    flags: None,
};
/// Paints nothing scan-like, but a verbatim copy would still drop it.
const LINK: Annot = Annot {
    subtype: "Link",
    flags: None,
};

impl Default for Fx {
    fn default() -> Self {
        Self {
            img_bytes: tiny_jpeg(128, 96, false),
            img_w: 128,
            img_h: 96,
            filter: "DCTDecode",
            colorspace: "DeviceRGB",
            bpc: 8,
            page_w: 144.0,
            page_h: 216.0,
            cover: 1.0,
            rotate: None,
            crop: None,
            second_image: false,
            smask: false,
            extra_text_op: false,
            decode_parms: None,
            offset: None,
            flip_vertical: false,
            raw_content_suffix: None,
            annot: None,
            content_streams: None,
            raw_content: None,
            skew: None,
            content_pad: 0,
            flate_content: false,
        }
    }
}

fn zlib(data: &[u8]) -> Vec<u8> {
    use flate2::{write::ZlibEncoder, Compression};
    use std::io::Write as _;
    let mut enc = ZlibEncoder::new(Vec::new(), Compression::default());
    enc.write_all(data).expect("deflate");
    enc.finish().expect("deflate finish")
}

fn tiny_jpeg(w: u32, h: u32, gray: bool) -> Vec<u8> {
    let mut out = Vec::new();
    let enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 70);
    if gray {
        image::GrayImage::from_fn(w, h, |x, y| image::Luma([(x * 3 + y * 5) as u8]))
            .write_with_encoder(enc)
            .expect("jpeg encode gray");
    } else {
        image::RgbImage::from_fn(w, h, |x, y| image::Rgb([(x * 2) as u8, (y * 2) as u8, 40]))
            .write_with_encoder(enc)
            .expect("jpeg encode rgb");
    }
    out
}

fn save_fx(fx: &Fx, path: &Path) {
    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();

    let mut img_dict = dictionary! {
        "Type" => "XObject",
        "Subtype" => "Image",
        "Width" => fx.img_w,
        "Height" => fx.img_h,
        "ColorSpace" => fx.colorspace,
        "BitsPerComponent" => fx.bpc,
        "Filter" => fx.filter,
    };
    if let Some(dp) = &fx.decode_parms {
        img_dict.set("DecodeParms", dp.clone());
    }
    if fx.smask {
        // Probe must reject on the key's presence alone — a dangling
        // reference is fine.
        img_dict.set("SMask", Object::Reference((9999, 0)));
    }
    let img_id =
        doc.add_object(Stream::new(img_dict, fx.img_bytes.clone()).with_compression(false));

    let mut xobjects = dictionary! { "Im0" => img_id };
    if fx.second_image {
        let img2 = dictionary! {
            "Type" => "XObject", "Subtype" => "Image",
            "Width" => 8i64, "Height" => 8i64,
            "ColorSpace" => "DeviceRGB", "BitsPerComponent" => 8i64,
            "Filter" => "DCTDecode",
        };
        let img2_id =
            doc.add_object(Stream::new(img2, tiny_jpeg(8, 8, false)).with_compression(false));
        xobjects.set("Im1", img2_id);
    }

    let dw = fx.page_w * fx.cover;
    let dh = fx.page_h * fx.cover;

    // cm operands honor offset/flip (Amendment A.3 + B.1):
    let (a_val, d_val, e_val, f_val) = if fx.flip_vertical {
        (dw, -dh, 0.0f32, dh)
    } else {
        let (ox, oy) = fx.offset.unwrap_or((0.0, 0.0));
        (dw, dh, ox, oy)
    };
    // Skew rides the off-diagonal (b, c) slots, composing with offset/flip.
    let (b_val, c_val) = fx.skew.unwrap_or((0.0, 0.0));

    let mut ops = vec![
        Operation::new("q", vec![]),
        Operation::new(
            "cm",
            vec![
                a_val.into(),
                b_val.into(),
                c_val.into(),
                d_val.into(),
                e_val.into(),
                f_val.into(),
            ],
        ),
        Operation::new("Do", vec![Object::Name(b"Im0".to_vec())]),
        Operation::new("Q", vec![]),
    ];
    if fx.extra_text_op {
        ops.push(Operation::new("BT", vec![]));
        ops.push(Operation::new("ET", vec![]));
    }
    let content = Content { operations: ops };

    // Content-stream selection precedence (B.1):
    //   explicit streams > raw_content > built ops (+ optional suffix)
    let content_ids: Vec<ObjectId> = if let Some(streams) = &fx.content_streams {
        streams
            .iter()
            .map(|b| {
                doc.add_object(Stream::new(dictionary! {}, b.to_vec()).with_compression(false))
            })
            .collect()
    } else {
        let mut bytes = if let Some(raw) = fx.raw_content {
            raw.to_vec()
        } else {
            let mut b = content.encode().expect("encode content ops");
            if let Some(sfx) = fx.raw_content_suffix {
                b.extend_from_slice(sfx);
            }
            b
        };
        bytes.resize(bytes.len() + fx.content_pad, b' ');
        let stream = if fx.flate_content {
            Stream::new(dictionary! { "Filter" => "FlateDecode" }, zlib(&bytes))
        } else {
            Stream::new(dictionary! {}, bytes)
        };
        vec![doc.add_object(stream.with_compression(false))]
    };

    let mut page = dictionary! {
        "Type" => "Page",
        "Parent" => pages_id,
        "MediaBox" => vec![
            0.into(), 0.into(), fx.page_w.into(), fx.page_h.into(),
        ],
        "Resources" => dictionary! { "XObject" => xobjects },
    };
    if let Some(r) = fx.rotate {
        page.set("Rotate", r);
    }
    if let Some(cb) = fx.crop {
        page.set(
            "CropBox",
            vec![
                Object::from(cb[0]),
                Object::from(cb[1]),
                Object::from(cb[2]),
                Object::from(cb[3]),
            ],
        );
    }

    // Annots object (B.1):
    if let Some(a) = fx.annot {
        let mut dict = dictionary! {
            "Type" => "Annot", "Subtype" => a.subtype,
            "Rect" => vec![0.into(), 0.into(), 10.into(), 10.into()],
        };
        if let Some(f) = a.flags {
            dict.set("F", f);
        }
        let annot = doc.add_object(dict);
        page.set("Annots", vec![Object::Reference(annot)]);
    }

    // Contents: single ref or array of refs (B.1):
    if content_ids.len() == 1 {
        page.set("Contents", content_ids[0]);
    } else {
        page.set(
            "Contents",
            content_ids
                .iter()
                .map(|id| Object::Reference(*id))
                .collect::<Vec<_>>(),
        );
    }

    let page_id = doc.add_object(page);

    doc.objects.insert(
        pages_id,
        Object::Dictionary(dictionary! {
            "Type" => "Pages",
            "Kids" => vec![page_id.into()],
            "Count" => 1,
        }),
    );
    let catalog_id = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages_id });
    doc.trailer.set("Root", catalog_id);
    doc.save(path).expect("save fixture pdf");
}

fn probe(fx: &Fx) -> paddleknife_core::source_pdf::PageProbe {
    let dir = tempfile::tempdir().expect("create tempdir");
    let p = dir.path().join("fx.pdf");
    save_fx(fx, &p);
    let src = SourcePdf::open(&p).expect("open fixture");
    assert_eq!(src.page_count(), 1);
    src.probe_page(0)
}

// ── Base tests (Task 3) ───────────────────────────────────────────────────────

#[test]
fn accepts_full_page_rgb_jpeg_byte_identical() {
    let fx = Fx::default();
    let want = fx.img_bytes.clone();
    let pr = probe(&fx);
    let pt = pr.passthrough.expect("qualifies");
    assert_eq!(pt.data, want, "bytes must be copied verbatim");
    assert!(!pt.gray);
    assert_eq!((pt.width, pt.height), (128, 96));
    assert_eq!(pt.filter, PtFilter::Dct);
    // 128 px over 2 in (144 pt) → 64 dpi; 96 px over 3 in → 32 dpi.
    let dpi = pr.effective_dpi.expect("hint");
    assert!((dpi - 64.0).abs() < 1.0, "dpi {dpi}");
}

#[test]
fn accepts_gray_jpeg() {
    let fx = Fx {
        img_bytes: tiny_jpeg(128, 96, true),
        colorspace: "DeviceGray",
        ..Fx::default()
    };
    let pt = probe(&fx).passthrough.expect("gray qualifies");
    assert!(pt.gray);
}

#[test]
fn accepts_ccitt_and_copies_parms() {
    let fx = Fx {
        img_bytes: b"\x00\x01fake-ccitt-bytes".to_vec(),
        filter: "CCITTFaxDecode",
        colorspace: "DeviceGray",
        bpc: 1,
        decode_parms: Some(dictionary! {
            "K" => -1i64, "Columns" => 1728i64, "BlackIs1" => true,
        }),
        ..Fx::default()
    };
    let pt = probe(&fx).passthrough.expect("ccitt qualifies");
    match pt.filter {
        PtFilter::Ccitt(c) => {
            assert_eq!(c.k, -1);
            assert_eq!(c.columns, 1728);
            assert!(c.black_is_1);
            assert!(c.end_of_block, "spec default");
            assert_eq!(c.rows, None);
        }
        other => unreachable!("expected ccitt, got {other:?}"),
    }
}

#[test]
fn accepts_raw_flate() {
    let fx = Fx {
        img_bytes: b"raw-flate-payload".to_vec(),
        filter: "FlateDecode",
        colorspace: "DeviceGray",
        ..Fx::default()
    };
    assert_eq!(
        probe(&fx).passthrough.expect("flate qualifies").filter,
        PtFilter::Flate
    );
}

#[test]
fn rejects_unknown_ccitt_parm_keys() {
    let fx = Fx {
        filter: "CCITTFaxDecode",
        colorspace: "DeviceGray",
        bpc: 1,
        decode_parms: Some(dictionary! { "K" => -1i64, "EndOfLine" => true }),
        ..Fx::default()
    };
    let pr = probe(&fx);
    assert!(pr.passthrough.is_none());
    // Geometry must have passed — the reject decision belongs to the
    // byte-level gate, and the surviving DPI hint proves it.
    assert!(pr.effective_dpi.is_some());
}

#[test]
fn rejects_flate_with_decode_parms() {
    let fx = Fx {
        filter: "FlateDecode",
        colorspace: "DeviceGray",
        decode_parms: Some(dictionary! { "Predictor" => 15i64, "Columns" => 128i64 }),
        ..Fx::default()
    };
    let pr = probe(&fx);
    assert!(pr.passthrough.is_none());
    // Geometry must have passed — the reject decision belongs to the
    // byte-level gate, and the surviving DPI hint proves it.
    assert!(pr.effective_dpi.is_some());
}

#[test]
fn partial_coverage_rejects_passthrough_but_keeps_hint() {
    // 80 % linear coverage fails the ±3 % passthrough size gate, so we
    // can't copy verbatim. But the image is still substantially the page
    // (within the ½–1½× DPI-hint window), so its scan resolution is known
    // and the re-encode can use it instead of the full OCR DPI.
    let pr = probe(&Fx {
        cover: 0.8,
        ..Fx::default()
    });
    assert!(pr.passthrough.is_none());
    assert!(pr.effective_dpi.is_some());
}

#[test]
fn rejects_rotated_page() {
    let pr = probe(&Fx {
        rotate: Some(90),
        ..Fx::default()
    });
    assert!(pr.passthrough.is_none());
}

#[test]
fn accepts_rotate_zero_and_360() {
    assert!(probe(&Fx {
        rotate: Some(0),
        ..Fx::default()
    })
    .passthrough
    .is_some());
    assert!(probe(&Fx {
        rotate: Some(360),
        ..Fx::default()
    })
    .passthrough
    .is_some());
}

#[test]
fn rejects_cropbox_smaller_than_mediabox() {
    let pr = probe(&Fx {
        crop: Some([10.0, 10.0, 100.0, 100.0]),
        ..Fx::default()
    });
    assert!(pr.passthrough.is_none());
}

#[test]
fn accepts_cropbox_equal_to_mediabox() {
    let pr = probe(&Fx {
        crop: Some([0.0, 0.0, 144.0, 216.0]),
        ..Fx::default()
    });
    assert!(pr.passthrough.is_some());
}

#[test]
fn rejects_second_image() {
    assert!(probe(&Fx {
        second_image: true,
        ..Fx::default()
    })
    .passthrough
    .is_none());
}

#[test]
fn rejects_extra_content_ops() {
    // BT is rejected by the pre-lexer first; the analyzer whitelist is the backstop (unit-tested in source_pdf.rs).
    assert!(probe(&Fx {
        extra_text_op: true,
        ..Fx::default()
    })
    .passthrough
    .is_none());
}

#[test]
fn rejects_smask() {
    let pr = probe(&Fx {
        smask: true,
        ..Fx::default()
    });
    assert!(pr.passthrough.is_none());
    // Geometry must have passed — the reject decision belongs to the
    // byte-level gate, and the surviving DPI hint proves it.
    assert!(pr.effective_dpi.is_some());
}

#[test]
fn unsupported_filter_still_yields_dpi_hint() {
    let fx = Fx {
        img_bytes: b"jbig2-payload".to_vec(),
        filter: "JBIG2Decode",
        colorspace: "DeviceGray",
        bpc: 1,
        ..Fx::default()
    };
    let pr = probe(&fx);
    assert!(pr.passthrough.is_none());
    let dpi = pr.effective_dpi.expect("geometry parsed → hint survives");
    assert!((dpi - 64.0).abs() < 1.0, "dpi {dpi}");
}

#[test]
fn rejects_indexed_colorspace_but_keeps_hint() {
    // ColorSpace as an array (Indexed) → as_name fails → byte-reject,
    // geometry hint survives.
    let dir = tempfile::tempdir().expect("create tempdir");
    let p = dir.path().join("fx.pdf");
    let fx = Fx::default();
    save_fx(&fx, &p);
    // Patch the saved fixture: replace the ColorSpace name with an array.
    let mut doc = Document::load(&p).expect("load fixture pdf");
    let img_id = {
        let pages = doc.get_pages();
        let page_id = *pages.values().next().expect("page entry");
        let res = doc
            .get_dictionary(page_id)
            .expect("page dict")
            .get(b"Resources")
            .expect("Resources key")
            .as_dict()
            .expect("Resources as dict");
        res.get(b"XObject")
            .expect("XObject key")
            .as_dict()
            .expect("XObject as dict")
            .get(b"Im0")
            .expect("Im0 key")
            .as_reference()
            .expect("Im0 as reference")
    };
    if let Ok(Object::Stream(s)) = doc.get_object_mut(img_id) {
        s.dict.set(
            "ColorSpace",
            vec![Object::from("Indexed"), Object::from("DeviceRGB")],
        );
    }
    doc.save(&p).expect("save patched fixture");

    let pr = SourcePdf::open(&p)
        .expect("open patched fixture")
        .probe_page(0);
    assert!(pr.passthrough.is_none());
    assert!(pr.effective_dpi.is_some());
}

// ── Amendment A.3 tests ───────────────────────────────────────────────────────

#[test]
fn rejects_translated_draw() {
    let pr = probe(&Fx {
        offset: Some((-72.0, 0.0)),
        ..Fx::default()
    });
    assert!(pr.passthrough.is_none());
}

#[test]
fn rejects_flipped_draw() {
    let pr = probe(&Fx {
        flip_vertical: true,
        ..Fx::default()
    });
    assert!(pr.passthrough.is_none());
}

#[test]
fn rejects_oversized_draw() {
    // 2× page on both axes: old area gate would have passed this.
    let pr = probe(&Fx {
        cover: 2.0,
        ..Fx::default()
    });
    assert!(pr.passthrough.is_none());
}

#[test]
fn skewed_full_page_keeps_hint_but_rejects_passthrough() {
    // Real scans bake a sub-degree deskew into the image CTM. The
    // off-diagonal terms exceed the passthrough axis-alignment gate
    // (so we must re-encode, not copy — the writer can't reproduce the
    // rotation), but the rotation-robust DPI magnitude is unaffected,
    // so the scan's resolution is still reported for the re-encode cap.
    let pr = probe(&Fx {
        skew: Some((4.0, -4.0)),
        ..Fx::default()
    });
    assert!(pr.passthrough.is_none(), "skew must reject verbatim copy");
    let dpi = pr.effective_dpi.expect("skew keeps the DPI hint");
    // 128 px over ~144 pt (2 in) ⇒ ~64 dpi, barely changed by the skew.
    assert!((dpi - 64.0).abs() < 1.0, "dpi {dpi}");
}

#[test]
fn rejects_garbage_suffix_in_content() {
    // Valid scan prefix + a token lopdf cannot parse + drawing ops:
    // lopdf's truncating parser would only surface the prefix — the
    // pre-lexer must reject the whole page.
    let pr = probe(&Fx {
        raw_content_suffix: Some(b" \x80 0 0 100 100 re f"),
        ..Fx::default()
    });
    assert!(pr.passthrough.is_none());
}

// ── Amendment B.1 tests ───────────────────────────────────────────────────────

#[test]
fn rejects_annotated_page() {
    // pdfium renders annotations into the raster; passthrough would
    // drop them (stamps, signature widgets). Any annotation at all
    // disqualifies — even a link, which paints nothing.
    for annot in [SQUARE, LINK] {
        let pr = probe(&Fx {
            annot: Some(annot),
            ..Fx::default()
        });
        assert!(pr.passthrough.is_none(), "{} annotation", annot.subtype);
    }
}

#[test]
fn rejects_token_split_across_streams() {
    // pdfium puts whitespace between /Contents streams; gluing them
    // raw would weld "21"+"6" into one 216. With \n-joining the cm
    // gets 7 operands and the analyzer rejects.
    let pr = probe(&Fx {
        content_streams: Some(vec![b"q 144 0 0 21", b"6 0 0 cm /Im0 Do Q"]),
        ..Fx::default()
    });
    assert!(pr.passthrough.is_none());
}

#[test]
fn rejects_nan_matrix() {
    // Four ~1e12 scales overflow f32 to inf, then a zero matrix makes
    // NaN — reject-on-true gates are blind to NaN without the guard.
    let pr = probe(&Fx {
        raw_content: Some(
            b"999999999999 0 0 999999999999 0 0 cm \
              999999999999 0 0 999999999999 0 0 cm \
              999999999999 0 0 999999999999 0 0 cm \
              999999999999 0 0 999999999999 0 0 cm \
              0 0 0 0 0 0 cm /Im0 Do",
        ),
        ..Fx::default()
    });
    assert!(pr.passthrough.is_none());
}

#[test]
fn rejects_huge_integer_lexer_cap() {
    // >12-digit numeral: lopdf's i64 parser would halt and silently
    // truncate, hiding the trailing draw — the lexer rejects first.
    let pr = probe(&Fx {
        raw_content: Some(
            b"q 144 0 0 216 0 0 cm /Im0 Do Q 12345678901234567890 0 0 9 0 0 cm /Im0 Do",
        ),
        ..Fx::default()
    });
    assert!(pr.passthrough.is_none());
}

#[test]
fn rejects_formfeed_hidden_draw() {
    // Form feed is whitespace to pdfium but halts lopdf's parser —
    // everything after it would be invisible to the whitelist.
    let pr = probe(&Fx {
        raw_content: Some(b"q 144 0 0 216 0 0 cm /Im0 Do Q\x0c9 0 0 9 0 0 cm /Im0 Do"),
        ..Fx::default()
    });
    assert!(pr.passthrough.is_none());
}

// ── Screenshot size regression (2026-09-21) ───────────────────────────────────
//
// A 9-page PDF of ~96-DPI desktop screenshots came out 4.5× larger than
// its source: three pages missed passthrough AND lost their DPI hint, so
// the full 300-DPI OCR raster (9.8× the source's pixels) was embedded.

/// A 317×239 px screenshot placed at 96.012 DPI — what "96 DPI" becomes
/// after a round trip through a PNG `pHYs` chunk (integer 3780 px/m).
/// Page size = px × 72 / 96.012, which the producer prints at full f64
/// precision: 17 significant digits per operand.
fn screenshot_fx() -> Fx {
    Fx {
        img_bytes: tiny_jpeg(317, 239, false),
        img_w: 317,
        img_h: 239,
        page_w: 237.720_28,
        page_h: 179.227_6,
        raw_content: Some(b"q\n237.72028496437946 0 0 179.2275965504312 0 0 cm\n/Im0 Do\nQ\n"),
        ..Fx::default()
    }
}

/// The screenshot page as the OCR pass rasterized it: 300 DPI, 991×747 —
/// 3.1× the source per axis.
fn screenshot_raster() -> RasterPage {
    RasterPage {
        index: 0,
        width: 991,
        height: 747,
        image: DynamicImage::ImageRgb8(RgbImage::from_fn(991, 747, |x, y| {
            image::Rgb([(x % 251) as u8, (y % 241) as u8, 90])
        })),
        dpi: 300,
        native_dpi: None,
    }
}

#[test]
fn long_fraction_cm_operands_still_qualify() {
    // Long *fractions* are harmless: lopdf parses reals with
    // f32::from_str, which cannot overflow. Only long *integers* hit the
    // i64 parser that silently truncates the op list.
    let fx = screenshot_fx();
    let want = fx.img_bytes.clone();
    let pr = probe(&fx);
    let pt = pr
        .passthrough
        .expect("full-precision reals must not reject");
    assert_eq!(pt.data, want, "bytes must be copied verbatim");
    assert_eq!((pt.width, pt.height), (317, 239));
    let dpi = pr.effective_dpi.expect("hint");
    assert!((dpi - 96.0).abs() < 0.5, "dpi {dpi}");
}

#[test]
fn passthrough_only_gates_keep_the_dpi_hint() {
    // None of these change what resolution the scan has: the page is
    // still one image of known pixel size drawn at a known size. They
    // only make a verbatim byte copy unsafe, so they must not take the
    // DPI hint down with them (the hint is what stops the re-encode
    // path from embedding the inflated OCR raster).
    let cases: [(&str, Fx); 3] = [
        (
            "rotated page",
            Fx {
                rotate: Some(90),
                ..Fx::default()
            },
        ),
        (
            "cropping CropBox",
            Fx {
                crop: Some([10.0, 10.0, 100.0, 100.0]),
                ..Fx::default()
            },
        ),
        (
            "link-annotated page",
            Fx {
                annot: Some(LINK),
                ..Fx::default()
            },
        ),
    ];
    for (what, fx) in cases {
        let pr = probe(&fx);
        assert!(pr.passthrough.is_none(), "{what}: must not copy verbatim");
        assert!(pr.effective_dpi.is_some(), "{what}: DPI hint must survive");
        let dpi = pr.effective_dpi.expect("asserted above");
        assert!((dpi - 64.0).abs() < 1.0, "{what}: dpi {dpi}");
    }
}

#[test]
fn unanalyzable_content_yields_no_hint() {
    // The flip side: when the content stream can't be trusted (extra
    // operators may be text or vector art that genuinely benefits from
    // the full OCR DPI), the probe must NOT guess a hint.
    let pr = probe(&Fx {
        extra_text_op: true,
        ..Fx::default()
    });
    assert!(pr.passthrough.is_none());
    assert!(pr.effective_dpi.is_none());
}

#[test]
fn declined_screenshot_never_embeds_more_pixels_than_its_source() {
    // Same screenshot, but passthrough is declined: a link annotation
    // would vanish from a verbatim copy, though it paints nothing, so the
    // scan's resolution still bounds the raster. The page was rasterized
    // at 300 DPI for OCR: 991×747 — 3.1× the source per axis.
    let fx = Fx {
        annot: Some(LINK),
        ..screenshot_fx()
    };
    let pr = probe(&fx);
    assert!(pr.passthrough.is_none());

    let raster = screenshot_raster();
    let enc = encode_page_opts(&raster, Some(&pr), &EmbedOptions::default(), 300).expect("encode");

    // MediaBox basis is untouched — the text layer stays aligned.
    assert_eq!((enc.width, enc.height), (991, 747));
    assert!(matches!(enc.image, PageImage::Jpeg { .. }));
    let (w, h) = enc.image.dims();
    let (embedded, source) = (w as u64 * h as u64, 317u64 * 239);
    assert!(
        embedded <= source * 102 / 100,
        "embedded {w}×{h} = {embedded} px exceeds the source's {source} px"
    );
    // …and it must not collapse either: the cap is the source's
    // resolution, not an arbitrary floor.
    assert!(embedded >= source * 95 / 100, "over-shrunk to {w}×{h}");
}

#[test]
fn accepts_draw_split_cleanly_across_streams() {
    // Two /Contents streams that are individually token-complete: the
    // \n join must read them as one valid full-page draw.
    let pr = probe(&Fx {
        content_streams: Some(vec![b"q 144 0 0 216 0 0 cm", b"/Im0 Do Q"]),
        ..Fx::default()
    });
    assert!(pr.passthrough.is_some());
}

// ── Annotations and the DPI cap (2026-09-30 audit) ────────────────────────────
//
// pdfium renders annotations into the OCR raster, but they are not part of
// the scan: a stamp or a filled form field drawn over a 96-DPI screenshot
// is as sharp as the 300-DPI raster. Capping that page at the scan's DPI
// would blur exactly the part a person added.

#[test]
fn visible_annotation_drops_the_dpi_hint() {
    let pr = probe(&Fx {
        annot: Some(SQUARE),
        ..Fx::default()
    });
    assert!(pr.passthrough.is_none());
    assert!(pr.effective_dpi.is_none(), "a stamp is not at scan DPI");
}

#[test]
fn links_popups_and_hidden_annotations_keep_the_dpi_hint() {
    let quiet = [
        LINK,
        Annot {
            subtype: "Popup",
            flags: None,
        },
        // Hidden (bit 2) and NoView (bit 6): never on screen.
        Annot {
            subtype: "Square",
            flags: Some(2),
        },
        Annot {
            subtype: "Stamp",
            flags: Some(32 | 4),
        },
    ];
    for annot in quiet {
        let pr = probe(&Fx {
            annot: Some(annot),
            ..Fx::default()
        });
        let what = annot.subtype;
        assert!(pr.passthrough.is_none(), "{what}: no verbatim copy");
        let dpi = pr.effective_dpi.expect("the scan still bounds the raster");
        assert!((dpi - 64.0).abs() < 1.0, "{what}: dpi {dpi}");
    }
}

#[test]
fn annotated_screenshot_embeds_at_full_resolution() {
    // The screenshot of the size regression, with a visible annotation:
    // no hint, and (no pdfium here) no native DPI either, so the raster is
    // embedded as rendered rather than capped at the scan's 96 DPI.
    let pr = probe(&Fx {
        annot: Some(SQUARE),
        ..screenshot_fx()
    });
    assert!(pr.effective_dpi.is_none());
    let enc = encode_page_opts(
        &screenshot_raster(),
        Some(&pr),
        &EmbedOptions::default(),
        300,
    )
    .expect("encode");
    assert_eq!(enc.image.dims(), (991, 747));
}

// ── Untrusted-input bounds (2026-09-30 audit) ─────────────────────────────────

#[test]
fn flate_compressed_content_still_qualifies() {
    // Control for the size caps below: compression itself is fine.
    let pr = probe(&Fx {
        flate_content: true,
        ..Fx::default()
    });
    assert!(pr.passthrough.is_some());
    assert!(pr.effective_dpi.is_some());
}

#[test]
fn oversized_content_stream_is_declined_unread() {
    // A valid draw padded past 64 KiB of raw bytes: no plain scan page is
    // that big, so the probe declines before decoding or lexing it.
    let pr = probe(&Fx {
        content_pad: 70 * 1024,
        ..Fx::default()
    });
    assert!(pr.passthrough.is_none());
    assert!(pr.effective_dpi.is_none());
}

#[test]
fn content_that_inflates_past_the_cap_is_declined() {
    // 1 MiB of spaces deflates to about a kilobyte: small on disk, so only
    // the decoded-size cap can stop it (a zlib bomb has the same shape).
    let pr = probe(&Fx {
        content_pad: 1 << 20,
        flate_content: true,
        ..Fx::default()
    });
    assert!(pr.passthrough.is_none());
    assert!(pr.effective_dpi.is_none());
    // Under the cap, the same padding is harmless.
    let pr = probe(&Fx {
        content_pad: 200 * 1024,
        flate_content: true,
        ..Fx::default()
    });
    assert!(pr.passthrough.is_some());
}

#[test]
fn image_width_overflowing_u32_is_declined() {
    // 2³² + 128 used to wrap to 128 and pass as the 128-px image it
    // pretends to be — passthrough included.
    let pr = probe(&Fx {
        img_w: (1i64 << 32) + 128,
        ..Fx::default()
    });
    assert!(pr.passthrough.is_none());
    assert!(pr.effective_dpi.is_none());
}
