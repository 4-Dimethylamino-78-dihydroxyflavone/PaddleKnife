//! Glyphless CID font for the invisible text layer.
//!
//! One 572-byte TrueType font (Tesseract's `pdf.ttf`, see [`FONT`])
//! covers every script: each distinct grapheme cluster in the document
//! is assigned a CID, every CID maps to glyph 1 (a degenerate outline —
//! the text is invisible anyway), and a ToUnicode CMap maps each CID
//! back to the cluster's full codepoint sequence so copy/search
//! round-trips Unicode exactly.
//!
//! This is the Tesseract `pdf.ttf` architecture. What it buys us:
//! - **No coverage problem.** CJK, Devanagari, Arabic, Thai — all the
//!   same empty glyph. (The old path measured CJK with a font that had
//!   no ideographs at all.)
//! - **No shaping problem.** A conjunct cluster is one CID; ToUnicode
//!   `bfchar` entries may map one CID to many codepoints.
//! - **No metric dependence.** Every (cluster, width-bucket) pair gets
//!   its own CID whose `/W` entry IS the measured advance — the same
//!   mechanism real fonts use, so every viewer's text layer (pdfium,
//!   MuPDF, pdf.js) scales selection boxes identically.
//!
//! Vertical metrics are the one designed-in convention: the font's
//! ascent is one full em and its descent ≈ 0, so a glyph shown at size
//! `h` with its pen on a line band's *bottom* edge claims exactly the
//! band `[bottom, bottom + h]` as its selection rectangle.

use crate::error::{OcrError, Result};
use std::collections::HashMap;

/// Fallback advance (`/DW`) for any CID without a `/W` entry — never
/// hit in practice because every allocated CID gets an explicit width.
pub const DEFAULT_WIDTH: u16 = 1000;

/// Width-bucket size in 1/1000 em. Cluster advances are quantized to
/// this grid so the same cluster at slightly different detected widths
/// shares a CID. 25‰ em ⇒ worst-case width error 1.25 % of font size —
/// invisible next to detector noise, and it keeps the CID count down.
const WIDTH_BUCKET: u32 = 25;

/// Quantize an advance (1/1000 em) to the bucket grid, clamped to a
/// sane positive range.
fn quantize_advance(em_millis: f32) -> u16 {
    let b = (em_millis / WIDTH_BUCKET as f32).round() as i64 * WIDTH_BUCKET as i64;
    b.clamp(WIDTH_BUCKET as i64, 20_000) as u16
}

/// Assigns one CID per distinct (grapheme cluster, advance-bucket)
/// pair, per document. CID 0 is reserved (.notdef by CID convention).
///
/// **Why width is part of the key (pdf.js interop):** the original
/// design gave every CID a fixed 1000-unit advance and dialed the real
/// width per *instance* with the `Tz` operator. pdfium and MuPDF honor
/// that; pdf.js computes its selection-layer span widths from glyph
/// metrics in a way that mis-scales per-instance `Tz`, producing
/// sliver-width selection rectangles ("patchy highlights"). Encoding
/// the advance in the font's `/W` array — the mechanism every real
/// font uses — renders identically everywhere, with no `Tz` at all.
#[derive(Debug, Default)]
pub struct CidAllocator {
    map: HashMap<(String, u16), u16>,
    /// In allocation order; index = CID - 1.
    entries: Vec<(String, u16)>,
}

impl CidAllocator {
    pub fn new() -> Self {
        Self::default()
    }

    /// CID for `cluster` shown at `advance_em_millis` (advance in
    /// 1/1000 em, i.e. `width_pts / font_size_pts × 1000`), allocating
    /// on first sight of the (cluster, bucket) pair.
    pub fn cid_for(&mut self, cluster: &str, advance_em_millis: f32) -> Result<u16> {
        let bucket = quantize_advance(advance_em_millis);
        if let Some(&cid) = self.map.get(&(cluster.to_string(), bucket)) {
            return Ok(cid);
        }
        let next = self.entries.len() + 1;
        if next > u16::MAX as usize {
            // 65,535 distinct (cluster, width) pairs in one document
            // would need a second descendant font; full CJK coverage
            // at a few buckets per glyph stays well under this.
            return Err(OcrError::Engine(
                "glyphless font: CID space exhausted (>65535 cluster/width pairs)".into(),
            ));
        }
        let cid = next as u16;
        self.map.insert((cluster.to_string(), bucket), cid);
        self.entries.push((cluster.to_string(), bucket));
        Ok(cid)
    }

    /// `/W` widths for CIDs 1..=max in allocation order (CID `i` has
    /// width `widths()[i-1]`, in 1/1000 em).
    pub fn widths(&self) -> Vec<f32> {
        self.entries.iter().map(|(_, b)| *b as f32).collect()
    }

    /// `/CIDToGIDMap` stream payload: every CID → glyph **1**.
    /// Two big-endian bytes per CID, CIDs 0..=max.
    ///
    /// Glyph 1 — not 0 — because pdfium's text page drops characters
    /// whose glyph has no outline when the font is embedded; glyph 1
    /// in the glyphless font carries a degenerate outline exactly so
    /// extraction keeps the character (same trick as Tesseract's
    /// all-→-GID-1 map).
    pub fn cid_to_gid_map(&self) -> Vec<u8> {
        let mut map = Vec::with_capacity(2 * (self.entries.len() + 1));
        for _ in 0..=self.entries.len() {
            map.extend_from_slice(&1u16.to_be_bytes());
        }
        map
    }

    /// ToUnicode CMap stream payload (CID → UTF-16BE codepoint
    /// sequence), via pdf-writer's `UnicodeCmap` builder.
    pub fn to_unicode_cmap(&self) -> Vec<u8> {
        use pdf_writer::types::{SystemInfo, UnicodeCmap};
        use pdf_writer::Name;
        let info = SystemInfo {
            registry: pdf_writer::Str(b"Adobe"),
            ordering: pdf_writer::Str(b"UCS"),
            supplement: 0,
        };
        let mut cmap = UnicodeCmap::<u16>::new(Name(b"Adobe-Identity-UCS"), info);
        for (i, (cluster, _)) in self.entries.iter().enumerate() {
            cmap.pair_with_multiple((i + 1) as u16, cluster.chars());
        }
        cmap.finish().into_vec()
    }
}

/// The embedded glyphless TrueType font — Tesseract's `pdf.ttf`,
/// vendored verbatim (572 bytes).
///
/// Source: <https://github.com/tesseract-ocr/tesseract/blob/main/tessdata/pdf.ttf>
/// License: Apache-2.0 (Copyright the Tesseract authors), compatible
/// with this crate's AGPL-3.0-or-later. Vendored as a byte array so a
/// fresh clone builds with no out-of-band font downloads.
///
/// **Local modifications to the upstream bytes** (checksums repaired,
/// whole-file checksum invariant still holds — see the test):
/// - glyph 1's hmtx advance 1024 → 0 and its outline quad shrunk
///   1024×2048 → 50×2048 font units. pdfium's loose char box is
///   `max(/W advance, glyph outline width)`; the upstream 0.5-em-wide
///   glyph clamped every narrower character's selection box to 0.5 em.
///   50 units (24‰ em) sits below the smallest /W bucket (25‰), so /W
///   always rules — while the outline stays tall enough that pdfium
///   doesn't classify the glyph as empty and drop the character.
///
/// Why this exact artifact instead of a hand-built minimal TTF: pdfium
/// (FreeType) accepts it — it has shipped in millions of OCRmyPDF /
/// Tesseract documents. Structure: 2 glyphs (glyph 1 carries a
/// degenerate outline; see [`CidAllocator::cid_to_gid_map`]), OS/2
/// table, unitsPerEm 2048, hhea ascent 2048 / descent -1 — i.e. the
/// *(glyph 1's hmtx advance is patched 1024→0 from the upstream file:
/// pdfium floors each loose char box at the font-program advance, which
/// clamped every narrow character's selection width to 0.5 em; with 0,
/// the PDF /W array rules in every reader.)*
/// "selection rect = [baseline, baseline + size]" convention the
/// writer's placement math assumes.
#[rustfmt::skip]
pub const FONT: &[u8; 572] = &[
    0x00, 0x01, 0x00, 0x00, 0x00, 0x0A, 0x00, 0x80, 0x00, 0x03, 0x00, 0x20, 0x4F, 0x53, 0x2F, 0x32,
    0x56, 0xDE, 0xC8, 0x94, 0x00, 0x00, 0x01, 0x28, 0x00, 0x00, 0x00, 0x60, 0x63, 0x6D, 0x61, 0x70,
    0x00, 0x0A, 0x00, 0x34, 0x00, 0x00, 0x01, 0x90, 0x00, 0x00, 0x00, 0x1E, 0x67, 0x6C, 0x79, 0x66,
    0x18, 0xF0, 0x39, 0x88, 0x00, 0x00, 0x01, 0xB8, 0x00, 0x00, 0x00, 0x18, 0x68, 0x65, 0x61, 0x64,
    0x0B, 0x78, 0xF1, 0x65, 0x00, 0x00, 0x00, 0xAC, 0x00, 0x00, 0x00, 0x36, 0x68, 0x68, 0x65, 0x61,
    0x0C, 0x02, 0x04, 0x02, 0x00, 0x00, 0x00, 0xE4, 0x00, 0x00, 0x00, 0x24, 0x68, 0x6D, 0x74, 0x78,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x88, 0x00, 0x00, 0x00, 0x08, 0x6C, 0x6F, 0x63, 0x61,
    0x00, 0x0C, 0x00, 0x00, 0x00, 0x00, 0x01, 0xB0, 0x00, 0x00, 0x00, 0x06, 0x6D, 0x61, 0x78, 0x70,
    0x00, 0x04, 0x00, 0x05, 0x00, 0x00, 0x01, 0x08, 0x00, 0x00, 0x00, 0x20, 0x6E, 0x61, 0x6D, 0x65,
    0xF2, 0xEB, 0x16, 0xDA, 0x00, 0x00, 0x01, 0xD0, 0x00, 0x00, 0x00, 0x4B, 0x70, 0x6F, 0x73, 0x74,
    0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x02, 0x1C, 0x00, 0x00, 0x00, 0x20, 0x00, 0x01, 0x00, 0x00,
    0x00, 0x01, 0x00, 0x00, 0xB0, 0xF8, 0x80, 0x48, 0x5F, 0x0F, 0x3C, 0xF5, 0x04, 0x07, 0x08, 0x00,
    0x00, 0x00, 0x00, 0x00, 0xCF, 0x9A, 0xFC, 0x6E, 0x00, 0x00, 0x00, 0x00, 0xD4, 0xC3, 0xA7, 0xF2,
    0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x02, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x08, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0x04, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x04,
    0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x01, 0x90, 0x00, 0x05,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x05, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x47, 0x4F, 0x4F, 0x47, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0xFF, 0xFF,
    0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x02, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x14, 0x00, 0x03, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x14, 0x00, 0x06, 0x00, 0x0A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x32,
    0x08, 0x00, 0x00, 0x03, 0x00, 0x00, 0x31, 0x21, 0x11, 0x21, 0x00, 0x32, 0xFF, 0xCE, 0x08, 0x00,
    0x00, 0x00, 0x00, 0x03, 0x00, 0x2A, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x05, 0x00, 0x16,
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x05, 0x00, 0x0B, 0x00, 0x16, 0x00, 0x03,
    0x00, 0x01, 0x04, 0x09, 0x00, 0x05, 0x00, 0x16, 0x00, 0x00, 0x00, 0x56, 0x00, 0x65, 0x00, 0x72,
    0x00, 0x73, 0x00, 0x69, 0x00, 0x6F, 0x00, 0x6E, 0x00, 0x20, 0x00, 0x31, 0x00, 0x2E, 0x00, 0x30,
    0x56, 0x65, 0x72, 0x73, 0x69, 0x6F, 0x6E, 0x20, 0x31, 0x2E, 0x30, 0x00, 0x00, 0x01, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

#[cfg(test)]
#[allow(clippy::unwrap_used)] // tests panic-on-fail by design
mod tests {
    use super::*;

    /// Sum of u32 big-endian words, the TrueType checksum primitive.
    fn table_checksum(data: &[u8]) -> u32 {
        let mut sum = 0u32;
        for chunk in data.chunks(4) {
            let mut word = [0u8; 4];
            word[..chunk.len()].copy_from_slice(chunk);
            sum = sum.wrapping_add(u32::from_be_bytes(word));
        }
        sum
    }

    #[test]
    fn vendored_font_is_uncorrupted() {
        // sfnt magic + table count + the whole-file checksum invariant
        // (head.checkSumAdjustment makes the file sum to 0xB1B0AFBA).
        // Guards the byte array against accidental edits.
        assert_eq!(&FONT[0..4], &0x0001_0000u32.to_be_bytes());
        assert_eq!(u16::from_be_bytes([FONT[4], FONT[5]]), 10, "10 tables");
        assert_eq!(table_checksum(FONT), 0xB1B0_AFBA);
        // numGlyphs = 2 (maxp at offset from its directory entry) —
        // glyph 1 must exist for the CIDToGIDMap convention.
        let num = u16::from_be_bytes([FONT[4], FONT[5]]) as usize;
        let mut num_glyphs = 0u16;
        for i in 0..num {
            let rec = 12 + i * 16;
            if &FONT[rec..rec + 4] == b"maxp" {
                let off = u32::from_be_bytes(FONT[rec + 8..rec + 12].try_into().unwrap()) as usize;
                num_glyphs = u16::from_be_bytes([FONT[off + 4], FONT[off + 5]]);
            }
        }
        assert_eq!(num_glyphs, 2);
    }

    #[test]
    fn allocator_dedups_by_cluster_and_width_bucket() {
        let mut a = CidAllocator::new();
        let c1 = a.cid_for("क्", 500.0).unwrap();
        let c2 = a.cid_for("a", 500.0).unwrap();
        // Same cluster, same bucket (510 rounds to 500) → same CID.
        let c3 = a.cid_for("क्", 510.0).unwrap();
        // Same cluster, clearly different width → new CID.
        let c4 = a.cid_for("क्", 800.0).unwrap();
        assert_eq!(c1, 1);
        assert_eq!(c2, 2);
        assert_eq!(c3, c1);
        assert_eq!(c4, 3);
        assert_eq!(a.cid_to_gid_map(), vec![0, 1, 0, 1, 0, 1, 0, 1]);
        assert_eq!(a.widths(), vec![500.0, 500.0, 800.0]);
    }

    #[test]
    fn advance_quantization_clamps_and_rounds() {
        assert_eq!(quantize_advance(0.0), 25);
        assert_eq!(quantize_advance(512.0), 500);
        assert_eq!(quantize_advance(513.0), 525);
        assert_eq!(quantize_advance(1e9), 20_000);
    }

    #[test]
    fn to_unicode_maps_multi_codepoint_clusters() {
        let mut a = CidAllocator::new();
        a.cid_for("कि", 600.0).unwrap(); // U+0915 U+093F — one cluster, two codepoints
        a.cid_for(" ", 300.0).unwrap();
        let cmap = a.to_unicode_cmap();
        let text = String::from_utf8_lossy(&cmap);
        // bfchar entry: CID 0001 → UTF-16BE "0915 093F".
        assert!(
            text.contains("<0001> <0915093f>") || text.contains("<0001> <0915093F>"),
            "cmap was:\n{text}"
        );
        assert!(
            text.contains("<0002> <0020>"),
            "space mapping present:\n{text}"
        );
    }
}
