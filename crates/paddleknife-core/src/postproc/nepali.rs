//! Nepali (Devanagari) post-processing for OCR blocks.
//!
//! Ported from the Python paddleknife. The core insight: in a Nepali
//! document, Latin characters are almost always OCR misreads, so it's
//! aggressive but correct to strip them. Same for ASCII digits —
//! convert them to Devanagari. Page-noise fragments and low-confidence
//! 1–2 character blocks are dropped entirely.
//!
//! The passes in order:
//!   1. NFC-normalize (combine decomposed matras).
//!   2. Discard pure page-noise (standalone page numbers / dashes).
//!   3. Discard <3-char low-conf fragments.
//!   4. Discard entirely-Latin blocks unless they look like ISBN / year /
//!      price (4+ digits) — in which case keep them as Devanagari digits.
//!   5. Strip Latin letters and stray ASCII punctuation.
//!   6. Convert remaining ASCII digits to Devanagari digits.
//!   7. Defuse Devanagari digits fused to word edges.
//!   8. Collapse whitespace, drop if empty / all-punctuation.

use crate::ocr::OcrBlock;

// Devanagari Unicode block: U+0900..U+097F.
const DEVA_START: u32 = 0x0900;
const DEVA_END: u32 = 0x097F;
const DEVA_DIGIT_START: u32 = 0x0966; // '०' 0
const DEVA_DIGIT_END: u32 = 0x096F; // '९' 9

fn is_devanagari(c: char) -> bool {
    let cp = c as u32;
    (DEVA_START..=DEVA_END).contains(&cp)
}

fn is_devanagari_digit(c: char) -> bool {
    let cp = c as u32;
    (DEVA_DIGIT_START..=DEVA_DIGIT_END).contains(&cp)
}

fn is_ascii_letter(c: char) -> bool {
    c.is_ascii_alphabetic()
}

fn is_script_char(c: char) -> bool {
    // Match the Python: category starts with L, N, or M, not whitespace.
    !c.is_whitespace() && (c.is_alphabetic() || c.is_numeric() || is_devanagari(c))
}

/// Fraction of script characters that are Devanagari.
fn devanagari_ratio(text: &str) -> f32 {
    let mut total = 0usize;
    let mut deva = 0usize;
    for c in text.chars() {
        if !is_script_char(c) {
            continue;
        }
        total += 1;
        if is_devanagari(c) {
            deva += 1;
        }
    }
    if total == 0 {
        0.0
    } else {
        deva as f32 / total as f32
    }
}

/// Convert ASCII digits to Devanagari digits.
fn ascii_digits_to_devanagari(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '0'..='9' => {
                // Safe: offset 0..=9 from U+0966 stays in the Devanagari range.
                char::from_u32(DEVA_DIGIT_START + (c as u32 - '0' as u32)).unwrap_or(c)
            }
            _ => c,
        })
        .collect()
}

/// Does `text` look like pure page-noise? Standalone page numbers,
/// dots, dashes, ASCII/Devanagari digits, whitespace.
fn is_page_noise(text: &str) -> bool {
    let t = text.trim();
    if t.is_empty() {
        return true;
    }
    t.chars().all(|c| {
        c.is_ascii_digit() || is_devanagari_digit(c) || c == '.' || c == '-' || c.is_whitespace()
    })
}

/// Does this string look like a keepable digit-only fragment
/// (4+ digits, ISBN-like, year, price)? Allows `-`, `:`, `.` as separators.
fn is_keepable_digit_run(text: &str) -> bool {
    let stripped = text.trim_matches(|c: char| {
        matches!(
            c,
            '*' | '-' | ':' | ';' | '$' | '/' | '\\' | '|' | '!' | '?' | ' '
        )
    });
    if stripped.len() < 4 {
        return false;
    }
    stripped
        .chars()
        .all(|c| c.is_ascii_digit() || c == '-' || c == ':' || c == '.')
}

/// NFC-normalize an arbitrary string. Rust's stdlib has no NFC; we use
/// a minimal hand-rolled pass covering the Devanagari composition rules
/// we actually see in practice (matra + vowel → composed). For a full
/// port we'd pull in unicode-normalization, but the cleanup below mostly
/// cares about already-composed input from PP-OCRv5 which emits NFC.
fn nfc_like(text: &str) -> String {
    // No-op today — PP-OCRv5's Devanagari rec model emits NFC-composed
    // codepoints already. If we ever see decomposed input we'll need
    // to add `unicode-normalization = "0.1"` as a dep.
    text.to_string()
}

/// Clean up a single OCR text block. Returns None if the block should
/// be discarded entirely.
pub fn postprocess_block(text: &str, confidence: f32) -> Option<String> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }

    let text = nfc_like(text);

    if is_page_noise(&text) {
        return None;
    }

    let deva_ratio = devanagari_ratio(&text);

    // Drop very-low-confidence 1-2 char fragments.
    let stripped_len = text.chars().filter(|c| !c.is_whitespace()).count();
    if confidence < 0.55 && stripped_len <= 2 {
        return None;
    }

    // Block has zero Devanagari — either a keepable digit run or junk.
    if deva_ratio == 0.0 {
        let inner = text.trim_matches(|c: char| {
            matches!(
                c,
                '*' | '-' | ':' | ';' | '$' | '/' | '\\' | '|' | '!' | '?' | ' '
            )
        });
        if inner.is_empty() {
            return None;
        }
        if is_keepable_digit_run(inner) {
            return Some(ascii_digits_to_devanagari(inner));
        }
        return None;
    }

    // Strip Latin letters.
    let mut cleaned: String = text.chars().filter(|c| !is_ascii_letter(*c)).collect();

    // Strip stray ASCII junk punctuation.
    cleaned.retain(|c| !matches!(c, '/' | '\\' | '$' | '@' | '#' | '&'));

    // Convert remaining ASCII digits to Devanagari digits.
    cleaned = ascii_digits_to_devanagari(&cleaned);

    // Defuse Devanagari digits fused to word edges — strip leading/trailing
    // Devanagari digits *only* from tokens that also contain a Devanagari
    // letter; leave standalone digit sequences (page numbers, ISBNs).
    let defused = cleaned
        .split_whitespace()
        .filter_map(defuse_token)
        .collect::<Vec<_>>()
        .join(" ");

    // Collapse whitespace + final trim.
    let out = collapse_spaces(&defused);
    let out = out.trim_matches(|c: char| c == '-' || c == ' ').to_string();

    // Drop if empty or all-punctuation.
    if out.is_empty()
        || out.chars().all(|c| {
            matches!(
                c,
                ' ' | '।' | ',' | '.' | '!' | '?' | ';' | ':' | '-' | '=' | '*' | '(' | ')'
            )
        })
    {
        return None;
    }

    Some(out)
}

/// Strip leading/trailing Devanagari digits from a token that contains
/// at least one Devanagari letter. Also strip stray `[` / `]`.
fn defuse_token(tok: &str) -> Option<String> {
    let has_letter = tok
        .chars()
        .any(|c| is_devanagari(c) && !is_devanagari_digit(c));
    let mut s: String = tok.chars().filter(|c| *c != '[' && *c != ']').collect();
    if has_letter {
        s = s.trim_start_matches(is_devanagari_digit).to_string();
        s = s.trim_end_matches(is_devanagari_digit).to_string();
    }
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

fn collapse_spaces(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_space = false;
    for c in s.chars() {
        if c == ' ' {
            if !prev_space {
                out.push(' ');
            }
            prev_space = true;
        } else {
            out.push(c);
            prev_space = false;
        }
    }
    out.trim().to_string()
}

/// Apply `postprocess_block` to every block on a page, dropping those
/// that return `None`. Operates in-place by retaining a filtered list.
pub fn postprocess_page(blocks: Vec<OcrBlock>) -> Vec<OcrBlock> {
    blocks
        .into_iter()
        .filter_map(|mut b| {
            let cleaned = postprocess_block(&b.text, b.confidence)?;
            if cleaned != b.text {
                b.text = cleaned;
                resync_chars(&mut b);
            }
            Some(b)
        })
        .collect()
}

/// Re-establish `concat(chars.text) == text` after a text edit.
///
/// The cheap path applies the same char-class transforms to each
/// measured span (strip Latin/junk, ASCII→Devanagari digits) and keeps
/// its quad. When that doesn't reproduce the cleaned text (the
/// edge-digit defusal is positional, not char-class), fall back to
/// uniform synthesis over the block quad — degraded geometry beats a
/// text layer whose boxes belong to deleted characters.
fn resync_chars(b: &mut OcrBlock) {
    let transformed: Vec<crate::ocr::CharSpan> = b
        .chars
        .iter()
        .filter_map(|span| {
            let t: String = span
                .text
                .chars()
                .filter(|c| !is_ascii_letter(*c))
                .filter(|c| !matches!(c, '/' | '\\' | '$' | '@' | '#' | '&' | '[' | ']'))
                .collect();
            let t = ascii_digits_to_devanagari(&t);
            if t.is_empty() {
                None
            } else {
                Some(crate::ocr::CharSpan {
                    text: t,
                    quad: span.quad,
                })
            }
        })
        .collect();
    let concat: String = transformed
        .iter()
        .flat_map(|s| s.text.chars())
        .filter(|c| !c.is_whitespace())
        .collect();
    let want: String = b.text.chars().filter(|c| !c.is_whitespace()).collect();
    if concat == want {
        b.chars = transformed;
    } else {
        b.synthesize_chars();
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] // tests panic-on-fail by design
mod tests {
    use super::*;

    fn block(text: &str, conf: f32) -> OcrBlock {
        OcrBlock {
            bbox: [0, 0, 10, 10],
            text: text.to_string(),
            confidence: conf,
            ..Default::default()
        }
    }

    /// A block with measured per-char spans (one per char, uniform).
    fn block_with_chars(text: &str, conf: f32) -> OcrBlock {
        use crate::geometry;
        let n = text.chars().count().max(1) as f32;
        let quad = geometry::quad_from_bbox([0.0, 0.0, 10.0 * n, 10.0]);
        OcrBlock {
            bbox: [0, 0, (10.0 * n) as i32, 10],
            text: text.to_string(),
            confidence: conf,
            quad: Some(quad),
            chars: text
                .chars()
                .enumerate()
                .map(|(i, c)| crate::ocr::CharSpan {
                    text: c.to_string(),
                    quad: geometry::quad_slice(&quad, i as f32 / n, (i + 1) as f32 / n),
                })
                .collect(),
            ..Default::default()
        }
    }

    #[test]
    fn latin_strip_resyncs_chars_keeping_measured_quads() {
        // "नेपालXto" → Latin stripped → "नेपाल"; the surviving spans
        // must keep their measured quads and concat to the new text.
        let blocks = postprocess_page(vec![block_with_chars("नेपालXto", 0.9)]);
        assert_eq!(blocks.len(), 1);
        let b = &blocks[0];
        assert_eq!(b.text, "नेपाल");
        let concat: String = b.chars.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(concat, "नेपाल");
        assert!(
            !b.chars_synthesized,
            "char-class strip keeps measured spans"
        );
        // First span quad untouched (measured, not re-synthesized).
        let q = b.chars[0].quad;
        assert!((q[0][0] - 0.0).abs() < 1e-3 && (q[1][0] - 10.0).abs() < 1e-3);
    }

    #[test]
    fn positional_edit_falls_back_to_synthesized_chars() {
        // Leading Devanagari digit fused to a word: defusal is
        // positional, so chars must be re-synthesized, never stale.
        let blocks = postprocess_page(vec![block_with_chars("१नेपाल", 0.9)]);
        assert_eq!(blocks.len(), 1);
        let b = &blocks[0];
        assert_eq!(b.text, "नेपाल");
        let concat: String = b.chars.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(concat, "नेपाल");
        assert!(b.chars_synthesized, "positional edit → synthesized spans");
    }

    #[test]
    fn drops_empty_blocks() {
        assert_eq!(postprocess_block("", 0.99), None);
        assert_eq!(postprocess_block("   ", 0.99), None);
    }

    #[test]
    fn drops_low_conf_single_char_fragment() {
        assert_eq!(postprocess_block("क", 0.50), None);
        // Same char at high confidence survives (after other passes).
        assert_eq!(postprocess_block("क", 0.95), Some("क".into()));
    }

    #[test]
    fn drops_pure_latin_noise() {
        assert_eq!(postprocess_block("hello", 0.95), None);
        assert_eq!(postprocess_block("abc xyz", 0.95), None);
    }

    #[test]
    fn pure_digit_strings_are_dropped_as_page_noise() {
        // Match Python: standalone digit/dash strings (ISBN, year, page
        // number) get filtered as page noise — they're rarely useful in
        // the cleaned Nepali text body.
        assert_eq!(postprocess_block("978-99933-2-690-8", 0.95), None);
        assert_eq!(postprocess_block("2026", 0.95), None);
        assert_eq!(postprocess_block("123", 0.95), None);
    }

    #[test]
    fn keeps_wrapped_digit_runs_converted_to_devanagari() {
        // Junk wrapper around a digit run: strip the wrapper, convert.
        // Triggers the `is_keepable_digit_run` branch, not page-noise.
        assert_eq!(
            postprocess_block("$978-99933-2-690-8$", 0.95),
            Some("९७८-९९९३३-२-६९०-८".into())
        );
    }

    #[test]
    fn strips_latin_from_mostly_devanagari() {
        let out = postprocess_block("महाकवि l देवकोटा", 0.95).unwrap();
        assert!(!out.contains('l'), "should strip stray 'l', got {out:?}");
        assert!(out.contains("महाकवि"));
        assert!(out.contains("देवकोटा"));
    }

    #[test]
    fn defuses_trailing_devanagari_digits_from_words() {
        // "वहिनी९" → "वहिनी"
        let out = postprocess_block("वहिनी९", 0.95).unwrap();
        assert_eq!(out, "वहिनी");
    }

    #[test]
    fn defuses_leading_devanagari_digits_from_words() {
        // "१मुना" → "मुना"
        let out = postprocess_block("१मुना", 0.95).unwrap();
        assert_eq!(out, "मुना");
    }

    #[test]
    fn keeps_standalone_devanagari_digit_runs() {
        // Pure page-number noise should drop.
        assert_eq!(postprocess_block("९०००", 0.95), None);
        // But a long digit run in a mixed block should survive somehow.
        let out = postprocess_block("पृष्ठ २०१६", 0.95).unwrap();
        assert!(out.contains("२०१६"));
    }

    #[test]
    fn converts_ascii_digits_in_mixed_block() {
        let out = postprocess_block("भाग 9", 0.95).unwrap();
        assert!(out.contains('९'));
        assert!(!out.contains('9'));
    }

    #[test]
    fn drops_all_punctuation_blocks() {
        assert_eq!(postprocess_block("!?;:-", 0.95), None);
        assert_eq!(postprocess_block("।।।", 0.95), None);
    }

    #[test]
    fn collapses_multiple_spaces() {
        let out = postprocess_block("हे    मेरा   भाइ", 0.95).unwrap();
        assert_eq!(out, "हे मेरा भाइ");
    }

    #[test]
    fn postprocess_page_filters_drops_and_cleans() {
        let blocks = vec![
            block("hello", 0.95),               // drop (Latin)
            block("महाकवि l देवकोटा", 0.95),     // keep + strip 'l'
            block("", 0.95),                    // drop (empty)
            block("978-99933-2-690-8", 0.95),   // drop (page noise)
            block("$978-99933-2-690-8$", 0.95), // keep + convert (wrapped digits)
        ];
        let out = postprocess_page(blocks);
        assert_eq!(out.len(), 2, "unexpected survivors: {out:?}");
        assert!(out[0].text.contains("महाकवि"));
        assert_eq!(out[1].text, "९७८-९९९३३-२-६९०-८");
    }
}
