//! Normalize fullwidth punctuation to ASCII in non-CJK text.
//!
//! PP-OCR recognizers are trained on a joint CJK + Latin corpus, and their
//! dictionary carries both `:` (U+003A) and `：` (U+FF1A FULLWIDTH COLON).
//! On Latin documents the model sometimes picks the fullwidth variant —
//! PP-OCRv6 emits `Ph： 02 …` and `Date：01/01/26` on an English receipt
//! where it emitted plain `:` two lines earlier. The glyphs look nearly
//! identical, so the error is invisible on inspection but breaks every
//! consumer: a search for "Date:" misses, invoice field parsers fail, and
//! the searchable-PDF text layer holds a character the reader never typed.
//!
//! **Why this runs here and not in [`crate::postproc`] proper.** The Nepali
//! pass deletes and reorders characters, so it must stay away from the PDF
//! path where `concat(chars.text) == text` is a hard contract. This pass is
//! a strict 1:1 codepoint substitution — same character count, same
//! grapheme clusters, same CTC box alignment — so it is safe to apply to
//! every output format, including the glyphless text layer.
//!
//! **Why it is conditional.** In genuine CJK text fullwidth punctuation is
//! correct typography, not an error: `日本：東京` should keep its `：`.
//! Normalization therefore applies per region, only when that region holds
//! no CJK ideographs, kana or Hangul.

/// Distance from a fullwidth ASCII variant to its ASCII original.
/// U+FF01 `！` → U+0021 `!`, through U+FF5E `～` → U+007E `~`.
const FULLWIDTH_OFFSET: u32 = 0xFEE0;

/// True for scripts that legitimately use fullwidth punctuation.
///
/// Deliberately narrower than [`crate::script::is_unspaced_script`]: Thai,
/// Lao, Khmer and Myanmar are written without spaces but do not use
/// fullwidth forms, so their text should still be normalized.
fn uses_fullwidth_punctuation(c: char) -> bool {
    matches!(c as u32,
        0x3040..=0x30FF |   // Hiragana + Katakana
        0x3400..=0x4DBF |   // CJK Unified Extension A
        0x4E00..=0x9FFF |   // CJK Unified Ideographs
        0xF900..=0xFAFF |   // CJK Compatibility Ideographs
        0xAC00..=0xD7AF |   // Hangul Syllables
        0x1100..=0x11FF     // Hangul Jamo
    )
}

/// Map one fullwidth codepoint to its ASCII equivalent, if it has one.
///
/// Covers the fullwidth ASCII variants (U+FF01–U+FF5E) plus the
/// ideographic space. Ideographic punctuation without an unambiguous ASCII
/// counterpart (`。`, `、`, `「`) is deliberately left alone.
fn to_ascii(c: char) -> Option<char> {
    match c as u32 {
        0xFF01..=0xFF5E => char::from_u32(c as u32 - FULLWIDTH_OFFSET),
        0x3000 => Some(' '), // IDEOGRAPHIC SPACE
        _ => None,
    }
}

/// True when `text` contains a fullwidth form that this pass would rewrite.
pub fn needs_normalizing(text: &str) -> bool {
    !text.chars().any(uses_fullwidth_punctuation) && text.chars().any(|c| to_ascii(c).is_some())
}

/// Rewrite fullwidth punctuation to ASCII, unless `text` looks like CJK.
///
/// Returns the input unchanged when it holds CJK/kana/Hangul, or when
/// there is nothing to rewrite. The output always has the same `chars()`
/// count as the input.
pub fn normalize(text: &str) -> std::borrow::Cow<'_, str> {
    if !needs_normalizing(text) {
        return std::borrow::Cow::Borrowed(text);
    }
    std::borrow::Cow::Owned(text.chars().map(|c| to_ascii(c).unwrap_or(c)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rewrites_fullwidth_colon_in_latin_text() {
        // The defect as observed on an English receipt (values are synthetic).
        assert_eq!(normalize("Ph： 02 55500115"), "Ph: 02 55500115");
        assert_eq!(normalize("Date：01/01/26"), "Date:01/01/26");
    }

    #[test]
    fn leaves_cjk_punctuation_alone() {
        // Fullwidth punctuation is correct typography here.
        let jp = "日本：東京";
        assert_eq!(normalize(jp), jp);
        let kr = "한국：서울";
        assert_eq!(normalize(kr), kr);
    }

    #[test]
    fn rewrites_the_whole_fullwidth_ascii_range() {
        assert_eq!(normalize("ＡＢＣ１２３（ｘ）"), "ABC123(x)");
        assert_eq!(normalize("＄１２３．４５"), "$123.45");
    }

    #[test]
    fn ideographic_space_becomes_a_normal_space() {
        assert_eq!(normalize("TOTAL\u{3000}AUD"), "TOTAL AUD");
    }

    #[test]
    fn leaves_ideographic_punctuation_without_ascii_equivalent() {
        // No unambiguous ASCII counterpart — must survive untouched.
        assert_eq!(normalize("hello。world、"), "hello。world、");
    }

    #[test]
    fn plain_ascii_is_untouched_and_not_reallocated() {
        let s = "Total: $125.00";
        assert!(matches!(normalize(s), std::borrow::Cow::Borrowed(_)));
        assert_eq!(normalize(s), s);
    }

    /// The searchable-PDF text layer relies on character counts lining up
    /// with the recognizer's CTC boxes, so the mapping must be strictly 1:1.
    #[test]
    fn character_count_is_preserved() {
        for s in [
            "Ph： 02",
            "ＡＢＣ１２３（ｘ）",
            "TOTAL\u{3000}AUD",
            "日本：東京",
        ] {
            assert_eq!(
                normalize(s).chars().count(),
                s.chars().count(),
                "1:1 mapping required for {s:?}"
            );
        }
    }
}
