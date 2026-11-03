//! Detect the dominant script of a string via Unicode codepoint ranges.
//!
//! Port of `_detect_script` from the retired Python tool
//! (`paddleknife.py:162-174` before commit bb551a6).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Script {
    English,
    Chinese,
    Devanagari,
    Arabic,
    Cyrillic,
    Japanese,
    Korean,
    Tamil,
    Telugu,
    Thai,
    Greek,
}

/// Scripts that do not separate words with spaces (CJK, kana, Thai,
/// Lao, Khmer, Myanmar). Hangul is *not* here — Korean uses spaces.
/// Drives line-join policy and searchable-PDF space insertion.
pub fn is_unspaced_script(c: char) -> bool {
    matches!(c as u32,
        0x2E80..=0x303F      // CJK radicals, Kangxi, CJK punctuation
        | 0x3040..=0x30FF    // hiragana, katakana
        | 0x3400..=0x4DBF    // CJK ext A
        | 0x4E00..=0x9FFF    // CJK unified
        | 0xF900..=0xFAFF    // CJK compat ideographs
        | 0xFF00..=0xFF60    // fullwidth forms
        | 0x0E00..=0x0E7F    // Thai
        | 0x0E80..=0x0EFF    // Lao
        | 0x1780..=0x17FF    // Khmer
        | 0x1000..=0x109F    // Myanmar
    )
}

/// Strong right-to-left characters (Hebrew, Arabic, and their
/// extended/presentation blocks). Drives reading-order policy.
pub fn is_rtl_char(c: char) -> bool {
    matches!(c as u32,
        0x0590..=0x05FF      // Hebrew
        | 0x0600..=0x06FF    // Arabic
        | 0x0750..=0x077F    // Arabic supplement
        | 0x08A0..=0x08FF    // Arabic extended-A
        | 0xFB1D..=0xFDFF    // Hebrew/Arabic presentation forms A
        | 0xFE70..=0xFEFF    // Arabic presentation forms B
    )
}

pub fn detect_script(text: &str) -> Script {
    let mut counts = [0u32; 11];
    for ch in text.chars() {
        let cp = ch as u32;
        let idx = match cp {
            0x0900..=0x097F => 2,  // Devanagari
            0x0600..=0x06FF => 3,  // Arabic
            0x0400..=0x04FF => 4,  // Cyrillic
            0x4E00..=0x9FFF => 1,  // CJK (Chinese)
            0x3040..=0x30FF => 5,  // Hiragana + Katakana
            0xAC00..=0xD7AF => 6,  // Hangul
            0x0B80..=0x0BFF => 7,  // Tamil
            0x0C00..=0x0C7F => 8,  // Telugu
            0x0E00..=0x0E7F => 9,  // Thai
            0x0370..=0x03FF => 10, // Greek
            _ => continue,
        };
        counts[idx] += 1;
    }
    // Safe to unwrap: counts is a fixed-size array of 11 elements, so max_by_key
    // on an enumeration of a non-empty iterator will always return Some.
    #[allow(clippy::unwrap_used)]
    let (max_idx, max_count) = counts.iter().enumerate().max_by_key(|(_, c)| *c).unwrap();
    if *max_count == 0 {
        return Script::English;
    }
    match max_idx {
        1 => Script::Chinese,
        2 => Script::Devanagari,
        3 => Script::Arabic,
        4 => Script::Cyrillic,
        5 => Script::Japanese,
        6 => Script::Korean,
        7 => Script::Tamil,
        8 => Script::Telugu,
        9 => Script::Thai,
        10 => Script::Greek,
        _ => Script::English,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_defaults_to_english() {
        assert_eq!(detect_script(""), Script::English);
    }

    #[test]
    fn ascii_defaults_to_english() {
        assert_eq!(detect_script("Hello World"), Script::English);
    }

    #[test]
    fn devanagari_detected() {
        assert_eq!(detect_script("मुना मदन"), Script::Devanagari);
    }

    #[test]
    fn arabic_detected() {
        assert_eq!(detect_script("السلام عليكم"), Script::Arabic);
    }

    #[test]
    fn mixed_picks_dominant() {
        // 10 Devanagari chars vs 5 Latin — Devanagari wins.
        assert_eq!(detect_script("मुनामदन hello"), Script::Devanagari);
    }
}
