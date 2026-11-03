//! Resolve model file paths from the project's `models/` directory.
//!
//! PP-OCR splits its pipeline into a script-agnostic detector plus a
//! per-script recognizer. The detector finds text regions; the recognizer
//! decodes glyphs against a fixed dictionary. Mismatching recognizer ↔
//! dictionary silently emits garbage rather than erroring, so the two
//! must be selected as an atomic unit. `RecognizerScript` enforces that.
//!
//! **Two orthogonal axes (Plan 9).** PP-OCRv6 introduced a *size* axis
//! that is independent of the *script* axis:
//!
//! * [`Quality`] picks the tier — tiny / small / medium. Detection is
//!   script-agnostic, so the tier selects the detector outright.
//! * [`RecognizerScript`] picks the recognizer + dictionary.
//!
//! The axes meet at one place: PP-OCRv6's general recognizer covers
//! Latin, CJK and Greek, so for those scripts the tier picks the
//! recognizer too. It contains **no** Devanagari, Cyrillic or Hangul
//! (verified by codepoint census of `ppocrv6_dict.txt`), so those scripts
//! keep their PP-OCRv5 recognizer at every tier — see
//! [`ScriptDescriptor::v6_general`]. That pairing is legal because every
//! PP-OCR recognizer takes the same `[N, 3, 48, W]` crop, so a v6
//! detector feeds a v5 recognizer without an adapter.
//!
//! **Adding a script** is still a one-row edit to [`DESCRIPTORS`]; adding
//! a tier is a one-row edit to [`QUALITIES`]. Weights download on demand,
//! as do the optional orientation classifiers in [`CLASSIFIERS`].

use crate::error::{OcrError, Result};
use std::path::{Path, PathBuf};

/// Which recognizer family to use. The detector is shared.
///
/// Each variant is backed by a static [`ScriptDescriptor`] — the enum is
/// just a typed handle into [`DESCRIPTORS`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecognizerScript {
    /// PP-OCRv5 mobile rec — Simplified/Traditional Chinese, English,
    /// Japanese, Pinyin. The broad default and auto-detect first pass.
    LatinCjk,
    /// PP-OCRv5 Devanagari rec — Hindi, Nepali, Sanskrit, Marathi, etc.
    Devanagari,
    /// PP-OCRv5 Latin rec — European Latin scripts with full diacritics
    /// (Czech, Polish, Slovak, Vietnamese, …). Covers letters the broad
    /// CJK model's dictionary omits (ď ť ů ž …).
    Latin,
    /// PP-OCRv5 Cyrillic rec — broad Cyrillic coverage (30+ languages:
    /// Serbian, Bulgarian, Mongolian, …).
    Cyrillic,
    /// PP-OCRv5 East-Slavic rec — Russian, Ukrainian, Belarusian, tuned
    /// tighter than the broad Cyrillic model. Preferred for RU/UK.
    Eslav,
    /// PP-OCRv5 Korean rec — Hangul + Hanja + English.
    Korean,
}

/// PP-OCRv6 model size tier. Orthogonal to [`RecognizerScript`].
///
/// Selects the detector always, and the recognizer for scripts inside
/// PP-OCRv6's repertoire (see [`ScriptDescriptor::v6_general`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Quality {
    /// 1.7 MiB det + 4.3 MiB rec. Smallest and fastest — measured faster
    /// than PP-OCRv5 mobile on CPU. Its 6 904-char dictionary drops kana,
    /// so it cannot read Japanese.
    Tiny,
    /// 9.4 MiB det + 20.2 MiB rec. Near-medium accuracy at roughly
    /// PP-OCRv5 mobile's latency.
    Small,
    /// 59.2 MiB det + 73.0 MiB rec. Highest accuracy and the tightest
    /// boxes — which matters here, because a tight quad makes a tighter
    /// searchable-PDF selection rectangle. The default.
    #[default]
    Medium,
}

/// Everything that varies per size tier. One static row per variant in
/// [`QUALITIES`].
#[derive(Debug, Clone, Copy)]
pub struct QualityDescriptor {
    pub id: Quality,
    /// `--quality` value that selects this tier.
    pub cli_value: &'static str,
    /// Human-readable name for CLI output.
    pub display_name: &'static str,
    /// Detector ONNX file name under `models/`.
    pub det_filename: &'static str,
    /// PP-OCRv6 general recognizer at this tier (Latin + CJK + Greek).
    pub rec_filename: &'static str,
    /// Dictionary paired with `rec_filename`. Tiny has its own smaller
    /// one — pairing it with the 18 708-entry dict yields garbage, not an
    /// error, so the two travel together.
    pub dict_filename: &'static str,
}

/// The tier registry. Adding a tier = adding a row here.
pub static QUALITIES: &[QualityDescriptor] = &[
    QualityDescriptor {
        id: Quality::Tiny,
        cli_value: "tiny",
        display_name: "PP-OCRv6 tiny",
        det_filename: "pp-ocrv6_tiny_det.onnx",
        rec_filename: "pp-ocrv6_tiny_rec.onnx",
        dict_filename: "ppocrv6_tiny_dict.txt",
    },
    QualityDescriptor {
        id: Quality::Small,
        cli_value: "small",
        display_name: "PP-OCRv6 small",
        det_filename: "pp-ocrv6_small_det.onnx",
        rec_filename: "pp-ocrv6_small_rec.onnx",
        dict_filename: "ppocrv6_dict.txt",
    },
    QualityDescriptor {
        id: Quality::Medium,
        cli_value: "medium",
        display_name: "PP-OCRv6 medium",
        det_filename: "pp-ocrv6_medium_det.onnx",
        rec_filename: "pp-ocrv6_medium_rec.onnx",
        dict_filename: "ppocrv6_dict.txt",
    },
];

impl Quality {
    /// The static descriptor for this tier.
    pub fn descriptor(self) -> &'static QualityDescriptor {
        QUALITIES
            .iter()
            .find(|q| q.id == self)
            .expect("every Quality variant has a QUALITIES row")
    }

    /// All tiers, smallest first.
    pub fn all() -> impl Iterator<Item = Quality> {
        QUALITIES.iter().map(|q| q.id)
    }

    /// Resolve a `--quality` value to a tier.
    pub fn from_cli_value(value: &str) -> Option<Quality> {
        let v = value.to_ascii_lowercase();
        QUALITIES.iter().find(|q| q.cli_value == v).map(|q| q.id)
    }

    /// Detector ONNX file name under `models/`.
    pub fn det_filename(self) -> &'static str {
        self.descriptor().det_filename
    }

    /// Human-readable name for CLI output.
    pub fn display_name(self) -> &'static str {
        self.descriptor().display_name
    }

    /// URL of this tier's detector ONNX (download-on-demand).
    pub fn det_onnx_url(self) -> String {
        v6_release_url(self.descriptor().det_filename)
    }

    /// URL of the PP-OCRv6 general recognizer at this tier.
    pub fn rec_onnx_url(self) -> String {
        v6_release_url(self.descriptor().rec_filename)
    }

    /// URL of the dictionary paired with this tier's recognizer.
    pub fn dict_url(self) -> String {
        v6_release_url(self.descriptor().dict_filename)
    }
}

/// PP-OCRv6 weights are published as flat ONNX on oar-ocr's `v0.7.0`
/// release (the upstream PaddlePaddle artifacts are `.tar` bundles that
/// would need unpacking). Same files, same hashes, no tar dependency.
fn v6_release_url(filename: &str) -> String {
    format!("https://github.com/GreatV/oar-ocr/releases/download/v0.7.0/{filename}")
}

/// An optional PP-LCNet orientation classifier. Neither is needed to read
/// a page: each backs one `ocr` flag and is downloaded when that flag is
/// first used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Classifier {
    /// Whole-page orientation, 4 classes (0/90/180/270), run before
    /// detection. Backs `--detect-page-orientation`.
    DocOrientation,
    /// Per-line orientation, 2 classes (0/180), run after detection. It
    /// catches upside-down lines only; sideways text is handled by quad
    /// ordering in `ocr::normalize_quad`. Backs `--detect-line-orientation`.
    TextlineOrientation,
}

/// Everything that varies per orientation classifier. One static row per
/// variant in [`CLASSIFIERS`].
#[derive(Debug, Clone, Copy)]
pub struct ClassifierDescriptor {
    pub id: Classifier,
    /// Human-readable name for CLI output.
    pub display_name: &'static str,
    /// ONNX file name under `models/`.
    pub filename: &'static str,
    /// The `ocr` flag that loads this classifier.
    pub cli_flag: &'static str,
    /// SHA-256 of the file, lowercase hex. A download that doesn't match is
    /// discarded rather than renamed into place.
    pub sha256: &'static str,
}

/// The classifier registry.
///
/// Both files are assets of oar-ocr's `v0.3.0` release (listed in oar-ocr
/// 0.8.1's `docs/models.md`). The checksums are oar-ocr-core 0.8.1's
/// registry entries for the same file names (`core::download::registry`),
/// and the copies this project has run on match them byte for byte.
pub static CLASSIFIERS: &[ClassifierDescriptor] = &[
    ClassifierDescriptor {
        id: Classifier::DocOrientation,
        display_name: "page-orientation classifier",
        filename: "pp-lcnet_x1_0_doc_ori.onnx",
        cli_flag: "--detect-page-orientation",
        sha256: "bbcd6c2b43ab15d2e605455aef2cd280ab87b570824522d24a69e9298875a1ac",
    },
    ClassifierDescriptor {
        id: Classifier::TextlineOrientation,
        display_name: "line-orientation classifier",
        filename: "pp-lcnet_x1_0_textline_ori.onnx",
        cli_flag: "--detect-line-orientation",
        sha256: "6b02efabbedd6be69e3de4c86b8dceed2d7329e75c12a796e6717bfb0d646950",
    },
];

impl Classifier {
    /// The static descriptor for this classifier.
    pub fn descriptor(self) -> &'static ClassifierDescriptor {
        CLASSIFIERS
            .iter()
            .find(|c| c.id == self)
            .expect("every Classifier variant has a CLASSIFIERS row")
    }

    /// All classifiers, in registry order.
    pub fn all() -> impl Iterator<Item = Classifier> {
        CLASSIFIERS.iter().map(|c| c.id)
    }

    /// ONNX file name under `models/`.
    pub fn filename(self) -> &'static str {
        self.descriptor().filename
    }

    /// Human-readable name for CLI output.
    pub fn display_name(self) -> &'static str {
        self.descriptor().display_name
    }

    /// Expected SHA-256 of the file, lowercase hex.
    pub fn sha256(self) -> &'static str {
        self.descriptor().sha256
    }

    /// URL of the ONNX file (download-on-demand).
    pub fn url(self) -> String {
        format!(
            "https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/{}",
            self.filename()
        )
    }
}

/// Every file name the registries in this module can place in `models/`:
/// each tier's detector, recognizer and dictionary, each script's
/// PP-OCRv5 pair, and the orientation classifiers. A directory holding
/// any one of them is a models directory. May repeat a name that two rows
/// share.
pub fn known_model_filenames() -> impl Iterator<Item = &'static str> {
    QUALITIES
        .iter()
        .flat_map(|q| [q.det_filename, q.rec_filename, q.dict_filename])
        .chain(
            DESCRIPTORS
                .iter()
                .flat_map(|d| [d.rec_filename, d.dict_filename]),
        )
        .chain(CLASSIFIERS.iter().map(|c| c.filename))
}

/// Post-processing policy applied between OCR and band-sort for text
/// exports (txt/json/md). PDF output skips postproc entirely (Plan 5),
/// so this never affects the searchable-PDF text layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PostProc {
    /// No cleanup — emit recognizer output verbatim.
    None,
    /// Aggressive Nepali/Devanagari cleanup (strip Latin misreads, drop
    /// page-noise). Correct for Nepali books, destructive elsewhere.
    Nepali,
}

/// Everything that varies per recognizer script. One static row per
/// variant in [`DESCRIPTORS`].
#[derive(Debug, Clone, Copy)]
pub struct ScriptDescriptor {
    pub id: RecognizerScript,
    /// Human-readable name for CLI output.
    pub display_name: &'static str,
    /// `--script` value(s) that select this recognizer. First is
    /// canonical; the rest are aliases (e.g. russian/ukrainian → eslav).
    pub cli_aliases: &'static [&'static str],
    /// Recognizer ONNX file name under `models/`.
    pub rec_filename: &'static str,
    /// Dictionary file name under `models/` (pairs with the recognizer).
    pub dict_filename: &'static str,
    /// HuggingFace repo holding the official pre-converted ONNX
    /// (`PaddlePaddle/<repo>`, file `inference.onnx`). Empty if the model
    /// ships bundled and has no download source wired.
    pub hf_onnx_repo: &'static str,
    /// Dictionary file name in PaddleOCR's `ppocr/utils/dict/` on GitHub.
    pub dict_github_name: &'static str,
    /// Text-export cleanup policy.
    pub postproc: PostProc,
    /// True when PP-OCRv6's *general* recognizer covers this script, in
    /// which case [`Quality`] selects the recognizer and `rec_filename` /
    /// `dict_filename` are only the PP-OCRv5 fallback.
    ///
    /// Set from a codepoint census of `ppocrv6_dict.txt` (18 708 entries:
    /// Latin, Greek, CJK, kana — and nothing else). Scripts whose glyphs
    /// are absent from that dictionary can never be emitted by the v6 CTC
    /// head, so they stay on PP-OCRv5 at every tier.
    pub v6_general: bool,
}

/// The script registry. Adding a language = adding a row here.
pub static DESCRIPTORS: &[ScriptDescriptor] = &[
    ScriptDescriptor {
        id: RecognizerScript::LatinCjk,
        display_name: "Latin/CJK (Chinese, Japanese, English)",
        cli_aliases: &["latin-cjk", "chinese", "japanese", "english"],
        rec_filename: "pp-ocrv5_mobile_rec.onnx",
        dict_filename: "ppocrv5_dict.txt",
        hf_onnx_repo: "PP-OCRv5_mobile_rec_onnx",
        dict_github_name: "ppocrv5_dict.txt",
        postproc: PostProc::None,
        // PP-OCRv6 covers this row outright and widens Latin coverage a
        // long way: Latin Extended-A 29 → 128, Extended-B 13 → 208,
        // Latin-1 Supplement 58 → 93. Verified PASS for English, German,
        // French, Italian, Portuguese, Dutch, Polish, Czech, Slovak,
        // Hungarian, Romanian, Turkish and the Nordic set.
        v6_general: true,
    },
    ScriptDescriptor {
        id: RecognizerScript::Devanagari,
        display_name: "Devanagari (Hindi, Nepali, Sanskrit, Marathi)",
        cli_aliases: &["devanagari", "hindi", "nepali", "sanskrit", "marathi"],
        rec_filename: "devanagari_PP-OCRv5_rec.onnx",
        dict_filename: "ppocrv5_devanagari_dict.txt",
        hf_onnx_repo: "devanagari_PP-OCRv5_mobile_rec_onnx",
        dict_github_name: "ppocrv5_devanagari_dict.txt",
        postproc: PostProc::Nepali,
        // `ppocrv6_dict.txt` contains ZERO Devanagari codepoints, so the
        // v6 head has no class to emit them. PP-OCRv5 only.
        v6_general: false,
    },
    ScriptDescriptor {
        id: RecognizerScript::Latin,
        display_name: "Latin-extended (Czech, Polish, Slovak, …)",
        cli_aliases: &["latin", "czech", "polish", "slovak", "vietnamese"],
        rec_filename: "latin_PP-OCRv5_mobile_rec.onnx",
        dict_filename: "ppocrv5_latin_dict.txt",
        hf_onnx_repo: "latin_PP-OCRv5_mobile_rec_onnx",
        dict_github_name: "ppocrv5_latin_dict.txt",
        postproc: PostProc::None,
        // Deliberately NOT v6, even though v6 handles Czech/Polish/Slovak
        // well: this row advertises `vietnamese`, and `ppocrv6_dict.txt`
        // is missing 46 of Vietnamese's tonal characters — routing it to
        // v6 would silently mangle every tone mark. The v5 Latin dict
        // also keeps 30 characters v6 dropped (typographic dashes and
        // quotes ‐ ‑ ‒ „ ‹ › ‼ №, Greek final sigma ς) plus Spanish `¡`.
        v6_general: false,
    },
    ScriptDescriptor {
        id: RecognizerScript::Cyrillic,
        display_name: "Cyrillic (broad: Serbian, Bulgarian, …)",
        cli_aliases: &["cyrillic"],
        rec_filename: "cyrillic_PP-OCRv5_mobile_rec.onnx",
        dict_filename: "ppocrv5_cyrillic_dict.txt",
        hf_onnx_repo: "cyrillic_PP-OCRv5_mobile_rec_onnx",
        dict_github_name: "ppocrv5_cyrillic_dict.txt",
        postproc: PostProc::None,
        // `ppocrv6_dict.txt` contains ZERO Cyrillic codepoints. PP-OCRv5 only.
        v6_general: false,
    },
    ScriptDescriptor {
        id: RecognizerScript::Eslav,
        display_name: "East Slavic (Russian, Ukrainian, Belarusian)",
        cli_aliases: &["eslav", "russian", "ukrainian", "belarusian"],
        rec_filename: "eslav_PP-OCRv5_mobile_rec.onnx",
        dict_filename: "ppocrv5_eslav_dict.txt",
        hf_onnx_repo: "eslav_PP-OCRv5_mobile_rec_onnx",
        dict_github_name: "ppocrv5_eslav_dict.txt",
        postproc: PostProc::None,
        // `ppocrv6_dict.txt` contains ZERO Cyrillic codepoints. PP-OCRv5 only.
        v6_general: false,
    },
    ScriptDescriptor {
        id: RecognizerScript::Korean,
        display_name: "Korean (Hangul + Hanja)",
        cli_aliases: &["korean"],
        rec_filename: "korean_PP-OCRv5_mobile_rec.onnx",
        dict_filename: "ppocrv5_korean_dict.txt",
        hf_onnx_repo: "korean_PP-OCRv5_mobile_rec_onnx",
        dict_github_name: "ppocrv5_korean_dict.txt",
        postproc: PostProc::None,
        // `ppocrv6_dict.txt` contains ZERO Hangul codepoints. PP-OCRv5 only.
        v6_general: false,
    },
];

impl RecognizerScript {
    /// The static descriptor for this script.
    pub fn descriptor(self) -> &'static ScriptDescriptor {
        // Linear scan over a 6-element static table — trivial, and keeps
        // the enum↔row mapping in one place (the table itself).
        DESCRIPTORS
            .iter()
            .find(|d| d.id == self)
            .expect("every RecognizerScript variant has a DESCRIPTORS row")
    }

    /// All recognizer scripts, in registry order.
    pub fn all() -> impl Iterator<Item = RecognizerScript> {
        DESCRIPTORS.iter().map(|d| d.id)
    }

    /// Resolve a `--script` value (canonical name or alias) to a script.
    pub fn from_cli_value(value: &str) -> Option<RecognizerScript> {
        let v = value.to_ascii_lowercase();
        DESCRIPTORS
            .iter()
            .find(|d| d.cli_aliases.contains(&v.as_str()))
            .map(|d| d.id)
    }

    /// File name of the recognizer ONNX model under `models/`.
    pub fn rec_filename(self) -> &'static str {
        self.descriptor().rec_filename
    }

    /// File name of the dictionary that pairs with `rec_filename()`.
    pub fn dict_filename(self) -> &'static str {
        self.descriptor().dict_filename
    }

    /// Text-export post-processing policy.
    pub fn postproc(self) -> PostProc {
        self.descriptor().postproc
    }

    /// Human-readable name for CLI output.
    pub fn display_name(self) -> &'static str {
        self.descriptor().display_name
    }

    /// URL of the recognizer ONNX on HuggingFace (download-on-demand).
    pub fn rec_onnx_url(self) -> String {
        format!(
            "https://huggingface.co/PaddlePaddle/{}/resolve/main/inference.onnx",
            self.descriptor().hf_onnx_repo
        )
    }

    /// URL of the dictionary `.txt` on PaddleOCR's GitHub (download-on-demand).
    pub fn dict_url(self) -> String {
        format!(
            "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/utils/dict/{}",
            self.descriptor().dict_github_name
        )
    }

    /// True when PP-OCRv6's general recognizer covers this script.
    pub fn uses_v6_general(self) -> bool {
        self.descriptor().v6_general
    }

    /// The (recognizer, dictionary) file names to use at `quality`.
    ///
    /// This is the single place the script and tier axes combine: v6-covered
    /// scripts follow the tier, everything else pins to its PP-OCRv5 pair.
    /// Returning both together is deliberate — a recognizer paired with the
    /// wrong dictionary emits plausible garbage instead of failing, so they
    /// must never be resolved independently.
    pub fn rec_and_dict_for(self, quality: Quality) -> (&'static str, &'static str) {
        if self.uses_v6_general() {
            let q = quality.descriptor();
            (q.rec_filename, q.dict_filename)
        } else {
            let d = self.descriptor();
            (d.rec_filename, d.dict_filename)
        }
    }

    /// Download URLs for [`Self::rec_and_dict_for`]'s two files.
    pub fn rec_and_dict_urls(self, quality: Quality) -> (String, String) {
        if self.uses_v6_general() {
            (quality.rec_onnx_url(), quality.dict_url())
        } else {
            (self.rec_onnx_url(), self.dict_url())
        }
    }
}

/// Path layout oar-ocr expects.
#[derive(Debug, Clone)]
pub struct ModelPaths {
    pub det_onnx: PathBuf,
    pub rec_onnx: PathBuf,
    pub keys_txt: PathBuf,
    /// Optional text-line orientation classifier (PP-LCNet). When
    /// present, the pipeline tags each detected region with a 0° or 180°
    /// angle before recognition. `pp-lcnet_x1_0_textline_ori.onnx` takes
    /// 80x160 input and emits 2 classes — it detects upside-down lines,
    /// NOT 90° rotation. Sideways text is handled by quad ordering in
    /// `ocr::normalize_quad` instead.
    pub textline_ori_onnx: Option<PathBuf>,
    /// Optional document-orientation classifier (PP-LCNet, 4 classes:
    /// 0/90/180/270). When present and enabled, oar-ocr rotates the whole
    /// page upright before detection and maps every box (incl. per-char
    /// boxes) back to original coordinates — the fix for fully-rotated
    /// pages like sideways tables (Plan 7).
    pub doc_ori_onnx: Option<PathBuf>,
}

impl ModelPaths {
    /// Resolve paths for the default Latin/CJK recognizer at the default
    /// tier. Kept for back-compat with existing callers and tests.
    pub fn in_dir(models_dir: &Path) -> Result<Self> {
        Self::resolve(models_dir, RecognizerScript::LatinCjk, Quality::default())
    }

    /// Resolve paths for a specific script at the default tier.
    pub fn in_dir_for(models_dir: &Path, script: RecognizerScript) -> Result<Self> {
        Self::resolve(models_dir, script, Quality::default())
    }

    /// Resolve paths for a specific (script, tier) pair.
    ///
    /// The three core files (det, rec, dict) must exist on disk. The
    /// optional orientation classifiers are loaded if present in the same
    /// directory; if absent, the pipeline skips them silently.
    pub fn resolve(models_dir: &Path, script: RecognizerScript, quality: Quality) -> Result<Self> {
        let (rec_name, dict_name) = script.rec_and_dict_for(quality);
        let det = models_dir.join(quality.det_filename());
        let rec = models_dir.join(rec_name);
        let keys = models_dir.join(dict_name);
        for p in [&det, &rec, &keys] {
            if !p.exists() {
                return Err(OcrError::ModelNotFound(p.display().to_string()));
            }
        }
        // Optional orientation classifiers.
        let textline_ori = models_dir.join(Classifier::TextlineOrientation.filename());
        let textline_ori = textline_ori.exists().then_some(textline_ori);
        let doc_ori = models_dir.join(Classifier::DocOrientation.filename());
        let doc_ori = doc_ori.exists().then_some(doc_ori);
        Ok(Self {
            det_onnx: det,
            rec_onnx: rec,
            keys_txt: keys,
            textline_ori_onnx: textline_ori,
            doc_ori_onnx: doc_ori,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Helper: stage empty placeholder files so existence checks pass
    /// without committing real model weights into the test fixture.
    fn stage_dir(names: &[&str]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        for n in names {
            fs::write(dir.path().join(n), b"").expect("write stub");
        }
        dir
    }

    #[test]
    fn every_variant_has_exactly_one_descriptor() {
        for script in RecognizerScript::all() {
            let matches = DESCRIPTORS.iter().filter(|d| d.id == script).count();
            assert_eq!(matches, 1, "{script:?} must have exactly one row");
        }
        assert_eq!(DESCRIPTORS.len(), 6, "registry size");
    }

    #[test]
    fn cli_aliases_are_unique_across_scripts() {
        let mut seen = std::collections::HashSet::new();
        for d in DESCRIPTORS {
            for alias in d.cli_aliases {
                assert!(seen.insert(*alias), "duplicate CLI alias: {alias}");
            }
        }
    }

    #[test]
    fn from_cli_value_resolves_aliases_case_insensitively() {
        assert_eq!(
            RecognizerScript::from_cli_value("russian"),
            Some(RecognizerScript::Eslav)
        );
        assert_eq!(
            RecognizerScript::from_cli_value("Ukrainian"),
            Some(RecognizerScript::Eslav)
        );
        assert_eq!(
            RecognizerScript::from_cli_value("czech"),
            Some(RecognizerScript::Latin)
        );
        assert_eq!(
            RecognizerScript::from_cli_value("japanese"),
            Some(RecognizerScript::LatinCjk)
        );
        assert_eq!(RecognizerScript::from_cli_value("klingon"), None);
    }

    #[test]
    fn only_devanagari_uses_nepali_postproc() {
        for script in RecognizerScript::all() {
            let expected = if script == RecognizerScript::Devanagari {
                PostProc::Nepali
            } else {
                PostProc::None
            };
            assert_eq!(script.postproc(), expected, "{script:?}");
        }
    }

    #[test]
    fn download_urls_are_well_formed() {
        let k = RecognizerScript::Korean;
        assert_eq!(
            k.rec_onnx_url(),
            "https://huggingface.co/PaddlePaddle/korean_PP-OCRv5_mobile_rec_onnx/resolve/main/inference.onnx"
        );
        assert_eq!(
            k.dict_url(),
            "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/utils/dict/ppocrv5_korean_dict.txt"
        );
    }

    #[test]
    fn latin_cjk_resolves_to_the_tier_rec_and_dict() {
        let dir = stage_dir(&[
            "pp-ocrv6_medium_det.onnx",
            "pp-ocrv6_medium_rec.onnx",
            "ppocrv6_dict.txt",
        ]);
        let paths = ModelPaths::in_dir_for(dir.path(), RecognizerScript::LatinCjk)
            .expect("latin-cjk paths resolve");
        assert!(paths.rec_onnx.ends_with("pp-ocrv6_medium_rec.onnx"));
        assert!(paths.keys_txt.ends_with("ppocrv6_dict.txt"));
    }

    #[test]
    fn devanagari_resolves_to_devanagari_rec_and_dict() {
        let dir = stage_dir(&[
            "pp-ocrv6_medium_det.onnx",
            "devanagari_PP-OCRv5_rec.onnx",
            "ppocrv5_devanagari_dict.txt",
        ]);
        let paths = ModelPaths::in_dir_for(dir.path(), RecognizerScript::Devanagari)
            .expect("devanagari paths resolve");
        assert!(paths.rec_onnx.ends_with("devanagari_PP-OCRv5_rec.onnx"));
        assert!(paths.keys_txt.ends_with("ppocrv5_devanagari_dict.txt"));
    }

    #[test]
    fn korean_resolves_to_korean_rec_and_dict() {
        let dir = stage_dir(&[
            "pp-ocrv6_medium_det.onnx",
            "korean_PP-OCRv5_mobile_rec.onnx",
            "ppocrv5_korean_dict.txt",
        ]);
        let paths = ModelPaths::in_dir_for(dir.path(), RecognizerScript::Korean)
            .expect("korean paths resolve");
        assert!(paths.rec_onnx.ends_with("korean_PP-OCRv5_mobile_rec.onnx"));
        assert!(paths.keys_txt.ends_with("ppocrv5_korean_dict.txt"));
    }

    #[test]
    fn missing_devanagari_dict_errors_with_modelnotfound() {
        // Stage rec but not dict — must fail loud, not pick up the wrong dict.
        let dir = stage_dir(&[
            "pp-ocrv6_medium_det.onnx",
            "devanagari_PP-OCRv5_rec.onnx",
            "ppocrv5_dict.txt", // wrong dict; should NOT be silently used
        ]);
        let err = ModelPaths::in_dir_for(dir.path(), RecognizerScript::Devanagari)
            .expect_err("missing devanagari dict must error");
        assert!(matches!(err, OcrError::ModelNotFound(_)));
    }

    #[test]
    fn legacy_in_dir_still_returns_latin_cjk() {
        let dir = stage_dir(&[
            "pp-ocrv6_medium_det.onnx",
            "pp-ocrv6_medium_rec.onnx",
            "ppocrv6_dict.txt",
        ]);
        let paths = ModelPaths::in_dir(dir.path()).expect("legacy in_dir");
        assert!(paths.rec_onnx.ends_with("pp-ocrv6_medium_rec.onnx"));
        assert!(paths.keys_txt.ends_with("ppocrv6_dict.txt"));
    }

    // ---- the two-axis contract -------------------------------------

    #[test]
    fn every_quality_has_exactly_one_descriptor() {
        for q in Quality::all() {
            assert_eq!(QUALITIES.iter().filter(|d| d.id == q).count(), 1, "{q:?}");
        }
        assert_eq!(QUALITIES.len(), 3, "tier registry size");
        assert_eq!(Quality::default(), Quality::Medium, "medium is the default");
    }

    #[test]
    fn quality_cli_values_round_trip() {
        for q in Quality::all() {
            assert_eq!(Quality::from_cli_value(q.descriptor().cli_value), Some(q));
        }
        assert_eq!(Quality::from_cli_value("MEDIUM"), Some(Quality::Medium));
        assert_eq!(Quality::from_cli_value("enormous"), None);
    }

    /// The detector follows the tier for EVERY script — detection is
    /// script-agnostic, so a v5-only recognizer still gets a v6 detector.
    #[test]
    fn detector_follows_tier_even_for_v5_only_scripts() {
        for q in Quality::all() {
            for script in RecognizerScript::all() {
                let (rec, _) = script.rec_and_dict_for(q);
                let expected_rec = if script.uses_v6_general() {
                    q.descriptor().rec_filename
                } else {
                    script.descriptor().rec_filename
                };
                assert_eq!(rec, expected_rec, "{script:?} @ {q:?}");
            }
        }
        // Concretely: Devanagari at medium = v6 detector + v5 Devanagari rec.
        let dir = stage_dir(&[
            "pp-ocrv6_medium_det.onnx",
            "devanagari_PP-OCRv5_rec.onnx",
            "ppocrv5_devanagari_dict.txt",
        ]);
        let paths = ModelPaths::resolve(dir.path(), RecognizerScript::Devanagari, Quality::Medium)
            .expect("mixed v6 det + v5 deva rec resolves");
        assert!(paths.det_onnx.ends_with("pp-ocrv6_medium_det.onnx"));
        assert!(paths.rec_onnx.ends_with("devanagari_PP-OCRv5_rec.onnx"));
    }

    /// Non-Latin scripts must never be routed to the v6 general model —
    /// its dictionary has no codepoints for them, so it would emit
    /// confident nonsense rather than fail.
    #[test]
    fn only_latin_cjk_is_marked_v6_general() {
        for script in RecognizerScript::all() {
            let expected = script == RecognizerScript::LatinCjk;
            assert_eq!(script.uses_v6_general(), expected, "{script:?}");
        }
    }

    /// Tier and dictionary travel together: tiny has its own 6 904-entry
    /// dict, small and medium share the 18 708-entry one.
    #[test]
    fn tier_dictionaries_pair_correctly() {
        let latin_cjk = RecognizerScript::LatinCjk;
        assert_eq!(
            latin_cjk.rec_and_dict_for(Quality::Tiny).1,
            "ppocrv6_tiny_dict.txt"
        );
        assert_eq!(
            latin_cjk.rec_and_dict_for(Quality::Small).1,
            "ppocrv6_dict.txt"
        );
        assert_eq!(
            latin_cjk.rec_and_dict_for(Quality::Medium).1,
            "ppocrv6_dict.txt"
        );
    }

    #[test]
    fn v6_urls_point_at_the_release_assets() {
        assert_eq!(
            Quality::Medium.det_onnx_url(),
            "https://github.com/GreatV/oar-ocr/releases/download/v0.7.0/pp-ocrv6_medium_det.onnx"
        );
        // A v5-only script keeps its HuggingFace / PaddleOCR sources.
        let (rec, dict) = RecognizerScript::Korean.rec_and_dict_urls(Quality::Medium);
        assert!(rec.contains("huggingface.co/PaddlePaddle/korean_PP-OCRv5"));
        assert!(dict.contains("raw.githubusercontent.com/PaddlePaddle/PaddleOCR"));
    }

    // ---- orientation classifiers -------------------------------------

    #[test]
    fn every_classifier_has_exactly_one_descriptor() {
        for c in Classifier::all() {
            assert_eq!(CLASSIFIERS.iter().filter(|d| d.id == c).count(), 1, "{c:?}");
        }
        assert_eq!(CLASSIFIERS.len(), 2, "classifier registry size");
    }

    #[test]
    fn classifier_urls_and_checksums_are_well_formed() {
        assert_eq!(
            Classifier::DocOrientation.url(),
            "https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-lcnet_x1_0_doc_ori.onnx"
        );
        assert_eq!(
            Classifier::TextlineOrientation.url(),
            "https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-lcnet_x1_0_textline_ori.onnx"
        );
        for c in Classifier::all() {
            let hash = c.sha256();
            assert_eq!(hash.len(), 64, "{c:?}");
            assert!(
                hash.chars()
                    .all(|ch| ch.is_ascii_digit() || ('a'..='f').contains(&ch)),
                "{c:?}: sha256 must be lowercase hex"
            );
        }
    }

    /// `ModelPaths` picks the classifiers up by their registry names.
    #[test]
    fn resolve_finds_classifiers_by_registry_filename() {
        let dir = stage_dir(&[
            "pp-ocrv6_medium_det.onnx",
            "pp-ocrv6_medium_rec.onnx",
            "ppocrv6_dict.txt",
            Classifier::DocOrientation.filename(),
        ]);
        let paths = ModelPaths::in_dir(dir.path()).expect("resolves");
        assert!(paths
            .doc_ori_onnx
            .is_some_and(|p| p.ends_with("pp-lcnet_x1_0_doc_ori.onnx")));
        assert!(paths.textline_ori_onnx.is_none(), "absent file stays None");
    }

    #[test]
    fn known_model_filenames_cover_every_registry() {
        let known: std::collections::HashSet<&str> = known_model_filenames().collect();
        for name in [
            "pp-ocrv6_tiny_det.onnx",
            "pp-ocrv6_medium_rec.onnx",
            "ppocrv6_tiny_dict.txt",
            "devanagari_PP-OCRv5_rec.onnx",
            "ppocrv5_korean_dict.txt",
            // The PP-OCRv5 Latin/CJK pair, so a pre-Plan-9 models/ still counts.
            "pp-ocrv5_mobile_rec.onnx",
            "ppocrv5_dict.txt",
            "pp-lcnet_x1_0_doc_ori.onnx",
            "pp-lcnet_x1_0_textline_ori.onnx",
        ] {
            assert!(known.contains(name), "{name} missing");
        }
        assert!(!known.contains("README.md"));
        assert!(known
            .iter()
            .all(|n| n.ends_with(".onnx") || n.ends_with(".txt")));
    }
}
