//! Download-on-demand for recognizer models (Plan 6).
//!
//! New scripts ship as official PaddleOCR ONNX on HuggingFace plus a
//! dictionary `.txt` from PaddleOCR's GitHub. Rather than vendoring
//! ~28 MB of weights, we fetch them into `models/` the first time a
//! recognizer is selected. Downloads are atomic (temp file + rename) so
//! an interrupted fetch never leaves a half-written model that would
//! then fail to load with a cryptic ONNX error.
//!
//! The optional orientation classifiers are fetched the same way, and
//! checked against a SHA-256 from the registry before they are renamed
//! into place.

use crate::{
    error::{OcrError, Result},
    model::{Classifier, Quality, RecognizerScript},
};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Read;
use std::path::Path;

/// Progress callback: `(downloaded_bytes, total_bytes_if_known)`.
pub type ProgressFn<'a> = dyn FnMut(u64, Option<u64>) + 'a;

/// One file to fetch: a URL and its destination filename under `models/`.
pub struct Fetch {
    pub url: String,
    pub filename: String,
    /// Human label for progress display (e.g. "korean recognizer").
    pub label: String,
    /// Expected SHA-256 (lowercase hex), when the registry knows it.
    pub sha256: Option<&'static str>,
}

/// The files a `(script, quality)` pair needs that are **missing** from
/// `models_dir`. Empty when everything is already present. The detector is
/// included if absent (a fresh clone may lack it too).
pub fn missing_files(models_dir: &Path, script: RecognizerScript, quality: Quality) -> Vec<Fetch> {
    let mut out = Vec::new();
    let (rec_name, dict_name) = script.rec_and_dict_for(quality);
    let (rec_url, dict_url) = script.rec_and_dict_urls(quality);

    if !models_dir.join(quality.det_filename()).exists() {
        out.push(Fetch {
            url: quality.det_onnx_url(),
            filename: quality.det_filename().to_string(),
            label: format!("{} text detector", quality.display_name()),
            sha256: None,
        });
    }
    // The recognizer's display name follows whichever model actually runs:
    // v6-covered scripts report the tier, the rest report the script.
    let rec_label = if script.uses_v6_general() {
        quality.display_name().to_string()
    } else {
        script.display_name().to_string()
    };
    if !models_dir.join(rec_name).exists() {
        out.push(Fetch {
            url: rec_url,
            filename: rec_name.to_string(),
            label: format!("{rec_label} recognizer"),
            sha256: None,
        });
    }
    if !models_dir.join(dict_name).exists() {
        out.push(Fetch {
            url: dict_url,
            filename: dict_name.to_string(),
            label: format!("{rec_label} dictionary"),
            sha256: None,
        });
    }
    out
}

/// The classifiers among `wanted` that are **missing** from `models_dir`,
/// in the order given. Empty when all are present.
pub fn missing_classifiers(
    models_dir: &Path,
    wanted: impl IntoIterator<Item = Classifier>,
) -> Vec<Fetch> {
    wanted
        .into_iter()
        .filter(|c| !models_dir.join(c.filename()).exists())
        .map(|c| Fetch {
            url: c.url(),
            filename: c.filename().to_string(),
            label: c.display_name().to_string(),
            sha256: Some(c.sha256()),
        })
        .collect()
}

/// Download `url` to `models_dir/filename` atomically.
///
/// Writes to `<filename>.part` then renames on success, so a partial
/// download is never visible under the real name. When `sha256` is given,
/// a download whose checksum differs is discarded instead. `on_progress`
/// fires as bytes arrive (total is `None` when the server omits
/// Content-Length).
pub fn download_file(
    url: &str,
    models_dir: &Path,
    filename: &str,
    sha256: Option<&str>,
    on_progress: &mut ProgressFn,
) -> Result<()> {
    fs::create_dir_all(models_dir)
        .map_err(|e| OcrError::Engine(format!("create {}: {e}", models_dir.display())))?;

    let resp = ureq::get(url)
        .call()
        .map_err(|e| OcrError::Engine(format!("GET {url}: {e}")))?;
    let total = resp
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok());

    save_verified(
        resp.into_body().into_reader(),
        models_dir,
        filename,
        total,
        sha256,
        on_progress,
    )
}

/// Stream `reader` into `models_dir/filename` through a `.part` file,
/// renaming it into place only once its length (`total`, when known) and
/// checksum (`sha256`, when given) check out. A failed check removes the
/// `.part` file.
fn save_verified(
    mut reader: impl Read,
    models_dir: &Path,
    filename: &str,
    total: Option<u64>,
    sha256: Option<&str>,
    on_progress: &mut ProgressFn,
) -> Result<()> {
    let dest = models_dir.join(filename);
    let tmp = models_dir.join(format!("{filename}.part"));

    let mut file = fs::File::create(&tmp)
        .map_err(|e| OcrError::Engine(format!("create {}: {e}", tmp.display())))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    let mut downloaded = 0u64;
    loop {
        let n = reader
            .read(&mut buf)
            .map_err(|e| OcrError::Engine(format!("read {filename}: {e}")))?;
        if n == 0 {
            break;
        }
        use std::io::Write as _;
        file.write_all(&buf[..n])
            .map_err(|e| OcrError::Engine(format!("write {}: {e}", tmp.display())))?;
        hasher.update(&buf[..n]);
        downloaded += n as u64;
        on_progress(downloaded, total);
    }
    file.sync_all()
        .map_err(|e| OcrError::Engine(format!("sync {}: {e}", tmp.display())))?;
    drop(file);

    // A truncated download (server hiccup) would otherwise rename into
    // place and fail later at model load. Guard the obvious case.
    if let Some(total) = total {
        if downloaded != total {
            let _ = fs::remove_file(&tmp);
            return Err(OcrError::Engine(format!(
                "incomplete download of {filename}: got {downloaded} of {total} bytes"
            )));
        }
    }
    // A changed upstream asset loads fine and then misbehaves (a
    // recognizer paired with the wrong dictionary emits plausible
    // garbage), so a known checksum is enforced, not merely logged.
    if let Some(expected) = sha256 {
        let actual = hex(&hasher.finalize());
        if !actual.eq_ignore_ascii_case(expected) {
            let _ = fs::remove_file(&tmp);
            return Err(OcrError::Engine(format!(
                "checksum mismatch for {filename}: expected sha256 {expected}, got {actual}"
            )));
        }
    }
    fs::rename(&tmp, &dest)
        .map_err(|e| OcrError::Engine(format!("rename into {}: {e}", dest.display())))?;
    Ok(())
}

/// Lowercase hex of `bytes`.
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const Q: Quality = Quality::Medium;

    /// SHA-256 of `b"hello\n"`.
    const HELLO_SHA256: &str = "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03";

    #[test]
    fn missing_files_lists_rec_and_dict_when_absent() {
        let dir = tempfile::tempdir().expect("tempdir");
        // Stage only the detector → rec + dict are missing.
        fs::write(dir.path().join(Q.det_filename()), b"x").expect("stage det");
        let fetches = missing_files(dir.path(), RecognizerScript::Korean, Q);
        let names: Vec<&str> = fetches.iter().map(|f| f.filename.as_str()).collect();
        assert_eq!(
            names,
            vec!["korean_PP-OCRv5_mobile_rec.onnx", "ppocrv5_korean_dict.txt"]
        );
        // URLs point at the right places.
        assert!(fetches[0].url.contains("korean_PP-OCRv5_mobile_rec_onnx"));
        assert!(fetches[1].url.ends_with("ppocrv5_korean_dict.txt"));
    }

    #[test]
    fn missing_files_includes_tier_detector_when_absent() {
        let dir = tempfile::tempdir().expect("tempdir");
        let fetches = missing_files(dir.path(), RecognizerScript::Eslav, Q);
        assert_eq!(fetches[0].filename, "pp-ocrv6_medium_det.onnx");
        assert!(fetches[0].url.contains("oar-ocr/releases/download/v0.7.0/"));
    }

    /// A v6-covered script must fetch the *tier's* recognizer, not the
    /// script row's PP-OCRv5 fallback.
    #[test]
    fn v6_general_script_fetches_tier_recognizer() {
        let dir = tempfile::tempdir().expect("tempdir");
        let fetches = missing_files(dir.path(), RecognizerScript::LatinCjk, Quality::Small);
        let names: Vec<&str> = fetches.iter().map(|f| f.filename.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "pp-ocrv6_small_det.onnx",
                "pp-ocrv6_small_rec.onnx",
                "ppocrv6_dict.txt",
            ]
        );
    }

    /// The tiny tier pairs with its own smaller dictionary.
    #[test]
    fn tiny_tier_uses_its_own_dictionary() {
        let dir = tempfile::tempdir().expect("tempdir");
        let fetches = missing_files(dir.path(), RecognizerScript::LatinCjk, Quality::Tiny);
        let names: Vec<&str> = fetches.iter().map(|f| f.filename.as_str()).collect();
        assert!(names.contains(&"ppocrv6_tiny_dict.txt"));
        assert!(!names.contains(&"ppocrv6_dict.txt"));
    }

    #[test]
    fn missing_files_empty_when_all_present() {
        let dir = tempfile::tempdir().expect("tempdir");
        for f in [
            Q.det_filename(),
            "korean_PP-OCRv5_mobile_rec.onnx",
            "ppocrv5_korean_dict.txt",
        ] {
            fs::write(dir.path().join(f), b"x").expect("stage");
        }
        assert!(missing_files(dir.path(), RecognizerScript::Korean, Q).is_empty());
    }

    #[test]
    fn missing_classifiers_lists_only_absent_ones_with_checksums() {
        let dir = tempfile::tempdir().expect("tempdir");
        fs::write(dir.path().join(Classifier::DocOrientation.filename()), b"x").expect("stage");
        let fetches = missing_classifiers(dir.path(), Classifier::all());
        assert_eq!(fetches.len(), 1);
        assert_eq!(fetches[0].filename, "pp-lcnet_x1_0_textline_ori.onnx");
        assert_eq!(fetches[0].url, Classifier::TextlineOrientation.url());
        assert_eq!(
            fetches[0].sha256,
            Some(Classifier::TextlineOrientation.sha256())
        );
        // Nothing wanted → nothing to fetch, whatever is on disk.
        assert!(missing_classifiers(dir.path(), std::iter::empty::<Classifier>()).is_empty());
    }

    #[test]
    fn save_verified_renames_a_matching_download_into_place() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut last = 0;
        save_verified(
            &b"hello\n"[..],
            dir.path(),
            "m.onnx",
            Some(6),
            Some(HELLO_SHA256),
            &mut |done, _| last = done,
        )
        .expect("verified save");
        assert_eq!(
            fs::read(dir.path().join("m.onnx")).expect("read"),
            b"hello\n"
        );
        assert!(!dir.path().join("m.onnx.part").exists());
        assert_eq!(last, 6, "progress reached the full length");
    }

    #[test]
    fn save_verified_discards_a_checksum_mismatch() {
        let dir = tempfile::tempdir().expect("tempdir");
        let err = save_verified(
            &b"hullo\n"[..],
            dir.path(),
            "m.onnx",
            None,
            Some(HELLO_SHA256),
            &mut |_, _| {},
        )
        .expect_err("wrong bytes must be rejected");
        assert!(err.to_string().contains("checksum mismatch"), "{err}");
        assert!(!dir.path().join("m.onnx").exists());
        assert!(!dir.path().join("m.onnx.part").exists());
    }

    #[test]
    fn save_verified_discards_a_truncated_download() {
        let dir = tempfile::tempdir().expect("tempdir");
        let err = save_verified(
            &b"hel"[..],
            dir.path(),
            "m.onnx",
            Some(6),
            None,
            &mut |_, _| {},
        )
        .expect_err("short body must be rejected");
        assert!(err.to_string().contains("incomplete download"), "{err}");
        assert!(!dir.path().join("m.onnx").exists());
        assert!(!dir.path().join("m.onnx.part").exists());
    }

    #[test]
    fn save_verified_without_a_checksum_accepts_any_bytes() {
        let dir = tempfile::tempdir().expect("tempdir");
        save_verified(&b"abc"[..], dir.path(), "d.txt", None, None, &mut |_, _| {})
            .expect("unchecked save");
        assert_eq!(fs::read(dir.path().join("d.txt")).expect("read"), b"abc");
    }
}
