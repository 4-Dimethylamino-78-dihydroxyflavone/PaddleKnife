//! Input discovery — turn a CLI path (a *file* or a *directory*) into an
//! ordered list of OCR-able documents.
//!
//! A directory may hold a single format or a mix of PDFs and images;
//! discovery classifies each entry and returns them in **natural** order
//! (`page2.png` before `page10.png`). That ordering is load-bearing: a
//! folder of scanned book pages is exactly the case where plain
//! lexicographic sorting silently shuffles the book.
//!
//! What counts as OCR-able is not a hand-maintained extension list. PDFs
//! are matched by extension; every other candidate is passed through
//! [`image::ImageFormat`], and only formats whose decoder both *exists*
//! (`can_read`) and is *compiled in* (`reading_enabled`) are accepted, so
//! the accepted set can never drift from what `image::open` will do.

use crate::error::{OcrError, Result};
use std::cmp::Ordering;
use std::path::{Path, PathBuf};

/// Suffix PaddleKnife stamps onto its own outputs. Discovery skips files
/// whose stem ends with it, so re-running over a folder never re-OCRs a
/// previous run's `*_paddleknife_OCR.pdf` — a real hazard, since our own
/// searchable-PDF output is itself a perfectly valid input.
pub const OUTPUT_STEM_SUFFIX: &str = "_paddleknife_OCR";

/// How a discovered file will be read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputKind {
    /// Multi-page PDF — rasterized page by page via pdfium.
    Pdf,
    /// Single-page raster read by the `image` crate.
    Image,
}

/// One OCR-able file found by [`discover`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputFile {
    /// Path to open.
    pub path: PathBuf,
    /// Path relative to the discovery root — just the file name for a
    /// single-file or non-recursive scan. Drives output-tree mirroring so
    /// `a/page.png` and `b/page.png` can't collide on one output name.
    pub relative: PathBuf,
    pub kind: InputKind,
}

/// Classify one path by extension. `None` means "not something we OCR".
///
/// Case-insensitive: `image::ImageFormat::from_extension` lowercases
/// internally, and the PDF check uses `eq_ignore_ascii_case`.
pub fn classify(path: &Path) -> Option<InputKind> {
    let ext = path.extension()?.to_str()?;
    if ext.eq_ignore_ascii_case("pdf") {
        return Some(InputKind::Pdf);
    }
    let fmt = image::ImageFormat::from_extension(ext)?;
    // `can_read`: a decoder exists at all. `reading_enabled`: it survived
    // this build's feature selection. Both must hold or `image::open`
    // would fail later with a confusing error.
    if fmt.can_read() && fmt.reading_enabled() {
        Some(InputKind::Image)
    } else {
        None
    }
}

/// Every extension [`classify`] accepts, sorted and de-duplicated.
/// Used to build actionable "nothing to OCR here" messages.
pub fn supported_extensions() -> Vec<&'static str> {
    let mut v = vec!["pdf"];
    for fmt in image::ImageFormat::all() {
        if fmt.can_read() && fmt.reading_enabled() {
            v.extend_from_slice(fmt.extensions_str());
        }
    }
    v.sort_unstable();
    v.dedup();
    v
}

/// True when `path` looks like one of our own previous outputs.
pub fn is_paddleknife_output(path: &Path) -> bool {
    path.file_stem()
        .and_then(|s| s.to_str())
        .is_some_and(|s| s.ends_with(OUTPUT_STEM_SUFFIX))
}

/// Resolve `root` into the ordered list of files to OCR.
///
/// * `root` is a file → a one-element list (errors if unsupported).
/// * `root` is a directory → every OCR-able file inside it, natural-sorted
///   by relative path. `recursive` descends into sub-directories.
///
/// Dot-files, non-OCR-able files, and previous PaddleKnife outputs are
/// skipped silently (logged at debug). An empty result is an error rather
/// than a silent no-op, because a run that OCRs nothing is never intended.
pub fn discover(root: &Path, recursive: bool) -> Result<Vec<InputFile>> {
    if root.is_dir() {
        let mut out = Vec::new();
        walk(root, Path::new(""), recursive, &mut out)?;
        out.sort_by(|a, b| relative_cmp(&a.relative, &b.relative));
        if out.is_empty() {
            return Err(OcrError::UnsupportedFormat(format!(
                "no OCR-able files in {}{}. Supported extensions: {}",
                root.display(),
                if recursive {
                    ""
                } else {
                    " (use --recursive to include sub-directories)"
                },
                supported_extensions().join(", "),
            )));
        }
        return Ok(out);
    }

    // Single file. Missing paths get a plain not-found rather than an
    // "unsupported format" red herring.
    if !root.exists() {
        return Err(OcrError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("input not found: {}", root.display()),
        )));
    }
    let kind = classify(root).ok_or_else(|| {
        OcrError::UnsupportedFormat(format!(
            "{} — supported extensions: {}",
            root.display(),
            supported_extensions().join(", "),
        ))
    })?;
    let relative = root
        .file_name()
        .map(PathBuf::from)
        .unwrap_or_else(|| root.to_path_buf());
    Ok(vec![InputFile {
        path: root.to_path_buf(),
        relative,
        kind,
    }])
}

/// Depth-first directory walk collecting OCR-able files. Order is not
/// maintained here — [`discover`] sorts the flat result — so this only
/// has to be exhaustive, not ordered.
fn walk(dir: &Path, rel: &Path, recursive: bool, out: &mut Vec<InputFile>) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        // Dot-files: `.DS_Store`, `.thumbnails`, editor scratch, VCS dirs.
        if name_str.starts_with('.') {
            continue;
        }
        let file_type = entry.file_type()?;
        let path = entry.path();

        if file_type.is_dir() {
            if recursive {
                walk(&path, &rel.join(&name), recursive, out)?;
            }
            continue;
        }
        // A symlink *to* a directory is not followed: symlink loops would
        // otherwise recurse forever. Symlinks to files are fine — they
        // fall through and open normally.
        if file_type.is_symlink() && path.is_dir() {
            continue;
        }
        if is_paddleknife_output(&path) {
            tracing::debug!(path = %path.display(), "skipping previous paddleknife output");
            continue;
        }
        match classify(&path) {
            Some(kind) => out.push(InputFile {
                path,
                relative: rel.join(&name),
                kind,
            }),
            None => tracing::debug!(path = %path.display(), "skipping unsupported file"),
        }
    }
    Ok(())
}

/// Natural-order comparison of two relative paths, component by
/// component. Shorter paths sort first when one is a prefix of the other.
pub fn relative_cmp(a: &Path, b: &Path) -> Ordering {
    let mut ac = a.components();
    let mut bc = b.components();
    loop {
        match (ac.next(), bc.next()) {
            (None, None) => return Ordering::Equal,
            (None, Some(_)) => return Ordering::Less,
            (Some(_), None) => return Ordering::Greater,
            (Some(x), Some(y)) => {
                let ord = natural_cmp(
                    &x.as_os_str().to_string_lossy(),
                    &y.as_os_str().to_string_lossy(),
                );
                if ord != Ordering::Equal {
                    return ord;
                }
            }
        }
    }
}

/// "Natural" string ordering: runs of ASCII digits compare as numbers, so
/// `page2` < `page10`. Everything else compares case-insensitively, with
/// the exact string as a final tiebreak so the result is a total order
/// (required for `sort_by` to be well-defined).
///
/// Equal numeric values order by digit-run length, i.e. `7` before `007`.
pub fn natural_cmp(a: &str, b: &str) -> Ordering {
    let av: Vec<char> = a.chars().collect();
    let bv: Vec<char> = b.chars().collect();
    let (mut i, mut j) = (0usize, 0usize);

    while i < av.len() && j < bv.len() {
        if av[i].is_ascii_digit() && bv[j].is_ascii_digit() {
            let (si, sj) = (i, j);
            while i < av.len() && av[i].is_ascii_digit() {
                i += 1;
            }
            while j < bv.len() && bv[j].is_ascii_digit() {
                j += 1;
            }
            let da: String = av[si..i].iter().collect();
            let db: String = bv[sj..j].iter().collect();
            // Strip leading zeros, then longer = larger, then lexicographic
            // — which on equal-length digit strings *is* numeric order, and
            // never overflows on absurdly long runs.
            let ta = da.trim_start_matches('0');
            let tb = db.trim_start_matches('0');
            let ord = ta.len().cmp(&tb.len()).then_with(|| ta.cmp(tb));
            if ord != Ordering::Equal {
                return ord;
            }
            // Same value: fewer leading zeros first.
            let ord = da.len().cmp(&db.len());
            if ord != Ordering::Equal {
                return ord;
            }
        } else {
            let la = av[i].to_lowercase().next().unwrap_or(av[i]);
            let lb = bv[j].to_lowercase().next().unwrap_or(bv[j]);
            let ord = la.cmp(&lb);
            if ord != Ordering::Equal {
                return ord;
            }
            i += 1;
            j += 1;
        }
    }

    // One side ran out: the shorter remainder sorts first. Exact-string
    // compare breaks case-only ties ("A.png" vs "a.png") deterministically.
    (av.len() - i).cmp(&(bv.len() - j)).then_with(|| a.cmp(b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn classify_matches_pdf_and_images_case_insensitively() {
        assert_eq!(classify(Path::new("a.pdf")), Some(InputKind::Pdf));
        assert_eq!(classify(Path::new("a.PDF")), Some(InputKind::Pdf));
        assert_eq!(classify(Path::new("a.png")), Some(InputKind::Image));
        assert_eq!(classify(Path::new("a.JPG")), Some(InputKind::Image));
        assert_eq!(classify(Path::new("a.tiff")), Some(InputKind::Image));
        assert_eq!(classify(Path::new("a.webp")), Some(InputKind::Image));
    }

    #[test]
    fn classify_rejects_non_image_and_extensionless() {
        assert_eq!(classify(Path::new("notes.txt")), None);
        assert_eq!(classify(Path::new("book.epub")), None);
        assert_eq!(classify(Path::new("README")), None);
        // Decoder-less format: `image` knows the extension but can't read it.
        assert_eq!(classify(Path::new("tex.dds")), None);
    }

    #[test]
    fn supported_extensions_lists_pdf_and_png() {
        let v = supported_extensions();
        assert!(v.contains(&"pdf"), "{v:?}");
        assert!(v.contains(&"png"), "{v:?}");
        assert!(v.contains(&"jpg"), "{v:?}");
        // Sorted + de-duplicated.
        let mut sorted = v.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(v, sorted);
    }

    #[test]
    fn natural_cmp_orders_numbers_numerically() {
        assert_eq!(natural_cmp("page2.png", "page10.png"), Ordering::Less);
        assert_eq!(natural_cmp("page10.png", "page9.png"), Ordering::Greater);
        assert_eq!(natural_cmp("p1", "p1"), Ordering::Equal);
    }

    #[test]
    fn natural_cmp_handles_leading_zeros() {
        // Same value → fewer leading zeros first, but never Equal, so the
        // ordering stays total.
        assert_eq!(natural_cmp("p7.png", "p007.png"), Ordering::Less);
        assert_eq!(natural_cmp("p007.png", "p8.png"), Ordering::Less);
    }

    #[test]
    fn natural_cmp_is_case_insensitive_with_exact_tiebreak() {
        assert_eq!(natural_cmp("Alpha.png", "beta.png"), Ordering::Less);
        // Case-only difference still yields a stable, non-Equal order.
        assert_ne!(natural_cmp("A.png", "a.png"), Ordering::Equal);
    }

    #[test]
    fn natural_cmp_survives_huge_digit_runs() {
        // 40-digit numbers would overflow u64 parsing; length-first
        // comparison handles them.
        let a = format!("scan{}.png", "9".repeat(40));
        let b = format!("scan{}.png", "1".repeat(41));
        assert_eq!(natural_cmp(&a, &b), Ordering::Less);
    }

    #[test]
    #[allow(clippy::unwrap_used)]
    fn discover_sorts_mixed_formats_naturally() {
        let dir = tempfile::tempdir().unwrap();
        for name in ["page10.png", "page2.jpg", "scan.pdf", "page1.tiff"] {
            fs::write(dir.path().join(name), b"x").unwrap();
        }
        // Not OCR-able — must not appear.
        fs::write(dir.path().join("notes.txt"), b"x").unwrap();

        let found = discover(dir.path(), false).unwrap();
        let names: Vec<String> = found
            .iter()
            .map(|f| f.relative.display().to_string())
            .collect();
        assert_eq!(names, ["page1.tiff", "page2.jpg", "page10.png", "scan.pdf"]);
        assert_eq!(found[0].kind, InputKind::Image);
        assert_eq!(found[3].kind, InputKind::Pdf);
    }

    #[test]
    #[allow(clippy::unwrap_used)]
    fn discover_skips_previous_paddleknife_output() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("book.pdf"), b"x").unwrap();
        fs::write(dir.path().join("book_paddleknife_OCR.pdf"), b"x").unwrap();

        let found = discover(dir.path(), false).unwrap();
        assert_eq!(found.len(), 1, "{found:?}");
        assert_eq!(found[0].relative, Path::new("book.pdf"));
    }

    #[test]
    #[allow(clippy::unwrap_used)]
    fn discover_non_recursive_ignores_subdirs() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("top.png"), b"x").unwrap();
        fs::create_dir(dir.path().join("sub")).unwrap();
        fs::write(dir.path().join("sub").join("deep.png"), b"x").unwrap();

        let flat = discover(dir.path(), false).unwrap();
        assert_eq!(flat.len(), 1);

        let deep = discover(dir.path(), true).unwrap();
        let names: Vec<String> = deep
            .iter()
            .map(|f| f.relative.display().to_string())
            .collect();
        assert_eq!(deep.len(), 2, "{names:?}");
        // Relative paths keep the sub-directory so outputs can mirror it.
        assert!(deep
            .iter()
            .any(|f| f.relative == Path::new("sub").join("deep.png")));
    }

    #[test]
    #[allow(clippy::unwrap_used)]
    fn discover_empty_dir_is_an_error_that_names_the_formats() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("readme.md"), b"x").unwrap();
        let err = discover(dir.path(), false).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("no OCR-able files"), "{msg}");
        assert!(msg.contains("pdf"), "{msg}");
        assert!(msg.contains("--recursive"), "{msg}");
    }

    #[test]
    #[allow(clippy::unwrap_used)]
    fn discover_single_file_yields_one_entry() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("one.png");
        fs::write(&f, b"x").unwrap();
        let found = discover(&f, false).unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].relative, Path::new("one.png"));
        assert_eq!(found[0].kind, InputKind::Image);
    }

    #[test]
    #[allow(clippy::unwrap_used)]
    fn discover_missing_path_reports_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let err = discover(&dir.path().join("nope.png"), false).unwrap_err();
        assert!(err.to_string().contains("not found"), "{err}");
    }

    #[test]
    #[allow(clippy::unwrap_used)]
    fn discover_unsupported_single_file_lists_supported_extensions() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("notes.txt");
        fs::write(&f, b"x").unwrap();
        let err = discover(&f, false).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("unsupported input format"), "{msg}");
        assert!(msg.contains("png"), "{msg}");
    }

    #[test]
    fn relative_cmp_puts_shallow_before_deep_on_shared_prefix() {
        assert_eq!(
            relative_cmp(Path::new("a.png"), &Path::new("a.png").join("x")),
            Ordering::Less
        );
        assert_eq!(
            relative_cmp(
                &Path::new("sub").join("2.png"),
                &Path::new("sub").join("10.png")
            ),
            Ordering::Less
        );
    }
}
