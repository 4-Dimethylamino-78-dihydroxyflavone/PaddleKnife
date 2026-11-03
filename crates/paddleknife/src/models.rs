//! Model acquisition: the `models` subcommand + the auto-fetch helpers
//! the OCR path calls when a selected recognizer or orientation
//! classifier isn't on disk.

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use paddleknife_core::{
    download::{self, Fetch},
    model::{Classifier, Quality, RecognizerScript},
};
use std::path::Path;

use crate::cli::{ModelsAction, ModelsArgs};

/// Handle `paddleknife models …`.
pub fn run(args: ModelsArgs, models_dir: &Path) -> Result<()> {
    match args.action {
        ModelsAction::List => list(models_dir),
        ModelsAction::Pull(pull) => {
            if !pull.all && pull.script.is_none() && !pull.orientation {
                anyhow::bail!(
                    "specify a script, --orientation or --all (e.g. `models pull korean`)"
                );
            }
            if pull.all {
                for script in RecognizerScript::all() {
                    ensure_script(models_dir, script, pull.quality)?;
                }
            } else if let Some(script) = pull.script {
                ensure_script(models_dir, script, pull.quality)?;
            }
            if pull.all || pull.orientation {
                ensure_classifiers(models_dir, Classifier::all())?;
            }
            Ok(())
        }
    }
}

/// Print where models live, then each tier, script and classifier with
/// whether its files are present.
fn list(models_dir: &Path) -> Result<()> {
    println!("models directory: {}", models_dir.display());

    println!();
    println!(
        "{:<8} {:<26} {:<26} status",
        "tier", "detector", "Latin/CJK recognizer"
    );
    for quality in Quality::all() {
        let q = quality.descriptor();
        let status = file_status(
            models_dir,
            &[q.det_filename, q.rec_filename, q.dict_filename],
            &format!("models pull latin-cjk --quality {}", q.cli_value),
        );
        println!(
            "{:<8} {:<26} {:<26} {}",
            q.cli_value, q.det_filename, q.rec_filename, status
        );
    }

    println!();
    println!(
        "{:<14} {:<46} status",
        "script", "recognizer (at --quality medium)"
    );
    for script in RecognizerScript::all() {
        let d = script.descriptor();
        // Recognizer + dictionary only; the detector is listed above and is
        // shared across every script.
        let (rec_name, dict_name) = script.rec_and_dict_for(Quality::default());
        let status = file_status(
            models_dir,
            &[rec_name, dict_name],
            &format!("models pull {}", d.cli_aliases[0]),
        );
        println!("{:<14} {:<46} {}", d.cli_aliases[0], rec_name, status);
    }

    println!();
    println!("{:<29} {:<33} status", "classifier", "file");
    for classifier in Classifier::all() {
        let c = classifier.descriptor();
        let status = file_status(models_dir, &[c.filename], "models pull --orientation");
        println!(
            "{:<29} {:<33} {} (used by {})",
            c.display_name, c.filename, status, c.cli_flag
        );
    }
    Ok(())
}

/// `present` when every one of `files` is in `models_dir`; otherwise
/// `absent` or how many are missing, with the `pull` command that
/// fetches them.
fn file_status(models_dir: &Path, files: &[&str], pull: &str) -> String {
    let missing = files
        .iter()
        .filter(|f| !models_dir.join(f).is_file())
        .count();
    match missing {
        0 => "present".to_string(),
        m if m == files.len() => format!("absent — `{pull}`"),
        m => format!("incomplete ({m} of {} missing) — `{pull}`", files.len()),
    }
}

/// Ensure `script`'s models exist under `models_dir`, downloading any
/// that are missing with a progress bar. No-op (and silent) when the
/// recognizer is already fully present — so the OCR hot path can call
/// this unconditionally.
pub fn ensure_script(models_dir: &Path, script: RecognizerScript, quality: Quality) -> Result<()> {
    let missing = download::missing_files(models_dir, script, quality);
    if missing.is_empty() {
        return Ok(());
    }
    println!(
        "Downloading {} model(s) for {} at {} into {} …",
        missing.len(),
        script.display_name(),
        quality.display_name(),
        models_dir.display()
    );
    for fetch in &missing {
        download_one(models_dir, fetch)?;
    }
    Ok(())
}

/// Ensure the `wanted` orientation classifiers exist under `models_dir`,
/// downloading (and checksumming) any that are missing. No-op when all
/// are present or none are wanted.
pub fn ensure_classifiers(
    models_dir: &Path,
    wanted: impl IntoIterator<Item = Classifier>,
) -> Result<()> {
    let missing = download::missing_classifiers(models_dir, wanted);
    if missing.is_empty() {
        return Ok(());
    }
    println!(
        "Downloading {} orientation classifier(s) into {} …",
        missing.len(),
        models_dir.display()
    );
    for fetch in &missing {
        download_one(models_dir, fetch)?;
    }
    Ok(())
}

fn download_one(models_dir: &Path, fetch: &Fetch) -> Result<()> {
    let bar = ProgressBar::new(0);
    bar.set_style(
        ProgressStyle::with_template(
            "  {msg} [{bar:30.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, eta {eta})",
        )
        .expect("static template")
        .progress_chars("=>-"),
    );
    bar.set_message(fetch.label.clone());

    let mut on_progress = |downloaded: u64, total: Option<u64>| {
        if let Some(t) = total {
            bar.set_length(t);
        }
        bar.set_position(downloaded);
    };
    download::download_file(
        &fetch.url,
        models_dir,
        &fetch.filename,
        fetch.sha256,
        &mut on_progress,
    )
    .with_context(|| format!("downloading {}", fetch.label))?;
    bar.finish_with_message(format!("{} ✓", fetch.label));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_status_reports_present_absent_and_incomplete() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("a.onnx"), b"x").expect("stage");
        let pull = "models pull x";

        assert_eq!(file_status(dir.path(), &["a.onnx"], pull), "present");
        assert_eq!(
            file_status(dir.path(), &["b.onnx", "c.txt"], pull),
            "absent — `models pull x`"
        );
        assert_eq!(
            file_status(dir.path(), &["a.onnx", "b.onnx"], pull),
            "incomplete (1 of 2 missing) — `models pull x`"
        );
    }

    /// A directory named like a model file is not the model.
    #[test]
    fn file_status_ignores_a_directory_with_the_file_name() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir(dir.path().join("a.onnx")).expect("mkdir");
        assert!(file_status(dir.path(), &["a.onnx"], "p").starts_with("absent"));
    }
}
