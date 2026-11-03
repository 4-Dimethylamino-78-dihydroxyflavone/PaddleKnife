mod cli;
mod models;

use anyhow::{bail, Context, Result};
use clap::Parser;
use cli::{Cli, Command, FormatArg, OcrArgs};
use indicatif::{ProgressBar, ProgressStyle};
use paddleknife_core::{
    image_io,
    input::{self, InputFile, InputKind},
    layout::{band_sort, DocPage, PageMetrics},
    model::{self, Classifier, ModelPaths, Quality, RecognizerScript},
    ocr::{OcrBlock, OcrDevice, OcrTuning, PageOcr},
    output::{self, searchable_pdf::EncodedPage},
    pdf, postproc,
    resilient::{ResilienceReport, ResilientEngine},
};
use std::time::Duration;
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

fn main() -> Result<()> {
    let cli = Cli::parse();
    init_logging(matches!(&cli.command, Some(Command::Ocr(args)) if args.verbose));

    match cli.command {
        Some(Command::Ocr(args)) => run_ocr(args),
        Some(Command::Models(args)) => models::run(args, &resolve_models_dir()?),
        None => enter_no_args_mode(),
    }
}

/// Environment variable that names the models directory outright.
const MODELS_DIR_ENV: &str = "PADDLEKNIFE_MODELS_DIR";

/// Directories searched for `models/`, starting with the exe's own and
/// walking up: enough to reach the workspace root from
/// `target/<profile>/` or `target/<triple>/<profile>/` under `cargo run`.
const EXE_SEARCH_LEVELS: usize = 4;

/// How [`select_models_dir`] arrived at its answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelsDirSource {
    /// Named by [`MODELS_DIR_ENV`].
    Env,
    /// An existing `models/` holding at least one known model file.
    Found,
    /// Nothing found: `models/` next to the exe, where downloads go.
    Default,
}

/// The models directory, for `ocr` and `models` alike — one resolver, so
/// `models pull` always writes where `ocr` will look. It is not created
/// here: a download creates it when it first writes a model
/// (`download::download_file`), so a read-only command such as
/// `models list` still works on a fresh install in a directory the user
/// can't write to.
fn resolve_models_dir() -> Result<PathBuf> {
    let exe = std::env::current_exe().ok();
    let cwd = std::env::current_dir().ok();
    let (dir, source) = select_models_dir(
        std::env::var_os(MODELS_DIR_ENV).map(PathBuf::from),
        exe.as_deref(),
        cwd.as_deref(),
    );
    if source == ModelsDirSource::Default {
        tracing::info!(
            dir = %dir.display(),
            "no models directory found; models will be downloaded here as needed"
        );
    }
    tracing::debug!(dir = %dir.display(), ?source, "models directory");
    Ok(dir)
}

/// Pick the models directory. Pure apart from reading the candidate
/// directories, so the order is unit-tested:
///
/// 1. `env` (the value of [`MODELS_DIR_ENV`]) when set and non-empty;
/// 2. the first `models/` holding any known model file, looking beside the
///    exe and up [`EXE_SEARCH_LEVELS`] directories, then in `cwd`;
/// 3. otherwise `models/` next to the exe (or in `cwd` when the exe's
///    location is unknown).
fn select_models_dir(
    env: Option<PathBuf>,
    exe: Option<&Path>,
    cwd: Option<&Path>,
) -> (PathBuf, ModelsDirSource) {
    if let Some(dir) = env.filter(|d| !d.as_os_str().is_empty()) {
        return (dir, ModelsDirSource::Env);
    }
    let exe_dir = exe.and_then(Path::parent);
    let found = exe_dir
        .into_iter()
        .flat_map(|d| d.ancestors().take(EXE_SEARCH_LEVELS))
        .chain(cwd)
        .map(|d| d.join("models"))
        .find(|candidate| holds_known_model(candidate));
    if let Some(dir) = found {
        return (dir, ModelsDirSource::Found);
    }
    let default = exe_dir
        .or(cwd)
        .map_or_else(|| PathBuf::from("models"), |d| d.join("models"));
    (default, ModelsDirSource::Default)
}

/// True when `dir` holds at least one file the model registry knows —
/// any tier, script or classifier, so a directory filled by any version
/// of `models pull` (or a fresh clone's bundled Devanagari model) counts.
fn holds_known_model(dir: &Path) -> bool {
    model::known_model_filenames().any(|name| dir.join(name).is_file())
}

fn init_logging(verbose: bool) {
    // Default: our own crates at info, but ort's session-init firehose
    // (hundreds of INFO lines per model load) only from WARN up.
    // `--verbose` raises our own crates to debug. RUST_LOG overrides
    // everything, e.g. RUST_LOG=debug or RUST_LOG=ort=info.
    let default = if verbose {
        "info,paddleknife=debug,paddleknife_core=debug,ort=warn"
    } else {
        "info,ort=warn"
    };
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(default));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .init();
}

/// Default DPI for PDF rasterization. 300 matches the Python port and
/// produces decent OCR accuracy on scanned pages; 200 is a good trade-off
/// for throughput and will probably become an option in Plan 2.5.
const DEFAULT_DPI: u32 = 300;

/// A loaded recognizer plus the script it was built for.
///
/// Batch runs hold this across files so a folder of same-script
/// documents pays exactly one ORT session init; the engine is rebuilt
/// only when the script in force actually changes.
struct EngineSlot {
    script: RecognizerScript,
    engine: ResilientEngine,
}

/// A page that could not be OCR'd on any rung of the retry ladder. The
/// document is still written — this page just has no text.
struct PageFailure {
    file: PathBuf,
    /// 1-based page number within `file`.
    page: usize,
    error: String,
}

/// Everything one `ocr` invocation needs, plus the engine it reuses.
struct RunCtx<'a> {
    args: &'a OcrArgs,
    models_dir: PathBuf,
    device: OcrDevice,
    embed: output::searchable_pdf::EmbedOptions,
    slot: Option<EngineSlot>,
    /// Recovery counts of engines already retired by a script change, so
    /// the end-of-run summary covers every file, not just the last engine.
    retired_recoveries: ResilienceReport,
    /// Script auto-detected from the first file that had text, reused for
    /// the rest unless `--script-per-file` asked for per-file detection.
    detected: Option<RecognizerScript>,
    /// Pages lost across the whole run, for the final verdict.
    page_failures: Vec<PageFailure>,
}

impl RunCtx<'_> {
    /// The engine tuning this invocation asked for. One definition, so the
    /// auto-detect sample pass can never drift from the main engine — it
    /// used to run with defaults, which meant `--region-batch-size` could
    /// not rescue a document whose *first* page was the heavy one.
    fn tuning(&self) -> OcrTuning {
        OcrTuning {
            region_batch_size: self.args.region_batch_size,
            region_ref_width: self.args.region_ref_width,
            image_batch_size: self.args.image_batch_size,
            detect_line_orientation: self.args.detect_line_orientation,
            detect_page_orientation: self.args.detect_page_orientation,
        }
    }

    /// Resolve the script for `file` and make sure `self.slot` holds an
    /// engine that can read it. Returns the script in force.
    fn prepare(&mut self, file: &InputFile) -> Result<RecognizerScript> {
        // Auto-detect's first-pass engine, when it is exactly the engine
        // the run needs — saves building an identical one straight after.
        let mut ready = None;
        let script = if let Some(pinned) = self.args.script.explicit() {
            pinned
        } else if let (Some(cached), false) = (self.detected, self.args.script_per_file) {
            cached
        } else {
            let warm = self.engine_for_sampling();
            let found = resolve_script(
                file,
                &self.models_dir,
                self.device,
                self.args.quality,
                self.tuning(),
                warm,
                self.args.fail_fast,
            )?;
            if found.text_page.is_some() {
                println!("auto-detected script: {:?}", found.script);
                self.detected = Some(found.script);
            } else {
                // Not cached: the next file gets a chance to show its text.
                println!(
                    "no text on the sampled page(s) to detect a script from; using the \
                     default ({:?})",
                    found.script
                );
            }
            ready = found.engine;
            found.script
        };

        // Already holding the right engine? Nothing to do — this is the
        // common case for every file after the first.
        if matches!(&self.slot, Some(s) if s.script == script) {
            return Ok(script);
        }

        // Download-on-demand: a pinned (or auto-selected) script whose
        // weights aren't on disk is fetched now, before the engine loads.
        models::ensure_script(&self.models_dir, script, self.args.quality)
            .with_context(|| format!("fetching {} models", script.display_name()))?;
        // Drop the outgoing engine *before* building the next one so two
        // ORT sessions never hold GPU memory simultaneously.
        self.retire_engine();
        let engine = match ready {
            Some(mut engine) => {
                // Its sample reads are of pages this run reads again, and a
                // page counts once: start the run's book-keeping afresh.
                engine.reset_recovery_state();
                engine
            }
            None => {
                let paths = ModelPaths::resolve(&self.models_dir, script, self.args.quality)
                    .with_context(|| {
                        format!(
                            "resolving {:?} models in {}",
                            script,
                            self.models_dir.display()
                        )
                    })?;
                ResilientEngine::from_paths(&paths, self.device, self.tuning()).with_context(
                    || format!("initializing OCR engine on {}", self.device.label()),
                )?
            }
        };
        self.slot = Some(EngineSlot { script, engine });
        Ok(script)
    }

    /// The engine auto-detect should sample with, taken out of the slot.
    ///
    /// Sampling reads with a LatinCjk engine minus the orientation steps.
    /// The live engine is exactly that when it is LatinCjk and no
    /// orientation step is on, so it is lent to the sampler (its counts so
    /// far banked, its book-keeping reset for the sample reads) instead of
    /// a second session being built beside it. Any other live engine is
    /// retired first: two ORT sessions must never hold GPU memory at once.
    fn engine_for_sampling(&mut self) -> Option<ResilientEngine> {
        let lendable = sample_engine_is_run_engine(&self.tuning())
            && matches!(&self.slot, Some(s) if s.script == RecognizerScript::LatinCjk);
        if !lendable {
            self.retire_engine();
            return None;
        }
        let mut engine = self.slot.take()?.engine;
        self.retired_recoveries.absorb(engine.report());
        engine.reset_recovery_state();
        Some(engine)
    }

    /// Drop the current engine, keeping its recovery counts for the
    /// end-of-run summary.
    fn retire_engine(&mut self) {
        if let Some(slot) = self.slot.take() {
            self.retired_recoveries.absorb(slot.engine.report());
        }
    }

    /// What the retry ladder and orientation check did over the whole run:
    /// every retired engine plus the live one.
    fn recoveries(&self) -> ResilienceReport {
        let mut total = self.retired_recoveries.clone();
        if let Some(slot) = &self.slot {
            total.absorb(slot.engine.report());
        }
        total
    }

    /// The engine built by the last [`RunCtx::prepare`] call.
    fn engine(&mut self) -> &mut ResilientEngine {
        &mut self
            .slot
            .as_mut()
            .expect("prepare() runs before engine()")
            .engine
    }
}

fn run_ocr(args: OcrArgs) -> Result<()> {
    let device: OcrDevice = args.device.into();

    // One code path for files and directories: discovery turns either
    // into an ordered list. A directory may mix PDFs and images freely.
    let inputs = input::discover(&args.input, args.recursive)
        .with_context(|| format!("resolving input {}", args.input.display()))?;
    let input_is_dir = args.input.is_dir();

    tracing::info!(
        input = %args.input.display(),
        files = inputs.len(),
        format = ?args.format,
        script = ?args.script,
        device = %device.label(),
        "starting OCR"
    );

    if input_is_dir {
        let pdfs = inputs.iter().filter(|f| f.kind == InputKind::Pdf).count();
        println!(
            "Found {} file(s) in {}{}: {} PDF, {} image.",
            inputs.len(),
            args.input.display(),
            if args.recursive { " (recursive)" } else { "" },
            pdfs,
            inputs.len() - pdfs,
        );
    }

    let models_dir = resolve_models_dir()?;
    // Download-on-demand, as for recognizers: without its classifier an
    // orientation flag would silently do nothing. Once per run — the
    // classifiers are shared by every script.
    models::ensure_classifiers(&models_dir, requested_classifiers(&args))
        .context("fetching orientation classifiers")?;

    let mut ctx = RunCtx {
        args: &args,
        models_dir,
        device,
        // Searchable PDF needs both the DocPages AND the page images, so
        // its path keeps an encoded raster per page. The other formats
        // only need the DocPages and drop bitmaps as they go.
        embed: output::searchable_pdf::EmbedOptions {
            allow_passthrough: args.pdf_image == cli::PdfImageArg::Auto,
            jpeg_quality: args.jpeg_quality,
            embed_dpi_override: args.embed_dpi,
        },
        slot: None,
        retired_recoveries: ResilienceReport::default(),
        detected: None,
        page_failures: Vec::new(),
    };

    // A single file is inherently one document, so it takes the merged
    // path too — which keeps its behaviour byte-identical to before.
    if args.merge || !input_is_dir {
        run_merged(&mut ctx, &inputs)
    } else {
        run_per_file(&mut ctx, &inputs)
    }
}

/// The orientation classifiers the `ocr` flags ask for.
fn requested_classifiers(args: &OcrArgs) -> Vec<Classifier> {
    let mut wanted = Vec::new();
    if args.detect_page_orientation {
        wanted.push(Classifier::DocOrientation);
    }
    if args.detect_line_orientation {
        wanted.push(Classifier::TextlineOrientation);
    }
    wanted
}

/// OCR every input into **one** document and write a single output.
/// Page numbers run continuously across files, so a folder of page
/// scans reads as one book.
fn run_merged(ctx: &mut RunCtx, inputs: &[InputFile]) -> Result<()> {
    let args = ctx.args;
    let keep_rasters = args.format == FormatArg::Pdf;
    let mut doc_pages: Vec<DocPage> = Vec::new();
    let mut encoded: Vec<EncodedPage> = Vec::new();
    let mut failures: Vec<(PathBuf, String)> = Vec::new();

    for (i, file) in inputs.iter().enumerate() {
        announce(i, inputs.len(), file);
        let offset = doc_pages.len();
        match ocr_one(ctx, file, keep_rasters, offset) {
            Ok((pages, enc)) => {
                doc_pages.extend(pages);
                encoded.extend(enc);
            }
            Err(e) => record_failure(args, &mut failures, file, e)?,
        }
    }

    if doc_pages.is_empty() {
        // Reached only when every input failed outright (each error is
        // already printed above). A file with *some* unreadable pages still
        // yields its other pages, so it never lands here.
        if inputs.len() == 1 {
            bail!("nothing to write: the input could not be OCR'd — see the error above");
        }
        bail!(
            "nothing to write: all {} inputs failed — see the errors above",
            inputs.len()
        );
    }

    let output_path = merged_output_path(args);
    ensure_parent_dir(&output_path)?;
    write_output(&output_path, args.format, &doc_pages, encoded)?;
    report_document(&doc_pages, &output_path, ctx.device);
    report_recoveries(ctx);
    finish(failures, &ctx.page_failures)
}

/// OCR each input into its **own** output file, mirroring the input
/// tree under `--output` when one was given.
fn run_per_file(ctx: &mut RunCtx, inputs: &[InputFile]) -> Result<()> {
    let args = ctx.args;
    let out_root = args.output.as_deref();
    if let Some(dir) = out_root {
        if dir.is_file() {
            bail!(
                "--output {} is an existing file, but OCRing a directory writes one \
                 output per input. Pass a directory instead, or add --merge to \
                 combine every input into that single file.",
                dir.display()
            );
        }
    }

    let keep_rasters = args.format == FormatArg::Pdf;
    let mut failures: Vec<(PathBuf, String)> = Vec::new();
    let (mut ok_files, mut ok_pages) = (0usize, 0usize);

    for (i, file) in inputs.iter().enumerate() {
        announce(i, inputs.len(), file);
        match process_one_file(ctx, file, out_root, keep_rasters) {
            Ok(pages) => {
                ok_files += 1;
                ok_pages += pages;
            }
            Err(e) => record_failure(args, &mut failures, file, e)?,
        }
    }

    println!();
    println!(
        "Batch complete: {}/{} file(s) OCR'd, {} page(s) total.",
        ok_files,
        inputs.len(),
        ok_pages
    );
    report_recoveries(ctx);
    finish(failures, &ctx.page_failures)
}

/// OCR one input and write its own output. Returns the page count.
fn process_one_file(
    ctx: &mut RunCtx,
    file: &InputFile,
    out_root: Option<&Path>,
    keep_rasters: bool,
) -> Result<usize> {
    let format = ctx.args.format;
    let device = ctx.device;
    let (doc_pages, encoded) = ocr_one(ctx, file, keep_rasters, 0)?;
    let output_path = per_file_output_path(file, out_root, format);
    ensure_parent_dir(&output_path)?;
    write_output(&output_path, format, &doc_pages, encoded)?;
    report_document(&doc_pages, &output_path, device);
    Ok(doc_pages.len())
}

/// Prepare the engine for `file` and OCR it, offsetting page indices by
/// `page_offset` (non-zero only when merging into one document).
fn ocr_one(
    ctx: &mut RunCtx,
    file: &InputFile,
    keep_rasters: bool,
    page_offset: usize,
) -> Result<(Vec<DocPage>, Vec<EncodedPage>)> {
    let script = ctx.prepare(file)?;
    let format = ctx.args.format;
    let fail_fast = ctx.args.fail_fast;
    let embed = ctx.embed.clone();
    let engine = ctx.engine();
    let FileOcr {
        pages,
        encoded,
        failed_pages,
    } = ocr_file(
        file,
        engine,
        keep_rasters,
        script,
        format,
        &embed,
        page_offset,
        fail_fast,
    )?;
    ctx.page_failures
        .extend(failed_pages.into_iter().map(|(page, error)| PageFailure {
            file: file.path.clone(),
            page,
            error,
        }));
    Ok((pages, encoded))
}

/// `[3/12] scans/page3.png` before each file of a multi-file run.
fn announce(index: usize, total: usize, file: &InputFile) {
    if total > 1 {
        println!();
        println!("[{}/{}] {}", index + 1, total, file.relative.display());
    }
}

/// Either re-raise (with `--fail-fast`) or note the failure and carry on.
fn record_failure(
    args: &OcrArgs,
    failures: &mut Vec<(PathBuf, String)>,
    file: &InputFile,
    err: anyhow::Error,
) -> Result<()> {
    if args.fail_fast {
        return Err(err).with_context(|| format!("processing {}", file.path.display()));
    }
    eprintln!("  ! {}: {err:#}", file.relative.display());
    failures.push((file.path.clone(), format!("{err:#}")));
    Ok(())
}

/// Final verdict for a batch: list every failure and exit non-zero if
/// there were any. Successful outputs are already on disk either way.
///
/// Two kinds of failure, reported separately because they mean different
/// things: a *file* that produced no output at all, and a *page* that was
/// left without text inside an output that was otherwise written.
fn finish(failures: Vec<(PathBuf, String)>, page_failures: &[PageFailure]) -> Result<()> {
    if failures.is_empty() && page_failures.is_empty() {
        return Ok(());
    }
    if !page_failures.is_empty() {
        eprintln!();
        eprintln!(
            "{} page(s) could not be OCR'd and were written without text:",
            page_failures.len()
        );
        for f in page_failures.iter().take(20) {
            eprintln!("  {} page {}: {}", f.file.display(), f.page, f.error);
        }
        if page_failures.len() > 20 {
            eprintln!("  … and {} more", page_failures.len() - 20);
        }
    }
    if !failures.is_empty() {
        eprintln!();
        eprintln!("{} file(s) failed:", failures.len());
        for (path, err) in &failures {
            eprintln!("  {}: {err}", path.display());
        }
    }
    match (failures.len(), page_failures.len()) {
        (0, pages) => {
            bail!("output written, but {pages} page(s) have no text — see the list above")
        }
        (files, 0) => bail!("{files} input file(s) failed — see the list above"),
        (files, pages) => bail!(
            "{files} input file(s) failed and {pages} page(s) have no text — see the lists above"
        ),
    }
}

/// A line or two on what the retry ladder and orientation check had to do,
/// so a run that quietly degraded is visible — and actionable — afterwards.
fn report_recoveries(ctx: &RunCtx) {
    let r = ctx.recoveries();
    if r.is_quiet() {
        return;
    }
    if r.retried_pages > 0 {
        let cpu_note = if r.cpu_pages > 0 {
            format!(" ({} fell back to CPU)", r.cpu_pages)
        } else {
            String::new()
        };
        println!(
            "{} page(s) {} and were retried with a smaller batch{cpu_note}.",
            r.retried_pages,
            r.retry_cause()
        );
    }
    if let Some(batch) = r.sticky_batch {
        println!(
            "Tip: this run settled on --region-batch-size {batch}; passing it up front skips \
             the failed first attempts."
        );
    }
    if r.reoriented_pages > 0 {
        println!(
            "{} page(s) were read the other way up from the orientation classifier's verdict \
             (recognition confidence decided).",
            r.reoriented_pages
        );
    }
}

/// Create the output's parent directory when `--output` points into a
/// tree that doesn't exist yet (or that we're mirroring into).
fn ensure_parent_dir(output_path: &Path) -> Result<()> {
    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating output directory {}", parent.display()))?;
        }
    }
    Ok(())
}

/// Per-document stats: page/line counts, blank pages, weak pages.
fn report_document(doc_pages: &[DocPage], output_path: &Path, device: OcrDevice) {
    let total_lines: usize = doc_pages.iter().map(|p| p.lines.len()).sum();
    let total_blocks: usize = doc_pages.iter().map(|p| p.metrics.total_blocks).sum();
    let avg_conf = if total_blocks > 0 {
        let sum: f32 = doc_pages
            .iter()
            .map(|p| p.metrics.avg_confidence * p.metrics.total_blocks as f32)
            .sum();
        sum / total_blocks as f32
    } else {
        0.0
    };
    println!(
        "Wrote {} pages ({} lines, avg confidence {:.2}) to {} [device: {}]",
        doc_pages.len(),
        total_lines,
        avg_conf,
        output_path.display(),
        device.label()
    );

    // Pages with no text at all (blank/illustration pages) are normal —
    // list them separately instead of flagging them as avg=0.00.
    let empty: Vec<usize> = doc_pages
        .iter()
        .filter(|p| p.metrics.total_blocks == 0)
        .map(|p| p.index + 1)
        .collect();
    if !empty.is_empty() {
        println!(
            "{} page(s) with no detected text (blank or image-only): {:?}",
            empty.len(),
            empty
        );
    }

    // Per-page breakdown for any pages that look weak (< 0.7 average or
    // > 0 low-conf lines). Helps the user spot scan-quality outliers.
    let weak: Vec<&DocPage> = doc_pages
        .iter()
        .filter(|p| {
            p.metrics.total_blocks > 0
                && (p.metrics.avg_confidence < 0.7 || p.metrics.low_conf_lines > 0)
        })
        .collect();
    if !weak.is_empty() {
        println!(
            "{} page(s) flagged for review (low confidence):",
            weak.len()
        );
        for p in weak.iter().take(20) {
            println!(
                "  page {:>3}: avg={:.2} min={:.2} low_conf_lines={}",
                p.index + 1,
                p.metrics.avg_confidence,
                p.metrics.min_confidence,
                p.metrics.low_conf_lines
            );
        }
        if weak.len() > 20 {
            println!("  … and {} more", weak.len() - 20);
        }
    }
}

/// Confidence below which the broad LatinCjk recognizer is considered to
/// have *failed* — the page is a script it can't emit, so we run the
/// specialist race.
const LOW_CONF: f32 = 0.7;
/// Fraction of Latin letters that must be accented/extended for a
/// high-confidence Latin page to prefer the diacritic-complete `Latin`
/// model over the broad CJK model (whose dict omits ď ť ů ž …).
const EXTENDED_LATIN_PREFER_LATIN: f32 = 0.06;
/// A race candidate this confident ends the race early (no point
/// downloading + trying the rest). Set above the ceiling that a
/// *wrong-script* recognizer reaches on confident-garbage (~0.86 for
/// Devanagari on Cyrillic) but below correct-script scores (0.97–1.00),
/// so only the right model short-circuits the race. The winner is still
/// the max over everything tried, so a missed early-exit never picks
/// the wrong script — it only costs extra passes.
const RACE_EARLY_EXIT: f32 = 0.95;
/// Mid-word-capital ratio above which a high-confidence "Latin" result
/// is treated as transliterated Cyrillic and sent to the race.
const MIDWORD_CAP_SUSPECT: f32 = 0.08;
/// Share of the detector's text lines the broad pass lost (see
/// [`lost_text_ratio`]) above which its confident read is not trusted and
/// the race runs. Its confidence only covers the lines it kept, so on a
/// cover with big Latin titles over Cyrillic copy it reads ~0.91 while
/// having discarded every Cyrillic line.
const LOST_TEXT_RACE: f32 = 0.2;
/// How many times the broad read's [`ReadQuality`] score a specialist must
/// reach to beat a broad read that lost lines ([`RaceGate::Score`]). Above
/// 1 so a near-tie keeps LatinCjk: a specialist whose dictionary also
/// holds Latin (Eslav, Cyrillic) reads a Latin page about as well, and
/// winning on noise would put Cyrillic homoglyphs into an English text
/// layer. Measured: the Passat cover's winner reached 2.2×, a sparse Latin
/// scan's best specialist 0.93×.
const RACE_MIN_GAIN: f32 = 1.25;

/// What a specialist must beat for the race to drop LatinCjk.
#[derive(Debug, Clone, Copy, PartialEq)]
enum RaceGate {
    /// The broad read's mean confidence: it read the page, but poorly.
    Confidence(f32),
    /// A [`ReadQuality`] score — [`RACE_MIN_GAIN`] × the broad read's: it
    /// read confidently but lost lines, so the confidence of what it kept
    /// says nothing, and a specialist must recover clearly more text.
    Score(f32),
    /// Nothing: the broad read is known wrong (confident transliterated
    /// Cyrillic), so any specialist that reads something wins.
    Open,
}

impl RaceGate {
    fn admits(self, conf: f32, score: f32) -> bool {
        match self {
            RaceGate::Confidence(lc_conf) => conf > lc_conf,
            RaceGate::Score(min_score) => score > 0.0 && score >= min_score,
            RaceGate::Open => score > 0.0,
        }
    }
}

/// Layer-1 decision after the broad first pass — pure, unit-tested.
#[derive(Debug, PartialEq)]
enum FirstPass {
    /// Confident enough to commit to this recognizer with no further passes.
    Keep(RecognizerScript),
    /// Run the specialist race, admitting only specialists past the gate.
    Race(RaceGate),
}

/// Decide from the broad LatinCjk pass — its `text`, mean block
/// confidence, [`lost_text_ratio`] and [`ReadQuality`] score — whether to
/// keep a recognizer or fall through to the race. See module spec.
fn classify_first_pass(text: &str, avg_conf: f32, lost_ratio: f32, score: f32) -> FirstPass {
    use paddleknife_core::script::{detect_script, Script};
    if avg_conf < LOW_CONF {
        return FirstPass::Race(RaceGate::Confidence(avg_conf));
    }
    if lost_ratio > LOST_TEXT_RACE {
        return FirstPass::Race(RaceGate::Score(score * RACE_MIN_GAIN));
    }
    // High confidence: the broad model read the page. Refine only where
    // its dictionary is known to be incomplete.
    match detect_script(text) {
        // Extremely unlikely at high conf from the broad model (its dict
        // can't emit these), but if it happens, trust the detection.
        Script::Cyrillic => FirstPass::Keep(RecognizerScript::Eslav),
        Script::Devanagari => FirstPass::Keep(RecognizerScript::Devanagari),
        Script::Korean => FirstPass::Keep(RecognizerScript::Korean),
        // Latin / CJK / Japanese / English / Greek.
        _ => {
            // Cyrillic read *as Latin lookalikes* (П→N, и→n, etc.) scores
            // high but is gibberish. Its tell is many mid-word capitals
            // ("NpnBeT", "MoBa") — which real Latin text never has — so
            // a high mid-word-capital ratio means "not really Latin":
            // race to find the true script (Cyrillic/Eslav).
            if midword_capital_ratio(text) > MIDWORD_CAP_SUSPECT {
                // Confident garbage — don't let it set the bar.
                return FirstPass::Race(RaceGate::Open);
            }
            if extended_latin_ratio(text) > EXTENDED_LATIN_PREFER_LATIN {
                FirstPass::Keep(RecognizerScript::Latin)
            } else {
                FirstPass::Keep(RecognizerScript::LatinCjk)
            }
        }
    }
}

/// Fraction of letters that are an uppercase letter immediately
/// following a lowercase letter (a mid-word capital). Near zero for
/// real Latin-script text; high for Cyrillic transliterated into Latin
/// homoglyphs by the broad recognizer.
fn midword_capital_ratio(text: &str) -> f32 {
    let mut letters = 0usize;
    let mut midcaps = 0usize;
    let mut prev_lower = false;
    for c in text.chars() {
        if c.is_alphabetic() {
            letters += 1;
            if c.is_uppercase() && prev_lower {
                midcaps += 1;
            }
            prev_lower = c.is_lowercase();
        } else {
            prev_lower = false;
        }
    }
    if letters == 0 {
        0.0
    } else {
        midcaps as f32 / letters as f32
    }
}

/// Fraction of Latin-script letters that are accented/extended
/// (U+00C0–U+024F). ~0 for plain English, high for Czech/Polish/etc.
/// CJK and other scripts are excluded from the denominator so a
/// CJK-dominant page doesn't dilute the signal.
fn extended_latin_ratio(text: &str) -> f32 {
    let mut latin = 0usize;
    let mut ext = 0usize;
    for c in text.chars() {
        let cp = c as u32;
        let is_ext_latin = (0x00C0..=0x024F).contains(&cp) && c.is_alphabetic();
        if c.is_ascii_alphabetic() || is_ext_latin {
            latin += 1;
            if is_ext_latin {
                ext += 1;
            }
        }
    }
    if latin == 0 {
        0.0
    } else {
        ext as f32 / latin as f32
    }
}

/// Share of the text lines the detector found that a read lost: lines
/// recognised as nothing (`unread_regions`) or as punctuation alone, the
/// residue of a line whose letters the dictionary could not emit
/// ("ул. Ленинская, д. 28" → ", . 28" keeps its digits and counts as read).
/// 0 for a page with no detected lines.
fn lost_text_ratio(blocks: &[OcrBlock], unread_regions: usize) -> f32 {
    let residue = blocks
        .iter()
        .filter(|b| !b.text.chars().any(char::is_alphanumeric))
        .count();
    let detected = blocks.len() + unread_regions;
    if detected == 0 {
        0.0
    } else {
        (residue + unread_regions) as f32 / detected as f32
    }
}

/// One specialist's read of the sample page.
#[derive(Debug, Clone, Copy)]
struct RaceEntry {
    script: RecognizerScript,
    /// Mean block confidence: what tells specialists apart. A wrong-script
    /// specialist still emits a lookalike per glyph — Devanagari read the
    /// Passat pages with as many characters as Eslav — but tops out around
    /// 0.86, under the right script's 0.92–1.00.
    conf: f32,
    /// [`ReadQuality`] score: what the gate may judge against the broad read.
    score: f32,
}

/// Pick the race winner: the most confident candidate the gate admits,
/// else fall back to LatinCjk. Pure, unit-tested.
fn race_winner(candidates: &[RaceEntry], gate: RaceGate) -> RecognizerScript {
    candidates
        .iter()
        .filter(|c| gate.admits(c.conf, c.score))
        .max_by(|a, b| a.conf.total_cmp(&b.conf))
        .map(|c| c.script)
        .unwrap_or(RecognizerScript::LatinCjk)
}

/// Specialist recognizers tried in the low-confidence race, in order.
/// Devanagari is bundled (free); the rest download on demand. Ordered so
/// the most common non-Latin scripts are tried before the rarer ones.
const RACE_ORDER: &[RecognizerScript] = &[
    RecognizerScript::Devanagari,
    RecognizerScript::Eslav,
    RecognizerScript::Korean,
    RecognizerScript::Cyrillic,
    RecognizerScript::Latin,
];

/// Pages auto-detect may sample, from the front. A blank cover or title
/// verso must not decide the script, but a document with no text in its
/// first few pages isn't worth rasterizing further for a guess.
const SAMPLE_MAX_PAGES: usize = 3;

/// What auto-detect settled on for one file.
struct Detection {
    script: RecognizerScript,
    /// 0-based page the verdict was read from; `None` when every sampled
    /// page was blank and `script` is only the default.
    text_page: Option<usize>,
    /// The first-pass engine, handed on when it is exactly the engine the
    /// run needs (the default script, and no orientation step it lacks).
    engine: Option<ResilientEngine>,
}

/// Resolve `--script auto` by sampling (Plan 6: layered).
///
/// 1. Broad LatinCjk pass over the first page that has any text, trying
///    up to [`SAMPLE_MAX_PAGES`]. When none has, keep the default: there is
///    nothing to judge a script by, so no specialist is downloaded.
/// 2. High confidence and few lost lines → keep LatinCjk (or upgrade to
///    the Latin model for diacritic-heavy pages). This is the fast path.
/// 3. Low confidence, or many detected lines read as nothing → the broad
///    model can't read this script. Race the specialist recognizers over
///    the same page (downloading each on demand), early-exiting on a
///    confident win, and take the most confident one past the
///    [`RaceGate`].
///
/// The caller handles pinned `--script` values, which skip sampling.
/// `warm` is a live LatinCjk engine to sample with instead of building one
/// (see [`RunCtx::engine_for_sampling`]). Without `fail_fast`, a sampled
/// page that can't be OCR'd reads as blank ([`sample_blocks`]).
fn resolve_script(
    file: &InputFile,
    models_dir: &Path,
    device: OcrDevice,
    quality: Quality,
    tuning: OcrTuning,
    warm: Option<ResilientEngine>,
    fail_fast: bool,
) -> Result<Detection> {
    let mut engine = match warm {
        Some(engine) => engine,
        None => {
            // The first pass needs the broad model, which a fresh install lacks.
            models::ensure_script(models_dir, RecognizerScript::LatinCjk, quality)
                .context("auto-detect: fetching the Latin/CJK models")?;
            sample_engine(
                models_dir,
                RecognizerScript::LatinCjk,
                device,
                quality,
                tuning,
            )?
        }
    };
    let reusable = sample_engine_is_run_engine(&tuning);

    // Pass 1: broad LatinCjk, over the first page with any text.
    let pages = sample_page_count(file)?;
    let sample = first_page_with_text(pages, |index| {
        let img = sample_page_image(file, index)?;
        let page = sample_blocks(engine.ocr_page(&img), index, fail_fast)?;
        Ok(((img, page.unread_regions), page.blocks))
    })?;
    let Some((text_page, (img, unread_regions), blocks)) = sample else {
        tracing::info!(
            pages,
            "auto-detect: no text on the sampled pages; keeping the default"
        );
        return Ok(Detection {
            script: RecognizerScript::LatinCjk,
            text_page: None,
            engine: reusable.then_some(engine),
        });
    };
    let lc_conf = avg_confidence(&blocks);
    let lc_text = joined_text(&blocks);
    let lost = lost_text_ratio(&blocks, unread_regions);
    let broad = PageOcr {
        blocks,
        unread_regions,
        ..PageOcr::default()
    };
    let lc_score = broad.read_quality().score;

    let gate = match classify_first_pass(&lc_text, lc_conf, lost, lc_score) {
        FirstPass::Keep(script) => {
            tracing::info!(
                conf = lc_conf,
                lost,
                page = text_page + 1,
                ?script,
                "auto-detect: first pass decisive"
            );
            return Ok(Detection {
                script,
                text_page: Some(text_page),
                engine: (reusable && script == RecognizerScript::LatinCjk).then_some(engine),
            });
        }
        FirstPass::Race(gate) => gate,
    };
    // Free the first-pass session before any specialist binds its own.
    drop(engine);

    // Pass 2: specialist race.
    println!(
        "Base recognizer unsure (score {lc_conf:.2}, {:.0}% of lines lost) — trying \
         specialist models (downloading as needed)…",
        lost * 100.0
    );
    let mut scored: Vec<RaceEntry> = Vec::new();
    for &script in RACE_ORDER {
        if let Err(e) = models::ensure_script(models_dir, script, quality) {
            tracing::warn!(?script, error = %e, "auto-detect: could not fetch specialist; skipping");
            continue;
        }
        match sample_read(&img, models_dir, script, device, quality, tuning) {
            Ok(page) => {
                let conf = page.mean_confidence().unwrap_or(0.0);
                let score = page.read_quality().score;
                tracing::info!(?script, conf, score, "auto-detect: specialist score");
                scored.push(RaceEntry {
                    script,
                    conf,
                    score,
                });
                if conf >= RACE_EARLY_EXIT && gate.admits(conf, score) {
                    break; // confident win — stop downloading/trying more
                }
            }
            Err(e) => {
                tracing::warn!(?script, error = %e, "auto-detect: specialist pass failed; skipping")
            }
        }
    }

    let winner = race_winner(&scored, gate);
    tracing::info!(?winner, ?gate, "auto-detect: race resolved");
    Ok(Detection {
        script: winner,
        text_page: Some(text_page),
        engine: None,
    })
}

/// Read sample pages `0..pages` in order until one yields text, and return
/// that page's index, image and blocks — or `None` when every page came
/// back empty. `read` does the loading and OCR, so the stopping rule is
/// testable without a model.
fn first_page_with_text<P>(
    pages: usize,
    mut read: impl FnMut(usize) -> Result<(P, Vec<OcrBlock>)>,
) -> Result<Option<(usize, P, Vec<OcrBlock>)>> {
    for index in 0..pages {
        let (page, blocks) = read(index)?;
        if has_text(&blocks) {
            return Ok(Some((index, page, blocks)));
        }
    }
    Ok(None)
}

/// A sampled page's OCR result as auto-detect uses it. Without
/// `fail_fast`, a page that can't be OCR'd even after the retries reads as
/// blank, so sampling moves on to the next page: the run itself writes such
/// a page without text and carries on, and sampling must not fail the
/// whole file over it. With `fail_fast` the error stops the run, as it
/// would on that page in the main pass.
fn sample_blocks<T: Default>(
    read: std::result::Result<T, paddleknife_core::error::OcrError>,
    index: usize,
    fail_fast: bool,
) -> Result<T> {
    match read {
        Ok(read) => Ok(read),
        Err(e) if !fail_fast => {
            tracing::warn!(
                page = index + 1,
                error = %e,
                "auto-detect: could not OCR a sampled page; treating it as blank"
            );
            Ok(T::default())
        }
        Err(e) => {
            Err(e).with_context(|| format!("auto-detect: LatinCjk pass over page {}", index + 1))
        }
    }
}

/// True when the auto-detect sample engine is the engine the run would
/// build for LatinCjk: it is the run's engine minus its orientation steps,
/// so with none requested the two are identical and one can serve as the
/// other.
fn sample_engine_is_run_engine(tuning: &OcrTuning) -> bool {
    !tuning.detect_line_orientation && !tuning.detect_page_orientation
}

/// True when any block recognised a non-whitespace character.
fn has_text(blocks: &[OcrBlock]) -> bool {
    blocks.iter().any(|b| !b.text.trim().is_empty())
}

/// Build an engine for sampling `script`. Same batch limits and retry
/// ladder as the real run — a sample page is a real page, and can be just
/// as heavy — but the orientation steps stay off: the sample only has to
/// tell scripts apart.
fn sample_engine(
    models_dir: &Path,
    script: RecognizerScript,
    device: OcrDevice,
    quality: Quality,
    tuning: OcrTuning,
) -> Result<ResilientEngine> {
    let paths = ModelPaths::resolve(models_dir, script, quality)
        .with_context(|| format!("loading {script:?} sample model"))?;
    let tuning = OcrTuning {
        detect_line_orientation: false,
        detect_page_orientation: false,
        ..tuning
    };
    ResilientEngine::from_paths(&paths, device, tuning)
        .with_context(|| format!("initializing {script:?} sample engine"))
}

/// Run one specialist recognizer over the sample image and return its
/// mean block confidence. The engine is dropped before returning so the
/// next candidate can bind its own ORT session.
fn sample_read(
    img: &image::DynamicImage,
    models_dir: &Path,
    script: RecognizerScript,
    device: OcrDevice,
    quality: Quality,
    tuning: OcrTuning,
) -> Result<PageOcr> {
    let mut engine = sample_engine(models_dir, script, device, quality, tuning)?;
    engine
        .ocr_page(img)
        .with_context(|| format!("{script:?} sample pass"))
}

fn avg_confidence(blocks: &[OcrBlock]) -> f32 {
    if blocks.is_empty() {
        return 0.0;
    }
    blocks.iter().map(|b| b.confidence).sum::<f32>() / blocks.len() as f32
}

/// Every block's text, space-separated, for the script heuristics.
fn joined_text(blocks: &[OcrBlock]) -> String {
    blocks
        .iter()
        .map(|b| b.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

/// How many pages of `file` auto-detect may sample: an image is one page,
/// a PDF up to [`SAMPLE_MAX_PAGES`].
fn sample_page_count(file: &InputFile) -> Result<usize> {
    let pages = match file.kind {
        InputKind::Pdf => pdf::page_count(&file.path).context("auto-detect: counting pages")?,
        InputKind::Image => 1,
    };
    Ok(pages.min(SAMPLE_MAX_PAGES))
}

/// Load page `index` (0-based) of `file` for the auto-detect sample pass.
/// Uses the `pdf::rasterize_page_with` shortcut so we don't pay for
/// full-document rasterization, and skips the native-DPI metadata fetches
/// only the searchable-PDF writer reads.
fn sample_page_image(file: &InputFile, index: usize) -> Result<image::DynamicImage> {
    match file.kind {
        InputKind::Pdf => Ok(pdf::rasterize_page_with(
            &file.path,
            index,
            DEFAULT_DPI,
            pdf::RasterOptions { native_dpi: false },
        )
        .with_context(|| format!("auto-detect: rasterizing page {}", index + 1))?
        .image),
        InputKind::Image => Ok(image_io::load_image(&file.path)?.image),
    }
}

/// OCR every page of one input file. If `keep_rasters` is true, also
/// return each page pre-encoded for embedding (needed by the
/// searchable-PDF writer). Otherwise each raster is dropped after its
/// page is OCR'd. When `script` is `Devanagari`, blocks pass through
/// `postproc::nepali::postprocess_page` before band-sort.
///
/// `page_offset` is added to every emitted `DocPage.index`, so merging a
/// folder produces one continuously-numbered document. Note it is *not*
/// applied to the passthrough probe or the progress bar — both are
/// inherently file-local.
#[allow(clippy::too_many_arguments)]
fn ocr_file(
    file: &InputFile,
    engine: &mut ResilientEngine,
    keep_rasters: bool,
    script: RecognizerScript,
    format: FormatArg,
    embed: &output::searchable_pdf::EmbedOptions,
    page_offset: usize,
    fail_fast: bool,
) -> Result<FileOcr> {
    let input = file.path.as_path();

    match file.kind {
        InputKind::Pdf => {
            let total = pdf::page_count(input).context("counting pdf pages")?;
            let bar = page_progress_bar(total);
            let mut doc_pages = Vec::new();
            // Searchable-PDF output keeps each page as its embeddable
            // JPEG (~0.5 MB) instead of the raw bitmap (~25 MB at
            // 300 dpi) so 1000-page books don't exhaust RAM.
            let mut encoded: Vec<EncodedPage> = Vec::new();
            let mut failed_pages: Vec<(usize, String)> = Vec::new();
            // Passthrough probe: read-only view of the source PDF. Any
            // open failure (encrypted, malformed) downgrades gracefully
            // to the re-encode path. The page-count gate guards against
            // lopdf/pdfium disagreeing on page order (typeless page-tree
            // nodes) — a mismatch would pair the wrong image (and the
            // wrong DPI) with a raster, so the probe is dropped for the
            // whole document.
            //
            // Opened even under `--pdf-image reencode`: the probe also
            // measures each page's effective DPI, which caps the re-encode
            // at the scan's true resolution. `encode_page_opts` ignores
            // `probe.passthrough` itself when passthrough is disallowed —
            // gating the open on it left reencode mode with no DPI hint,
            // embedding every page at the full OCR DPI.
            let source = if keep_rasters {
                match paddleknife_core::source_pdf::SourcePdf::open(input) {
                    Ok(s) if s.page_count() == total => Some(s),
                    Ok(s) => {
                        tracing::warn!(
                            source_pages = s.page_count(),
                            pdfium_pages = total,
                            "source PDF page count disagrees with pdfium; not using its images \
                             or DPI hints — pages will be re-encoded, sized by pdfium's own \
                             image resolution where it can tell"
                        );
                        None
                    }
                    Err(e) => {
                        tracing::warn!(error = %e,
                            "source-PDF inspection unavailable; pages will be re-encoded, sized \
                             by pdfium's own image resolution where it can tell");
                        None
                    }
                }
            } else {
                None
            };
            // Only the searchable-PDF writer reads `native_dpi`, and it
            // costs image-metadata fetches per page, so text exports skip it.
            let raster_opts = pdf::RasterOptions {
                native_dpi: keep_rasters,
            };
            // for_each_page_with expects Result<(), OcrError>; propagate
            // OCR errors directly, and wrap the whole call in anyhow
            // context at the outer `?`.
            pdf::for_each_page_with(input, DEFAULT_DPI, raster_opts, |raster| {
                // One unreadable page must not cost the other 999. By the
                // time an error reaches here the engine has already retried
                // on every rung down to CPU, so keep the page — image and
                // all, just without text — and carry on. `--fail-fast`
                // restores abort-on-first-error.
                let blocks = match engine.ocr(&raster.image) {
                    Ok(blocks) => blocks,
                    Err(e) if fail_fast => return Err(e),
                    Err(e) => {
                        bar.println(format!(
                            "  ! page {}: could not be OCR'd, leaving it without text: {e}",
                            raster.index + 1
                        ));
                        failed_pages.push((raster.index + 1, e.to_string()));
                        Vec::new()
                    }
                };
                let blocks = apply_script_postproc(blocks, script, format);
                let metrics = PageMetrics::from_blocks(&blocks);
                let lines = band_sort(blocks);
                bar.set_message(format!(
                    "p{}: {} lines, conf {:.2}",
                    raster.index + 1,
                    lines.len(),
                    metrics.avg_confidence
                ));
                doc_pages.push(DocPage {
                    index: page_offset + raster.index,
                    size: (raster.width, raster.height),
                    lines,
                    metrics,
                });
                if keep_rasters {
                    let probe = source.as_ref().map(|s| s.probe_page(raster.index));
                    encoded.push(output::searchable_pdf::encode_page_opts(
                        &raster,
                        probe.as_ref(),
                        embed,
                        DEFAULT_DPI,
                    )?);
                }
                bar.inc(1);
                Ok(())
            })
            .context("streaming OCR over pdf pages")?;
            bar.finish_with_message("OCR complete");
            // Nothing readable at all is a failed *file*, not a document of
            // blank pages: don't write an output that is 100% placeholder.
            if !doc_pages.is_empty() && failed_pages.len() == doc_pages.len() {
                bail!(
                    "every page failed; first error: {}",
                    failed_pages.first().map_or("unknown", |(_, e)| e.as_str())
                );
            }
            Ok(FileOcr {
                pages: doc_pages,
                encoded,
                failed_pages,
            })
        }
        InputKind::Image => {
            // Single-image input → one page.
            let page = image_io::load_image(input)?;
            let raster = pdf::RasterPage {
                index: 0,
                width: page.width,
                height: page.height,
                image: page.image,
                // An image file has no page size to measure against: the
                // writer lays it out at its document-wide DPI.
                dpi: 0,
                native_dpi: None,
            };
            let blocks = engine.ocr(&raster.image).context("running OCR on page")?;
            let blocks = apply_script_postproc(blocks, script, format);
            let metrics = PageMetrics::from_blocks(&blocks);
            let lines = band_sort(blocks);
            let doc_page = DocPage {
                index: page_offset,
                size: (raster.width, raster.height),
                lines,
                metrics,
            };
            // Image inputs have no source PDF and are already native
            // resolution → `ocr_dpi = 0` skips DPI capping.
            let encoded = if keep_rasters {
                vec![
                    output::searchable_pdf::encode_page_opts(&raster, None, embed, 0)
                        .context("encoding page raster")?,
                ]
            } else {
                Vec::new()
            };
            Ok(FileOcr {
                pages: vec![doc_page],
                encoded,
                failed_pages: Vec::new(),
            })
        }
    }
}

/// What OCRing one input file produced.
struct FileOcr {
    pages: Vec<DocPage>,
    encoded: Vec<EncodedPage>,
    /// `(1-based page number, error)` for pages left without text.
    failed_pages: Vec<(usize, String)>,
}

/// Per-page progress bar on stderr: count, percent, ETA, and a
/// per-page lines/confidence message. indicatif hides the bar when
/// stderr is not an interactive terminal.
fn page_progress_bar(total: usize) -> ProgressBar {
    let bar = ProgressBar::new(total as u64);
    bar.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{bar:30.cyan/blue}] {pos}/{len} pages ({percent}%) {msg} [elapsed {elapsed_precise}, eta {eta}]",
        )
        .expect("static progress template")
        .progress_chars("=>-"),
    );
    bar.enable_steady_tick(Duration::from_millis(150));
    bar
}

/// Script-specific cleanup pass between OCR and band-sort.
///
/// **Skipped entirely for searchable-PDF output.** The Nepali cleanup
/// deletes "page-noise" blocks — standalone digits, years (२०२५),
/// point sizes, rule lines — which is right for text exports but wrong
/// for the invisible text layer: the layer must mirror the page, and a
/// deleted block leaves its printed word unselectable (the missing-box
/// holes reported on Muna Madan's publisher page). Cleanup remains in
/// effect for txt / json / md.
fn apply_script_postproc(
    blocks: Vec<paddleknife_core::ocr::OcrBlock>,
    script: RecognizerScript,
    format: FormatArg,
) -> Vec<paddleknife_core::ocr::OcrBlock> {
    if format == FormatArg::Pdf {
        return blocks;
    }
    match script.postproc() {
        paddleknife_core::model::PostProc::Nepali => postproc::nepali::postprocess_page(blocks),
        paddleknife_core::model::PostProc::None => blocks,
    }
}

fn write_output(
    output_path: &std::path::Path,
    format: FormatArg,
    doc_pages: &[DocPage],
    rasters_for_pdf: Vec<EncodedPage>,
) -> Result<()> {
    match format {
        FormatArg::Pdf => {
            let spinner = ProgressBar::new_spinner();
            spinner.set_message("assembling searchable PDF…");
            spinner.enable_steady_tick(Duration::from_millis(120));
            output::searchable_pdf::write_searchable_pdf_encoded(
                output_path,
                doc_pages,
                rasters_for_pdf,
                DEFAULT_DPI,
            )
            .context("writing searchable PDF")?;
            spinner.finish_and_clear();
        }
        FormatArg::Txt => {
            write_buffered(create_output(output_path)?, |w| {
                output::text::write_text(w, doc_pages)
            })
            .context("writing txt")?;
        }
        FormatArg::Json => {
            write_buffered(create_output(output_path)?, |w| {
                output::json::write_json(w, doc_pages)
            })
            .context("writing json")?;
        }
        FormatArg::Md => {
            write_buffered(create_output(output_path)?, |w| {
                output::markdown::write_markdown(w, doc_pages)
            })
            .context("writing md")?;
        }
    }
    Ok(())
}

fn create_output(path: &Path) -> Result<File> {
    File::create(path).with_context(|| format!("creating {}", path.display()))
}

/// Run `write` into a buffer over `inner`, then flush it explicitly.
/// `BufWriter` flushes on drop as well, but discards any error there: a
/// full disk would leave a truncated output behind a zero exit status.
fn write_buffered<W: Write>(
    inner: W,
    write: impl FnOnce(&mut BufWriter<W>) -> paddleknife_core::error::Result<()>,
) -> Result<()> {
    let mut w = BufWriter::new(inner);
    write(&mut w)?;
    w.flush()?;
    Ok(())
}

fn default_output_path(input: &Path, format: FormatArg) -> PathBuf {
    let stem = input.file_stem().unwrap_or_default().to_string_lossy();
    let parent = input.parent().unwrap_or(Path::new("."));
    parent.join(format!(
        "{stem}_paddleknife_OCR.{}",
        format.default_extension()
    ))
}

/// Where a run that produces exactly one document writes.
///
/// A directory input names its output after the directory and drops it
/// **beside** the directory, mirroring the single-file rule of "next to
/// the input". Re-running is safe: discovery skips `*_paddleknife_OCR.*`.
fn merged_output_path(args: &OcrArgs) -> PathBuf {
    if let Some(explicit) = &args.output {
        return explicit.clone();
    }
    if !args.input.is_dir() {
        return default_output_path(&args.input, args.format);
    }
    // `Path::parent` yields `Some("")` for a bare name like `scans` and
    // for `.`; both mean "the current directory".
    let parent = args
        .input
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    parent.join(format!(
        "{}_paddleknife_OCR.{}",
        dir_stem(&args.input),
        args.format.default_extension()
    ))
}

/// A directory's own name, resolving `.`, `..`, and trailing separators
/// via canonicalization so `paddleknife ocr .` still names its output
/// after the real folder.
fn dir_stem(dir: &Path) -> String {
    dir.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .or_else(|| {
            dir.canonicalize()
                .ok()
                .and_then(|c| c.file_name().map(|n| n.to_string_lossy().into_owned()))
        })
        .unwrap_or_else(|| "paddleknife".to_string())
}

/// Where one file of a batch run writes.
///
/// Without `--output` each result lands beside its own input. With one,
/// the input's sub-folder layout is mirrored under it — otherwise a
/// recursive run over `a/page.png` and `b/page.png` would have both
/// results fighting over the same `page_paddleknife_OCR.txt`.
fn per_file_output_path(file: &InputFile, out_root: Option<&Path>, format: FormatArg) -> PathBuf {
    let Some(root) = out_root else {
        return default_output_path(&file.path, format);
    };
    let sub = file.relative.parent().unwrap_or(Path::new(""));
    let stem = file.path.file_stem().unwrap_or_default().to_string_lossy();
    root.join(sub).join(format!(
        "{stem}_paddleknife_OCR.{}",
        format.default_extension()
    ))
}

/// With no sub-command, print a short usage overview. On an interactive
/// console, pause before exiting so double-click users on Windows see the
/// output before the window closes.
fn enter_no_args_mode() -> Result<()> {
    use std::io::IsTerminal as _;

    println!("paddleknife v{}", paddleknife_core::VERSION);
    println!();
    println!("No sub-command given. Available commands:");
    println!("  paddleknife ocr <file|dir>  Run OCR and write output");
    println!("                              --format {{txt|pdf|json|md}}");
    println!("                              --script {{latin-cjk|devanagari|…}}");
    println!("  paddleknife models list     Show which models are installed");
    println!("  paddleknife models pull     Download models ahead of first use");
    println!();
    println!("A directory input OCRs every PDF and image inside it (mixed");
    println!("formats welcome), in natural page order:");
    println!("  paddleknife ocr ./scans/              one output per file");
    println!("  paddleknife ocr ./scans/ --recursive  …including sub-folders");
    println!("  paddleknife ocr ./book/ --merge       one document from all files");
    println!("  paddleknife ocr ./scans/ -o ./out/    write results into ./out/");
    println!();
    println!("Run `paddleknife help ocr` for every option.");

    if std::io::stdin().is_terminal() {
        println!();
        println!("Press Enter to exit...");
        let mut buf = String::new();
        let _ = std::io::stdin().read_line(&mut buf);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extended_latin_ratio_zero_for_plain_english() {
        assert_eq!(extended_latin_ratio("Hello world this is plain"), 0.0);
    }

    #[test]
    fn extended_latin_ratio_high_for_czech() {
        // Czech: many accented letters.
        let r = extended_latin_ratio("Příliš žluťoučký kůň úpěl ďábelské ódy");
        assert!(r > 0.2, "czech ratio {r}");
    }

    #[test]
    fn extended_latin_ratio_ignores_cjk() {
        // A mostly-CJK line with one English word → denominator is the
        // Latin letters only, so ratio stays 0 (no accents).
        assert_eq!(extended_latin_ratio("中国 GDP 增长"), 0.0);
    }

    #[test]
    fn first_pass_low_confidence_races_against_the_broad_confidence() {
        assert_eq!(
            classify_first_pass("anything", 0.4, 0.0, 10.0),
            FirstPass::Race(RaceGate::Confidence(0.4))
        );
    }

    #[test]
    fn first_pass_plain_english_keeps_latincjk() {
        assert_eq!(
            classify_first_pass("The quick brown fox jumps", 0.95, 0.0, 20.0),
            FirstPass::Keep(RecognizerScript::LatinCjk)
        );
    }

    #[test]
    fn first_pass_diacritic_latin_upgrades_to_latin_model() {
        assert_eq!(
            classify_first_pass("Příliš žluťoučký kůň úpěl", 0.9, 0.0, 20.0),
            FirstPass::Keep(RecognizerScript::Latin)
        );
    }

    #[test]
    fn first_pass_transliterated_cyrillic_races_with_an_open_gate() {
        // Latin lookalikes the broad model emits for Cyrillic (no actual
        // Cyrillic chars) — confident garbage → the broad read sets no
        // bar, so a real specialist can win despite its high score.
        assert_eq!(
            classify_first_pass("NpnBeT Mnp nHTepecHbiu", 0.88, 0.0, 18.0),
            FirstPass::Race(RaceGate::Open)
        );
        assert_eq!(
            classify_first_pass("YkpaiHcbka MoBa LikaBa", 0.93, 0.0, 18.0),
            FirstPass::Race(RaceGate::Open)
        );
    }

    #[test]
    fn midword_capital_ratio_zero_for_normal_text() {
        assert_eq!(midword_capital_ratio("Hello World Foo Bar"), 0.0);
        assert_eq!(midword_capital_ratio("Příliš žluťoučký kůň"), 0.0);
    }

    #[test]
    fn first_pass_high_conf_chinese_keeps_latincjk() {
        assert_eq!(
            classify_first_pass("中华人民共和国成立了", 0.92, 0.0, 9.0),
            FirstPass::Keep(RecognizerScript::LatinCjk)
        );
    }

    #[test]
    fn first_pass_high_conf_cyrillic_routes_eslav() {
        // If the broad pass somehow emits confident Cyrillic, trust it.
        assert_eq!(
            classify_first_pass("Привет мир как дела", 0.9, 0.0, 15.0),
            FirstPass::Keep(RecognizerScript::Eslav)
        );
    }

    #[test]
    fn first_pass_that_lost_lines_races_despite_high_confidence() {
        // The VW Passat B6 cover: big Latin titles read at 0.91 while 30%
        // of the lines — every Cyrillic one — came back empty or as bare
        // punctuation.
        assert_eq!(
            classify_first_pass("VW Passat B6 2005. NEW", 0.91, 0.30, 77.4),
            FirstPass::Race(RaceGate::Score(77.4 * RACE_MIN_GAIN))
        );
        assert_eq!(
            classify_first_pass("VW Passat B6 2005. NEW", 0.91, LOST_TEXT_RACE, 77.4),
            FirstPass::Keep(RecognizerScript::LatinCjk),
            "at the threshold the broad read stands"
        );
    }

    #[test]
    fn lost_text_counts_unread_lines_and_punctuation_residue() {
        let blocks = [
            block("VW"),
            block("Passat B6"),
            block(", . 28 . (495) 105-64-44"), // digits survive: read
            block(", ,"),                      // letters all dropped
            block("."),
        ];
        // 2 residue + 3 unread of 5 + 3 detected lines.
        assert!((lost_text_ratio(&blocks, 3) - 5.0 / 8.0).abs() < 1e-6);
        assert_eq!(lost_text_ratio(&blocks[..3], 0), 0.0);
        assert_eq!(lost_text_ratio(&[], 0), 0.0);
    }

    fn entry(script: RecognizerScript, conf: f32, score: f32) -> RaceEntry {
        RaceEntry {
            script,
            conf,
            score,
        }
    }

    #[test]
    fn race_winner_after_lost_lines_needs_more_text_then_ranks_by_confidence() {
        use RecognizerScript::*;
        // Measured on the Passat cover (broad read: score 77.4).
        let gate = RaceGate::Score(77.4 * RACE_MIN_GAIN);
        let cands = [
            entry(Devanagari, 0.851, 157.1),
            entry(Eslav, 0.920, 172.3),
            entry(Korean, 0.829, 153.4),
            entry(Cyrillic, 0.923, 172.6),
            entry(Latin, 0.884, 71.9),
        ];
        assert_eq!(race_winner(&cands, gate), Cyrillic);
        // Score alone must not pick: wrong-script Devanagari can out-count
        // the right model's characters, and it still loses on confidence.
        let cands = [
            entry(Devanagari, 0.834, 4500.0),
            entry(Eslav, 0.979, 4069.0),
        ];
        assert_eq!(race_winner(&cands, gate), Eslav);
        // A sparse Latin scan (broad score 66.4): every specialist ties the
        // broad read, so LatinCjk stays — no Cyrillic homoglyphs.
        let gate = RaceGate::Score(66.4 * RACE_MIN_GAIN);
        let cands = [entry(Eslav, 0.872, 61.5), entry(Cyrillic, 0.940, 60.5)];
        assert_eq!(race_winner(&cands, gate), LatinCjk);
    }

    #[test]
    fn race_winner_picks_most_confident_above_the_broad_confidence() {
        let cands = [
            entry(RecognizerScript::Devanagari, 0.55, 10.0),
            entry(RecognizerScript::Eslav, 0.91, 9.0),
            entry(RecognizerScript::Korean, 0.48, 12.0),
        ];
        let gate = RaceGate::Confidence(0.50);
        assert_eq!(race_winner(&cands, gate), RecognizerScript::Eslav);
    }

    #[test]
    fn race_winner_falls_back_to_latincjk_when_none_pass_the_gate() {
        let cands = [
            entry(RecognizerScript::Devanagari, 0.40, 10.0),
            entry(RecognizerScript::Korean, 0.35, 10.0),
        ];
        // Broad confidence 0.60 beats both → fall back.
        let gate = RaceGate::Confidence(0.60);
        assert_eq!(race_winner(&cands, gate), RecognizerScript::LatinCjk);
        // An open gate still refuses a read of nothing.
        let cands = [entry(RecognizerScript::Korean, 0.0, 0.0)];
        assert_eq!(
            race_winner(&cands, RaceGate::Open),
            RecognizerScript::LatinCjk
        );
    }

    #[test]
    fn race_winner_empty_falls_back() {
        let gate = RaceGate::Confidence(0.30);
        assert_eq!(race_winner(&[], gate), RecognizerScript::LatinCjk);
    }

    // ---- directory input: CLI wiring + output-path resolution ----

    /// Parse a real `paddleknife ocr …` argv into `OcrArgs`, so these
    /// tests also pin the flag spellings the docs promise.
    fn ocr_args(argv: &[&str]) -> OcrArgs {
        let cli = Cli::try_parse_from(argv).expect("argv parses");
        match cli.command {
            Some(Command::Ocr(a)) => a,
            other => unreachable!("expected the ocr sub-command, got {other:?}"),
        }
    }

    #[test]
    fn cli_directory_flags_default_off() {
        let a = ocr_args(&["paddleknife", "ocr", "book.pdf"]);
        assert!(!a.recursive);
        assert!(!a.merge);
        assert!(!a.fail_fast);
        assert!(!a.script_per_file);
    }

    #[test]
    fn cli_accepts_directory_flags_including_short_recursive() {
        let a = ocr_args(&[
            "paddleknife",
            "ocr",
            "./scans",
            "-r",
            "--merge",
            "--fail-fast",
            "--script-per-file",
        ]);
        assert!(a.recursive);
        assert!(a.merge);
        assert!(a.fail_fast);
        assert!(a.script_per_file);
    }

    #[test]
    fn merged_output_path_for_a_file_is_unchanged() {
        // Regression pin: single-file runs must keep their old name.
        let a = ocr_args(&["paddleknife", "ocr", "docs/book.pdf", "-f", "txt"]);
        assert_eq!(
            merged_output_path(&a),
            Path::new("docs").join("book_paddleknife_OCR.txt")
        );
    }

    #[test]
    fn merged_output_path_for_a_directory_is_named_after_it() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let scans = tmp.path().join("scans");
        std::fs::create_dir(&scans).expect("mkdir");
        let a = ocr_args(&[
            "paddleknife",
            "ocr",
            &scans.display().to_string(),
            "--merge",
            "-f",
            "pdf",
        ]);
        // Beside the directory, named after it.
        assert_eq!(
            merged_output_path(&a),
            tmp.path().join("scans_paddleknife_OCR.pdf")
        );
    }

    #[test]
    fn merged_output_path_honours_explicit_output() {
        let a = ocr_args(&["paddleknife", "ocr", "./scans", "-o", "out/book.md"]);
        assert_eq!(merged_output_path(&a), Path::new("out/book.md"));
    }

    #[test]
    fn per_file_output_path_defaults_beside_its_input() {
        let file = InputFile {
            path: Path::new("scans").join("page1.png"),
            relative: PathBuf::from("page1.png"),
            kind: InputKind::Image,
        };
        assert_eq!(
            per_file_output_path(&file, None, FormatArg::Txt),
            Path::new("scans").join("page1_paddleknife_OCR.txt")
        );
    }

    #[test]
    fn per_file_output_path_mirrors_subdirs_under_the_output_root() {
        // Two same-named files in different sub-folders must not collide.
        let a = InputFile {
            path: Path::new("scans/a").join("page.png"),
            relative: Path::new("a").join("page.png"),
            kind: InputKind::Image,
        };
        let b = InputFile {
            path: Path::new("scans/b").join("page.png"),
            relative: Path::new("b").join("page.png"),
            kind: InputKind::Image,
        };
        let root = Path::new("out");
        let pa = per_file_output_path(&a, Some(root), FormatArg::Json);
        let pb = per_file_output_path(&b, Some(root), FormatArg::Json);
        assert_eq!(
            pa,
            Path::new("out").join("a").join("page_paddleknife_OCR.json")
        );
        assert_eq!(
            pb,
            Path::new("out").join("b").join("page_paddleknife_OCR.json")
        );
        assert_ne!(pa, pb);
    }

    #[test]
    fn dir_stem_resolves_dot_to_the_real_folder_name() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let named = tmp.path().join("mybook");
        std::fs::create_dir(&named).expect("mkdir");
        assert_eq!(dir_stem(&named), "mybook");
        // Trailing separator must not swallow the name.
        assert_eq!(dir_stem(&named.join("")), "mybook");
    }

    // ---- models directory resolution ----

    /// Create `dir` holding `files` (empty placeholders), return it.
    fn stage(dir: PathBuf, files: &[&str]) -> PathBuf {
        std::fs::create_dir_all(&dir).expect("mkdir");
        for f in files {
            std::fs::write(dir.join(f), b"").expect("stage file");
        }
        dir
    }

    /// An exe path deep enough inside `root` that the whole ancestor
    /// search stays inside it — so a `models/` that happens to exist above
    /// the system temp dir can't leak into a test expecting "not found".
    fn deep_exe(root: &Path) -> PathBuf {
        root.join("a")
            .join("b")
            .join("c")
            .join("install")
            .join("paddleknife.exe")
    }

    #[test]
    fn models_dir_env_var_wins_over_a_found_dir() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let exe = tmp.path().join("bin").join("paddleknife.exe");
        stage(
            tmp.path().join("bin").join("models"),
            &["pp-ocrv6_medium_det.onnx"],
        );
        let chosen = tmp.path().join("elsewhere");
        assert_eq!(
            select_models_dir(Some(chosen.clone()), Some(exe.as_path()), None),
            (chosen, ModelsDirSource::Env)
        );
    }

    #[test]
    fn models_dir_empty_env_var_is_ignored() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let exe = deep_exe(tmp.path());
        let (_, source) = select_models_dir(Some(PathBuf::new()), Some(exe.as_path()), None);
        assert_eq!(source, ModelsDirSource::Default);
    }

    /// `cargo run`: the exe is in `target/<profile>/`, the models at the
    /// workspace root — and any one registry file identifies them, not
    /// just the legacy PP-OCRv5 detector.
    #[test]
    fn models_dir_found_up_the_exe_ancestors_by_any_known_file() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let exe = tmp
            .path()
            .join("target")
            .join("debug")
            .join("paddleknife.exe");
        let models = stage(tmp.path().join("models"), &["ppocrv6_tiny_dict.txt"]);
        assert_eq!(
            select_models_dir(None, Some(exe.as_path()), None),
            (models, ModelsDirSource::Found)
        );
    }

    #[test]
    fn models_dir_without_a_known_file_is_skipped() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let bin = tmp.path().join("bin");
        let exe = bin.join("paddleknife.exe");
        stage(bin.join("models"), &["README.md", "notes.txt"]);
        let models = stage(tmp.path().join("models"), &["pp-lcnet_x1_0_doc_ori.onnx"]);
        assert_eq!(
            select_models_dir(None, Some(exe.as_path()), None),
            (models, ModelsDirSource::Found)
        );
    }

    #[test]
    fn models_dir_nearest_to_the_exe_wins() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let bin = tmp.path().join("bin");
        let exe = bin.join("paddleknife.exe");
        let near = stage(bin.join("models"), &["devanagari_PP-OCRv5_rec.onnx"]);
        stage(tmp.path().join("models"), &["pp-ocrv6_medium_det.onnx"]);
        assert_eq!(
            select_models_dir(None, Some(exe.as_path()), None),
            (near, ModelsDirSource::Found)
        );
    }

    #[test]
    fn models_dir_search_stops_after_the_level_limit() {
        let tmp = tempfile::tempdir().expect("tempdir");
        // The exe's dir plus three ancestors are searched; `tmp` is the
        // fifth directory up, so its `models/` is out of reach.
        let deep = tmp.path().join("a").join("b").join("c").join("d");
        let exe = deep.join("paddleknife.exe");
        stage(tmp.path().join("models"), &["pp-ocrv6_medium_det.onnx"]);
        assert_eq!(
            select_models_dir(None, Some(exe.as_path()), None),
            (deep.join("models"), ModelsDirSource::Default)
        );
        // One level shallower, it is found.
        let exe = tmp
            .path()
            .join("a")
            .join("b")
            .join("c")
            .join("paddleknife.exe");
        assert_eq!(
            select_models_dir(None, Some(exe.as_path()), None).1,
            ModelsDirSource::Found
        );
    }

    #[test]
    fn models_dir_falls_back_to_the_working_directory() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let exe = deep_exe(tmp.path());
        let cwd = tmp.path().join("work");
        let models = stage(cwd.join("models"), &["ppocrv5_korean_dict.txt"]);
        assert_eq!(
            select_models_dir(None, Some(exe.as_path()), Some(cwd.as_path())),
            (models, ModelsDirSource::Found)
        );
    }

    /// A fresh install has no models anywhere: the answer is the directory
    /// `models pull` and on-demand downloads write to, next to the exe.
    #[test]
    fn models_dir_defaults_next_to_the_exe() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let exe = deep_exe(tmp.path());
        let install = exe.parent().expect("exe has a directory").to_path_buf();
        let cwd = tmp.path().join("work");
        assert_eq!(
            select_models_dir(None, Some(exe.as_path()), Some(cwd.as_path())),
            (install.join("models"), ModelsDirSource::Default)
        );
        // Exe location unknown: the working directory instead.
        assert_eq!(
            select_models_dir(None, None, Some(cwd.as_path())),
            (cwd.join("models"), ModelsDirSource::Default)
        );
    }

    // ---- orientation classifiers ----

    #[test]
    fn requested_classifiers_follow_the_orientation_flags() {
        let none = ocr_args(&["paddleknife", "ocr", "book.pdf"]);
        assert!(requested_classifiers(&none).is_empty());
        let both = ocr_args(&[
            "paddleknife",
            "ocr",
            "book.pdf",
            "--detect-page-orientation",
            "--detect-line-orientation",
        ]);
        assert_eq!(
            requested_classifiers(&both),
            vec![Classifier::DocOrientation, Classifier::TextlineOrientation]
        );
    }

    #[test]
    fn cli_models_pull_accepts_orientation_without_a_script() {
        let cli = Cli::try_parse_from(["paddleknife", "models", "pull", "--orientation"])
            .expect("argv parses");
        match cli.command {
            Some(Command::Models(cli::ModelsArgs {
                action: cli::ModelsAction::Pull(p),
            })) => {
                assert!(p.orientation);
                assert!(!p.all);
                assert!(p.script.is_none());
            }
            other => unreachable!("expected models pull, got {other:?}"),
        }
    }

    // ---- auto-detect sampling ----

    fn block(text: &str) -> OcrBlock {
        OcrBlock {
            text: text.into(),
            confidence: 0.9,
            ..OcrBlock::default()
        }
    }

    #[test]
    fn has_text_ignores_empty_and_whitespace_blocks() {
        assert!(!has_text(&[]));
        assert!(!has_text(&[block(""), block("  ")]));
        assert!(has_text(&[block(" "), block("a")]));
    }

    /// A blank cover must not decide the script: sampling moves on to the
    /// first page with text, and stops reading there.
    #[test]
    fn first_page_with_text_skips_blank_pages_and_stops_at_text() {
        let pages = [
            vec![],
            vec![block(" ")],
            vec![block("मुना मदन")],
            vec![block("x")],
        ];
        let mut reads = Vec::new();
        let found = first_page_with_text(pages.len(), |i| {
            reads.push(i);
            Ok((i * 10, pages[i].clone()))
        })
        .expect("no read error")
        .expect("page 3 has text");
        assert_eq!(found.0, 2, "index of the first page with text");
        assert_eq!(found.1, 20, "that page's own image comes back");
        assert_eq!(found.2[0].text, "मुना मदन");
        assert_eq!(reads, vec![0, 1, 2], "page 4 is never read");
    }

    #[test]
    fn first_page_with_text_is_none_when_every_sampled_page_is_blank() {
        let mut reads = 0;
        let found = first_page_with_text(SAMPLE_MAX_PAGES, |_| {
            reads += 1;
            Ok(((), Vec::new()))
        })
        .expect("no read error");
        assert!(found.is_none());
        assert_eq!(reads, SAMPLE_MAX_PAGES);
        // A zero-page document reads nothing.
        let found = first_page_with_text::<()>(0, |_| unreachable!("no pages to read"))
            .expect("no read error");
        assert!(found.is_none());
    }

    #[test]
    fn a_sampled_page_that_cannot_be_ocrd_reads_as_blank_unless_fail_fast() {
        use paddleknife_core::error::OcrError;
        // Page 1 fails on every rung; page 2 has text. The run would write
        // page 1 without text and carry on, so sampling moves past it too.
        let read = |i: usize| match i {
            0 => Err(OcrError::Engine("adapter execution failed".into())),
            _ => Ok(vec![block("Hello")]),
        };
        let found = first_page_with_text(2, |i| Ok(((), sample_blocks(read(i), i, false)?)))
            .expect("the failure is not fatal")
            .expect("page 2 decides");
        assert_eq!(found.0, 1);

        // `--fail-fast`: the same page stops the run, naming it.
        let err = first_page_with_text(2, |i| Ok(((), sample_blocks(read(i), i, true)?)))
            .expect_err("fail-fast surfaces it");
        assert!(format!("{err:#}").contains("page 1"), "{err:#}");
    }

    #[test]
    fn first_page_with_text_propagates_read_errors() {
        let err = first_page_with_text::<()>(3, |i| {
            if i == 1 {
                bail!("page 2 would not rasterize")
            }
            Ok(((), Vec::new()))
        })
        .expect_err("the error surfaces");
        assert!(err.to_string().contains("page 2"));
    }

    // ---- text-export writer ----

    /// A sink whose every write fails, like a full disk.
    struct FullDisk;

    impl Write for FullDisk {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("disk full"))
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// The bytes fit in `BufWriter`'s buffer, so every write "succeeds" and
    /// only the explicit flush can report that none of them landed.
    #[test]
    fn write_buffered_reports_an_error_raised_only_at_flush() {
        let err = write_buffered(FullDisk, |w| {
            w.write_all(b"--- Page 1 ---\nhello\n")?;
            Ok(())
        })
        .expect_err("a failed flush must not pass silently");
        assert!(format!("{err:#}").contains("disk full"), "{err:#}");
    }

    #[test]
    fn write_buffered_delivers_everything_written() {
        let mut out = Vec::new();
        write_buffered(&mut out, |w| {
            w.write_all(b"hello ")?;
            w.write_all(b"world")?;
            Ok(())
        })
        .expect("writes to memory succeed");
        assert_eq!(out, b"hello world");
    }
}
