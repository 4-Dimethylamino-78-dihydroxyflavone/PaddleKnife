//! Top-level clap definitions: the `ocr` and `models` sub-commands.

use clap::{Args, Parser, Subcommand, ValueEnum};
use paddleknife_core::{
    model::{Quality, RecognizerScript},
    ocr::OcrDevice,
};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "paddleknife", version, about = "Swiss-army-knife OCR tool")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Run OCR on a file or a directory of files.
    Ocr(OcrArgs),
    /// List installed models, or download them ahead of first use.
    ///
    /// `ocr` downloads whatever it needs on first use, into the same
    /// directory: the one named by the PADDLEKNIFE_MODELS_DIR environment
    /// variable; else the nearest `models/` holding a model file, beside
    /// or up to three levels above the executable, or in the working
    /// directory; else a new `models/` next to the executable.
    Models(ModelsArgs),
}

#[derive(Args, Debug)]
pub struct ModelsArgs {
    #[command(subcommand)]
    pub action: ModelsAction,
}

#[derive(Subcommand, Debug)]
pub enum ModelsAction {
    /// Download a script's recognizer + dictionary (and the tier's
    /// detector if missing), and/or the orientation classifiers.
    /// `--all` fetches every script and both classifiers.
    Pull(ModelsPullArgs),
    /// List detectors, recognizers and orientation classifiers, and
    /// whether each is present locally.
    List,
}

#[derive(Args, Debug)]
pub struct ModelsPullArgs {
    /// Script to fetch (e.g. korean, russian, czech, latin-cjk). Omit
    /// with --all or --orientation.
    #[arg(value_parser = parse_pinned_script)]
    pub script: Option<RecognizerScript>,
    /// Fetch every registered script, plus the orientation classifiers.
    #[arg(long)]
    pub all: bool,
    /// Fetch the orientation classifiers used by
    /// --detect-page-orientation and --detect-line-orientation.
    #[arg(long)]
    pub orientation: bool,
    /// Size tier whose detector (and Latin/CJK recognizer) to fetch.
    #[arg(long, default_value = "medium", value_parser = parse_quality, value_name = "TIER")]
    pub quality: Quality,
}

/// clap parser for a pinned script (no `auto`) — used by `models pull`.
pub fn parse_pinned_script(value: &str) -> Result<RecognizerScript, String> {
    RecognizerScript::from_cli_value(value)
        .ok_or_else(|| format!("unknown script '{value}'. {}", script_help()))
}

/// Output format chosen at the CLI boundary. Each variant maps to one
/// writer in `paddleknife_core::output`.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum FormatArg {
    /// Plain text with `--- Page N ---` separators.
    Txt,
    /// Searchable PDF: the original raster + invisible OCR text layer.
    Pdf,
    /// Pretty-printed JSON array of DocPage objects.
    Json,
    /// Markdown with `# Page N` headings.
    Md,
}

impl FormatArg {
    /// File extension (without the dot) used when `--output` is omitted.
    pub fn default_extension(self) -> &'static str {
        match self {
            Self::Txt => "txt",
            Self::Pdf => "pdf",
            Self::Json => "json",
            Self::Md => "md",
        }
    }
}

/// Page-image embedding strategy for searchable-PDF output.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum PdfImageArg {
    /// Passthrough original scan bytes when safe, else re-encode.
    Auto,
    /// Always re-encode from the OCR raster.
    Reencode,
}

/// Recognizer script selection. `auto` (default) samples the first page
/// that has text and routes to the right recognizer; any registry alias
/// pins a script and skips detection.
///
/// Parsed against `model::DESCRIPTORS` so the accepted names live in one
/// place — see `parse_script`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScriptArg {
    /// Detect script from a sample page (default).
    Auto,
    /// User pinned a specific recognizer.
    Pinned(RecognizerScript),
}

impl ScriptArg {
    /// Returns the explicit `RecognizerScript` if the user pinned one,
    /// or `None` for `Auto` so callers know to detect.
    pub fn explicit(self) -> Option<RecognizerScript> {
        match self {
            Self::Auto => None,
            Self::Pinned(s) => Some(s),
        }
    }
}

/// clap value parser for `--quality`: a PP-OCRv6 size tier.
///
/// Parsed against `model::QUALITIES` for the same reason `--script` is
/// parsed against `DESCRIPTORS` — the accepted names live in one place.
pub fn parse_quality(value: &str) -> Result<Quality, String> {
    Quality::from_cli_value(value)
        .ok_or_else(|| format!("unknown quality '{value}'. {}", quality_help()))
}

/// Help string listing every tier, built from the registry.
pub fn quality_help() -> String {
    let names: Vec<&str> = paddleknife_core::model::QUALITIES
        .iter()
        .map(|q| q.cli_value)
        .collect();
    format!("Valid values: {}", names.join(", "))
}

/// clap value parser for `--script`: `auto` or any registry alias.
pub fn parse_script(value: &str) -> Result<ScriptArg, String> {
    if value.eq_ignore_ascii_case("auto") {
        return Ok(ScriptArg::Auto);
    }
    RecognizerScript::from_cli_value(value)
        .map(ScriptArg::Pinned)
        .ok_or_else(|| format!("unknown script '{value}'. {}", script_help()))
}

/// Help string listing `auto` + every canonical alias, built from the
/// registry so it can never drift from what `parse_script` accepts.
pub fn script_help() -> String {
    let mut names = vec!["auto".to_string()];
    for d in paddleknife_core::model::DESCRIPTORS {
        // First alias is canonical; list it with its display name.
        if let Some(canon) = d.cli_aliases.first() {
            names.push((*canon).to_string());
        }
    }
    format!("valid: {}", names.join(", "))
}

/// Execution-provider choice at the CLI. Maps 1:1 onto `OcrDevice`.
#[derive(ValueEnum, Clone, Copy, Debug)]
pub enum DeviceArg {
    /// Pick the best available: DirectML on Windows, CoreML on macOS, else CPU.
    Auto,
    /// Force CPU — matches Plan 1 / 2 behavior.
    Cpu,
    /// DirectML — Windows, any DX12 GPU (Intel iGPU, AMD, NVIDIA).
    Dml,
    /// CUDA — NVIDIA only. Requires building with `--features cuda`.
    Cuda,
}

impl From<DeviceArg> for OcrDevice {
    fn from(a: DeviceArg) -> Self {
        match a {
            DeviceArg::Auto => Self::Auto,
            DeviceArg::Cpu => Self::Cpu,
            DeviceArg::Dml => Self::DirectMl,
            DeviceArg::Cuda => Self::Cuda,
        }
    }
}

#[derive(Args, Debug)]
pub struct OcrArgs {
    /// Input file *or directory* — PDF or image (PNG / JPG / TIFF /
    /// WebP / …). A directory is scanned for every OCR-able file it
    /// holds, in natural order (page2 before page10); single-format and
    /// mixed PDF+image folders are both fine. See --recursive, --merge.
    pub input: PathBuf,

    /// Output path. For a single file (or --merge) this is the output
    /// file, defaulting to <input stem>_paddleknife_OCR.<ext> next to
    /// the input. For a directory without --merge it is the output
    /// *directory* — the input's sub-folder layout is mirrored inside
    /// it — defaulting to writing each result beside its own input.
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Descend into sub-directories when INPUT is a directory.
    #[arg(short, long)]
    pub recursive: bool,

    /// Treat every discovered file as one document: pages are
    /// concatenated in natural order and written to a single output.
    /// Use this for a folder holding one book's page scans. Without it,
    /// each input file gets its own output file.
    #[arg(long)]
    pub merge: bool,

    /// Abort on the first failure: a file that cannot be read or
    /// written, or a single page that cannot be OCR'd even after the
    /// automatic retries. Default: a failed PDF page is written without
    /// text, a failed file is skipped, the run carries on, and every
    /// failure is listed at the end (exit status is still non-zero if
    /// anything failed).
    #[arg(long)]
    pub fail_fast: bool,

    /// Re-run script auto-detection for every file, for folders that
    /// mix languages. Costs one extra sample pass (and possibly an
    /// engine rebuild) per file. Default: detect once, from the first
    /// file that has text, and reuse it. Ignored when --script pins a
    /// language.
    #[arg(long)]
    pub script_per_file: bool,

    /// Output format.
    #[arg(short, long, value_enum, default_value_t = FormatArg::Txt)]
    pub format: FormatArg,

    /// Recognizer script. `auto` (default) samples the first page that
    /// has text (of the first three) and routes automatically; if none
    /// has, it keeps latin-cjk. Pin a language to skip detection:
    /// devanagari/hindi/nepali, latin/czech/polish, cyrillic,
    /// eslav/russian/ukrainian, korean, chinese/japanese (latin-cjk).
    #[arg(
        short,
        long,
        default_value = "auto",
        value_parser = parse_script,
    )]
    pub script: ScriptArg,

    /// Model size tier: `tiny`, `small` or `medium` (default).
    ///
    /// Picks the PP-OCRv6 detector, and the PP-OCRv6 recognizer for
    /// Latin/CJK. Scripts outside PP-OCRv6's dictionary (devanagari,
    /// cyrillic, eslav, korean) keep their PP-OCRv5 recognizer at every
    /// tier and are affected only in detection.
    #[arg(
        long,
        default_value = "medium",
        value_parser = parse_quality,
        value_name = "TIER",
    )]
    pub quality: Quality,

    /// Execution provider. `auto` picks the GPU path where available
    /// (DirectML on Windows, CoreML on macOS) and falls back to CPU
    /// if that GPU runtime isn't present.
    #[arg(short, long, value_enum, default_value_t = DeviceArg::Auto)]
    pub device: DeviceArg,

    /// Recognizer batch size — the one memory knob. Default: 64 on GPU.
    ///
    /// Peak GPU memory follows it (measured: 3.6 GiB at 64, 2.2 at 32,
    /// 1.5 at 16, 1.2 at 8), so on a GPU with little free memory start
    /// lower, e.g. 16 for a 4 GB card. You rarely need to: a page that runs out
    /// of memory is retried automatically at half this, then at 8, then
    /// on CPU, and a run that keeps hitting the limit settles on the
    /// size that works. Setting it just skips the failed first attempts.
    #[arg(long, value_name = "N")]
    pub region_batch_size: Option<usize>,

    /// Advanced: reference line width (px, at the recognizer's 48 px
    /// input height) for the batch memory budget. A batch holds the full
    /// --region-batch-size only while its widest line is at most this
    /// wide; batches of wider lines hold proportionally fewer, because a
    /// batch is padded to its widest member. Default: 896 on GPU. 0
    /// turns the budget off (batches are then limited by count alone,
    /// which runs out of GPU memory on pages mixing short and very
    /// long lines).
    #[arg(long, value_name = "PX")]
    pub region_ref_width: Option<usize>,

    /// Override the detector's image batch size. Currently has no
    /// effect: each page is sent to the detector on its own, so there is
    /// never more than one image to batch. Default: oar-ocr's own value.
    #[arg(long, value_name = "N")]
    pub image_batch_size: Option<usize>,

    /// Enable whole-page orientation detection (0/90/180/270). Rotates
    /// fully-rotated pages (sideways tables, landscape scans) upright
    /// before detection — text on them detects at native quality — and
    /// maps all boxes back to the original page. Cheap (one tiny
    /// classifier pass per page). Its classifier,
    /// `pp-lcnet_x1_0_doc_ori.onnx` (6.5 MB), downloads on first use.
    ///
    /// The classifier is trained on scanned documents and often reads
    /// screenshots or sparse pages upside-down, so its verdict is
    /// checked: a page that comes back poorly recognised is re-read
    /// flipped 180° and the better result kept (one extra pass, only on
    /// such pages).
    #[arg(long)]
    pub detect_page_orientation: bool,

    /// Enable per-line orientation detection (upright vs upside-down;
    /// sideways lines are handled without it). Adds ~15% latency but
    /// makes upside-down lines in scans readable and correctly
    /// positioned in the searchable-PDF text layer. Its classifier,
    /// `pp-lcnet_x1_0_textline_ori.onnx` (6.5 MB), downloads on first use.
    #[arg(long)]
    pub detect_line_orientation: bool,

    /// How the page image is embedded in searchable-PDF output.
    /// `auto` copies the original scan bytes through verbatim when a
    /// page is a plain single-image scan (output stays ≈ input size);
    /// `reencode` always re-encodes from the OCR raster.
    #[arg(long, value_enum, default_value_t = PdfImageArg::Auto)]
    pub pdf_image: PdfImageArg,

    /// JPEG quality (1-100) for re-encoded page images in searchable-PDF
    /// output.
    #[arg(long, value_name = "Q", default_value_t = 85,
          value_parser = clap::value_parser!(u8).range(1..=100))]
    pub jpeg_quality: u8,

    /// Upper limit on the resolution (DPI) of PDF pages re-encoded into
    /// searchable-PDF output. A page is stored at min(DPI, the
    /// resolution detected for its source image), so this can shrink a
    /// page but never enlarge it past its source; where no source
    /// resolution is detected, DPI itself is the limit. Default: no
    /// limit beyond the detected source resolution, so scans are never
    /// stored above their true resolution. Image inputs are always
    /// embedded at their own resolution. OCR accuracy is unaffected —
    /// recognition always runs at full rasterization DPI.
    #[arg(long, value_name = "DPI")]
    pub embed_dpi: Option<u32>,

    /// Debug-level logging from PaddleKnife itself. RUST_LOG, when set,
    /// takes precedence.
    #[arg(short, long)]
    pub verbose: bool,
}
