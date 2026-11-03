//! OCR abstraction: `OcrBackend` trait + oar-ocr-backed implementation.
//!
//! The trait exists so alternative backends (e.g. a future GPU-tuned
//! implementation, or pure-Rust tract) can swap in without touching
//! the rest of the pipeline.

use crate::{
    error::{OcrError, Result},
    geometry::{self, Quad},
    model::ModelPaths,
};
use image::DynamicImage;
use oar_ocr::{
    core::config::{OrtExecutionProvider, OrtSessionConfig},
    oarocr::{DocumentOrientation, OAROCRBuilder, PredictOptions},
};
use unicode_segmentation::UnicodeSegmentation;

/// One grapheme cluster of recognized text with its detected quad.
///
/// The cluster is the atom of the searchable-PDF text layer: a
/// Devanagari conjunct or a base+matra sequence stays one span, so its
/// selection rectangle covers the whole drawn glyph and copy yields the
/// full codepoint sequence.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CharSpan {
    /// The cluster's text (one grapheme cluster, possibly several
    /// Unicode scalars).
    pub text: String,
    /// Detected quad in original-image pixels, clockwise from top-left.
    pub quad: Quad,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct OcrBlock {
    /// Axis-aligned bounding box in the original image coords.
    /// `[x1, y1, x2, y2]` — top-left, bottom-right. Derived from
    /// `quad`; kept for legacy consumers and JSON stability.
    pub bbox: [i32; 4],
    pub text: String,
    pub confidence: f32,
    /// Orientation angle in degrees. 0 = upright, 180 = upside-down.
    /// Only populated when `--detect-line-orientation` is on and the
    /// `pp-lcnet_x1_0_textline_ori.onnx` model is present — that model is
    /// a 2-class classifier, so 90/270 never appear here.
    #[serde(default)]
    pub rotation_deg: f32,
    /// Detector-derived quad (this block's slice of the region
    /// polygon), original-image pixels, clockwise from top-left.
    /// `None` on JSON produced before Plan 5.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quad: Option<Quad>,
    /// Per-grapheme-cluster geometry, in text order. Empty on legacy
    /// JSON. When `chars_synthesized` is true the quads are uniform
    /// slices of the block quad, not CTC-measured.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub chars: Vec<CharSpan>,
    /// True when the recognizer's char boxes were missing or did not
    /// match the text and `chars` was synthesized by uniform split.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub chars_synthesized: bool,
}

impl OcrBlock {
    /// The block's quad, falling back to the bbox rectangle for legacy
    /// data that predates the quad field.
    pub fn quad_or_bbox(&self) -> Quad {
        self.quad.unwrap_or_else(|| {
            geometry::quad_from_bbox([
                self.bbox[0] as f32,
                self.bbox[1] as f32,
                self.bbox[2] as f32,
                self.bbox[3] as f32,
            ])
        })
    }

    /// Replace `chars` with uniform grapheme-cluster slices of the
    /// block quad and set `chars_synthesized`.
    ///
    /// For callers that mutate `text` in a way that can't be mapped
    /// back onto the measured CTC spans (e.g. post-processing that
    /// reorders or deletes mid-token characters) — the geometry
    /// contract `concat(chars.text) == text` must hold for the
    /// searchable-PDF writer and JSON consumers.
    pub fn synthesize_chars(&mut self) {
        let quad = self.quad_or_bbox();
        let graphemes: Vec<&str> = self.text.graphemes(true).collect();
        let n = graphemes.len().max(1) as f32;
        self.chars = graphemes
            .iter()
            .enumerate()
            .filter(|(_, g)| !g.chars().all(char::is_whitespace))
            .map(|(i, g)| CharSpan {
                text: (*g).to_string(),
                quad: quad_slice(&quad, i as f32 / n, (i + 1) as f32 / n),
            })
            .collect();
        self.chars_synthesized = true;
    }
}

/// Execution-provider selection for the OCR backend.
///
/// Resolved into an `OrtSessionConfig` before the builder runs.
/// `Auto` tries DirectML first on Windows, falls back to CPU if DirectML
/// init fails at ORT runtime (driver missing, VM, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OcrDevice {
    /// Try DirectML (Windows) / CoreML (macOS), fall back to CPU.
    Auto,
    /// Force CPU.
    Cpu,
    /// DirectML (Windows, any DX12 GPU).
    DirectMl,
    /// CUDA (NVIDIA). Only available with the `cuda` Cargo feature.
    Cuda,
}

impl OcrDevice {
    /// Build the EP chain for this device. CPU is always appended as a
    /// final fallback so a missing GPU runtime degrades gracefully
    /// rather than panicking.
    ///
    /// All optional EP fields pass `None`, which tells ORT to use its
    /// own defaults (device 0, auto memory limits, etc.). We can expose
    /// knobs like `--gpu-mem-limit` later if anyone needs them.
    fn execution_providers(self) -> Vec<OrtExecutionProvider> {
        match self {
            Self::Cpu => vec![OrtExecutionProvider::CPU],
            Self::DirectMl => vec![dml_provider(), OrtExecutionProvider::CPU],
            Self::Cuda => vec![cuda_provider(), OrtExecutionProvider::CPU],
            Self::Auto => {
                // Platform-appropriate GPU EP, then CPU fallback.
                #[cfg(target_os = "windows")]
                {
                    vec![dml_provider(), OrtExecutionProvider::CPU]
                }
                #[cfg(target_os = "macos")]
                {
                    vec![coreml_provider(), OrtExecutionProvider::CPU]
                }
                #[cfg(not(any(target_os = "windows", target_os = "macos")))]
                {
                    vec![OrtExecutionProvider::CPU]
                }
            }
        }
    }

    /// Whether this device runs the recognizer on an accelerator, where
    /// memory is scarce and a batch that is too large fails outright
    /// instead of paging.
    pub fn is_accelerated(self) -> bool {
        !matches!(self, Self::Cpu)
    }

    /// Human-readable label used in CLI output ("running on DirectML", etc.).
    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "CPU",
            Self::DirectMl => "DirectML",
            Self::Cuda => "CUDA",
        }
    }
}

/// Aspect ratio at which oar-ocr treats a crop as rotated text.
///
/// Must stay in lockstep with `get_rotate_crop_image`, which applies
/// `rotate270` when `height >= width * 1.5`. If these disagree, the quad
/// we hand downstream describes a different orientation than the one the
/// recognizer actually read.
const ROTATED_CROP_ASPECT: f32 = 1.5;

/// Normalize a detector polygon to a quad ordered along the *reading*
/// direction: `[start-top, end-top, end-bottom, start-bottom]`.
///
/// The classic sum/difference trick finds the four extreme corners and is
/// exact for any tilt below 45°, but it always labels them as though the
/// text were upright — so for a 90°-rotated region the "top edge" comes
/// out as the short side. Everything downstream reads the top edge as the
/// text's advance direction: `quad_slice` would then carve per-character
/// boxes across the *height* of the line, collapsing every glyph into a
/// one-pixel-wide, full-height sliver.
///
/// oar-ocr resolves the same ambiguity in `get_rotate_crop_image` by
/// applying `rotate270` (90° counter-clockwise) to any crop at least 1.5×
/// taller than wide. Tracing those corners through that rotation, the
/// upright image the recognizer saw has `tr` at its top-left and `br` at
/// its top-right — so the reading-order quad is the label set rotated by
/// one position, and the text runs downward along the right edge.
///
/// Polygons with more than 4 points (curved-text detectors) are reduced to
/// their 4 extreme corners — every point participates, not just the first
/// four.
fn normalize_quad(points: &[oar_ocr::processors::Point]) -> Option<Quad> {
    if points.len() < 4 {
        return None;
    }
    let pts: Vec<[f32; 2]> = points.iter().map(|p| [p.x, p.y]).collect();
    let tl = *pts
        .iter()
        .min_by(|a, b| (a[0] + a[1]).total_cmp(&(b[0] + b[1])))?;
    let br = *pts
        .iter()
        .max_by(|a, b| (a[0] + a[1]).total_cmp(&(b[0] + b[1])))?;
    let tr = *pts
        .iter()
        .max_by(|a, b| (a[0] - a[1]).total_cmp(&(b[0] - b[1])))?;
    let bl = *pts
        .iter()
        .min_by(|a, b| (a[0] - a[1]).total_cmp(&(b[0] - b[1])))?;

    let width = edge_len(tl, tr).max(edge_len(bl, br));
    let height = edge_len(tl, bl).max(edge_len(tr, br));
    if height >= width * ROTATED_CROP_ASPECT {
        // Rotated: re-label so the long edge becomes the advance edge.
        Some([tr, br, bl, tl])
    } else {
        Some([tl, tr, br, bl])
    }
}

fn edge_len(a: [f32; 2], b: [f32; 2]) -> f32 {
    ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt()
}

// `geometry::quad_slice` re-attaches tilt to the CTC char boxes, which
// oar-ocr computes against the axis-aligned bound only.
use crate::geometry::quad_slice;

/// Per-`char` `[t0, t1]` positions (fraction of the region's reading
/// extent) from oar-ocr's CTC char boxes. Returns `None` when boxes are
/// missing or don't match the char count — the caller synthesizes.
fn char_t_ranges(
    text: &str,
    word_boxes: Option<&Vec<oar_ocr::processors::BoundingBox>>,
    region_bbox: &oar_ocr::processors::BoundingBox,
) -> Option<Vec<[f32; 2]>> {
    let boxes = word_boxes?;
    let char_count = text.chars().count();
    if boxes.len() != char_count || char_count == 0 {
        return None;
    }
    let x_min = region_bbox.x_min();
    let width = (region_bbox.x_max() - x_min).max(1e-3);
    Some(
        boxes
            .iter()
            .map(|b| {
                let t0 = ((b.x_min() - x_min) / width).clamp(0.0, 1.0);
                let t1 = ((b.x_max() - x_min) / width).clamp(t0, 1.0);
                [t0, t1]
            })
            .collect(),
    )
}

/// A grapheme cluster with its `[t0, t1]` extent along the region.
struct ClusterT {
    text: String,
    t: [f32; 2],
    is_whitespace: bool,
}

/// Group per-`char` t-ranges into grapheme clusters (cluster extent =
/// min..max of member chars).
fn cluster_t_ranges(text: &str, char_ts: &[[f32; 2]]) -> Vec<ClusterT> {
    let mut clusters = Vec::new();
    let mut char_idx = 0usize;
    for g in text.graphemes(true) {
        let n = g.chars().count();
        let members = &char_ts[char_idx..(char_idx + n).min(char_ts.len())];
        let t0 = members.iter().map(|t| t[0]).fold(f32::MAX, f32::min);
        let t1 = members.iter().map(|t| t[1]).fold(f32::MIN, f32::max);
        clusters.push(ClusterT {
            text: g.to_string(),
            t: [t0, t1],
            is_whitespace: g.chars().all(char::is_whitespace),
        });
        char_idx += n;
    }
    clusters
}

/// Split clusters into whitespace-delimited word `OcrBlock`s, each
/// carrying its cluster spans sliced out of the region quad.
fn word_blocks(
    clusters: &[ClusterT],
    region_quad: &Quad,
    confidence: f32,
    rotation_deg: f32,
    synthesized: bool,
) -> Vec<OcrBlock> {
    let mut blocks = Vec::new();
    let mut word: Vec<&ClusterT> = Vec::new();
    let mut flush = |word: &mut Vec<&ClusterT>| {
        if word.is_empty() {
            return;
        }
        let t0 = word.iter().map(|c| c.t[0]).fold(f32::MAX, f32::min);
        let t1 = word.iter().map(|c| c.t[1]).fold(f32::MIN, f32::max);
        let quad = quad_slice(region_quad, t0, t1);
        blocks.push(OcrBlock {
            bbox: geometry::bbox_i32_of(&quad),
            text: word.iter().map(|c| c.text.as_str()).collect(),
            confidence,
            rotation_deg,
            quad: Some(quad),
            chars: word
                .iter()
                .map(|c| CharSpan {
                    text: c.text.clone(),
                    quad: quad_slice(region_quad, c.t[0], c.t[1]),
                })
                .collect(),
            chars_synthesized: synthesized,
        });
        word.clear();
    };
    for cluster in clusters {
        if cluster.is_whitespace {
            flush(&mut word);
        } else {
            word.push(cluster);
        }
    }
    flush(&mut word);
    blocks
}

/// Uniform per-cluster t-ranges for regions whose char boxes are
/// missing or inconsistent: each cluster gets an equal slice of the
/// region's reading extent.
fn synthesized_clusters(text: &str) -> Vec<ClusterT> {
    let graphemes: Vec<&str> = text.graphemes(true).collect();
    let n = graphemes.len().max(1) as f32;
    graphemes
        .iter()
        .enumerate()
        .map(|(i, g)| ClusterT {
            text: (*g).to_string(),
            t: [i as f32 / n, (i + 1) as f32 / n],
            is_whitespace: g.chars().all(char::is_whitespace),
        })
        .collect()
}

// EP factory helpers — `#[allow(dead_code)]` on the ones that a given
// target doesn't use avoids target-specific `unused` warnings without
// making the code conditional.
#[allow(dead_code)]
fn dml_provider() -> OrtExecutionProvider {
    OrtExecutionProvider::DirectML { device_id: None }
}
#[allow(dead_code)]
fn cuda_provider() -> OrtExecutionProvider {
    OrtExecutionProvider::CUDA {
        device_id: None,
        gpu_mem_limit: None,
        arena_extend_strategy: None,
        cudnn_conv_algo_search: None,
        cudnn_conv_use_max_workspace: None,
    }
}
#[allow(dead_code)]
fn coreml_provider() -> OrtExecutionProvider {
    OrtExecutionProvider::CoreML {
        ane_only: None,
        subgraphs: None,
    }
}

/// How one page's orientation is decided.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PageOrientation {
    /// Use the page-orientation classifier when
    /// [`OcrTuning::detect_page_orientation`] enabled it; otherwise upright.
    #[default]
    Auto,
    /// Skip the classifier and treat the page as rotated by this many
    /// quarter turns (0 = upright, 1 = 90°, 2 = 180°, 3 = 270°). Geometry is
    /// still mapped back to the original page, exactly as for `Auto`.
    Forced(u8),
}

/// Per-page overrides. `Default` means "as the engine was built".
///
/// These are what make a failed page *retryable*: the same loaded models
/// can re-run it with a smaller recognizer batch, or the other way up.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PageOptions {
    /// Recognizer batch size for this page only. The padded-area budget
    /// scales with it (see [`OcrTuning::region_ref_width`]), so halving
    /// this halves peak recognizer memory whatever the page looks like.
    pub region_batch_size: Option<usize>,
    pub orientation: PageOrientation,
}

/// One page's OCR result plus what the engine decided about the page.
#[derive(Debug, Clone, Default)]
pub struct PageOcr {
    pub blocks: Vec<OcrBlock>,
    /// Page rotation that was corrected before detection, in degrees
    /// (0/90/180/270). `None` when no orientation step ran or the
    /// classifier's verdict was too weak to act on.
    pub page_angle_deg: Option<f32>,
    /// Text regions the detector found but the recognizer returned no text
    /// for (dropped from `blocks`). A recognizer whose dictionary lacks the
    /// page's script — the broad Latin/CJK model on Cyrillic — empties
    /// those lines instead of misreading them, which no confidence figure
    /// over `blocks` can show.
    pub unread_regions: usize,
}

impl PageOcr {
    /// Mean block confidence; `None` for a page with no text. A cheap
    /// signal that a page *may* be the wrong way up — measured: Devanagari
    /// book pages 0.85–0.91 read correctly vs 0.48–0.52 upside-down, chat
    /// screenshots ~0.97 vs 0.62–0.70. It weighs a one-letter block like a
    /// forty-letter one, so it must not *choose* between two reads; that is
    /// [`Self::read_quality`]'s job.
    pub fn mean_confidence(&self) -> Option<f32> {
        if self.blocks.is_empty() {
            return None;
        }
        let sum: f32 = self.blocks.iter().map(|b| b.confidence).sum();
        Some(sum / self.blocks.len() as f32)
    }

    /// How much text this read recovered, and how confidently — the
    /// yardstick for choosing between two reads of the same page.
    pub fn read_quality(&self) -> ReadQuality {
        self.blocks
            .iter()
            .fold(ReadQuality::default(), |acc, block| {
                let graphemes = block
                    .text
                    .graphemes(true)
                    .filter(|g| !g.chars().all(char::is_whitespace))
                    .count();
                ReadQuality {
                    graphemes: acc.graphemes + graphemes,
                    score: acc.score + graphemes as f32 * block.confidence,
                }
            })
    }
}

/// Text-weighted summary of one read of a page (see
/// [`PageOcr::read_quality`]).
///
/// Weighting by recognised characters is what makes two reads comparable: a
/// read that found two confident fragments has not "read the page better"
/// than one that found forty lines at slightly lower confidence.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ReadQuality {
    /// Recognised grapheme clusters, whitespace excluded.
    pub graphemes: usize,
    /// Σ over blocks of grapheme count × block confidence: "confidently
    /// read characters". 0 for a read with no text.
    pub score: f32,
}

impl ReadQuality {
    /// Confidence per recognised character; 0 for a read with no text.
    pub fn char_confidence(self) -> f32 {
        if self.graphemes == 0 {
            0.0
        } else {
            self.score / self.graphemes as f32
        }
    }
}

/// OCR backend: anything that turns a full-page image into a list of
/// recognised text blocks.
pub trait OcrBackend: Send {
    fn ocr(&mut self, img: &DynamicImage) -> Result<Vec<OcrBlock>>;

    /// OCR one page with per-page overrides. Backends that cannot honour
    /// them fall back to [`OcrBackend::ocr`].
    fn ocr_page(&mut self, img: &DynamicImage, _opts: &PageOptions) -> Result<PageOcr> {
        Ok(PageOcr {
            blocks: self.ocr(img)?,
            page_angle_deg: None,
            unread_regions: 0,
        })
    }
}

/// Tunable knobs for the OCR engine. All fields default to "let oar-ocr
/// pick" except where a GPU-safe value is hard-coded (region_batch_size
/// = 64 and region_ref_width = 896 on GPU). Set explicitly to override.
#[derive(Debug, Clone, Copy, Default)]
pub struct OcrTuning {
    /// Recognizer batch size. None → auto (64 on GPU, oar-ocr's own on CPU).
    pub region_batch_size: Option<usize>,
    /// Reference crop width (px at the recognizer's 48 px input height) for
    /// the padded-area budget: a recognition batch may hold
    /// `region_batch_size` crops only while padded to at most this width;
    /// wider batches hold proportionally fewer. None → auto
    /// ([`OarOcrBackend::DEFAULT_GPU_REGION_REF_WIDTH`] on GPU, off on
    /// CPU). `Some(0)` → off: count-only batching, which runs out of GPU
    /// memory on pages that mix short and very long lines.
    pub region_ref_width: Option<usize>,
    /// Detector batch size. None → oar-ocr's own default.
    pub image_batch_size: Option<usize>,
    /// Enable per-line orientation classification (0° vs 180° only —
    /// the PP-LCNet textline model has 2 classes, so it corrects
    /// upside-down lines but cannot detect sideways text).
    /// Requires `ModelPaths::textline_ori_onnx` to be Some.
    pub detect_line_orientation: bool,
    /// Enable whole-page orientation classification (0/90/180/270).
    /// oar-ocr rotates the page upright before detection and maps all
    /// geometry (incl. per-char boxes) back to original coordinates,
    /// so rotated tables detect at native quality. Requires
    /// `ModelPaths::doc_ori_onnx` to be Some.
    pub detect_page_orientation: bool,
}

/// Type-erased handle used by the rest of the pipeline.
pub struct OcrEngine {
    inner: Box<dyn OcrBackend>,
}

impl OcrEngine {
    /// Construct an engine backed by oar-ocr (ort/ONNX Runtime), CPU only.
    /// Kept for back-compat and simple tests.
    pub fn oar_ocr(paths: &ModelPaths) -> Result<Self> {
        Self::oar_ocr_with_device(paths, OcrDevice::Cpu)
    }

    /// Construct an engine backed by oar-ocr on the requested device.
    pub fn oar_ocr_with_device(paths: &ModelPaths, device: OcrDevice) -> Result<Self> {
        Self::oar_ocr_with_tuning(paths, device, OcrTuning::default())
    }

    /// Plan-4.5 constructor — full control over batch sizes etc.
    pub fn oar_ocr_with_tuning(
        paths: &ModelPaths,
        device: OcrDevice,
        tuning: OcrTuning,
    ) -> Result<Self> {
        Ok(Self {
            inner: Box::new(OarOcrBackend::from_paths_tuned(paths, device, tuning)?),
        })
    }

    /// Construct with any caller-supplied backend.
    pub fn with_backend(backend: Box<dyn OcrBackend>) -> Self {
        Self { inner: backend }
    }

    /// Default constructor — CPU oar-ocr (matches Plan 1 behavior).
    pub fn from_paths(paths: &ModelPaths) -> Result<Self> {
        Self::oar_ocr(paths)
    }

    /// Preferred constructor in Plan 4+: pick the device explicitly.
    /// Defaults to default tuning; use `from_paths_tuned` for overrides.
    pub fn from_paths_with_device(paths: &ModelPaths, device: OcrDevice) -> Result<Self> {
        Self::oar_ocr_with_device(paths, device)
    }

    /// Plan-4.5 preferred constructor — device + tuning.
    pub fn from_paths_tuned(
        paths: &ModelPaths,
        device: OcrDevice,
        tuning: OcrTuning,
    ) -> Result<Self> {
        Self::oar_ocr_with_tuning(paths, device, tuning)
    }

    pub fn ocr(&mut self, img: &DynamicImage) -> Result<Vec<OcrBlock>> {
        self.inner.ocr(img)
    }

    /// OCR one page with per-page overrides — see [`PageOptions`].
    pub fn ocr_page(&mut self, img: &DynamicImage, opts: &PageOptions) -> Result<PageOcr> {
        self.inner.ocr_page(img, opts)
    }
}

pub struct OarOcrBackend {
    // The concrete type returned by OAROCRBuilder::build().
    inner: oar_ocr::oarocr::OAROCR,
}

impl OarOcrBackend {
    pub fn from_paths(paths: &ModelPaths) -> Result<Self> {
        Self::from_paths_with_device(paths, OcrDevice::Cpu)
    }

    pub fn from_paths_with_device(paths: &ModelPaths, device: OcrDevice) -> Result<Self> {
        Self::from_paths_tuned(paths, device, OcrTuning::default())
    }

    /// Default GPU region batch size. Plan 4 set this conservatively at
    /// 32 because DirectML crashed on dense pages. The Plan 4.5 sweep
    /// found this machine tolerates 64 reliably; higher requires VRAM
    /// the user opts into via `--region-batch-size`.
    ///
    /// That sweep only ever saw narrow lines. A batch is padded to its
    /// widest crop, so 64 is safe only together with
    /// [`Self::DEFAULT_GPU_REGION_REF_WIDTH`] — see Plan 10.
    pub const DEFAULT_GPU_REGION_BATCH: usize = 64;

    /// Default reference width for the padded-area budget on GPU: a batch
    /// holds the full region batch only while padded to ≤ this many px.
    ///
    /// Recognizer memory is `batch × (width / 8) × vocab × 4` bytes for the
    /// CTC logits alone, materialised ~3× (MatMul → Add → Softmax), and
    /// ONNX Runtime's DirectML allocator rounds every buffer up to a power
    /// of two. 896 keeps the largest vocabulary we ship (PP-OCRv6, 18,710
    /// classes) at 64 × 112 × 18,710 × 4 = 511.6 MiB — just inside the
    /// 512 MiB bucket. Measured cost is ~66 KiB per padded column on
    /// DirectML, so the default budget of 64 × 896 columns peaks near
    /// 3.6 GiB; each halving of the batch size halves that.
    pub const DEFAULT_GPU_REGION_REF_WIDTH: usize = 896;

    /// Page-orientation verdicts scoring below this are ignored. The
    /// classifier's score does *not* separate right from wrong (wrong
    /// verdicts reach 0.91), so this only filters out outright guesses;
    /// `resilient::ResilientEngine` does the real checking by comparing the
    /// two orientations' reads with [`PageOcr::read_quality`].
    const DOC_ORIENTATION_MIN_SCORE: f32 = 0.5;

    /// The region batch size an engine built with `tuning` on `device`
    /// starts from, or `None` when oar-ocr picks (CPU).
    pub fn effective_region_batch(device: OcrDevice, tuning: &OcrTuning) -> Option<usize> {
        tuning.region_batch_size.or_else(|| {
            device
                .is_accelerated()
                .then_some(Self::DEFAULT_GPU_REGION_BATCH)
        })
    }

    pub fn from_paths_tuned(
        paths: &ModelPaths,
        device: OcrDevice,
        tuning: OcrTuning,
    ) -> Result<Self> {
        let session_cfg =
            OrtSessionConfig::new().with_execution_providers(device.execution_providers());

        let mut builder = OAROCRBuilder::new(&paths.det_onnx, &paths.rec_onnx, &paths.keys_txt)
            .ort_session(session_cfg)
            // Plan 2.7: word-level bboxes for tight text-layer fit.
            .return_word_box(true);

        // Plan 7: optional whole-page orientation classifier.
        if tuning.detect_page_orientation {
            if let Some(ref doc_ori_path) = paths.doc_ori_onnx {
                builder = builder
                    .with_document_image_orientation_classification(doc_ori_path)
                    .document_orientation_min_score(Self::DOC_ORIENTATION_MIN_SCORE);
                tracing::info!(
                    model = %doc_ori_path.display(),
                    "document orientation classification enabled"
                );
            } else {
                tracing::warn!(
                    "detect_page_orientation requested but doc_ori_onnx model \
                     not found — page orientation classification will be skipped"
                );
            }
        }

        // Plan 2.8: optional text-line orientation classifier.
        if tuning.detect_line_orientation {
            if let Some(ref ori_path) = paths.textline_ori_onnx {
                builder = builder.with_text_line_orientation_classification(ori_path);
                tracing::info!(
                    model = %ori_path.display(),
                    "text-line orientation classification enabled"
                );
            } else {
                tracing::warn!(
                    "detect_line_orientation requested but textline_ori_onnx \
                     model not found — orientation classification will be skipped"
                );
            }
        }

        // Region batch: explicit override > GPU default (64) > CPU default (none).
        // GPU defaults exist because DirectML had unbounded-batch crashes on
        // dense pages in Plan 4 testing.
        if let Some(n) = Self::effective_region_batch(device, &tuning) {
            builder = builder.region_batch_size(n);
        }
        // Padded-area budget (Plan 10): explicit override > GPU default > off.
        // The count cap alone bounds the wrong quantity — one 50:1 line pads
        // every crop in its batch to ~2500 px and exhausts VRAM.
        let region_ref_width = tuning.region_ref_width.or_else(|| {
            device
                .is_accelerated()
                .then_some(Self::DEFAULT_GPU_REGION_REF_WIDTH)
        });
        if let Some(px) = region_ref_width {
            builder = builder.region_ref_width(px);
        }
        if let Some(n) = tuning.image_batch_size {
            builder = builder.image_batch_size(n);
        }

        let ocr = builder
            .build()
            .map_err(|e| OcrError::Engine(format!("oar-ocr build: {e}")))?;
        Ok(Self { inner: ocr })
    }
}

impl OcrBackend for OarOcrBackend {
    fn ocr(&mut self, img: &DynamicImage) -> Result<Vec<OcrBlock>> {
        Ok(self.ocr_page(img, &PageOptions::default())?.blocks)
    }

    fn ocr_page(&mut self, img: &DynamicImage, opts: &PageOptions) -> Result<PageOcr> {
        let predict_opts = PredictOptions {
            region_batch_size: opts.region_batch_size,
            region_ref_width: None,
            document_orientation: match opts.orientation {
                PageOrientation::Auto => DocumentOrientation::Auto,
                PageOrientation::Forced(turns) => DocumentOrientation::Forced(turns),
            },
        };
        // predict() takes Vec<RgbImage>; convert DynamicImage → RgbImage.
        let rgb = img.to_rgb8();
        let results = self
            .inner
            .predict_with(vec![rgb], &predict_opts)
            .map_err(|e| OcrError::Engine(format!("oar-ocr predict: {e}")))?;

        let Some(first) = results.into_iter().next() else {
            return Ok(PageOcr::default());
        };
        let page_angle_deg = first.orientation_angle;

        let mut blocks = Vec::with_capacity(first.text_regions.len());
        let mut synthesized_regions = 0usize;
        let mut unread_regions = 0usize;
        for region in first.text_regions.iter() {
            let raw_text = region.text.as_deref().unwrap_or("");
            if raw_text.trim().is_empty() {
                unread_regions += 1;
                continue;
            }
            // Fold fullwidth punctuation to ASCII on non-CJK regions. Done
            // here, before clustering, so every output format benefits —
            // including the searchable-PDF text layer, which `postproc`
            // deliberately skips. Safe at this point precisely because the
            // substitution is 1:1: the char count still matches the CTC box
            // count that `char_t_ranges` validates against just below.
            let normalized = crate::postproc::fullwidth::normalize(raw_text);
            let region_text: &str = &normalized;
            let confidence = region.confidence.unwrap_or(0.0);
            let rotation_deg = region.orientation_angle.unwrap_or(0.0);

            // Plan 5: lossless geometry. The detector's 4-point polygon
            // (region.bounding_box.points — previously read only through
            // axis-aligned accessors) becomes the region quad; the CTC
            // char boxes become per-grapheme-cluster t-slices of it, so
            // tilt survives and every cluster keeps its measured
            // position for the writer.
            let bb = &region.bounding_box;
            let region_quad = normalize_quad(&bb.points).unwrap_or_else(|| {
                geometry::quad_from_bbox([bb.x_min(), bb.y_min(), bb.x_max(), bb.y_max()])
            });

            let (clusters, synthesized) =
                match char_t_ranges(region_text, region.word_boxes.as_ref(), bb) {
                    Some(char_ts) => (cluster_t_ranges(region_text, &char_ts), false),
                    None => {
                        synthesized_regions += 1;
                        tracing::warn!(
                            text = %region_text.chars().take(24).collect::<String>(),
                            boxes = region.word_boxes.as_ref().map_or(0, Vec::len),
                            chars = region_text.chars().count(),
                            "char boxes missing/mismatched; synthesizing uniform spans"
                        );
                        (synthesized_clusters(region_text), true)
                    }
                };
            blocks.extend(word_blocks(
                &clusters,
                &region_quad,
                confidence,
                rotation_deg,
                synthesized,
            ));
        }
        tracing::debug!(
            regions = first.text_regions.len(),
            synthesized_regions,
            unread_regions,
            "ocr page: char-box coverage"
        );
        Ok(PageOcr {
            blocks,
            page_angle_deg,
            unread_regions,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oar_ocr::processors::BoundingBox;

    fn aabb_quad(x1: f32, y1: f32, x2: f32, y2: f32) -> Quad {
        geometry::quad_from_bbox([x1, y1, x2, y2])
    }

    /// Per-char boxes mimicking oar-ocr CTC output across a region.
    fn char_boxes(region: [f32; 4], edges: &[[f32; 2]]) -> Vec<BoundingBox> {
        edges
            .iter()
            .map(|&[x0, x1]| BoundingBox::from_coords(x0, region[1], x1, region[3]))
            .collect()
    }

    #[test]
    fn word_split_uses_measured_char_boxes() {
        // "ab cd" across x 0..100, chars at uneven positions.
        let region = BoundingBox::from_coords(0.0, 10.0, 100.0, 40.0);
        let boxes = char_boxes(
            [0.0, 10.0, 100.0, 40.0],
            &[
                [0.0, 12.0],
                [12.0, 30.0],
                [30.0, 42.0],
                [42.0, 70.0],
                [70.0, 100.0],
            ],
        );
        let ts = char_t_ranges("ab cd", Some(&boxes), &region).expect("t ranges");
        let clusters = cluster_t_ranges("ab cd", &ts);
        let rquad = aabb_quad(0.0, 10.0, 100.0, 40.0);
        let blocks = word_blocks(&clusters, &rquad, 0.9, 0.0, false);

        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].text, "ab");
        assert_eq!(blocks[1].text, "cd");
        // "ab" spans x 0..30 (measured), not half the region.
        assert_eq!(blocks[0].bbox, [0, 10, 30, 40]);
        // "cd" spans x 42..100.
        assert_eq!(blocks[1].bbox, [42, 10, 100, 40]);
        // Per-cluster geometry survives.
        assert_eq!(blocks[0].chars.len(), 2);
        assert_eq!(blocks[0].chars[0].text, "a");
        assert_eq!(
            geometry::bbox_i32_of(&blocks[0].chars[1].quad),
            [12, 10, 30, 40]
        );
        assert!(!blocks[0].chars_synthesized);
    }

    #[test]
    fn devanagari_cluster_groups_base_and_matra() {
        // "कि" = क (base) + ि (matra): 2 chars, 1 grapheme cluster.
        let region = BoundingBox::from_coords(0.0, 0.0, 60.0, 30.0);
        let boxes = char_boxes([0.0, 0.0, 60.0, 30.0], &[[0.0, 30.0], [30.0, 60.0]]);
        let ts = char_t_ranges("कि", Some(&boxes), &region).expect("t ranges");
        let clusters = cluster_t_ranges("कि", &ts);
        assert_eq!(clusters.len(), 1, "one grapheme cluster");
        let blocks = word_blocks(&clusters, &aabb_quad(0.0, 0.0, 60.0, 30.0), 0.9, 0.0, false);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].chars.len(), 1);
        assert_eq!(blocks[0].chars[0].text, "कि");
        // Cluster quad = union of both char slices.
        assert_eq!(
            geometry::bbox_i32_of(&blocks[0].chars[0].quad),
            [0, 0, 60, 30]
        );
    }

    #[test]
    fn cjk_text_is_one_block_with_per_char_spans() {
        let region = BoundingBox::from_coords(0.0, 0.0, 90.0, 30.0);
        let boxes = char_boxes(
            [0.0, 0.0, 90.0, 30.0],
            &[[0.0, 30.0], [30.0, 60.0], [60.0, 90.0]],
        );
        let ts = char_t_ranges("中国人", Some(&boxes), &region).expect("t ranges");
        let clusters = cluster_t_ranges("中国人", &ts);
        let blocks = word_blocks(&clusters, &aabb_quad(0.0, 0.0, 90.0, 30.0), 0.9, 0.0, false);
        assert_eq!(blocks.len(), 1, "no whitespace → one block");
        assert_eq!(blocks[0].chars.len(), 3);
        assert_eq!(
            geometry::bbox_i32_of(&blocks[0].chars[1].quad),
            [30, 0, 60, 30]
        );
    }

    #[test]
    fn mismatched_boxes_fall_back_to_synthesized() {
        let region = BoundingBox::from_coords(0.0, 0.0, 100.0, 20.0);
        // 2 boxes for 5 chars → mismatch.
        let boxes = char_boxes([0.0, 0.0, 100.0, 20.0], &[[0.0, 50.0], [50.0, 100.0]]);
        assert!(char_t_ranges("hello", Some(&boxes), &region).is_none());
        assert!(char_t_ranges("hello", None, &region).is_none());

        let clusters = synthesized_clusters("ab cd");
        let blocks = word_blocks(&clusters, &aabb_quad(0.0, 0.0, 100.0, 20.0), 0.5, 0.0, true);
        assert_eq!(blocks.len(), 2);
        assert!(blocks[0].chars_synthesized);
        // Uniform slots: "ab" = clusters 0,1 of 5 → x 0..40.
        assert_eq!(blocks[0].bbox, [0, 0, 40, 20]);
        assert_eq!(blocks[1].bbox, [60, 0, 100, 20]);
    }

    #[test]
    fn tilted_region_quad_slices_inherit_slope() {
        // Parallelogram tilted: top edge y=0→8, bottom y=30→38.
        let rquad: Quad = [[0.0, 0.0], [100.0, 8.0], [100.0, 38.0], [0.0, 30.0]];
        let slice = quad_slice(&rquad, 0.5, 1.0);
        // Top-left of slice = midpoint of top edge = (50, 4).
        assert!((slice[0][0] - 50.0).abs() < 1e-3);
        assert!((slice[0][1] - 4.0).abs() < 1e-3);
        // Bottom-right = (100, 38).
        assert!((slice[2][0] - 100.0).abs() < 1e-3);
        assert!((slice[2][1] - 38.0).abs() < 1e-3);
    }

    #[test]
    fn normalize_quad_orders_clockwise_from_tl() {
        use oar_ocr::processors::Point;
        let pts = vec![
            Point::new(100.0, 38.0), // br
            Point::new(0.0, 0.0),    // tl
            Point::new(100.0, 8.0),  // tr
            Point::new(0.0, 30.0),   // bl
        ];
        let q = normalize_quad(&pts).expect("quad");
        assert_eq!(q, [[0.0, 0.0], [100.0, 8.0], [100.0, 38.0], [0.0, 30.0]]);
    }

    #[test]
    fn normalize_quad_puts_rotated_text_in_reading_order() {
        // A vertical line, 30 wide × 300 tall: oar-ocr rotates such a crop
        // 270° before recognition, so the text it read runs downward along
        // the right edge and the reading-order quad is [tr, br, bl, tl].
        use oar_ocr::processors::Point;
        let pts = vec![
            Point::new(100.0, 300.0), // bl
            Point::new(130.0, 0.0),   // tr
            Point::new(100.0, 0.0),   // tl
            Point::new(130.0, 300.0), // br
        ];
        let q = normalize_quad(&pts).expect("quad");
        assert_eq!(
            q,
            [[130.0, 0.0], [130.0, 300.0], [100.0, 300.0], [100.0, 0.0]]
        );

        // The advance edge is now the long one, so per-character slices run
        // down the line at full line thickness — not 1 px slivers across it.
        let first_half = geometry::bbox_i32_of(&quad_slice(&q, 0.0, 0.5));
        assert_eq!(first_half, [100, 0, 130, 150]);
    }

    #[test]
    fn normalize_quad_rotation_threshold_matches_oar_ocr() {
        use oar_ocr::processors::Point;
        let rect = |w: f32, h: f32| {
            vec![
                Point::new(0.0, 0.0),
                Point::new(w, 0.0),
                Point::new(w, h),
                Point::new(0.0, h),
            ]
        };
        // Exactly ROTATED_CROP_ASPECT (h >= 1.5 × w): rotated, like
        // `get_rotate_crop_image`.
        let at = normalize_quad(&rect(20.0, 30.0)).expect("quad");
        assert_eq!(at, [[20.0, 0.0], [20.0, 30.0], [0.0, 30.0], [0.0, 0.0]]);
        // Just under it: left upright.
        let under = normalize_quad(&rect(20.0, 29.0)).expect("quad");
        assert_eq!(under, [[0.0, 0.0], [20.0, 0.0], [20.0, 29.0], [0.0, 29.0]]);
    }

    #[test]
    fn read_quality_weights_confidence_by_recognised_characters() {
        let block = |text: &str, confidence: f32| OcrBlock {
            text: text.into(),
            confidence,
            ..OcrBlock::default()
        };
        let page = PageOcr {
            // "कि" is two chars but one grapheme cluster; whitespace is not
            // a recognised character.
            blocks: vec![block("abcd", 0.5), block("कि", 1.0), block(" a ", 1.0)],
            ..PageOcr::default()
        };
        let q = page.read_quality();
        assert_eq!(q.graphemes, 6);
        assert!((q.score - 4.0).abs() < 1e-6, "4 × 0.5 + 1 × 1.0 + 1 × 1.0");
        assert!((q.char_confidence() - 4.0 / 6.0).abs() < 1e-6);
        // The per-block mean would call this page 0.83; per character it is
        // 0.67, because the long block is the poorly-read one.
        let mean = page.mean_confidence().expect("has text");
        assert!((mean - 2.5 / 3.0).abs() < 1e-6);

        let empty = PageOcr::default().read_quality();
        assert_eq!(empty, ReadQuality::default());
        assert_eq!(empty.char_confidence(), 0.0);
    }
}
