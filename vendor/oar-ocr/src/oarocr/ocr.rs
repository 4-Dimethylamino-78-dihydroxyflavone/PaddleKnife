//! High-level OCR builder API.
//!
//! This module provides `OAROCRBuilder` for constructing OCR pipelines with a fluent API.
//! It simplifies the process of configuring text detection, recognition, and optional
//! preprocessing components.

// PaddleKnife patch (see vendor/oar-ocr/PADDLEKNIFE_PATCHES.md):
//  * recognition batches are bounded by padded *area* as well as crop count
//    (`region_ref_width`, `plan_recognition_chunks`);
//  * per-call overrides via `PredictOptions` / `OAROCR::predict_with`;
//  * word-box ratios are clamped to the recognizer's hard width cap;
//  * optional score floor for the document-orientation classifier.

use super::builder_utils::{
    build_optional_adapter, default_cpu_region_batch_size, resolve_device_batch_sizes,
    resolve_model_path, resolve_model_source,
};
use oar_ocr_core::core::ModelSource;
use oar_ocr_core::core::config::OrtSessionConfig;
use oar_ocr_core::core::constants::{DEFAULT_MAX_IMG_WIDTH, DEFAULT_REC_IMAGE_SHAPE};
use oar_ocr_core::core::errors::OCRError;
use oar_ocr_core::core::traits::OrtConfigurable;
use oar_ocr_core::core::traits::adapter::{AdapterBuilder, ModelAdapter};
use oar_ocr_core::core::traits::task::ImageTaskInput;
use oar_ocr_core::domain::adapters::{
    DocumentOrientationAdapter, DocumentOrientationAdapterBuilder, TextDetectionAdapter,
    TextDetectionAdapterBuilder, TextLineOrientationAdapter, TextLineOrientationAdapterBuilder,
    TextRecognitionAdapter, TextRecognitionAdapterBuilder, UVDocRectifierAdapter,
    UVDocRectifierAdapterBuilder,
};
use oar_ocr_core::domain::tasks::{TextDetectionConfig, TextRecognitionConfig};
use oar_ocr_core::processors::BoundingBox;
use std::ops::Range;
use std::path::PathBuf;
use std::sync::Arc;

pub use super::preprocess::DocumentOrientation;

/// Per-call overrides for [`OAROCR::predict_with`]. `Default` reproduces
/// [`OAROCR::predict`] exactly.
///
/// These exist so a caller can *retry* a page under tighter memory limits, or
/// at a different page orientation, without rebuilding the ONNX sessions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PredictOptions {
    /// Cap on crops per recognition batch. `None` = the builder's value.
    pub region_batch_size: Option<usize>,
    /// Reference crop width for the padded-area budget; see
    /// [`OAROCRBuilder::region_ref_width`]. `None` = the builder's value,
    /// `Some(0)` = no area budget (count-only batching, as upstream).
    pub region_ref_width: Option<usize>,
    /// How to decide the page orientation for this call.
    pub document_orientation: DocumentOrientation,
}

/// Width, in recognizer input pixels, of the tensor a batch is padded to when
/// its widest crop has aspect ratio `wh_ratio`. Mirrors `CRNNModel::preprocess`:
/// never narrower than the model's base width, never wider than its hard cap.
fn rec_tensor_width(wh_ratio: f32) -> usize {
    let [_, img_h, base_w] = DEFAULT_REC_IMAGE_SHAPE;
    let natural = (img_h as f32 * wh_ratio) as usize;
    natural.clamp(base_w, DEFAULT_MAX_IMG_WIDTH)
}

/// Clamps a crop's width/height ratio to what the recognizer can represent.
///
/// `CRNNModel::preprocess` squeezes any crop wider than its hard cap into the
/// cap, so past this ratio extra width no longer widens the tensor. The
/// word-box math undoes batch padding as `wh_ratio / max_wh_ratio`; both terms
/// must be clamped the same way or the boxes of a crop that shares a batch
/// with an over-cap line are scaled against a tensor wider than the real one.
fn clamp_rec_ratio(wh_ratio: f32) -> f32 {
    wh_ratio.min(DEFAULT_MAX_IMG_WIDTH as f32 / DEFAULT_REC_IMAGE_SHAPE[1] as f32)
}

/// Splits crops — already sorted by ascending width/height ratio — into
/// recognition batches.
///
/// A batch is padded to its *widest* member, so its memory cost is
/// `count x tensor_width`, not `count`. With a fixed count cap, one 50:1 line
/// drags every other crop in its batch up to a ~2500 px tensor; on an
/// 18k-class CTC head that is the difference between ~0.3 GiB and ~9 GiB of
/// activations, and it is what exhausts GPU memory on pages that mix short and
/// very long lines.
///
/// Each batch therefore satisfies both
///   * `count <= max_count`, and
///   * `count x tensor_width(widest) <= column_budget` (when a budget is set),
/// except that a batch always holds at least one crop. Because the input is
/// sorted ascending, the widest member of a batch is always its last, which
/// makes a single greedy pass optimal: narrow crops still fill `max_count`,
/// while wide ones are isolated into proportionally smaller batches.
fn plan_recognition_chunks(
    wh_ratios: &[f32],
    max_count: usize,
    column_budget: Option<usize>,
) -> Vec<Range<usize>> {
    let max_count = max_count.max(1);
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < wh_ratios.len() {
        let mut end = start + 1;
        while end < wh_ratios.len() && end - start < max_count {
            let padded_columns = (end - start + 1) * rec_tensor_width(wh_ratios[end]);
            if column_budget.is_some_and(|budget| padded_columns > budget) {
                break;
            }
            end += 1;
        }
        chunks.push(start..end);
        start = end;
    }
    chunks
}

/// Internal structure holding the OCR pipeline adapters.
#[derive(Debug)]
struct OCRPipeline {
    rectification_adapter: Option<UVDocRectifierAdapter>,
    document_orientation_adapter: Option<DocumentOrientationAdapter>,
    text_detection_adapter: TextDetectionAdapter,
    text_line_orientation_adapter: Option<TextLineOrientationAdapter>,
    text_recognition_adapter: TextRecognitionAdapter,
}

/// Builder for constructing OCR pipelines.
///
/// This builder provides a high-level API for configuring text detection and recognition
/// pipelines with optional preprocessing components like orientation classification and
/// image rectification.
///
/// # Example
///
/// ```no_run
/// use oar_ocr::oarocr::OAROCRBuilder;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let ocr = OAROCRBuilder::new(
///     "path/to/text_detection.onnx",
///     "path/to/text_recognition.onnx",
///     "path/to/character_dict.txt"
/// )
/// .with_document_image_orientation_classification("path/to/orientation.onnx")
/// .with_text_line_orientation_classification("path/to/line_orientation.onnx")
/// .image_batch_size(4)
/// .region_batch_size(32)
/// .build()?;
/// # let _ = ocr;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct OAROCRBuilder {
    // Required fields
    text_detection_model: ModelSource,
    text_recognition_model: ModelSource,
    character_dict_path: PathBuf,
    character_dict_content: Option<String>,

    // Optional components
    document_orientation_model: Option<ModelSource>,
    text_line_orientation_model: Option<ModelSource>,
    document_rectification_model: Option<ModelSource>,

    // Configuration
    ort_session_config: Option<OrtSessionConfig>,
    text_detection_config: Option<TextDetectionConfig>,
    text_recognition_config: Option<TextRecognitionConfig>,
    image_batch_size: Option<usize>,
    region_batch_size: Option<usize>,
    region_ref_width: Option<usize>,
    document_orientation_min_score: f32,

    // Text type and word box options
    text_type: Option<String>,
    return_word_box: bool,
}

impl OAROCRBuilder {
    // Guardrail against pathological user input. This is intentionally generous and
    // not a model-tuned throughput limit.
    const MAX_BATCH_SIZE: usize = 4096;

    /// Creates a new OCR builder with required components.
    ///
    /// # Arguments
    ///
    /// * `text_detection_model` - Text detection ONNX model: a path, or raw
    ///   model bytes (e.g. from `include_bytes!`)
    /// * `text_recognition_model` - Text recognition ONNX model: a path or
    ///   raw model bytes
    /// * `character_dict_path` - Path to the character dictionary file (see
    ///   [`Self::character_dict_content`] for the in-memory alternative)
    pub fn new(
        text_detection_model: impl Into<ModelSource>,
        text_recognition_model: impl Into<ModelSource>,
        character_dict_path: impl Into<PathBuf>,
    ) -> Self {
        Self {
            text_detection_model: text_detection_model.into(),
            text_recognition_model: text_recognition_model.into(),
            character_dict_path: character_dict_path.into(),
            character_dict_content: None,
            document_orientation_model: None,
            text_line_orientation_model: None,
            document_rectification_model: None,
            ort_session_config: None,
            text_detection_config: None,
            text_recognition_config: None,
            image_batch_size: None,
            region_batch_size: None,
            region_ref_width: None,
            document_orientation_min_score: 0.0,
            text_type: None,
            return_word_box: false,
        }
    }

    /// Sets the character dictionary from an in-memory string (e.g. from
    /// `include_str!`). When set, `character_dict_path` is ignored.
    pub fn character_dict_content(mut self, content: impl Into<String>) -> Self {
        self.character_dict_content = Some(content.into());
        self
    }

    /// Sets the ONNX Runtime session configuration.
    ///
    /// This configuration will be applied to all models in the pipeline.
    pub fn ort_session(mut self, config: OrtSessionConfig) -> Self {
        self.ort_session_config = Some(config);
        self
    }

    /// Sets the text detection model configuration.
    ///
    /// Controls text detection preprocessing and postprocessing.
    pub fn text_detection_config(mut self, config: TextDetectionConfig) -> Self {
        self.text_detection_config = Some(config);
        self
    }

    /// Sets the text recognition model configuration.
    ///
    /// Controls text recognition preprocessing and postprocessing.
    pub fn text_recognition_config(mut self, config: TextRecognitionConfig) -> Self {
        self.text_recognition_config = Some(config);
        self
    }

    /// Sets the batch size for processing input images during text detection.
    ///
    /// This controls how many images are sent to the text detection adapter per call.
    /// If a detector cannot batch the provided images (e.g., mismatched sizes), the
    /// pipeline falls back to per-image detection. Values are validated in `build()`
    /// and must be within `1..=MAX_BATCH_SIZE`. When unset, CPU execution uses `1`;
    /// explicitly configured accelerators use the adapter's throughput default.
    pub fn image_batch_size(mut self, size: usize) -> Self {
        self.image_batch_size = Some(size);
        self
    }

    /// Sets the batch size for processing detected text regions.
    ///
    /// Controls memory usage during text recognition. Smaller values use less memory.
    /// When unset, CPU execution uses `16` for PP-OCRv6 Tiny and `4` for other
    /// models; explicitly configured accelerators use the adapter's throughput
    /// default (`64`). Values are validated in `build()` and must be within
    /// `1..=MAX_BATCH_SIZE`.
    pub fn region_batch_size(mut self, size: usize) -> Self {
        self.region_batch_size = Some(size);
        self
    }

    /// Bounds each recognition batch by padded *area*, not just crop count.
    ///
    /// The budget is `region_batch_size x ref_width` tensor columns: a batch
    /// may hold the full `region_batch_size` crops only while it is padded to
    /// at most `ref_width` pixels; wider batches hold proportionally fewer
    /// (e.g. with 64 x 896, a batch padded to 2480 px holds at most 23 crops).
    /// Halving `region_batch_size` therefore halves peak recognizer memory for
    /// every page shape, which a count cap alone does not.
    ///
    /// `0` disables the area budget. Unset keeps upstream's count-only batching.
    pub fn region_ref_width(mut self, ref_width: usize) -> Self {
        self.region_ref_width = Some(ref_width);
        self
    }

    /// Minimum classifier score for a document-orientation verdict to be
    /// applied; lower-scoring verdicts leave the page untouched. Default `0.0`
    /// (top-1 always wins, as upstream).
    pub fn document_orientation_min_score(mut self, min_score: f32) -> Self {
        self.document_orientation_min_score = min_score;
        self
    }

    /// Adds document image orientation classification to the pipeline.
    ///
    /// This component detects and corrects document orientation before text detection.
    pub fn with_document_image_orientation_classification(
        mut self,
        model_source: impl Into<ModelSource>,
    ) -> Self {
        self.document_orientation_model = Some(model_source.into());
        self
    }

    /// Adds text line orientation classification to the pipeline.
    ///
    /// This component detects and corrects text line orientation after text detection.
    pub fn with_text_line_orientation_classification(
        mut self,
        model_source: impl Into<ModelSource>,
    ) -> Self {
        self.text_line_orientation_model = Some(model_source.into());
        self
    }

    /// Adds document image rectification to the pipeline.
    ///
    /// This component corrects document distortion before text detection.
    pub fn with_document_image_rectification(
        mut self,
        model_source: impl Into<ModelSource>,
    ) -> Self {
        self.document_rectification_model = Some(model_source.into());
        self
    }

    /// Sets the text type for sorting and cropping strategy.
    ///
    /// This matches the text_type parameter:
    /// - "seal": Uses polygon-based sorting/cropping for seal text (circular/curved)
    /// - "table": Uses table-friendly detection defaults (box_threshold=0.4)
    /// - Other values or None: Uses quad-based sorting (default)
    ///
    /// # Arguments
    ///
    /// * `text_type` - Text type identifier ("seal", etc.)
    pub fn text_type(mut self, text_type: impl Into<String>) -> Self {
        self.text_type = Some(text_type.into());
        self
    }

    /// Enables word-level bounding box detection.
    ///
    /// When enabled, the pipeline will attempt to detect individual words
    /// within each text line and populate the `word_boxes` field in `TextRegion`.
    ///
    /// Note: This feature requires word-level detection support in the recognition model.
    ///
    /// # Arguments
    ///
    /// * `enable` - Whether to enable word box detection
    pub fn return_word_box(mut self, enable: bool) -> Self {
        self.return_word_box = enable;
        self
    }

    /// Builds the OCR runtime.
    ///
    /// This instantiates all adapters and returns an `OAROCR` instance ready for prediction.
    pub fn build(self) -> Result<OAROCR, OCRError> {
        if let Some(size) = self.image_batch_size {
            Self::validate_batch_size("image_batch_size", size)?;
        }
        if let Some(size) = self.region_batch_size {
            Self::validate_batch_size("region_batch_size", size)?;
        }

        // Resolve required model paths through the auto-download cache when
        // the feature is enabled. With the feature off these are no-ops.
        let text_detection_model = resolve_model_source(&self.text_detection_model)?;
        let text_recognition_model = resolve_model_source(&self.text_recognition_model)?;

        // CPU operators already fan work out across ORT's intra-op pool. Tiny's
        // much cheaper recognizer benefits from a wider batch, while larger
        // models regress beyond four on Windows. Accelerators retain their
        // throughput-oriented adapter defaults (8 detection / 64 recognition).
        let cpu_region_batch_size =
            default_cpu_region_batch_size(Some(&text_recognition_model), None);
        let (image_batch_size, region_batch_size) = resolve_device_batch_sizes(
            self.ort_session_config.as_ref(),
            self.image_batch_size,
            self.region_batch_size,
            1,
            cpu_region_batch_size,
        );

        // Load character dictionary for text recognition
        let char_dict = match &self.character_dict_content {
            Some(content) => content.clone(),
            None => {
                let character_dict_path = resolve_model_path(&self.character_dict_path)?;
                std::fs::read_to_string(&character_dict_path).map_err(|e| {
                    OCRError::InvalidInput {
                        message: format!(
                            "Failed to read character dictionary from '{}': {}",
                            character_dict_path.display(),
                            e
                        ),
                    }
                })?
            }
        };

        // Build document rectification adapter if enabled
        let rectification_adapter = build_optional_adapter(
            self.document_rectification_model.as_ref(),
            self.ort_session_config.as_ref(),
            UVDocRectifierAdapterBuilder::new,
        )?;

        // Build document orientation adapter if enabled
        let document_orientation_adapter = build_optional_adapter(
            self.document_orientation_model.as_ref(),
            self.ort_session_config.as_ref(),
            DocumentOrientationAdapterBuilder::new,
        )?;

        // Build text detection adapter (required)
        let mut detection_builder = TextDetectionAdapterBuilder::new();

        if let Some(ref ort_config) = self.ort_session_config {
            detection_builder = detection_builder.with_ort_config(ort_config.clone());
        }

        // Align text detection defaults with OCR pipeline.
        // Defaults depend on text_type:
        // - general: limit_side_len=960, limit_type="max", thresh=0.3, box_thresh=0.6, unclip_ratio=2.0
        // - table: limit_side_len=960, limit_type="max", thresh=0.3, box_thresh=0.4, unclip_ratio=2.0
        // - seal: limit_side_len=736, limit_type="min", thresh=0.2, box_thresh=0.6, unclip_ratio=0.5
        let mut effective_det_cfg = self.text_detection_config.clone().unwrap_or_default();
        let has_explicit_det_cfg = self.text_detection_config.is_some();
        if !has_explicit_det_cfg {
            match self.text_type.as_deref().unwrap_or("general") {
                "table" => {
                    effective_det_cfg.score_threshold = 0.3;
                    effective_det_cfg.box_threshold = 0.4;
                    effective_det_cfg.unclip_ratio = 2.0;
                    if effective_det_cfg.limit_side_len.is_none() {
                        effective_det_cfg.limit_side_len = Some(960);
                    }
                    if effective_det_cfg.limit_type.is_none() {
                        effective_det_cfg.limit_type = Some(crate::processors::LimitType::Max);
                    }
                    if effective_det_cfg.max_side_len.is_none() {
                        effective_det_cfg.max_side_len = Some(4000);
                    }
                }
                "seal" => {
                    effective_det_cfg.score_threshold = 0.2;
                    effective_det_cfg.box_threshold = 0.6;
                    effective_det_cfg.unclip_ratio = 0.5;
                    if effective_det_cfg.limit_side_len.is_none() {
                        effective_det_cfg.limit_side_len = Some(736);
                    }
                    if effective_det_cfg.limit_type.is_none() {
                        effective_det_cfg.limit_type = Some(crate::processors::LimitType::Min);
                    }
                    if effective_det_cfg.max_side_len.is_none() {
                        effective_det_cfg.max_side_len = Some(4000);
                    }
                }
                _ => {
                    effective_det_cfg.score_threshold = 0.3;
                    effective_det_cfg.box_threshold = 0.6;
                    effective_det_cfg.unclip_ratio = 2.0;
                    if effective_det_cfg.limit_side_len.is_none() {
                        effective_det_cfg.limit_side_len = Some(960);
                    }
                    if effective_det_cfg.limit_type.is_none() {
                        effective_det_cfg.limit_type = Some(crate::processors::LimitType::Max);
                    }
                    if effective_det_cfg.max_side_len.is_none() {
                        effective_det_cfg.max_side_len = Some(4000);
                    }
                }
            }
        }

        detection_builder = detection_builder.with_config(effective_det_cfg);

        // Pass text_type to detection adapter for proper preprocessing configuration
        if let Some(ref text_type) = self.text_type {
            detection_builder = detection_builder.text_type(text_type.clone());
        }

        let text_detection_adapter = detection_builder.build(text_detection_model)?;

        // Build text line orientation adapter if enabled
        let text_line_orientation_adapter = build_optional_adapter(
            self.text_line_orientation_model.as_ref(),
            self.ort_session_config.as_ref(),
            TextLineOrientationAdapterBuilder::new,
        )?;

        // Build text recognition adapter (required)
        // Parse char_dict into Vec<String> - one character per line
        let char_dict_vec: Vec<String> = char_dict.lines().map(|s| s.to_string()).collect();

        let mut recognition_builder = TextRecognitionAdapterBuilder::new()
            .character_dict(char_dict_vec)
            .return_word_box(self.return_word_box);

        if let Some(ref ort_config) = self.ort_session_config {
            recognition_builder = recognition_builder.with_ort_config(ort_config.clone());
        }

        if let Some(ref rec_config) = self.text_recognition_config {
            recognition_builder = recognition_builder.with_config(rec_config.clone());
        }

        let text_recognition_adapter = recognition_builder.build(text_recognition_model)?;

        let pipeline = OCRPipeline {
            rectification_adapter,
            document_orientation_adapter,
            text_detection_adapter,
            text_line_orientation_adapter,
            text_recognition_adapter,
        };

        Ok(OAROCR {
            pipeline,
            text_type: self.text_type,
            return_word_box: self.return_word_box,
            image_batch_size,
            region_batch_size,
            region_ref_width: self.region_ref_width,
            document_orientation_min_score: self.document_orientation_min_score,
        })
    }

    fn validate_batch_size(field: &str, size: usize) -> Result<(), OCRError> {
        if size == 0 || size > Self::MAX_BATCH_SIZE {
            return Err(OCRError::validation_error(
                "OAROCRBuilder",
                field,
                &format!("1..={}", Self::MAX_BATCH_SIZE),
                &size.to_string(),
            ));
        }

        Ok(())
    }
}

/// OCR runtime for executing text detection and recognition.
///
/// This struct represents a configured OCR pipeline that can process images
/// to extract text.
#[derive(Debug)]
pub struct OAROCR {
    pipeline: OCRPipeline,
    text_type: Option<String>,
    return_word_box: bool,
    /// Text detection batch size for `predict(images)`.
    ///
    /// This controls how many preprocessed images are sent to the text detection adapter in a
    /// single call. CPU builders resolve this to `1`; accelerator builders leave it as `None`
    /// so the adapter's `recommended_batch_size()` is used.
    image_batch_size: Option<usize>,
    /// Batch size for text region recognition
    region_batch_size: Option<usize>,
    /// Reference width for the padded-area budget (`None`/`0` = count-only).
    region_ref_width: Option<usize>,
    /// Score floor for document-orientation verdicts.
    document_orientation_min_score: f32,
}

struct CroppedTextRegion {
    detection_index: usize,
    bbox: BoundingBox,
    image: Arc<image::RgbImage>,
    wh_ratio: f32,
    line_orientation_angle: Option<f32>,
}

impl std::fmt::Debug for CroppedTextRegion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CroppedTextRegion")
            .field("detection_index", &self.detection_index)
            .field("bbox", &self.bbox)
            .field(
                "image",
                &format_args!("RgbImage({}x{})", self.image.width(), self.image.height()),
            )
            .field("wh_ratio", &self.wh_ratio)
            .field("line_orientation_angle", &self.line_orientation_angle)
            .finish()
    }
}

impl OAROCR {
    /// Predicts text from images using the configured OCR pipeline.
    ///
    /// This method orchestrates the execution of all configured tasks in the pipeline,
    /// including optional components like document orientation, rectification, and
    /// text line orientation classification.
    ///
    /// # Arguments
    ///
    /// * `images` - Collection of RGB images to process
    ///
    /// # Returns
    ///
    /// A vector of `OAROCRResult` containing the OCR results for each image,
    /// or an error if processing fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use oar_ocr::oarocr::ocr::OAROCRBuilder;
    /// use oar_ocr::utils::load_image;
    /// use std::path::Path;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let ocr = OAROCRBuilder::new(
    ///     "det.onnx",
    ///     "rec.onnx",
    ///     "dict.txt",
    /// ).build()?;
    ///
    /// let image = load_image(Path::new("document.jpg"))?;
    /// let results = ocr.predict(vec![image])?;
    ///
    /// for result in results {
    ///     for region in result.text_regions {
    ///         if let Some(text) = region.text {
    ///             println!("Text: {}", text);
    ///         }
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn predict(
        &self,
        images: Vec<image::RgbImage>,
    ) -> Result<Vec<crate::oarocr::OAROCRResult>, OCRError> {
        self.predict_with(images, &PredictOptions::default())
    }

    /// [`Self::predict`] with per-call overrides — see [`PredictOptions`].
    pub fn predict_with(
        &self,
        images: Vec<image::RgbImage>,
        options: &PredictOptions,
    ) -> Result<Vec<crate::oarocr::OAROCRResult>, OCRError> {
        use crate::oarocr::preprocess::DocumentPreprocessor;
        use std::sync::Arc;

        if let Some(size) = options.region_batch_size {
            OAROCRBuilder::validate_batch_size("region_batch_size", size)?;
        }

        if images.is_empty() {
            return Err(OCRError::validation_error(
                "OCR Pipeline",
                "images",
                "non-empty slice",
                "empty slice",
            ));
        }

        let preprocessor = DocumentPreprocessor::new(
            self.pipeline.document_orientation_adapter.as_ref(),
            self.pipeline.rectification_adapter.as_ref(),
        )
        .with_orientation_mode(options.document_orientation)
        .with_orientation_min_score(self.document_orientation_min_score);

        let mut prepared: Vec<(
            Arc<image::RgbImage>,
            crate::oarocr::preprocess::PreprocessResult,
        )> = Vec::with_capacity(images.len());

        for image in images.into_iter() {
            let input_img_arc = Arc::new(image);
            let preprocess = preprocessor.preprocess(Arc::clone(&input_img_arc))?;
            prepared.push((input_img_arc, preprocess));
        }

        let det_batch_size = self
            .image_batch_size
            .unwrap_or_else(|| {
                self.pipeline
                    .text_detection_adapter
                    .recommended_batch_size()
            })
            .max(1);

        let mut all_detection_boxes: Vec<Vec<BoundingBox>> = vec![Vec::new(); prepared.len()];

        let mut start = 0usize;
        while start < prepared.len() {
            let end = (start + det_batch_size).min(prepared.len());

            let batch_images: Vec<Arc<image::RgbImage>> = prepared[start..end]
                .iter()
                .map(|(_, preprocess)| Arc::clone(&preprocess.image))
                .collect();

            match self.detect_sorted_text_boxes_batch(batch_images) {
                Ok(batch_boxes) => {
                    for (offset, boxes) in batch_boxes.into_iter().enumerate() {
                        all_detection_boxes[start + offset] = boxes;
                    }
                }
                Err(err) => {
                    tracing::warn!(
                        target: "ocr",
                        error = %err,
                        batch_start = start,
                        batch_end = end,
                        "Batched text detection failed; falling back to per-image detection"
                    );
                    for i in start..end {
                        all_detection_boxes[i] =
                            self.detect_sorted_text_boxes(&prepared[i].1.image)?;
                    }
                }
            }

            start = end;
        }

        // Phase 1: crop + line-orientation per image into a shared pool tagged by
        // image index. Recognizing crops together (vs one batch per image) yields
        // batches that are both larger (better GPU use) and width-tighter (less
        // padding waste/drift), since same-width crops are plentiful across pages.
        //
        // Bounded by `MAX_POOLED_CROPS`: each crop's `Arc<RgbImage>` lives until its
        // batch runs, so an unbounded pool grows peak memory with the input and can
        // OOM on big multi-page calls. We flush (recognize + scatter) on reaching
        // the cap; it's high enough that typical batches still pool fully.
        const MAX_POOLED_CROPS: usize = 4096;
        let mut per_image_results: Vec<Vec<Option<crate::oarocr::TextRegion>>> =
            all_detection_boxes
                .iter()
                .map(|b| vec![None; b.len()])
                .collect();
        let total_crops: usize = all_detection_boxes.iter().map(|b| b.len()).sum();
        let mut global_crops: Vec<(usize, CroppedTextRegion)> =
            Vec::with_capacity(total_crops.min(MAX_POOLED_CROPS));
        for (img_idx, (_, preprocess)) in prepared.iter().enumerate() {
            let mut crops =
                self.crop_text_regions(&preprocess.image, &all_detection_boxes[img_idx])?;
            self.classify_line_orientations(&mut crops)?;
            for crop in crops {
                global_crops.push((img_idx, crop));
                if global_crops.len() >= MAX_POOLED_CROPS {
                    // Flush mid-image as well: a single dense page can exceed the cap
                    // on its own, so checking only between images would let the pool
                    // (and its live `Arc<RgbImage>`s) grow past the bound before any
                    // flush runs. `replace` (not `take`) keeps a pre-sized buffer for
                    // the next wave, so repeated flushes don't re-grow from zero.
                    let pool =
                        std::mem::replace(&mut global_crops, Vec::with_capacity(MAX_POOLED_CROPS));
                    self.recognize_global(pool, &mut per_image_results, options)?;
                }
            }
        }

        // Phase 2: recognize the remaining pool and scatter results back per image.
        if !global_crops.is_empty() {
            self.recognize_global(global_crops, &mut per_image_results, options)?;
        }

        // Phase 3: assemble per-image results (reading order + rotate-back).
        let mut results = Vec::with_capacity(prepared.len());
        for (img_idx, ((input_img_arc, preprocess), image_results)) in
            prepared.into_iter().zip(per_image_results).enumerate()
        {
            let mut text_regions: Vec<crate::oarocr::TextRegion> =
                image_results.into_iter().flatten().collect();

            if let Some(rot) = preprocess.rotation {
                Self::rotate_text_regions_back(&mut text_regions, rot);
            }

            results.push(crate::oarocr::OAROCRResult {
                input_path: Arc::from(format!("image_{}", img_idx)),
                index: img_idx,
                input_img: input_img_arc,
                text_regions,
                orientation_angle: preprocess.orientation_angle,
                rectified_img: preprocess.rectified_img,
            });
        }

        Ok(results)
    }

    fn detect_sorted_text_boxes_batch(
        &self,
        images: Vec<Arc<image::RgbImage>>,
    ) -> Result<Vec<Vec<BoundingBox>>, OCRError> {
        if images.is_empty() {
            return Ok(Vec::new());
        }

        let input = ImageTaskInput::from_arc_images(images);
        let det = self.pipeline.text_detection_adapter.execute(input, None)?;

        let mut results: Vec<Vec<BoundingBox>> = Vec::with_capacity(det.detections.len());
        for detections in det.detections.into_iter() {
            let boxes = detections.into_iter().map(|d| d.bbox).collect::<Vec<_>>();
            results.push(self.sort_detection_boxes(&boxes));
        }

        Ok(results)
    }

    fn detect_sorted_text_boxes(
        &self,
        image: &Arc<image::RgbImage>,
    ) -> Result<Vec<BoundingBox>, OCRError> {
        let input = ImageTaskInput::from_arc_images(vec![Arc::clone(image)]);
        let det = self.pipeline.text_detection_adapter.execute(input, None)?;

        let boxes = det
            .detections
            .into_iter()
            .next()
            .unwrap_or_default()
            .into_iter()
            .map(|d| d.bbox)
            .collect::<Vec<_>>();

        Ok(self.sort_detection_boxes(&boxes))
    }

    fn sort_detection_boxes(&self, boxes: &[BoundingBox]) -> Vec<BoundingBox> {
        if boxes.is_empty() {
            return Vec::new();
        }

        let is_seal_text = self
            .text_type
            .as_ref()
            .map(|t| t.to_lowercase() == "seal")
            .unwrap_or(false);

        if is_seal_text {
            crate::processors::sort_poly_boxes(boxes)
        } else {
            crate::processors::sort_quad_boxes(boxes)
        }
    }

    fn crop_text_regions(
        &self,
        image: &Arc<image::RgbImage>,
        detection_boxes: &[BoundingBox],
    ) -> Result<Vec<CroppedTextRegion>, OCRError> {
        use crate::oarocr::EdgeProcessor;
        use crate::oarocr::TextCroppingProcessor;

        if detection_boxes.is_empty() {
            return Ok(Vec::new());
        }

        let processor = TextCroppingProcessor::new(true); // handle_rotation = true
        // Zero-copy: share Arc instead of cloning the image
        let cropped = processor.process((Arc::clone(image), detection_boxes.to_vec()))?;

        let mut regions = Vec::new();
        for (idx, crop_result) in cropped.into_iter().enumerate() {
            let Some(img) = crop_result else {
                continue;
            };
            let wh_ratio = img.width() as f32 / img.height().max(1) as f32;
            regions.push(CroppedTextRegion {
                detection_index: idx,
                bbox: detection_boxes
                    .get(idx)
                    .cloned()
                    .unwrap_or_else(|| BoundingBox::from_coords(0.0, 0.0, 0.0, 0.0)),
                image: img,
                wh_ratio,
                line_orientation_angle: None,
            });
        }

        Ok(regions)
    }

    fn classify_line_orientations(
        &self,
        regions: &mut [CroppedTextRegion],
    ) -> Result<(), OCRError> {
        let Some(ref line_orientation_adapter) = self.pipeline.text_line_orientation_adapter else {
            return Ok(());
        };

        if regions.is_empty() {
            return Ok(());
        }

        let input_images = regions.iter().map(|r| Arc::clone(&r.image)).collect();
        let input = ImageTaskInput::from_arc_images(input_images);
        let orient = line_orientation_adapter.execute(input, None)?;

        for (idx, classifications) in orient
            .classifications
            .iter()
            .enumerate()
            .take(regions.len())
        {
            let Some(top_class) = classifications.first() else {
                continue;
            };

            // Convert class_id to angle (0=0°, 1=180°)
            let angle = (top_class.class_id as f32) * 180.0;
            regions[idx].line_orientation_angle = Some(angle);

            if top_class.class_id == 1 {
                regions[idx].image =
                    Arc::new(image::imageops::rotate180(regions[idx].image.as_ref()));
            }
        }

        Ok(())
    }

    /// Recognizes a global pool of crops (gathered across all images) and scatters
    /// each result back into `per_image_results[image_index][detection_index]`.
    ///
    /// Crops are sorted by width/height ratio across the whole batch of images and
    /// chunked into fixed-size batches, so each recognition batch is
    /// width-homogeneous (minimal zero-padding) yet well-filled. Pooling across
    /// images gives many more same-width crops than any single page, which makes
    /// the batches both tighter and larger than per-image batching could.
    fn recognize_global(
        &self,
        mut crops: Vec<(usize, CroppedTextRegion)>,
        per_image_results: &mut [Vec<Option<crate::oarocr::TextRegion>>],
        options: &PredictOptions,
    ) -> Result<(), OCRError> {
        if crops.is_empty() {
            return Ok(());
        }

        crops.sort_by(|a, b| {
            a.1.wh_ratio
                .partial_cmp(&b.1.wh_ratio)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let base_rec_ratio = DEFAULT_REC_IMAGE_SHAPE[2] as f32 / DEFAULT_REC_IMAGE_SHAPE[1] as f32;
        let batch_size = options
            .region_batch_size
            .or(self.region_batch_size)
            .unwrap_or_else(|| {
                self.pipeline
                    .text_recognition_adapter
                    .recommended_batch_size()
            })
            .max(1);
        let column_budget = options
            .region_ref_width
            .or(self.region_ref_width)
            .filter(|&ref_width| ref_width > 0)
            .map(|ref_width| batch_size.saturating_mul(ref_width));

        let wh_ratios: Vec<f32> = crops.iter().map(|(_, r)| r.wh_ratio).collect();
        let plan = plan_recognition_chunks(&wh_ratios, batch_size, column_budget);
        tracing::debug!(
            crops = crops.len(),
            batches = plan.len(),
            batch_size,
            ?column_budget,
            widest_batch = plan.last().map_or(0, |r| r.len()),
            widest_tensor = wh_ratios.last().map_or(0, |&r| rec_tensor_width(r)),
            "recognition batch plan"
        );

        for range in plan {
            let chunk = &crops[range];
            let chunk_max_wh_ratio = clamp_rec_ratio(
                chunk
                    .iter()
                    .map(|(_, r)| r.wh_ratio)
                    .fold(base_rec_ratio, |acc, r| acc.max(r)),
            );

            let rec_input = ImageTaskInput::from_arc_images(
                chunk.iter().map(|(_, r)| Arc::clone(&r.image)).collect(),
            );

            let rec = self
                .pipeline
                .text_recognition_adapter
                .execute(rec_input, None)?;

            let n = rec.texts.len().min(chunk.len());
            for (i, (img_idx, region)) in chunk.iter().take(n).enumerate() {
                let text = rec.texts.get(i).map(String::as_str).unwrap_or("");
                let score = *rec.scores.get(i).unwrap_or(&0.0);

                let char_positions: &[f32] = rec
                    .char_positions
                    .get(i)
                    .map(|v| v.as_slice())
                    .unwrap_or(&[]);
                let col_indices: &[usize] = rec
                    .char_col_indices
                    .get(i)
                    .map(|v| v.as_slice())
                    .unwrap_or(&[]);
                let seq_len = *rec.sequence_lengths.get(i).unwrap_or(&0);

                let bbox = region.bbox.clone();
                let word_boxes = if self.return_word_box && !col_indices.is_empty() && seq_len > 0 {
                    Some(Self::ctc_word_boxes(
                        &bbox,
                        text,
                        col_indices,
                        seq_len,
                        clamp_rec_ratio(region.wh_ratio),
                        chunk_max_wh_ratio,
                    ))
                } else if self.return_word_box && !char_positions.is_empty() {
                    Some(Self::char_positions_to_word_boxes(
                        &bbox,
                        char_positions,
                        text.chars().count(),
                    ))
                } else {
                    None
                };

                if let Some(image_results) = per_image_results.get_mut(*img_idx)
                    && region.detection_index < image_results.len()
                {
                    image_results[region.detection_index] = Some(crate::oarocr::TextRegion {
                        bounding_box: bbox.clone(),
                        dt_poly: Some(bbox.clone()),
                        rec_poly: Some(bbox),
                        text: Some(std::sync::Arc::from(text)),
                        confidence: Some(score),
                        orientation_angle: region.line_orientation_angle,
                        word_boxes,
                        label: None,
                    });
                }
            }
        }

        Ok(())
    }

    fn rotate_text_regions_back(
        regions: &mut [crate::oarocr::TextRegion],
        rot: crate::oarocr::preprocess::OrientationCorrection,
    ) {
        for region in regions {
            region.dt_poly = region.dt_poly.take().map(|poly| {
                poly.rotate_back_to_original(rot.angle, rot.rotated_width, rot.rotated_height)
            });
            region.rec_poly = region.rec_poly.take().map(|poly| {
                poly.rotate_back_to_original(rot.angle, rot.rotated_width, rot.rotated_height)
            });
            region.bounding_box = region.bounding_box.rotate_back_to_original(
                rot.angle,
                rot.rotated_width,
                rot.rotated_height,
            );

            if let Some(ref word_boxes) = region.word_boxes {
                let transformed_word_boxes: Vec<_> = word_boxes
                    .iter()
                    .map(|wb| {
                        wb.rotate_back_to_original(rot.angle, rot.rotated_width, rot.rotated_height)
                    })
                    .collect();
                region.word_boxes = Some(transformed_word_boxes);
            }
        }
    }

    /// Converts CTC column indices to word-level bounding boxes using standard approach.
    ///
    /// This method calculates character-specific widths based on the column indices from CTC decoding,
    /// which provides more accurate word boxes than uniform distribution.
    ///
    /// It aligns with standard logic by distinguishing between CJK and other characters:
    /// - CJK characters use a center-based approach with average character width to avoid being too narrow.
    /// - Other characters use the standard column-based width.
    ///
    /// # Arguments
    ///
    /// * `line_bbox` - The bounding box of the entire text line
    /// * `text` - The recognized text string
    /// * `col_indices` - Column indices (timesteps) for each character from CTC output
    /// * `seq_len` - Total number of columns (sequence length) in the CTC output
    /// * `wh_ratio` - Width/height ratio of this region's crop
    /// * `max_wh_ratio` - Max width/height ratio in the batch (used to undo padding)
    ///
    /// # Returns
    ///
    /// A vector of bounding boxes, one for each character
    fn ctc_word_boxes(
        line_bbox: &BoundingBox,
        text: &str,
        col_indices: &[usize],
        seq_len: usize,
        wh_ratio: f32,
        max_wh_ratio: f32,
    ) -> Vec<BoundingBox> {
        if col_indices.is_empty() || seq_len == 0 || text.is_empty() {
            return Vec::new();
        }

        // Scale effective column count using standard logic (handles padding to max width)
        let effective_col_num = (seq_len as f32) * (wh_ratio / max_wh_ratio);
        if effective_col_num <= f32::EPSILON {
            return Vec::new();
        }

        // Get the line bounding box coordinates
        let x_min = line_bbox.x_min();
        let y_min = line_bbox.y_min();
        let x_max = line_bbox.x_max();
        let y_max = line_bbox.y_max();
        let width = x_max - x_min;

        // Calculate cell width (width of each column in the CTC output)
        let cell_width = width / effective_col_num.max(f32::EPSILON);

        let mut word_boxes = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let avg_char_width = width / chars.len().max(1) as f32;

        // Pre-calculate centers for all characters
        let centers: Vec<f32> = col_indices
            .iter()
            .map(|&idx| x_min + (idx as f32 + 0.5) * cell_width)
            .collect();

        for (i, _) in col_indices.iter().enumerate() {
            let ch = chars.get(i).copied().unwrap_or('?');
            let center_x = centers[i];

            if Self::is_cjk(ch) {
                let half_width = avg_char_width / 2.0;
                let char_x_min = (center_x - half_width).max(x_min);
                let char_x_max = (center_x + half_width).min(x_max);
                let char_box = BoundingBox::from_coords(char_x_min, y_min, char_x_max, y_max);
                word_boxes.push(char_box);
            } else {
                // For non-CJK characters, use the midpoint between adjacent character centers
                // to determine boundaries. This provides contiguous boxes that adapt to character density.
                let char_x_min = if i == 0 {
                    x_min
                } else {
                    (centers[i - 1] + center_x) / 2.0
                }
                .max(x_min);

                let char_x_max = if i == col_indices.len() - 1 {
                    x_max
                } else {
                    (center_x + centers[i + 1]) / 2.0
                }
                .min(x_max);

                let char_box = BoundingBox::from_coords(char_x_min, y_min, char_x_max, y_max);
                word_boxes.push(char_box);
            }
        }

        word_boxes
    }

    /// Converts normalized character positions to word-level bounding boxes.
    ///
    /// This is a fallback method that uses uniform character width distribution.
    /// Use `ctc_word_boxes` when CTC column indices are available for better accuracy.
    ///
    /// # Arguments
    ///
    /// * `line_bbox` - The bounding box of the entire text line
    /// * `char_positions` - Normalized x-positions (0.0-1.0) for each character
    /// * `char_count` - Number of characters in the text
    ///
    /// # Returns
    ///
    /// A vector of bounding boxes, one for each character/word
    fn char_positions_to_word_boxes(
        line_bbox: &BoundingBox,
        char_positions: &[f32],
        char_count: usize,
    ) -> Vec<BoundingBox> {
        if char_positions.is_empty() || char_count == 0 {
            return Vec::new();
        }

        // Get the line bounding box coordinates
        let x_min = line_bbox.x_min();
        let y_min = line_bbox.y_min();
        let x_max = line_bbox.x_max();
        let y_max = line_bbox.y_max();
        let width = x_max - x_min;

        // Calculate approximate character width
        let char_width = width / char_count as f32;

        // Create a bounding box for each character based on its position
        let mut word_boxes = Vec::new();
        for &pos in char_positions.iter() {
            // Calculate x position (pos is normalized 0.0-1.0)
            let char_x_center = x_min + (pos * width);

            // Estimate character box boundaries
            // Use half character width on each side of the position
            let char_x_min = (char_x_center - char_width / 2.0).max(x_min);
            let char_x_max = (char_x_center + char_width / 2.0).min(x_max);

            // Use the full height of the text line for each character
            let char_box = BoundingBox::from_coords(char_x_min, y_min, char_x_max, y_max);
            word_boxes.push(char_box);
        }

        word_boxes
    }

    /// Detect whether a character is CJK.
    fn is_cjk(c: char) -> bool {
        let u = c as u32;
        (0x4E00..=0x9FFF).contains(&u)
            || (0x3400..=0x4DBF).contains(&u)
            || (0x20000..=0x2A6DF).contains(&u)
            || (0x2A700..=0x2B73F).contains(&u)
            || (0x2B740..=0x2B81F).contains(&u)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oarocr_builder_new() {
        let builder = OAROCRBuilder::new("det.onnx", "rec.onnx", "dict.txt");

        assert_eq!(
            builder.text_detection_model.as_path(),
            Some(std::path::Path::new("det.onnx"))
        );
        assert_eq!(
            builder.text_recognition_model.as_path(),
            Some(std::path::Path::new("rec.onnx"))
        );
        assert_eq!(builder.character_dict_path, PathBuf::from("dict.txt"));
        assert!(builder.document_orientation_model.is_none());
        assert!(builder.text_line_orientation_model.is_none());
        assert!(builder.document_rectification_model.is_none());
    }

    #[test]
    fn test_oarocr_builder_with_optional_components() {
        let builder = OAROCRBuilder::new("det.onnx", "rec.onnx", "dict.txt")
            .with_document_image_orientation_classification("doc_orient.onnx")
            .with_text_line_orientation_classification("line_orient.onnx")
            .with_document_image_rectification("rectify.onnx");

        let Some(source) = builder.document_orientation_model.as_ref() else {
            panic!("expected document_orientation_model to be Some");
        };
        assert_eq!(
            source.as_path(),
            Some(std::path::Path::new("doc_orient.onnx"))
        );
        let Some(source) = builder.text_line_orientation_model.as_ref() else {
            panic!("expected text_line_orientation_model to be Some");
        };
        assert_eq!(
            source.as_path(),
            Some(std::path::Path::new("line_orient.onnx"))
        );
        let Some(source) = builder.document_rectification_model.as_ref() else {
            panic!("expected document_rectification_model to be Some");
        };
        assert_eq!(source.as_path(), Some(std::path::Path::new("rectify.onnx")));
    }

    #[test]
    fn test_oarocr_builder_with_configuration() {
        let det_config = TextDetectionConfig {
            score_threshold: 0.5,
            box_threshold: 0.6,
            unclip_ratio: 1.8,
            max_candidates: 1000,
            limit_side_len: None,
            limit_type: None,
            max_side_len: None,
        };

        let rec_config = TextRecognitionConfig {
            score_threshold: 0.7,
        };

        let builder = OAROCRBuilder::new("det.onnx", "rec.onnx", "dict.txt")
            .text_detection_config(det_config.clone())
            .text_recognition_config(rec_config.clone());

        assert!(builder.text_detection_config.is_some());
        assert!(builder.text_recognition_config.is_some());
    }

    #[test]
    fn test_oarocr_builder_with_batch_sizes() {
        let builder = OAROCRBuilder::new("det.onnx", "rec.onnx", "dict.txt")
            .image_batch_size(4)
            .region_batch_size(64);

        assert_eq!(builder.image_batch_size, Some(4));
        assert_eq!(builder.region_batch_size, Some(64));
    }

    #[test]
    fn test_validate_batch_size_accepts_bounds() {
        assert!(OAROCRBuilder::validate_batch_size("image_batch_size", 1).is_ok());
        assert!(
            OAROCRBuilder::validate_batch_size("region_batch_size", OAROCRBuilder::MAX_BATCH_SIZE,)
                .is_ok()
        );
    }

    #[test]
    fn test_validate_batch_size_rejects_zero() {
        let err = OAROCRBuilder::validate_batch_size("image_batch_size", 0).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("image_batch_size"));
        assert!(msg.contains(&format!("1..={}", OAROCRBuilder::MAX_BATCH_SIZE)));
    }

    #[test]
    fn test_validate_batch_size_rejects_values_above_max() {
        let err = OAROCRBuilder::validate_batch_size(
            "region_batch_size",
            OAROCRBuilder::MAX_BATCH_SIZE + 1,
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("region_batch_size"));
        assert!(msg.contains(&format!("1..={}", OAROCRBuilder::MAX_BATCH_SIZE)));
    }

    #[test]
    fn test_ctc_word_boxes_logic() {
        let line_bbox = BoundingBox::from_coords(0.0, 0.0, 100.0, 20.0);
        // seq_len=10, wh_ratio=5 (100/20), max_wh_ratio=5 -> effective_col_num = 10
        // cell_width = 100/10 = 10.0

        // Test 1: Non-CJK "ABC"
        // Indices: 1, 4, 7 (approx centers: 15, 45, 75)
        let text = "ABC";
        let col_indices = vec![1, 4, 7];
        let seq_len = 10;
        let wh_ratio = 5.0;
        let max_wh_ratio = 5.0;

        let boxes = OAROCR::ctc_word_boxes(
            &line_bbox,
            text,
            &col_indices,
            seq_len,
            wh_ratio,
            max_wh_ratio,
        );

        assert_eq!(boxes.len(), 3);
        // Center 0: 1.5 * 10 = 15. Center 1: 4.5 * 10 = 45. Center 2: 7.5 * 10 = 75.
        // Box 0: Left=0, Right=(15+45)/2 = 30.
        // Box 1: Left=30, Right=(45+75)/2 = 60.
        // Box 2: Left=60, Right=100.

        assert!((boxes[0].x_min() - 0.0).abs() < 1e-5);
        assert!((boxes[0].x_max() - 30.0).abs() < 1e-5);
        assert!((boxes[1].x_min() - 30.0).abs() < 1e-5);
        assert!((boxes[1].x_max() - 60.0).abs() < 1e-5);
        assert!((boxes[2].x_min() - 60.0).abs() < 1e-5);
        assert!((boxes[2].x_max() - 100.0).abs() < 1e-5);
    }

    // ---- PaddleKnife patch: area-budgeted recognition batching ----

    /// Total padded columns of one planned batch.
    fn padded_columns(ratios: &[f32], range: &std::ops::Range<usize>) -> usize {
        range.len() * rec_tensor_width(ratios[range.end - 1])
    }

    #[test]
    fn rec_tensor_width_is_floored_and_capped() {
        assert_eq!(rec_tensor_width(1.0), 320); // floor: model base width
        assert_eq!(rec_tensor_width(20.0), 960);
        assert_eq!(rec_tensor_width(51.6), 2476);
        assert_eq!(rec_tensor_width(500.0), 3200); // hard cap
    }

    #[test]
    fn plan_without_budget_matches_upstream_fixed_chunks() {
        let ratios: Vec<f32> = (0..10).map(|i| 1.0 + i as f32).collect();
        let plan = plan_recognition_chunks(&ratios, 4, None);
        assert_eq!(plan, vec![0..4, 4..8, 8..10]);
    }

    #[test]
    fn plan_covers_every_crop_exactly_once_in_order() {
        let mut ratios: Vec<f32> = (0..200).map(|i| 0.5 + (i as f32 * 0.37) % 70.0).collect();
        ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let plan = plan_recognition_chunks(&ratios, 64, Some(64 * 896));
        let mut next = 0;
        for range in &plan {
            assert_eq!(range.start, next, "batches must be contiguous");
            assert!(!range.is_empty());
            next = range.end;
        }
        assert_eq!(next, ratios.len());
    }

    #[test]
    fn plan_respects_both_count_and_area_limits() {
        let mut ratios: Vec<f32> = (0..300).map(|i| 0.5 + (i as f32 * 0.91) % 80.0).collect();
        ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let budget = 64 * 896;
        for range in plan_recognition_chunks(&ratios, 64, Some(budget)) {
            assert!(range.len() <= 64);
            // A lone crop is always allowed, however wide.
            assert!(range.len() == 1 || padded_columns(&ratios, &range) <= budget);
        }
    }

    #[test]
    fn narrow_crops_still_fill_the_count_cap() {
        // Invoice-like page: 157 short fields. The area budget must not slow
        // this case down — batching is identical to upstream's.
        let ratios = vec![4.0_f32; 157];
        let budgeted = plan_recognition_chunks(&ratios, 64, Some(64 * 896));
        let upstream = plan_recognition_chunks(&ratios, 64, None);
        assert_eq!(budgeted, upstream);
        assert_eq!(budgeted, vec![0..64, 64..128, 128..157]);
    }

    #[test]
    fn one_very_wide_line_no_longer_taxes_the_whole_page() {
        // The page that motivated this patch: 46 ordinary chat lines plus one
        // 51.6:1 header. Upstream pads all 47 crops to 2476 px (116k columns,
        // ~9 GiB on DirectML with an 18.7k-class head).
        let mut ratios = vec![5.5_f32; 46];
        ratios.push(51.6);
        let upstream = plan_recognition_chunks(&ratios, 64, None);
        assert_eq!(upstream, vec![0..47]);
        assert_eq!(padded_columns(&ratios, &upstream[0]), 47 * 2476);

        let plan = plan_recognition_chunks(&ratios, 64, Some(64 * 896));
        assert_eq!(plan, vec![0..46, 46..47]);
        assert_eq!(padded_columns(&ratios, &plan[0]), 46 * 320);
        assert_eq!(padded_columns(&ratios, &plan[1]), 2476);
    }

    #[test]
    fn halving_the_batch_size_halves_the_area_budget() {
        // A page of uniformly long lines: the count cap never binds, so only
        // the area budget keeps memory in check — and it must scale with the
        // batch size so a retry at half the batch really uses half the memory.
        let ratios = vec![51.6_f32; 100];
        let widest = |batch: usize| {
            plan_recognition_chunks(&ratios, batch, Some(batch * 896))
                .iter()
                .map(|r| padded_columns(&ratios, r))
                .max()
                .unwrap()
        };
        assert_eq!(widest(64), 23 * 2476); // 64 * 896 / 2476 = 23.16
        assert_eq!(widest(32), 11 * 2476);
        assert_eq!(widest(8), 2 * 2476);
    }

    #[test]
    fn a_single_crop_wider_than_the_budget_still_runs_alone() {
        let ratios = vec![3.0, 400.0];
        let plan = plan_recognition_chunks(&ratios, 8, Some(1000));
        assert_eq!(plan, vec![0..1, 1..2]);
    }

    #[test]
    fn plan_handles_empty_input_and_zero_count() {
        assert!(plan_recognition_chunks(&[], 64, Some(1)).is_empty());
        // max_count of 0 is treated as 1 rather than looping forever.
        assert_eq!(plan_recognition_chunks(&[1.0, 2.0], 0, None), vec![0..1, 1..2]);
    }

    #[test]
    fn word_boxes_use_the_capped_tensor_width() {
        // A 50:1 crop batched with a 100:1 crop. The recognizer caps the tensor
        // at 3200 px (ratio 66.67), so the 50:1 crop fills 75% of it: 300 of
        // 400 CTC columns. Unclamped, upstream computes 50/100 = 200 columns
        // and every box on the line is stretched 1.5x to the right.
        assert_eq!(clamp_rec_ratio(20.0), 20.0);
        assert!((clamp_rec_ratio(100.0) - 3200.0 / 48.0).abs() < 1e-4);

        let bbox = BoundingBox::from_coords(0.0, 0.0, 1000.0, 10.0);
        let centre_of_middle_char = |wh: f32, max_wh: f32| {
            let boxes = OAROCR::ctc_word_boxes(&bbox, "abc", &[0, 150, 299], 400, wh, max_wh);
            assert_eq!(boxes.len(), 3);
            (boxes[1].x_min() + boxes[1].x_max()) / 2.0
        };

        // Column 150 of 300 occupied columns is the middle of the line.
        let fixed = centre_of_middle_char(clamp_rec_ratio(50.0), clamp_rec_ratio(100.0));
        assert!((fixed - 500.0).abs() < 5.0, "clamped centre {fixed}");
        // Upstream's unclamped ratios put it far right of centre.
        let upstream = centre_of_middle_char(50.0, 100.0);
        assert!(upstream > 650.0, "unclamped centre {upstream}");
    }
}
