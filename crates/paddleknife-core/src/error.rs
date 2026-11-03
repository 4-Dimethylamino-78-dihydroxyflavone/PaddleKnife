//! Error types for the core pipeline.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum OcrError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("image: {0}")]
    Image(#[from] image::ImageError),

    #[error("unsupported input format: {0}")]
    UnsupportedFormat(String),

    #[error("model not found: {0}")]
    ModelNotFound(String),

    #[error("ocr engine: {0}")]
    Engine(String),
}

impl OcrError {
    /// Whether this reads like an allocation failure in the inference
    /// runtime rather than some other engine error.
    ///
    /// A heuristic over ONNX Runtime's message text — `Engine` carries no
    /// structured cause — so use it to choose *wording* only, never to decide
    /// whether a retry is worth making. On Windows the DirectML form is
    /// `8007000E Not enough memory resources are available` (`E_OUTOFMEMORY`),
    /// which may be system commit rather than VRAM; see `sysmem`.
    pub fn is_out_of_memory(&self) -> bool {
        /// Lower-case fragments of the allocation failures we have seen or
        /// that the runtimes document.
        const MARKERS: [&str; 5] = [
            "8007000e",
            "e_outofmemory",
            "not enough memory",
            "out of memory",
            "failed to allocate memory",
        ];
        let Self::Engine(message) = self else {
            return false;
        };
        let message = message.to_ascii_lowercase();
        MARKERS.iter().any(|marker| message.contains(marker))
    }
}

pub type Result<T> = std::result::Result<T, OcrError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directml_and_cuda_allocation_failures_read_as_out_of_memory() {
        for message in [
            "oar-ocr predict: Non-zero status code returned while running MatMul node. \
             Name:'MatMul.12' Status Message: Exception(3) tid(4f10) 8007000E \
             Not enough memory resources are available to complete this operation.",
            "adapter execution failed: E_OUTOFMEMORY",
            "CUDA failure 2: out of memory",
            "Failed to allocate memory for requested buffer of size 1073741824",
        ] {
            assert!(
                OcrError::Engine(message.into()).is_out_of_memory(),
                "should read as OOM: {message}"
            );
        }
    }

    #[test]
    fn other_errors_do_not_read_as_out_of_memory() {
        assert!(!OcrError::Engine("adapter execution failed".into()).is_out_of_memory());
        assert!(!OcrError::Engine("oar-ocr build: invalid model".into()).is_out_of_memory());
        // Only engine failures can be allocation failures; a missing model
        // whose *path* happens to contain the words is still a missing model.
        assert!(!OcrError::ModelNotFound("out of memory/rec.onnx".into()).is_out_of_memory());
    }
}
