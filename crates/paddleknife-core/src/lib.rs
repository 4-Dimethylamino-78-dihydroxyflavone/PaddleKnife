//! PaddleKnife core library — OCR pipeline, PDF I/O, output writers.
//!
//! Re-exports the minimum public surface for the `paddleknife` binary.

pub mod download;
pub mod error;
pub mod geometry;
pub mod image_io;
pub mod input;
pub mod layout;
pub mod model;
pub mod ocr;
pub mod output;
pub mod page;
pub mod pdf;
pub mod postproc;
pub mod resilient;
pub mod script;
pub mod source_pdf;
pub mod sysmem;

pub use error::OcrError;

/// Crate version from Cargo.toml (compile-time constant).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
