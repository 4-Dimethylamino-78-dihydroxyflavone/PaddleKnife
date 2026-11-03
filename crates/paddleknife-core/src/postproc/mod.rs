//! Post-processing passes that run on OCR output before layout.
//!
//! Keep these as pure functions on `Vec<OcrBlock>`: no calls back into
//! the OCR engine, no I/O. Lets callers compose passes and lets us
//! unit-test each pass with synthetic inputs.

pub mod fullwidth;
pub mod nepali;
