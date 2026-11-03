//! Output writers for each supported format.
//!
//! All writers take `&[DocPage]` (post-layout) and write to any
//! `impl Write`, so the library stays agnostic of files vs stdout vs
//! in-memory buffers.

pub mod json;
pub mod markdown;
pub mod searchable_pdf;
pub mod text;
