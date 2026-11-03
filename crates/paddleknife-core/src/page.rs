//! A single decoded image, as returned by `image_io::load_image`.

use image::DynamicImage;

pub struct Page {
    /// Zero-based index (always 0 for single images).
    pub index: usize,
    pub width: u32,
    pub height: u32,
    pub image: DynamicImage,
}
