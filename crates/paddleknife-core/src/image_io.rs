//! Single-image input loader (PNG, JPG, WebP, etc. — whatever `image` supports).

use crate::{error::Result, page::Page};
use std::path::Path;

pub fn load_image(path: &Path) -> Result<Page> {
    let img = image::open(path)?;
    Ok(Page {
        index: 0,
        width: img.width(),
        height: img.height(),
        image: img,
    })
}
