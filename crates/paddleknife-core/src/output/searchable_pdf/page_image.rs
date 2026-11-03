//! What gets embedded as a page's image XObject: the source scan's
//! original compressed bytes (passthrough — zero quality/size cost) or
//! a JPEG (re)encoded from the OCR raster.

use crate::source_pdf::PassthroughImage;

#[derive(Debug, Clone)]
pub enum PageImage {
    /// Original bytes copied verbatim from the source PDF.
    Passthrough(PassthroughImage),
    /// JPEG re-encoded from the raster. `gray` selects DeviceGray.
    Jpeg {
        data: Vec<u8>,
        width: u32,
        height: u32,
        gray: bool,
    },
}

impl PageImage {
    /// Pixel dimensions of the *embedded* image (NOT the raster).
    pub fn dims(&self) -> (u32, u32) {
        match self {
            Self::Passthrough(p) => (p.width, p.height),
            Self::Jpeg { width, height, .. } => (*width, *height),
        }
    }

    pub fn data(&self) -> &[u8] {
        match self {
            Self::Passthrough(p) => &p.data,
            Self::Jpeg { data, .. } => data,
        }
    }
}
