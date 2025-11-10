//! Image format definitions for OpenCV backend

use opencv::imgcodecs::{IMWRITE_JPEG_QUALITY, IMWRITE_WEBP_QUALITY};

/// Supported image formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    /// JPEG format
    Jpeg,
    /// WebP format
    WebP,
}

impl ImageFormat {
    pub(crate) fn extension(&self) -> &str {
        match self {
            ImageFormat::Jpeg => ".jpg",
            ImageFormat::WebP => ".webp",
        }
    }

    pub(crate) fn quality_param(&self) -> i32 {
        match self {
            ImageFormat::Jpeg => IMWRITE_JPEG_QUALITY,
            ImageFormat::WebP => IMWRITE_WEBP_QUALITY,
        }
    }
}
