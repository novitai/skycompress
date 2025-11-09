//! Image I/O utilities for OpenCV backend

use opencv::{
    core::Mat,
    imgcodecs::{imread, IMREAD_COLOR},
    prelude::*,
};

use super::opencv_backend_errors::{CompressionError, Result};

/// Load image from file path
///
/// # Arguments
///
/// * `path` - Path to image file
///
/// # Returns
///
/// Loaded image as OpenCV Mat
///
/// # Errors
///
/// Returns error if file doesn't exist or cannot be decoded
pub fn load_image(path: &str) -> Result<Mat> {
    let img = imread(path, IMREAD_COLOR)?;

    if img.empty() {
        return Err(CompressionError::InvalidFormat);
    }

    Ok(img)
}
