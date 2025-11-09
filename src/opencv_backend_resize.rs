//! Image resizing utilities for OpenCV backend

use opencv::{
    core::Mat,
    imgproc::{resize, INTER_LANCZOS4},
    prelude::*,
};

use super::opencv_backend_errors::Result;

/// Resize image by scale factor
///
/// Uses Lanczos4 interpolation for high-quality resizing
pub(crate) fn resize_image(image: &Mat, scale: f64) -> Result<Mat> {
    if scale == 1.0 {
        return Ok(image.clone());
    }

    let size = image.size()?;
    let new_width = (size.width as f64 * scale).round() as i32;
    let new_height = (size.height as f64 * scale).round() as i32;

    let mut resized = Mat::default();
    resize(
        image,
        &mut resized,
        opencv::core::Size::new(new_width, new_height),
        0.0,
        0.0,
        INTER_LANCZOS4,
    )?;

    Ok(resized)
}
