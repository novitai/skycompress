//! Image resizing utilities for OpenCV backend

use opencv::{
    core::Mat,
    imgproc::{resize, INTER_LANCZOS4},
    prelude::*,
};

use super::errors::Result;

/// Resize image by scale factor
///
/// Uses LANCZOS4 interpolation (highest quality, ~3-4ms slower than LINEAR)
///
/// For speed-critical applications, switch to INTER_LINEAR to match Python's speed.
/// Default prioritizes quality over the minimal performance difference.
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
        INTER_LANCZOS4,  // Highest quality (default)
    )?;

    Ok(resized)
}
