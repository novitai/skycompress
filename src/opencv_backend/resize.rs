//! Image resizing utilities for OpenCV backend

use opencv::{
    core::Mat,
    imgproc::{resize, INTER_LANCZOS4, INTER_LINEAR},
    prelude::*,
};

use super::errors::Result;

/// Interpolation mode for resizing
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterpolationMode {
    /// Highest quality (LANCZOS4) - ~3-4ms slower, professional grade
    /// Best for: Surveillance, security, archival
    Quality,

    /// Balanced quality/speed (LINEAR) - matches Python OpenCV default
    /// Best for: Real-time streams, live video
    Speed,
}

impl Default for InterpolationMode {
    fn default() -> Self {
        // Default to quality for surveillance/security use cases
        InterpolationMode::Quality
    }
}

impl InterpolationMode {
    fn to_opencv_flag(&self) -> i32 {
        match self {
            InterpolationMode::Quality => INTER_LANCZOS4,
            InterpolationMode::Speed => INTER_LINEAR,
        }
    }
}

/// Resize image by scale factor
///
/// Uses LANCZOS4 interpolation (highest quality, ~3-4ms slower than LINEAR)
///
/// For speed-critical applications, switch to INTER_LINEAR to match Python's speed.
/// Default prioritizes quality over the minimal performance difference.
pub(crate) fn resize_image(image: &Mat, scale: f64) -> Result<Mat> {
    resize_image_with_mode(image, scale, InterpolationMode::default())
}

/// Resize image with specific interpolation mode
pub(crate) fn resize_image_with_mode(
    image: &Mat,
    scale: f64,
    mode: InterpolationMode,
) -> Result<Mat> {
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
        mode.to_opencv_flag(),
    )?;

    Ok(resized)
}
