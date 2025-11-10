//! Main compression algorithm for OpenCV backend

use opencv::{core::Mat, prelude::*};

use super::errors::{CompressionError, Result};
use super::encode::encode_image;
use super::format::ImageFormat;
use super::resize::resize_image;

/// Compress an image to fit within a byte limit using OpenCV
///
/// Uses binary search to find optimal quality and scale parameters
///
/// # Arguments
///
/// * `image` - Input image as OpenCV Mat
/// * `byte_limit` - Maximum size in bytes
/// * `format` - Compression format
///
/// # Returns
///
/// Compressed image data as bytes
pub fn compress_image(image: &Mat, byte_limit: usize, format: ImageFormat) -> Result<Vec<u8>> {
    if byte_limit < 1024 {
        return Err(CompressionError::ByteLimitTooSmall { limit: byte_limit });
    }

    let start = std::time::Instant::now();
    let original_size = image.size()?;

    tracing::info!(
        width = original_size.width,
        height = original_size.height,
        target_bytes = byte_limit,
        "Starting OpenCV compression"
    );

    // Get original image size at max quality
    let original_encoded = encode_image(image, 100, format)?;
    let original_bytes = original_encoded.len();

    // If already under limit, return as-is
    if original_bytes <= byte_limit {
        tracing::info!(
            compression_time_ms = start.elapsed().as_millis(),
            "Image already under byte limit"
        );
        return Ok(original_encoded);
    }

    // Binary search on quality and scale
    let mut min_quality = 1;
    let mut max_quality = 100;
    let mut min_scale = 0.1f64;
    let mut max_scale = 1.0f64;

    let mut best_quality = min_quality;
    let mut best_scale = min_scale;
    let mut best_result: Vec<u8> = Vec::new();

    while min_quality <= max_quality && min_scale <= max_scale {
        let mid_quality = (min_quality + max_quality) / 2;
        let mid_scale = (min_scale + max_scale) / 2.0;

        // Resize image
        let scaled_img = resize_image(image, mid_scale)?;

        // Encode with current quality
        let compressed = encode_image(&scaled_img, mid_quality, format)?;
        let compressed_size = compressed.len();

        tracing::debug!(
            quality = mid_quality,
            scale = mid_scale,
            bytes = compressed_size,
            "Trying compression parameters"
        );

        if compressed_size == byte_limit {
            // Exact match
            tracing::info!(
                quality = mid_quality,
                scale = mid_scale,
                final_bytes = compressed_size,
                compression_time_ms = start.elapsed().as_millis(),
                "Exact byte limit reached"
            );
            return Ok(compressed);
        } else if compressed_size < byte_limit {
            // Under limit - try higher quality/scale
            if mid_quality > best_quality {
                best_quality = mid_quality;
                best_scale = mid_scale;
                best_result = compressed;
            }
            min_quality = mid_quality + 1;
            min_scale = (mid_scale + 0.01).min(1.0);
        } else {
            // Over limit - try lower quality/scale
            max_quality = mid_quality - 1;
            max_scale = (mid_scale - 0.01).max(0.1);
        }
    }

    // Final compression with best parameters
    if best_result.is_empty() {
        let scaled_img = resize_image(image, best_scale)?;
        best_result = encode_image(&scaled_img, best_quality, format)?;
    }

    let final_size = best_result.len();
    let reduction_ratio = original_bytes as f64 / final_size as f64;

    tracing::info!(
        quality = best_quality,
        scale = best_scale,
        original_bytes,
        final_bytes = final_size,
        reduction_ratio = format!("{:.2}x", reduction_ratio),
        compression_time_ms = start.elapsed().as_millis(),
        "OpenCV compression complete"
    );

    if final_size > byte_limit {
        return Err(CompressionError::OptimizationFailed);
    }

    Ok(best_result)
}
