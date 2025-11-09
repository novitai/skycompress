//! Image crate backend for compression
//!
//! Pure Rust implementation using the `image` crate

use image::{DynamicImage, codecs::jpeg::JpegEncoder};
use std::io::Cursor;
use thiserror::Error;

/// Supported image formats for compression
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    /// JPEG format
    Jpeg,
}

/// Errors that can occur during image compression
#[derive(Error, Debug)]
pub enum CompressionError {
    #[error("Image encoding failed: {0}")]
    EncodingFailed(String),

    #[error("Failed to resize image: {0}")]
    ResizeFailed(String),

    #[error("Byte limit too small: {limit} bytes (minimum ~1KB recommended)")]
    ByteLimitTooSmall { limit: usize },

    #[error("Compression optimization failed: could not reach target size")]
    OptimizationFailed,
}

/// Result type for compression operations
pub type Result<T> = std::result::Result<T, CompressionError>;

/// Compress an image to fit within a byte limit
///
/// Uses binary search to find optimal quality and scale parameters
/// to compress the image as close as possible to the target byte limit
/// without exceeding it.
///
/// # Arguments
///
/// * `image` - Input image to compress
/// * `byte_limit` - Maximum size in bytes for output
/// * `format` - Compression format (JPEG or WebP)
///
/// # Returns
///
/// Compressed image data as bytes
///
/// # Example
///
/// ```rust
/// use skycompress::{compress_image, ImageFormat};
/// use image::DynamicImage;
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let img = image::open("photo.jpg")?;
/// let compressed = compress_image(&img, 50_000, ImageFormat::Jpeg)?;
/// assert!(compressed.len() <= 50_000);
/// # Ok(())
/// # }
/// ```
pub fn compress_image(
    image: &DynamicImage,
    byte_limit: usize,
    format: ImageFormat,
) -> Result<Vec<u8>> {
    if byte_limit < 1024 {
        return Err(CompressionError::ByteLimitTooSmall { limit: byte_limit });
    }

    let start = std::time::Instant::now();

    // Get original image size
    let original_size = encode_image(image, 100, 1.0, format)?;
    let original_bytes = original_size.len();

    tracing::info!(
        original_width = image.width(),
        original_height = image.height(),
        original_bytes,
        target_bytes = byte_limit,
        "Starting image compression"
    );

    // If already under limit at max quality, return as-is
    if original_bytes <= byte_limit {
        tracing::info!(
            compression_time_ms = start.elapsed().as_millis(),
            "Image already under byte limit, no compression needed"
        );
        return Ok(original_size);
    }

    // Binary search on quality and scale
    let mut min_quality = 1u8;
    let mut max_quality = 100u8;
    let mut min_scale = 0.1f32;
    let mut max_scale = 1.0f32;

    let mut best_quality = min_quality;
    let mut best_scale = min_scale;
    let mut best_result: Vec<u8> = Vec::new();

    while min_quality <= max_quality && min_scale <= max_scale {
        let mid_quality = (min_quality + max_quality) / 2;
        let mid_scale = (min_scale + max_scale) / 2.0;

        // Resize image
        let scaled_img = resize_image(image, mid_scale)?;

        // Encode with current quality
        let compressed = encode_image(&scaled_img, mid_quality, mid_scale, format)?;
        let compressed_size = compressed.len();

        tracing::debug!(
            quality = mid_quality,
            scale = mid_scale,
            width = scaled_img.width(),
            height = scaled_img.height(),
            bytes = compressed_size,
            "Trying compression parameters"
        );

        if compressed_size == byte_limit {
            // Exact match - return immediately
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
            min_quality = mid_quality.saturating_add(1);
            min_scale = (mid_scale + 0.01).min(1.0);
        } else {
            // Over limit - try lower quality/scale
            max_quality = mid_quality.saturating_sub(1);
            max_scale = (mid_scale - 0.01).max(0.1);
        }
    }

    // Final compression with best parameters
    if best_result.is_empty() {
        let scaled_img = resize_image(image, best_scale)?;
        best_result = encode_image(&scaled_img, best_quality, best_scale, format)?;
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
        "Compression complete"
    );

    if final_size > byte_limit {
        return Err(CompressionError::OptimizationFailed);
    }

    Ok(best_result)
}

/// Resize image by scale factor
fn resize_image(image: &DynamicImage, scale: f32) -> Result<DynamicImage> {
    if scale == 1.0 {
        return Ok(image.clone());
    }

    let new_width = (image.width() as f32 * scale).round() as u32;
    let new_height = (image.height() as f32 * scale).round() as u32;

    if new_width == 0 || new_height == 0 {
        return Err(CompressionError::ResizeFailed(
            "Scaled dimensions would be zero".to_string()
        ));
    }

    Ok(image.resize(new_width, new_height, image::imageops::FilterType::Lanczos3))
}

/// Encode image with specified quality
fn encode_image(
    image: &DynamicImage,
    quality: u8,
    _scale: f32,
    format: ImageFormat,
) -> Result<Vec<u8>> {
    let mut buffer = Cursor::new(Vec::new());
    let rgb = image.to_rgb8();

    match format {
        ImageFormat::Jpeg => {
            let mut encoder = JpegEncoder::new_with_quality(&mut buffer, quality);
            encoder
                .encode(
                    rgb.as_raw(),
                    image.width(),
                    image.height(),
                    image::ExtendedColorType::Rgb8,
                )
                .map_err(|e| CompressionError::EncodingFailed(e.to_string()))?;
        }
    }

    Ok(buffer.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    fn create_test_image(width: u32, height: u32) -> DynamicImage {
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(width, height, |x, y| {
            let r = (x % 256) as u8;
            let g = (y % 256) as u8;
            let b = ((x + y) % 256) as u8;
            Rgb([r, g, b])
        });
        DynamicImage::ImageRgb8(img)
    }

    #[test]
    fn test_compress_image_under_limit() {
        let img = create_test_image(800, 600);
        let byte_limit = 50_000;

        let result = compress_image(&img, byte_limit, ImageFormat::Jpeg);
        assert!(result.is_ok());

        let compressed = result.unwrap();
        assert!(compressed.len() <= byte_limit);
    }

    #[test]
    fn test_compress_image_already_small() {
        let img = create_test_image(100, 100);
        let byte_limit = 1_000_000; // 1MB - way more than needed

        let result = compress_image(&img, byte_limit, ImageFormat::Jpeg);
        assert!(result.is_ok());
    }


    #[test]
    fn test_byte_limit_too_small() {
        let img = create_test_image(800, 600);
        let byte_limit = 500; // Too small

        let result = compress_image(&img, byte_limit, ImageFormat::Jpeg);
        assert!(matches!(result, Err(CompressionError::ByteLimitTooSmall { .. })));
    }

    #[test]
    fn test_resize_image() {
        let img = create_test_image(1000, 800);

        let scaled = resize_image(&img, 0.5).unwrap();
        assert_eq!(scaled.width(), 500);
        assert_eq!(scaled.height(), 400);
    }

    #[test]
    fn test_resize_image_no_scaling() {
        let img = create_test_image(800, 600);

        let scaled = resize_image(&img, 1.0).unwrap();
        assert_eq!(scaled.width(), 800);
        assert_eq!(scaled.height(), 600);
    }

    #[test]
    fn test_encode_jpeg() {
        let img = create_test_image(100, 100);

        let encoded = encode_image(&img, 90, 1.0, ImageFormat::Jpeg).unwrap();
        assert!(!encoded.is_empty());
    }

}
