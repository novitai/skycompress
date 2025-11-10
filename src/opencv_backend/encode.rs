//! Image encoding utilities for OpenCV backend

use opencv::{
    core::{Mat, Vector},
    imgcodecs::imencode,
    prelude::VectorToVec,
};

use super::errors::Result;
use super::format::ImageFormat;

/// Encode image with specified quality
///
/// # Arguments
///
/// * `image` - Image to encode
/// * `quality` - Quality level (1-100)
/// * `format` - Output format (JPEG or WebP)
///
/// # Returns
///
/// Encoded image data as bytes
pub(crate) fn encode_image(image: &Mat, quality: i32, format: ImageFormat) -> Result<Vec<u8>> {
    let mut buffer = Vector::<u8>::new();
    let params = Vector::from_slice(&[format.quality_param(), quality]);

    imencode(format.extension(), image, &mut buffer, &params)?;

    Ok(buffer.to_vec())
}
