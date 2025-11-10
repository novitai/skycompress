//! Python bindings for skycompress
//!
//! Usage from Python:
//!   import skycompress
//!   skycompress.compress_image("input.jpg", "output.jpg", 50000)

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::path::Path;

/// Compress an image file to a target byte size
///
/// Args:
///     input_path (str): Path to input image
///     output_path (str): Path to output image
///     target_size (int): Target size in bytes
///     format (str, optional): Output format ("jpeg", "webp", "png"). Defaults to "jpeg".
///
/// Returns:
///     int: Actual compressed size in bytes
///
/// Raises:
///     ValueError: If compression fails or parameters are invalid
///
/// Example:
///     >>> import skycompress
///     >>> size = skycompress.compress_image("photo.jpg", "compressed.jpg", 50000)
///     >>> print(f"Compressed to {size} bytes")
#[pyfunction]
#[pyo3(signature = (input_path, output_path, target_size, format="jpeg"))]
fn compress_image(
    input_path: &str,
    output_path: &str,
    target_size: usize,
    format: &str,
) -> PyResult<usize> {
    // Validate target size
    if target_size < 1024 {
        return Err(PyValueError::new_err(
            "Target size must be at least 1KB (1024 bytes)"
        ));
    }

    // Parse format
    let img_format = match format.to_lowercase().as_str() {
        "jpeg" | "jpg" => crate::ImageFormat::Jpeg,
        other => {
            return Err(PyValueError::new_err(
                format!("Unsupported format '{}'. Currently only 'jpeg' is supported.", other)
            ));
        }
    };

    // Load image
    let img = image::open(input_path)
        .map_err(|e| PyValueError::new_err(format!("Failed to load image: {}", e)))?;

    // Compress
    let compressed = crate::compress_image(&img, target_size, img_format)
        .map_err(|e| PyValueError::new_err(format!("Compression failed: {}", e)))?;

    // Save
    std::fs::write(output_path, &compressed)
        .map_err(|e| PyValueError::new_err(format!("Failed to save image: {}", e)))?;

    Ok(compressed.len())
}

/// Compress image bytes to a target byte size
///
/// Args:
///     image_bytes (bytes): Input image as bytes
///     target_size (int): Target size in bytes
///     format (str, optional): Output format ("jpeg", "webp", "png"). Defaults to "jpeg".
///
/// Returns:
///     bytes: Compressed image bytes
///
/// Raises:
///     ValueError: If compression fails or parameters are invalid
///
/// Example:
///     >>> with open("photo.jpg", "rb") as f:
///     ...     input_bytes = f.read()
///     >>> compressed = skycompress.compress_bytes(input_bytes, 50000)
///     >>> with open("compressed.jpg", "wb") as f:
///     ...     f.write(compressed)
#[pyfunction]
#[pyo3(signature = (image_bytes, target_size, format="jpeg"))]
fn compress_bytes(
    image_bytes: &[u8],
    target_size: usize,
    format: &str,
) -> PyResult<Vec<u8>> {
    // Validate target size
    if target_size < 1024 {
        return Err(PyValueError::new_err(
            "Target size must be at least 1KB (1024 bytes)"
        ));
    }

    // Parse format
    let img_format = match format.to_lowercase().as_str() {
        "jpeg" | "jpg" => crate::ImageFormat::Jpeg,
        other => {
            return Err(PyValueError::new_err(
                format!("Unsupported format '{}'. Currently only 'jpeg' is supported.", other)
            ));
        }
    };

    // Load image from bytes
    let img = image::load_from_memory(image_bytes)
        .map_err(|e| PyValueError::new_err(format!("Failed to load image: {}", e)))?;

    // Compress
    let compressed = crate::compress_image(&img, target_size, img_format)
        .map_err(|e| PyValueError::new_err(format!("Compression failed: {}", e)))?;

    Ok(compressed)
}

/// Get library version
///
/// Returns:
///     str: Version string (e.g., "0.1.0")
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Python module initialization
#[pymodule]
fn skycompress(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compress_image, m)?)?;
    m.add_function(wrap_pyfunction!(compress_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__doc__", "Image compression with byte-limit targeting using binary search optimization")?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert_eq!(version(), env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn test_compress_bytes_validates_target_size() {
        let result = compress_bytes(&[0u8; 100], 512, "jpeg");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("at least 1KB"));
    }

    #[test]
    fn test_compress_bytes_validates_format() {
        let result = compress_bytes(&[0u8; 100], 2048, "invalid");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unsupported format"));
    }
}
