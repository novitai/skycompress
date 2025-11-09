//! OpenCV-based compression backend
//!
//! High-performance compression using OpenCV bindings

mod opencv_backend_compress;
mod opencv_backend_encode;
mod opencv_backend_errors;
mod opencv_backend_format;
mod opencv_backend_io;
mod opencv_backend_resize;

pub use opencv_backend_compress::compress_image;
pub use opencv_backend_errors::{CompressionError, Result};
pub use opencv_backend_format::ImageFormat;
pub use opencv_backend_io::load_image;

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::Mat;

    fn create_test_image() -> Mat {
        Mat::new_rows_cols_with_default(
            800,
            1200,
            opencv::core::CV_8UC3,
            opencv::core::Scalar::new(128.0, 128.0, 128.0, 0.0),
        )
        .unwrap()
    }

    #[test]
    fn test_compress_image_under_limit() {
        let img = create_test_image();
        let byte_limit = 50_000;

        let result = compress_image(&img, byte_limit, ImageFormat::Jpeg);
        assert!(result.is_ok());

        let compressed = result.unwrap();
        assert!(compressed.len() <= byte_limit);
    }

    #[test]
    fn test_byte_limit_too_small() {
        let img = create_test_image();
        let result = compress_image(&img, 500, ImageFormat::Jpeg);

        assert!(matches!(
            result,
            Err(CompressionError::ByteLimitTooSmall { .. })
        ));
    }
}
