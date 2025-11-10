//! OpenCV-based compression backend
//!
//! High-performance compression using OpenCV bindings

mod adaptive;
mod cache;
mod compress;
mod encode;
mod errors;
mod format;
mod io;
mod io_mmap;
mod parallel;
pub mod resize;
mod rtsp;
mod similarity_cache;

pub use adaptive::compress_image_adaptive;
pub use cache::{cache_stats, clear_cache, compress_with_cache};
pub use compress::compress_image;
pub use errors::{CompressionError, Result};
pub use format::ImageFormat;
pub use io::load_image;
pub use io_mmap::{load_image_mmap, save_image};
pub use parallel::compress_image_parallel;
pub use resize::InterpolationMode;
pub use rtsp::{compress_rtsp_frame, compress_rtsp_frame_with_cache, compress_rtsp_frames_batch};
pub use similarity_cache::{CacheStats as SimilarityCacheStats, SimilarityCache};

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
