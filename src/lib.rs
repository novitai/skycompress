//! Image compression with byte-limit targeting
//!
//! This library provides image compression with a target byte size using
//! binary search optimization on both quality and image dimensions.
//!
//! Supports two backends:
//! - **image crate** (default, pure Rust): Uses image crate
//! - **OpenCV** (optional, fast): Uses opencv-rust bindings (enable with `opencv-backend` feature)

#[cfg(feature = "opencv-backend")]
pub mod opencv_backend;

#[cfg(feature = "opencv-backend")]
pub use opencv_backend::{
    cache_stats, clear_cache, compress_image, compress_image_adaptive, compress_image_parallel,
    compress_rtsp_frame, compress_rtsp_frame_with_cache, compress_rtsp_frames_batch,
    compress_with_cache, load_image, load_image_mmap, save_image, CompressionError, ImageFormat,
    InterpolationMode, Result, SimilarityCache, SimilarityCacheStats,
};

#[cfg(all(feature = "image-backend", not(feature = "opencv-backend")))]
pub mod image_backend;

#[cfg(all(feature = "image-backend", not(feature = "opencv-backend")))]
pub use image_backend::{compress_image, ImageFormat, CompressionError, Result};

// Python bindings
#[cfg(feature = "python")]
pub mod python;
