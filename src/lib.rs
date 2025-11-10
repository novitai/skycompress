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
pub use opencv_backend::{compress_image, load_image, ImageFormat, CompressionError, Result};

#[cfg(all(feature = "image-backend", not(feature = "opencv-backend")))]
pub mod image_backend;

#[cfg(all(feature = "image-backend", not(feature = "opencv-backend")))]
pub use image_backend::{compress_image, ImageFormat, CompressionError, Result};

// Python bindings
#[cfg(feature = "python")]
pub mod python;
