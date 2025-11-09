//! Error types for OpenCV backend

use thiserror::Error;

/// Errors that can occur during OpenCV compression
#[derive(Error, Debug)]
pub enum CompressionError {
    #[error("OpenCV error: {0}")]
    OpenCVError(#[from] opencv::Error),

    #[error("Byte limit too small: {limit} bytes (minimum ~1KB recommended)")]
    ByteLimitTooSmall { limit: usize },

    #[error("Compression optimization failed: could not reach target size")]
    OptimizationFailed,

    #[error("Invalid image format")]
    InvalidFormat,
}

/// Result type for compression operations
pub type Result<T> = std::result::Result<T, CompressionError>;
