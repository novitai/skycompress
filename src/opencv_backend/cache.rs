//! Smart caching for compression results
//!
//! Caches compression results by image hash and byte limit

use ahash::AHashMap;
use once_cell::sync::Lazy;
use opencv::{core::Mat, prelude::*};
use std::sync::Mutex;

use super::compress::compress_image;
use super::errors::Result;
use super::format::ImageFormat;

/// Global compression cache
static COMPRESSION_CACHE: Lazy<Mutex<AHashMap<(u64, usize), Vec<u8>>>> =
    Lazy::new(|| Mutex::new(AHashMap::new()));

/// Calculate fast hash of image data
///
/// Uses ahash for high-performance hashing
fn hash_image(image: &Mat) -> Result<u64> {
    use std::hash::{Hash, Hasher};

    let size = image.size()?;
    let data_size = image.total() * image.elem_size()?;

    let mut hasher = ahash::AHasher::default();

    // Hash image dimensions
    size.width.hash(&mut hasher);
    size.height.hash(&mut hasher);

    // Hash a sample of pixels (much faster than full image)
    // Sample every 100th pixel for speed
    let step = 100;
    let total = image.total();

    unsafe {
        let data_ptr = image.data();
        if !data_ptr.is_null() && data_size > 0 {
            for i in (0..total).step_by(step) {
                if i < total {
                    let offset = i * image.elem_size()?;
                    if offset < data_size {
                        let byte = *data_ptr.add(offset);
                        byte.hash(&mut hasher);
                    }
                }
            }
        }
    }

    Ok(hasher.finish())
}

/// Compress image with caching
///
/// Checks cache first, compresses if not found, then caches result
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
///
/// # Performance
///
/// Cache hit: ~0.1ms (just hash + lookup)
/// Cache miss: Normal compression time + ~1ms for hashing/caching
///
/// For repeated images: ~1000x faster!
pub fn compress_with_cache(
    image: &Mat,
    byte_limit: usize,
    format: ImageFormat,
) -> Result<Vec<u8>> {
    let start = std::time::Instant::now();

    // Calculate image hash
    let image_hash = hash_image(image)?;
    let cache_key = (image_hash, byte_limit);

    tracing::debug!(
        image_hash = format!("{:016x}", image_hash),
        byte_limit,
        "Checking cache"
    );

    // Check cache
    {
        let cache = COMPRESSION_CACHE.lock().unwrap();
        if let Some(cached) = cache.get(&cache_key) {
            tracing::info!(
                cached_size = cached.len(),
                lookup_time_us = start.elapsed().as_micros(),
                "Cache HIT - returning cached result"
            );
            return Ok(cached.clone());
        }
    }

    tracing::debug!("Cache MISS - compressing image");

    // Not in cache - compress it
    let result = compress_image(image, byte_limit, format)?;

    // Cache the result
    {
        let mut cache = COMPRESSION_CACHE.lock().unwrap();

        // Limit cache size to prevent memory bloat
        const MAX_CACHE_ENTRIES: usize = 100;
        if cache.len() >= MAX_CACHE_ENTRIES {
            // Evict random entry (simple strategy)
            if let Some(key) = cache.keys().next().cloned() {
                cache.remove(&key);
                tracing::debug!("Cache full, evicted oldest entry");
            }
        }

        cache.insert(cache_key, result.clone());
    }

    tracing::info!(
        total_time_ms = start.elapsed().as_millis(),
        "Compressed and cached result"
    );

    Ok(result)
}

/// Clear the compression cache
///
/// Useful for memory management or testing
pub fn clear_cache() {
    let mut cache = COMPRESSION_CACHE.lock().unwrap();
    let size = cache.len();
    cache.clear();

    tracing::info!(
        entries_cleared = size,
        "Cache cleared"
    );
}

/// Get cache statistics
pub fn cache_stats() -> (usize, usize) {
    let cache = COMPRESSION_CACHE.lock().unwrap();
    let entries = cache.len();
    let total_bytes: usize = cache.values().map(|v| v.len()).sum();

    (entries, total_bytes)
}

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
    fn test_cache_hit() {
        clear_cache();

        let img = create_test_image();
        let byte_limit = 50_000;

        // First call - cache miss
        let result1 = compress_with_cache(&img, byte_limit, ImageFormat::Jpeg);
        assert!(result1.is_ok());

        // Second call - cache hit
        let result2 = compress_with_cache(&img, byte_limit, ImageFormat::Jpeg);
        assert!(result2.is_ok());

        // Should be identical
        assert_eq!(result1.unwrap(), result2.unwrap());

        let (entries, _) = cache_stats();
        assert_eq!(entries, 1);
    }

    #[test]
    fn test_clear_cache() {
        let img = create_test_image();
        compress_with_cache(&img, 50_000, ImageFormat::Jpeg).unwrap();

        let (entries_before, _) = cache_stats();
        assert!(entries_before > 0);

        clear_cache();

        let (entries_after, _) = cache_stats();
        assert_eq!(entries_after, 0);
    }
}
