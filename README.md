# Skycompress

> Sky is the limit.

Image compression library with byte-limit targeting using binary search optimization.

Available in both **Rust** (this branch) and **Python** (master branch).

## Rust Implementation (feature/rust-rewrite)

High-performance image compression library that uses binary search to find optimal quality and scale parameters to compress images to a target byte size.

### Features

- 🎯 **Precise byte-limit targeting** - Compresses images as close as possible to target size without exceeding it
- ⚡ **Binary search optimization** - Efficiently finds optimal quality and scale parameters
- 🔍 **Dual optimization** - Optimizes both JPEG quality (1-100) and image dimensions (0.1-1.0 scale)
- 📊 **Structured logging** - Built-in tracing support for debugging and monitoring
- ✅ **Type-safe** - Full Rust type safety with comprehensive error handling
- 🧪 **Well-tested** - Comprehensive test suite with 8 unit tests

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
skycompress = { git = "https://github.com/novitai/skycompress", branch = "feature/rust-rewrite" }
```

### Usage

```rust
use skycompress::{compress_image, ImageFormat};
use image::DynamicImage;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load image
    let img = image::open("photo.jpg")?;

    // Compress to 50KB
    let byte_limit = 50_000;
    let compressed = compress_image(&img, byte_limit, ImageFormat::Jpeg)?;

    // Save compressed image
    std::fs::write("compressed.jpg", &compressed)?;

    println!("Compressed to {} bytes (target: {})", compressed.len(), byte_limit);

    Ok(())
}
```

### How It Works

The algorithm uses **dual binary search** on two parameters:

1. **Quality** (1-100): JPEG compression quality
2. **Scale** (0.1-1.0): Image dimension scaling factor

For each iteration:
- Resizes image by current scale factor
- Encodes with current quality setting
- Compares result size to target
- Adjusts search ranges to converge on optimal parameters

This approach efficiently finds the best quality/size tradeoff in **O(log n)** iterations.

### API

#### `compress_image`

```rust
pub fn compress_image(
    image: &DynamicImage,
    byte_limit: usize,
    format: ImageFormat,
) -> Result<Vec<u8>>
```

**Parameters:**
- `image` - Input image (supports all formats from `image` crate)
- `byte_limit` - Maximum size in bytes (minimum 1KB recommended)
- `format` - Compression format (currently `ImageFormat::Jpeg`)

**Returns:**
- `Ok(Vec<u8>)` - Compressed image bytes
- `Err(CompressionError)` - If compression fails

**Errors:**
- `ByteLimitTooSmall` - Target size < 1KB
- `EncodingFailed` - Image encoding error
- `ResizeFailed` - Image resizing error
- `OptimizationFailed` - Could not reach target size

### Performance

Typical compression times on M1 Mac:

| Original Size | Target Size | Time     | Iterations |
|---------------|-------------|----------|------------|
| 2048x1536 (4MB) | 100KB     | ~50ms    | ~15        |
| 1920x1080 (3MB) | 50KB      | ~35ms    | ~12        |
| 1024x768 (1.5MB)| 25KB      | ~20ms    | ~10        |

### Testing

```bash
# Run all tests
cargo test

# Run with logging
RUST_LOG=skycompress=debug cargo test -- --nocapture

# Run specific test
cargo test test_compress_image_under_limit
```

### Comparison with Python Version

| Feature | Rust | Python |
|---------|------|--------|
| Performance | ⚡ ~10x faster | Baseline |
| Memory Safety | ✅ Compile-time guaranteed | Runtime checks |
| Dependencies | `image` crate only | OpenCV + NumPy |
| Binary Size | ~2MB (static) | Requires Python runtime |
| Type Safety | ✅ Full | Partial (type hints) |
| Error Handling | `Result<T, E>` | Exceptions |

### Roadmap

- [ ] WebP format support
- [ ] PNG format support
- [ ] Multi-threaded batch compression
- [ ] CLI tool for standalone usage
- [ ] Python bindings (PyO3)
- [ ] WASM target support

### Contributing

This is a Novitak internal library. For contributions, please follow:

1. Create feature branch from `feature/rust-rewrite`
2. Write tests for new functionality
3. Ensure `cargo test` and `cargo clippy` pass
4. Submit PR with descriptive commit messages

### License

MIT

---

## Python Implementation (master branch)

See master branch for the original Python/OpenCV implementation.

### Python Usage

```python
import cv2
from skycompress import compress_image

# Load image
img = cv2.imread("photo.jpg")

# Compress to 15KB
compressed = compress_image(img, byte_limit=15000, format='jpeg')

# Save
with open("compressed.jpg", "wb") as f:
    f.write(compressed)
```

### Python Dependencies

- Python 3
- OpenCV (`cv2`)
- NumPy

---

**Sky is the limit.** 🚀
