//! Benchmark OpenCV backend
//!
//! Usage: cargo run --example opencv_benchmark --features opencv-backend --no-default-features

use skycompress::{compress_image, load_image, ImageFormat};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let targets = vec![
        ("15KB", 15_360),
        ("50KB", 51_200),
        ("100KB", 102_400),
    ];

    println!("Loading baseimg.webp...");
    let img = load_image("baseimg.webp")?;

    println!("\n{:<10} {:<15} {:<15}", "Target", "Time (ms)", "Result Size");
    println!("{}", "-".repeat(40));

    for (name, byte_limit) in targets {
        let start = Instant::now();
        let compressed = compress_image(&img, byte_limit, ImageFormat::Jpeg)?;
        let duration_ms = start.elapsed().as_millis();

        println!(
            "{:<10} {:<15} {:<15}",
            name,
            format!("{:.2}", duration_ms),
            format!("{} bytes", compressed.len())
        );

        // Save result
        std::fs::write(format!("rust_opencv_{}.jpg", name), &compressed)?;
    }

    Ok(())
}
