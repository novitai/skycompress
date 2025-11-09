//! Compare OpenCV compression performance

use skycompress::{compress_image, load_image, ImageFormat};
use opencv::prelude::*;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Load image
    println!("Loading baseimg.webp with OpenCV...");
    let img = load_image("baseimg.webp")?;

    let size = img.size()?;
    println!("Original image: {}x{}", size.width, size.height);

    // Test different byte limits
    let byte_limits = vec![10_000, 25_000, 50_000, 100_000];

    println!("\n{:<15} {:<15} {:<15} {:<15}", "Target Size", "Result Size", "Time (ms)", "Quality+Scale");
    println!("{}", "-".repeat(70));

    for &byte_limit in &byte_limits {
        let start = Instant::now();
        let compressed = compress_image(&img, byte_limit, ImageFormat::Jpeg)?;
        let duration = start.elapsed();

        println!(
            "{:<15} {:<15} {:<15.2}",
            format!("{}KB", byte_limit / 1024),
            format!("{}KB ({} bytes)", compressed.len() / 1024, compressed.len()),
            duration.as_secs_f64() * 1000.0,
        );

        // Save result
        let filename = format!("opencv_compressed_{}k.jpg", byte_limit / 1024);
        std::fs::write(&filename, &compressed)?;
        println!("  Saved to {}", filename);
    }

    Ok(())
}
