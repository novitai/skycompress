//! Compare compression performance

use skycompress::{compress_image, ImageFormat};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Load image
    println!("Loading baseimg.webp...");
    let img = image::open("baseimg.webp")?;

    println!("Original image: {}x{}", img.width(), img.height());

    // Test different byte limits
    let byte_limits = vec![10_000, 25_000, 50_000, 100_000];

    println!("\n{:<15} {:<15} {:<15} {:<15}", "Target Size", "Result Size", "Time (ms)", "Reduction");
    println!("{}", "-".repeat(60));

    for &byte_limit in &byte_limits {
        let start = Instant::now();
        let compressed = compress_image(&img, byte_limit, ImageFormat::Jpeg)?;
        let duration = start.elapsed();

        let reduction = (img.as_bytes().len() as f64) / (compressed.len() as f64);

        println!(
            "{:<15} {:<15} {:<15.2} {:<15.2}x",
            format!("{}KB", byte_limit / 1024),
            format!("{}KB ({} bytes)", compressed.len() / 1024, compressed.len()),
            duration.as_secs_f64() * 1000.0,
            reduction
        );

        // Save result
        let filename = format!("rust_compressed_{}k.jpg", byte_limit / 1024);
        std::fs::write(&filename, &compressed)?;
        println!("  Saved to {}", filename);
    }

    Ok(())
}
