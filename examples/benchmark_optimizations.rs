//! Benchmark comparing all optimization strategies
//!
//! Run with: cargo run --release --example benchmark_optimizations --features opencv-backend

use skycompress::*;
use opencv::prelude::*;
use std::time::Instant;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Setup tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("\n🚀 SKYCOMPRESS OPTIMIZATION BENCHMARK 🚀\n");
    println!("{}", "=".repeat(80));

    // Load test image
    let image_path = "baseimg.webp";
    println!("\n📷 Loading test image: {}", image_path);

    let img = load_image_mmap(image_path)?;
    let size = img.size()?;
    println!("   Image size: {}x{}", size.width, size.height);

    // Test different byte limits
    let byte_limits = vec![15_000, 50_000, 100_000];

    for &byte_limit in &byte_limits {
        println!("\n{}", "=".repeat(80));
        println!("🎯 TARGET SIZE: {} KB", byte_limit / 1000);
        println!("{}", "=".repeat(80));

        // 1. Sequential Binary Search (Original)
        println!("\n1️⃣  Sequential Binary Search (Original)");
        let start = Instant::now();
        let result_seq = compress_image(&img, byte_limit, ImageFormat::Jpeg)?;
        let time_seq = start.elapsed();
        println!("   ⏱  Time: {:?}", time_seq);
        println!("   📦 Size: {} bytes", result_seq.len());

        // 2. Adaptive Binary Search
        println!("\n2️⃣  Adaptive Binary Search");
        let start = Instant::now();
        let result_adaptive = compress_image_adaptive(&img, byte_limit, ImageFormat::Jpeg)?;
        let time_adaptive = start.elapsed();
        let speedup_adaptive = time_seq.as_secs_f64() / time_adaptive.as_secs_f64();
        println!("   ⏱  Time: {:?}", time_adaptive);
        println!("   📦 Size: {} bytes", result_adaptive.len());
        println!("   🚀 Speedup: {:.2}x faster", speedup_adaptive);

        // 3. Parallel Binary Search
        println!("\n3️⃣  Parallel Binary Search (Rayon)");
        let start = Instant::now();
        let result_parallel = compress_image_parallel(&img, byte_limit, ImageFormat::Jpeg)?;
        let time_parallel = start.elapsed();
        let speedup_parallel = time_seq.as_secs_f64() / time_parallel.as_secs_f64();
        println!("   ⏱  Time: {:?}", time_parallel);
        println!("   📦 Size: {} bytes", result_parallel.len());
        println!("   🚀 Speedup: {:.2}x faster", speedup_parallel);
        println!("   🧵 Threads: {}", rayon::current_num_threads());

        // 4. Cached Compression (second call)
        println!("\n4️⃣  Cached Compression");

        // First call (cache miss)
        let start = Instant::now();
        let _ = compress_with_cache(&img, byte_limit, ImageFormat::Jpeg)?;
        let time_cache_miss = start.elapsed();
        println!("   ⏱  Cache MISS: {:?}", time_cache_miss);

        // Second call (cache hit)
        let start = Instant::now();
        let result_cache = compress_with_cache(&img, byte_limit, ImageFormat::Jpeg)?;
        let time_cache_hit = start.elapsed();
        let speedup_cache = time_seq.as_secs_f64() / time_cache_hit.as_secs_f64();
        println!("   ⏱  Cache HIT:  {:?}", time_cache_hit);
        println!("   📦 Size: {} bytes", result_cache.len());
        println!("   🚀 Speedup: {:.2}x faster", speedup_cache);

        let (entries, total_bytes) = cache_stats();
        println!("   💾 Cache: {} entries, {} KB total", entries, total_bytes / 1000);

        // Summary
        println!("\n📊 SUMMARY FOR {} KB TARGET:", byte_limit / 1000);
        println!("   Sequential:  {:?}", time_seq);
        println!("   Adaptive:    {:?} ({:.2}x speedup)", time_adaptive, speedup_adaptive);
        println!("   Parallel:    {:?} ({:.2}x speedup)", time_parallel, speedup_parallel);
        println!("   Cache Hit:   {:?} ({:.2}x speedup)", time_cache_hit, speedup_cache);
    }

    // Overall summary
    println!("\n{}", "=".repeat(80));
    println!("✅ BENCHMARK COMPLETE");
    println!("{}", "=".repeat(80));
    println!("\n💡 RECOMMENDATIONS:");
    println!("   • For batch processing: Use Parallel");
    println!("   • For repeated images: Use Cache");
    println!("   • For faster convergence: Use Adaptive");
    println!("   • For production: Combine Parallel + Cache");
    println!();

    // Clear cache
    clear_cache();

    Ok(())
}
