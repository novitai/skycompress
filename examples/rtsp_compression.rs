//! RTSP stream compression example
//!
//! Demonstrates optimal compression for real-time video streams
//!
//! Run with: cargo run --release --example rtsp_compression --features opencv-backend

use opencv::prelude::*;
use skycompress::*;
use std::time::Instant;

fn create_mock_frame(width: i32, height: i32) -> opencv::core::Mat {
    opencv::core::Mat::new_rows_cols_with_default(
        height,
        width,
        opencv::core::CV_8UC3,
        opencv::core::Scalar::new(128.0, 128.0, 128.0, 0.0),
    )
    .unwrap()
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Setup tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("\n🎥 SKYCOMPRESS - RTSP STREAM COMPRESSION 🎥\n");
    println!("{}", "=".repeat(80));

    // Scenario 1: Single camera stream
    println!("\n📹 SCENARIO 1: Single Camera Stream (ARGONAUT-01)");
    println!("{}", "-".repeat(80));

    let frame = create_mock_frame(1200, 800);
    let byte_limit = 50_000;

    println!("   Frame size: 1200x800");
    println!("   Target size: {} KB", byte_limit / 1000);

    // Simulate 100 frames
    let num_frames = 100;
    let start = Instant::now();

    for i in 0..num_frames {
        let compressed = compress_rtsp_frame(&frame, byte_limit, ImageFormat::Jpeg)?;

        if i == 0 {
            println!("   First frame compressed to: {} bytes", compressed.len());
        }
    }

    let elapsed = start.elapsed();
    let avg_per_frame = elapsed.as_millis() as f64 / num_frames as f64;
    let fps = 1000.0 / avg_per_frame;

    println!("\n   Results:");
    println!("   ├─ Frames: {}", num_frames);
    println!("   ├─ Total time: {:?}", elapsed);
    println!("   ├─ Avg per frame: {:.2}ms", avg_per_frame);
    println!("   └─ Throughput: {:.1} FPS", fps);

    // Scenario 2: Multiple camera streams
    println!("\n\n📹 SCENARIO 2: Multiple Camera Streams (4 ARGONAUTs)");
    println!("{}", "-".repeat(80));

    let num_cameras = 4;
    let frames: Vec<_> = (0..num_cameras)
        .map(|_| create_mock_frame(1200, 800))
        .collect();

    println!("   Number of cameras: {}", num_cameras);
    println!("   Frame size: 1200x800 each");
    println!("   Target size: {} KB each", byte_limit / 1000);

    // Simulate 50 batches
    let num_batches = 50;
    let start = Instant::now();

    for _ in 0..num_batches {
        let _ = compress_rtsp_frames_batch(&frames, byte_limit, ImageFormat::Jpeg)?;
    }

    let elapsed = start.elapsed();
    let total_frames = num_batches * num_cameras;
    let avg_per_batch = elapsed.as_millis() as f64 / num_batches as f64;
    let avg_per_frame = elapsed.as_millis() as f64 / total_frames as f64;
    let total_fps = 1000.0 / avg_per_frame;

    println!("\n   Results:");
    println!("   ├─ Batches: {}", num_batches);
    println!("   ├─ Total frames: {}", total_frames);
    println!("   ├─ Total time: {:?}", elapsed);
    println!("   ├─ Avg per batch: {:.2}ms", avg_per_batch);
    println!("   ├─ Avg per frame: {:.2}ms", avg_per_frame);
    println!("   └─ Total throughput: {:.1} FPS", total_fps);

    // Comparison
    println!("\n\n📊 PERFORMANCE COMPARISON");
    println!("{}", "=".repeat(80));

    let sequential_time = avg_per_frame * num_cameras as f64;
    let parallel_speedup = sequential_time / avg_per_batch;

    println!("\n   Sequential (no batching):");
    println!("   └─ {} cameras × {:.2}ms = {:.2}ms per cycle",
        num_cameras, avg_per_frame, sequential_time);

    println!("\n   Parallel (batching):");
    println!("   └─ {:.2}ms per cycle ({:.2}x speedup!)", avg_per_batch, parallel_speedup);

    // Real-world application
    println!("\n\n💡 REAL-WORLD APPLICATION");
    println!("{}", "=".repeat(80));

    println!("\n   Recommended usage for TAK Listener:");
    println!();
    println!("   ```rust");
    println!("   // Single camera");
    println!("   loop {{");
    println!("       let frame = rtsp_stream.read_frame()?;");
    println!("       let compressed = compress_rtsp_frame(&frame, 50_000, ImageFormat::Jpeg)?;");
    println!("       send_to_tak(compressed)?;");
    println!("   }}");
    println!();
    println!("   // Multiple cameras (parallel)");
    println!("   let cameras = vec![argonaut_01, argonaut_02, argonaut_03, argonaut_04];");
    println!("   loop {{");
    println!("       let frames: Vec<Mat> = cameras.iter()");
    println!("           .map(|cam| cam.read_frame())");
    println!("           .collect()?;");
    println!();
    println!("       let compressed = compress_rtsp_frames_batch(&frames, 50_000, ImageFormat::Jpeg)?;");
    println!();
    println!("       for (i, data) in compressed.iter().enumerate() {{");
    println!("           send_to_tak(i, data)?;");
    println!("       }}");
    println!("   }}");
    println!("   ```");

    println!("\n{}", "=".repeat(80));
    println!("✅ RTSP compression benchmark complete!\n");

    Ok(())
}
