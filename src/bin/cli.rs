//! Skycompress CLI - Image compression with byte-limit targeting
//!
//! Usage:
//!   skycompress <input> <output> --target-size 50KB
//!   skycompress photo.jpg compressed.jpg -t 100000

use clap::Parser;
use skycompress::{compress_image, ImageFormat, Result};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "skycompress")]
#[command(author = "Novitak Team")]
#[command(version = "0.1.0")]
#[command(about = "Image compression with byte-limit targeting", long_about = None)]
struct Cli {
    /// Input image path
    #[arg(value_name = "INPUT")]
    input: PathBuf,

    /// Output image path
    #[arg(value_name = "OUTPUT")]
    output: PathBuf,

    /// Target size in bytes (supports KB, MB suffixes)
    /// Examples: 50000, 50KB, 1MB
    #[arg(short = 't', long = "target-size", value_name = "SIZE")]
    target_size: String,

    /// Output format (jpeg, webp, png)
    #[arg(short = 'f', long = "format", default_value = "jpeg")]
    format: String,

    /// Verbose output
    #[arg(short = 'v', long = "verbose")]
    verbose: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging if verbose
    if cli.verbose {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .init();
    }

    // Parse target size
    let byte_limit = parse_size(&cli.target_size)?;

    // Parse format
    let format = match cli.format.to_lowercase().as_str() {
        "jpeg" | "jpg" => ImageFormat::Jpeg,
        other => {
            eprintln!("Error: Unsupported format '{}'. Currently only 'jpeg' is supported.", other);
            std::process::exit(1);
        }
    };

    // Load image
    println!("Loading image: {}", cli.input.display());
    let img = image::open(&cli.input)
        .map_err(|e| skycompress::CompressionError::EncodingFailed(e.to_string()))?;

    let original_size = std::fs::metadata(&cli.input)
        .map(|m| m.len())
        .unwrap_or(0);

    // Compress
    println!("Compressing to target: {} bytes...", byte_limit);
    let compressed = compress_image(&img, byte_limit, format)?;

    // Save
    std::fs::write(&cli.output, &compressed)
        .map_err(|e| skycompress::CompressionError::EncodingFailed(e.to_string()))?;

    // Report
    let compressed_size = compressed.len();
    let ratio = if original_size > 0 {
        (compressed_size as f64 / original_size as f64) * 100.0
    } else {
        0.0
    };

    println!("✓ Success!");
    println!("  Original:   {} bytes", original_size);
    println!("  Compressed: {} bytes ({}%)", compressed_size, ratio as u32);
    println!("  Target:     {} bytes", byte_limit);
    println!("  Saved to:   {}", cli.output.display());

    Ok(())
}

/// Parse size string like "50KB", "1MB", "100000"
fn parse_size(size_str: &str) -> Result<usize> {
    let size_str = size_str.trim().to_uppercase();

    // Try to parse as raw number first
    if let Ok(bytes) = size_str.parse::<usize>() {
        return Ok(bytes);
    }

    // Parse with suffix (KB, MB)
    if size_str.ends_with("KB") {
        let num = size_str.trim_end_matches("KB").trim();
        let kb: f64 = num.parse()
            .map_err(|_| skycompress::CompressionError::EncodingFailed(
                format!("Invalid size format: {}", size_str)
            ))?;
        Ok((kb * 1024.0) as usize)
    } else if size_str.ends_with("MB") {
        let num = size_str.trim_end_matches("MB").trim();
        let mb: f64 = num.parse()
            .map_err(|_| skycompress::CompressionError::EncodingFailed(
                format!("Invalid size format: {}", size_str)
            ))?;
        Ok((mb * 1024.0 * 1024.0) as usize)
    } else {
        Err(skycompress::CompressionError::EncodingFailed(
            format!("Invalid size format: {}. Use bytes, KB, or MB (e.g., 50KB, 1MB)", size_str)
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_size_bytes() {
        assert_eq!(parse_size("50000").unwrap(), 50000);
        assert_eq!(parse_size("1024").unwrap(), 1024);
    }

    #[test]
    fn test_parse_size_kb() {
        assert_eq!(parse_size("50KB").unwrap(), 51200);
        assert_eq!(parse_size("1KB").unwrap(), 1024);
        assert_eq!(parse_size("0.5KB").unwrap(), 512);
    }

    #[test]
    fn test_parse_size_mb() {
        assert_eq!(parse_size("1MB").unwrap(), 1048576);
        assert_eq!(parse_size("2MB").unwrap(), 2097152);
        assert_eq!(parse_size("0.5MB").unwrap(), 524288);
    }

    #[test]
    fn test_parse_size_invalid() {
        assert!(parse_size("invalid").is_err());
        assert!(parse_size("50GB").is_err());
    }
}
