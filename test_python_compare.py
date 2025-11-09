#!/usr/bin/env python3
"""Compare Python skycompress performance"""

import time
import cv2
from skycompress import compress_image

def main():
    # Load image
    print("Loading baseimg.webp...")
    img = cv2.imread("baseimg.webp")

    if img is None:
        print("ERROR: Could not load baseimg.webp")
        return

    print(f"Original image: {img.shape[1]}x{img.shape[0]}")

    # Test different byte limits
    byte_limits = [10_000, 25_000, 50_000, 100_000]

    print(f"\n{'Target Size':<15} {'Result Size':<15} {'Time (ms)':<15} {'Reduction':<15}")
    print("-" * 60)

    for byte_limit in byte_limits:
        start = time.perf_counter()
        compressed = compress_image(img, byte_limit, format='jpeg')
        duration = (time.perf_counter() - start) * 1000  # Convert to ms

        # Calculate reduction
        original_size = img.nbytes
        compressed_size = len(compressed)
        reduction = original_size / compressed_size

        print(f"{byte_limit // 1024}KB{'':<12} "
              f"{compressed_size // 1024}KB ({compressed_size} bytes){'':<15} "
              f"{duration:<15.2f} "
              f"{reduction:<15.2f}x")

        # Save result
        filename = f"python_compressed_{byte_limit // 1024}k.jpg"
        with open(filename, "wb") as f:
            f.write(compressed)
        print(f"  Saved to {filename}")

if __name__ == "__main__":
    main()
