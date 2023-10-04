import numpy as np
from skycompress import compress_image  # Replace 'your_package_name' with your actual package name

def test_compress_image_basic():
    # Create a basic random image of float64 type with values between 0 and 1.
    original_img = np.random.random((100, 100, 3)).astype(np.float64)
    compressed_img = compress_image(original_img, "rgb", 5000)


    # Ensure we get some output
    assert compressed_img is not None, "Compression failed: No output received."
    # Ensure the output is a bytearray
    assert isinstance(compressed_img, np.ndarray), "Compression output type mismatch: Expected numpy array."
    # Ensure it respects the byte limit
    assert len(bytearray(compressed_img)) <= 5000, "Compression failed: Output exceeds byte limit."


def test_compress_image_shapes():
    # Square image
    square_img = np.ones((512, 512, 3), np.float64) * 255
    compressed_square = compress_image(square_img, 'rgb', 5000)
    assert len(bytearray(compressed_square)) <= 5000, "Failed for square image."

    # Rectangular image - Landscape
    landscape_img = np.ones((300, 600, 3), np.float64) * 255
    compressed_landscape = compress_image(landscape_img, 'rgb', 5000)
    assert len(bytearray(compressed_landscape)) <= 5000, "Failed for landscape image."

    # Rectangular image - Portrait
    portrait_img = np.ones((600, 300, 3), np.float64) * 255
    compressed_portrait = compress_image(portrait_img, 'rgb', 5000)
    assert len(bytearray(compressed_portrait)) <= 5000, "Failed for portrait image."


def test_compress_image_formats():
    # RGB format
    rgb_img = np.ones((512, 512, 3), np.float64) * 255
    compressed_rgb = compress_image(rgb_img, 'rgb', 5000)
    assert len(bytearray(compressed_rgb)) <= 5000, "Failed for RGB image."

    # Grey format
    grey_img = np.ones((512, 512, 1), np.float64) * 255
    compressed_grey = compress_image(grey_img, 'gry', 5000)
    assert len(bytearray(compressed_grey)) <= 5000, "Failed for Grey image."

    # Binary format
    binary_img = np.ones((512, 512, 3), np.float64) * 255
    compressed_bin = compress_image(binary_img, 'bin', 5000)
    assert len(bytearray(compressed_bin)) <= 5000, "Failed for Binary image."

def test_compress_image_edge_cases():
    # Really small image
    small_img = np.ones((50, 50, 3), np.float64) * 255
    compressed_small = compress_image(small_img, 'rgb', 5000)
    assert len(bytearray(compressed_small)) <= 5000, "Failed for small image."
    
    # Huge byte limit
    normal_img = np.ones((512, 512, 3), np.float64) * 255
    compressed_huge_limit = compress_image(normal_img, 'rgb', 100000)
    assert len(bytearray(compressed_huge_limit)) <= 100000, "Failed for huge byte limit."

    # Extremely low byte limit - this will likely fail, but it's good to be aware of the limitations
    compressed_low_limit = compress_image(normal_img, 'rgb', 10)
    assert len(bytearray(compressed_low_limit)) <= 10, "Failed for extremely low byte limit."

