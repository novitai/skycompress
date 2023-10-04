import numpy as np
from skycompress import compress_image  
import pytest

def generate_image(shape, value=255, dtype=np.float64):
    """Helper function to generate an image with given parameters."""
    return np.ones(shape, dtype) * value

def run_compression_test(img_shape, format, byte_limit):
    """Helper function to handle common compression testing logic."""
    img = generate_image(img_shape)
    compressed_img = compress_image(img, format, byte_limit)
    
    assert compressed_img is not None, f"Compression failed for {format} image of shape {img_shape} with byte limit {byte_limit}."
    assert isinstance(compressed_img, np.ndarray), "Compression output type mismatch: Expected numpy array."
    assert len(bytearray(compressed_img)) <= byte_limit, f"Compression exceeded byte limit for {format} image of shape {img_shape}."

# Test basic functionality of compress_image
def test_basic_compression():
    run_compression_test((100, 100, 3), 'rgb', 5000)

# Test compress_image with different shapes of input images
@pytest.mark.parametrize("img_shape", [(512, 512, 3), (300, 600, 3), (600, 300, 3)])
def test_shapes_compression(img_shape):
    run_compression_test(img_shape, 'rgb', 5000)

# Test compress_image with different image formats
@pytest.mark.parametrize("img_format", ['rgb', 'gry', 'bin'])
def test_formats_compression(img_format):
    run_compression_test((512, 512, 3 if img_format != 'gry' else 1), img_format, 5000)

# Test compression of small image
def test_small_image_compression():
    run_compression_test((50, 50, 3), 'rgb', 5000)

# Test compression with a large byte limit
def test_large_byte_limit_compression():
    run_compression_test((512, 512, 3), 'rgb', 100000)

# Test compression with an extremely small byte limit
@pytest.mark.xfail(reason="The compression might not be able to compress to such a low byte limit.")
def test_extremely_small_byte_limit_compression():
    run_compression_test((512, 512, 3), 'rgb', 10)
