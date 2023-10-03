# Skycompress

This module provides a function to compress images to a desired byte size using OpenCV.

Sky is the limit.

## Dependencies
- Python 3
- OpenCV
- NumPy

## Usage

The primary function in this module is `compress_image()`, which compresses an image to the desired byte size.

### Parameters:

- `original_img` (numpy.ndarray): The image to be compressed, represented as a numpy array.
- `color_fmt` (str): The desired color format. This should be one of 'rgb', 'gry', or 'bin'.
- `byte_limit` (int): The maximum number of bytes that the compressed image should be.

### Returns:

- `bytearray`: The compressed image, encoded in JPEG format.

### Example:

```python
import cv2
import numpy as np

from skycompress import compress_image

# Load the image
img = cv2.imread("path_to_your_image.jpg")

# Compress the image
compressed_img_data = compress_image.compress_image(img, "rgb", 15000)  # Compress to 15000 bytes in RGB format

# Save the compressed image
with open("compressed_image.jpg", "wb") as f:
    f.write(compressed_img_data)
