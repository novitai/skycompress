import cv2  # type: ignore
import numpy as np  # type: ignore
import pytest  # type: ignore

from skycompress import compress_image


class ImageTester():
    def __init__(self):
        self.i = 0

    def load_image(self, image_path="data/messi_612px_high_jpg100.jpg"):
        """Load an image ."""
        image = cv2.imread(image_path)
        assert image is not None, f"Image not loaded from {image_path}."
        return image

    def run_compress_test_and_save(self, byte_limit, format='jpeg'):
        """Helper function to handle common compression testing logic."""
        img = self.load_image()
        img_shape = img.shape
        compressed_img = compress_image(img, byte_limit, format)

        assert compressed_img is not None, \
            f"Compression failed for rgb image of shape {img_shape} with byte limit {byte_limit}."
        assert isinstance(compressed_img, np.ndarray), "Compression output type mismatch: Expected numpy array."
        assert len(bytearray(compressed_img)) <= byte_limit, \
            "Compression exceeded byte limit for rgb image of shape {img_shape}."
        assert len(bytearray(compressed_img)) > byte_limit * 0.75, \
            "Compression too high ({len(bytearray(compressed_img))}) for {img_shape}."
        with open(f"compressed_image_{self.i}_{byte_limit}.{format}", "wb") as f:
            f.write(compressed_img)
        self.i += 1


# Test compress_image on a real image at different target sizes and formats
@pytest.mark.parametrize("target_size", [3000, 5000, 15000])
@pytest.mark.parametrize("format", ['jpeg', 'webp'])
def test_format_compressions(target_size, format):
    it = ImageTester()
    it.run_compress_test_and_save(target_size, format=format)
