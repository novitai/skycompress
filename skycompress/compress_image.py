import logging
import time
import cv2  # type: ignore
import numpy as np  # type: ignore
import numpy.typing as npt  # type: ignore
from pathlib import Path


LOGGER = logging.getLogger(__name__)
LOGGER = logging.getLogger(Path(__file__).resolve().stem)
LOGGER.setLevel(logging.INFO)  # for debugging set logging to DEBUG
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logging.getLogger().addHandler(ch)


def compress_image(original_img: npt.NDArray[np.float64], byte_limit: int) -> bytearray:
    """
    Function to compress the image to a set number of bytes

    Inputs
    original_img = RGB image to be compressed, as a numpy array
    byte_limit = Int defining max number of bytes that output image should be

    Outputs
    jpeg_data = Compressed image, encoded as a jpeg format byte array
    """

    # Function variables
    start_time = time.perf_counter()
    byte_limit = byte_limit  # 340 for Iridium, 3800 for FiPy
    jpeg_quality = 100

    # Save the initial image chip for a size on disk reference
    _, jpeg_data = cv2.imencode('.jpg', original_img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    jpeg_data = bytearray(jpeg_data)
    out_image_size = len(jpeg_data)

    min_quality, max_quality = 0, 100
    min_dimension, max_dimension = 0.1, 1.0

    best_quality = min_quality
    best_dimension = min_dimension

    try:
        while min_quality <= max_quality and min_dimension <= max_dimension:
            # tune the mid quality option to balance what ratio you want quality & dimension
            mid_quality = (min_quality + max_quality) // 2
            # tune the mid quality option to balance what ratio you want quality / dimension
            mid_dimension = (min_dimension + max_dimension) / 2

            new_img = cv2.resize(original_img, (0, 0), fx=mid_dimension, fy=mid_dimension)
            _, jpeg_data = cv2.imencode('.jpg', new_img, [int(cv2.IMWRITE_JPEG_QUALITY), mid_quality])
            out_image_size = len(bytearray(jpeg_data))
            if out_image_size == byte_limit:
                return jpeg_data  # Exit early if we've hit the byte limit exactly
            elif out_image_size < byte_limit:
                if mid_quality > best_quality:
                    best_quality = mid_quality
                    best_dimension = mid_dimension
                min_quality = mid_quality + 1
                min_dimension = mid_dimension + 0.01
            else:
                max_quality = mid_quality - 1
                max_dimension = mid_dimension - 0.01

        # Recreate the image with the best parameters found
        best_img = cv2.resize(original_img, (0, 0), fx=best_dimension, fy=best_dimension)
        _, best_jpeg_data = cv2.imencode('.jpg', best_img, [int(cv2.IMWRITE_JPEG_QUALITY), best_quality])
        end_time = time.perf_counter()
        LOGGER.info(f'image compression took {end_time - start_time} seconds')
        return best_jpeg_data
    except Exception as e:
        LOGGER.warning(f'Failed to compress: \n {e}')
        return bytearray(b'')
