import logging
import time
from pathlib import Path

import cv2  # type: ignore
import numpy as np  # type: ignore
import numpy.typing as npt  # type: ignore
from turbojpeg import TurboJPEG, TJPF_BGR, TJFLAG_FASTDCT

# Initialize TurboJPEG
jpeg = TurboJPEG()

LOGGER = logging.getLogger(__name__)
LOGGER = logging.getLogger(Path(__file__).resolve().stem)
LOGGER.setLevel(logging.INFO)  # for debugging set logging to DEBUG
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logging.getLogger().addHandler(ch)


def compress_image(original_img: npt.NDArray[np.uint8], byte_limit: int, format: str = 'jpeg') -> bytearray:
    """
    Function to compress the image to a set number of bytes

    Inputs
    original_img = RGB image to be compressed, as a numpy array
    byte_limit = Int defining max number of bytes that output image should be
    format = Compression format, either 'jpeg' or 'webp'

    Outputs
    compressed_data = Compressed image, encoded as a jpeg or webp format byte array
    """
    start_time = time.perf_counter()
    byte_limit = byte_limit
    jpeg_quality = 100

    if format == 'jpeg':
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    elif format == 'webp':
        encode_param = [int(cv2.IMWRITE_WEBP_QUALITY), jpeg_quality]
    else:
        raise ValueError("Unsupported format. Use 'jpeg' or 'webp'.")

    _, compressed_data = cv2.imencode(f'.{format}', original_img, encode_param)
    compressed_data = bytearray(compressed_data)
    out_image_size = len(compressed_data)

    min_quality, max_quality = 0, 100
    min_dimension, max_dimension = 0.1, 1.0

    best_quality = min_quality
    best_dimension = min_dimension

    try:
        while min_quality <= max_quality and min_dimension <= max_dimension:
            mid_quality = (min_quality + max_quality) // 2
            mid_dimension = (min_dimension + max_dimension) / 2

            new_img = cv2.resize(original_img, (0, 0), fx=mid_dimension, fy=mid_dimension)

            if format == 'jpeg':
                _, compressed_data = cv2.imencode('.jpeg', new_img, [int(cv2.IMWRITE_JPEG_QUALITY), mid_quality])
            else:
                _, compressed_data = cv2.imencode('.webp', new_img, [int(cv2.IMWRITE_WEBP_QUALITY), mid_quality])

            out_image_size = len(bytearray(compressed_data))
            if out_image_size == byte_limit:
                return compressed_data  # Exit early if we've hit the byte limit exactly
            elif out_image_size < byte_limit:
                if mid_quality > best_quality:
                    best_quality = mid_quality
                    best_dimension = mid_dimension
                min_quality = mid_quality + 1
                min_dimension = mid_dimension + 0.01
            else:
                max_quality = mid_quality - 1
                max_dimension = mid_dimension - 0.01

        best_img = cv2.resize(original_img, (0, 0), fx=best_dimension, fy=best_dimension)

        if format == 'jpeg':
            best_compressed_data = jpeg.encode(best_img, quality=best_quality, flags=TJFLAG_FASTDCT)
        else:
            _, best_compressed_data = cv2.imencode('.webp', best_img, [int(cv2.IMWRITE_WEBP_QUALITY), best_quality])
            best_compressed_data = bytearray(best_compressed_data)

        end_time = time.perf_counter()
        LOGGER.info(f'image compression took {end_time - start_time} seconds')
        return best_compressed_data
    except Exception as e:
        LOGGER.warning(f'Failed to compress: \n {e}')
        return bytearray(b'')


if __name__ == "__main__":
    input_image_path = 'fig.jpeg'
    desired_size_kb = 15
    image = cv2.imread(input_image_path)

    compressed_data = compress_image(image, desired_size_kb * 1024, format=format)

    if compressed_data:
        print(f"Compressed data length: {len(compressed_data)} bytes")
