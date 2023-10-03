import logging
import cv2  # type: ignore
import numpy as np
import numpy.typing as npt

LOGGER = logging.getLogger(__name__)


def compress_image(original_img: npt.NDArray[np.float64], color_fmt: str, byte_limit: int) -> bytearray:
    """
    Function to compress the image to a set number of bytes

    Inputs
    original_img = Image to be compressed, as a numpy array
    color_fmt = String to define color format ('rgb', 'gry', 'bin')
    bbyte_limit = Int defining max number of bytes that output image should be

    Outputs
    jpeg_data = Compressed image, encoded as a jpeg format byte array
    """

    # Function variables
    byte_limit = byte_limit  # 340 for Iridium, 3800 for FiPy
    jpeg_quality = 100
    color_fmt = color_fmt  # bin, gry or rgb

    # Convert image to selected color format
    if color_fmt == 'bin':

        # Save the initial image chip for a size on disk reference
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        _, original_img = cv2.threshold(original_img, 127, 255, cv2.THRESH_BINARY)
        _, jpeg_data = cv2.imencode('.jpg', original_img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        jpeg_data = bytearray(jpeg_data)
        outImageSize = len(jpeg_data)

    if color_fmt == 'gry':

        # Save the initial image chip for a size on disk reference
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        _, jpeg_data = cv2.imencode('.jpg', original_img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        jpeg_data = bytearray(jpeg_data)
        outImageSize = len(jpeg_data)

    if color_fmt == 'rgb':

        # Save the initial image chip for a size on disk reference
        _, jpeg_data = cv2.imencode('.jpg', original_img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        jpeg_data = bytearray(jpeg_data)
        outImageSize = len(jpeg_data)

    else:
        LOGGER.warning("Not a valid color format")

    min_quality, max_quality = 0, 100
    min_dimension, max_dimension = 0.1, 1.0

    best_quality = min_quality
    best_dimension = min_dimension
    best_size = float('inf')

    while min_quality <= max_quality and min_dimension <= max_dimension:
        # tune the mid quality option to balance what ratio you want quality & dimension
        mid_quality = (min_quality + max_quality ) // 2
        # tune the mid quality option to balance what ratio you want quality / dimension
        mid_dimension = (max_dimension + max_dimension) / 2
        
        new_img = cv2.resize(original_img, (0, 0), fx=mid_dimension, fy=mid_dimension)
        _, jpeg_data = cv2.imencode('.jpg', new_img, [int(cv2.IMWRITE_JPEG_QUALITY), mid_quality])
        outImageSize = len(bytearray(jpeg_data))
        if outImageSize <= byte_limit:
            return jpeg_data  # Exit early if we've hit the byte limit exactly
        elif outImageSize < byte_limit:
            if mid_quality > best_quality:
                best_quality = mid_quality
                best_dimension = mid_dimension
                best_size = outImageSize
            min_quality = mid_quality + 10
            min_dimension = mid_dimension + 1
        else:
            max_quality = mid_quality - 10
            max_dimension = mid_dimension - 0.01

    # Recreate the image with the best parameters found
    best_img = cv2.resize(original_img, (0, 0), fx=best_dimension, fy=best_dimension)
    _, best_jpeg_data = cv2.imencode('.jpg', best_img, [int(cv2.IMWRITE_JPEG_QUALITY), best_quality])

    return best_jpeg_data
