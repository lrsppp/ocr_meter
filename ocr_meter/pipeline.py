from typing import Any, Dict

import cv2
import numpy as np

from ocr_meter.image_processor import ImageProcessor
from ocr_meter.models import ImageModel
from ocr_meter.preprocessing import (
    add_contours,
    extract_digit,
    get_largest_rectangle,
    resize_by_scale,
)


def pipeline(image_model: ImageModel, config: Dict[str, Any]) -> float:
    """
    Perform a pipeline of image processing steps to extract and process a digit from the input image model.

    :param image_model: Instance of the ImageModel class containing image data
    :param config: Configuration dictionary with parameters for processing steps
    :return: Extracted and processed digit
    """
    # Part I: Extract rectangle
    img = np.array(image_model.image_data, dtype=np.uint8)
    steps_img = [
        (cv2.cvtColor, {"code": cv2.COLOR_RGB2GRAY}),
        (
            cv2.adaptiveThreshold,
            {
                "maxValue": 255,
                "adaptiveMethod": cv2.ADAPTIVE_THRESH_MEAN_C,
                "thresholdType": cv2.THRESH_BINARY_INV,
                "blockSize": 51,
                "C": 9,
            },
        ),
        (add_contours, {"mode": cv2.RETR_EXTERNAL, "method": cv2.CHAIN_APPROX_SIMPLE}),
        (
            cv2.morphologyEx,
            {
                "op": cv2.MORPH_OPEN,
                "kernel": cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6)),
                "iterations": 7,
            },
        ),
        (get_largest_rectangle, {}),
    ]
    proc_img = ImageProcessor(steps_img)
    x, y, w, h = proc_img.process(img)
    rect = img[y : y + h, x : x + w]

    # Part II: Process rectangle
    steps_rect = [
        (resize_by_scale, {"w_scale": 32, "h_scale": 32}),
        (cv2.cvtColor, {"code": cv2.COLOR_BGR2GRAY}),
        (cv2.GaussianBlur, {"ksize": (0, 0), "sigmaX": 10, "sigmaY": 10}),
        (cv2.erode, {"kernel": None, "iterations": 20}),
        (cv2.threshold, {"thresh": 0, "maxval": 255, "type": cv2.THRESH_OTSU}),
        (lambda x: x[1], {}),
        (cv2.morphologyEx, {"op": cv2.MORPH_CLOSE, "kernel": (15, 15)}),
        (cv2.bitwise_not, {}),
    ]
    proc_rect = ImageProcessor(steps_rect)
    rect = proc_rect.process(rect)
    digit = extract_digit(rect, custom_config=config["custom_config"])
    return np.round(digit, 2)
