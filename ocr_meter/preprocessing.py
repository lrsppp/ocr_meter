import re

import cv2
import numpy as np
import pytesseract

from ocr_meter.const import CONFIG


def add_contours(
    img: np.ndarray,
    mode: int = cv2.RETR_EXTERNAL,
    method: int = cv2.CHAIN_APPROX_SIMPLE,
) -> np.ndarray:
    """
    Add contours to the input image.

    :param img: Input image
    :param mode: Retrieval mode for findContours function (default: cv2.RETR_EXTERNAL)
    :param method: Approximation method for findContours function (default: cv2.CHAIN_APPROX_SIMPLE)
    :return: Image with added contours
    """
    cnts = cv2.findContours(img, mode, method)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(img, [c], -1, (255, 255, 255), -1)
    return img


def get_largest_rectangle(
    img: np.ndarray, img_orig: np.ndarray | None = None
) -> tuple[int, int, int, int]:
    """
    Get the coordinates of the largest rectangle in the image.

    :param img: Input image
    :param img_orig: Original image (default: None)
    :return: Coordinates of the largest rectangle (x, y, w, h)
    """
    cnts = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    coords = []
    for c in cnts:
        x_, y_, w_, h_ = cv2.boundingRect(c)
        coords.append((x_, y_, w_, h_))

    x, y, w, h = coords[np.argmax([c[2] * c[3] for c in coords])]
    return x, y, w, h


def resize_by_scale(
    img: np.ndarray, w_scale: int = 32, h_scale: int = 32
) -> np.ndarray:
    """
    Resize the image by a given scale.

    :param img: Input image
    :param w_scale: Width scale factor (default: 32)
    :param h_scale: Height scale factor (default: 32)
    :return: Resized image
    """
    (h, w) = img.shape[:2]
    return cv2.resize(img, (w * w_scale, h * h_scale))


def extract_digit(
    rect: np.ndarray, custom_config: (str | None) = CONFIG["custom_config"]
) -> float:
    """
    Apply OCR using pytesseract to extract number from rectangle

    :param rect: Rectangle
    :param custom_config: Custom config passed to `pytesseract.image_to_string` function
    :return: Number
    """
    result = pytesseract.image_to_string(rect, config=custom_config)
    cleaned_result = parse_string(result)
    return cleaned_result


def parse_string(s: str) -> float:
    """
    Parse a string and convert it to a float.

    :param s: Input string
    :return: Parsed float value
    """
    clean_s = re.sub(r"[^\d,\.]", "", s)
    clean_s = s.replace(",", ".")
    try:
        digit = float(clean_s)
    except Exception:
        digit = np.nan
    finally:
        return digit
