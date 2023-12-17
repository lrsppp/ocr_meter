import math
import re

import cv2
import numpy as np
import pytesseract
from deskew import determine_skew

from ocr_meter.const import CONFIG


def resize_image(img: np.ndarray, size=(1024, 768)) -> np.ndarray:
    """
    :param img: Resize image to given size
    :param size: Target size
    :return: Resized image
    """
    return cv2.resize(img, size)


def find_meter_rectangle(img):
    """
    Find meter rectangle in image.

    :param: Image
    :return: Rectangle, i.e. cropped image
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 9
    )

    # Fill rectangular contours
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(thresh, [c], -1, (255, 255, 255), -1)

    # Morph open
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=7)

    # Draw rectangles
    cnts = cv2.findContours(opening, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    coords = []
    for c in cnts:
        x_, y_, w_, h_ = cv2.boundingRect(c)
        coords.append((x_, y_, w_, h_))

    x, y, w, h = coords[np.argmax([c[2] * c[3] for c in coords])]
    return img[y : y + h, x : x + w]


def rotate(image, angle, background):
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(
        np.cos(angle_radian) * old_width
    )
    height = abs(np.sin(angle_radian) * old_width) + abs(
        np.cos(angle_radian) * old_height
    )

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(
        image, rot_mat, (int(round(height)), int(round(width))), borderValue=background
    )


def deskew_image(img):
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(grayscale)
    rotated = rotate(img, angle, (0, 0, 0))
    return rotated


def morph_rect(rect, kernel_size=(3, 3)):
    """
    Preprocess rectangle (cropped image) using thresholding and morphological
    transformations.

    See also: https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html

    :param rect: Rectangle
    :param kernel_size: Kernel size passed to `cv2.morphologyEx(..., kernel_size)`
    :return: Transformed rectangle
    """
    (h, w) = rect.shape[:2]
    rect = cv2.resize(rect, (w * 32, h * 32))
    gray = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=10, sigmaY=10)
    gray = cv2.erode(gray, None, iterations=20)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]
    # thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    morph_close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, (15, 15))
    morph_close = cv2.bitwise_not(morph_close)
    return morph_close


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


def parse_string(s: str) -> str:
    clean_s = re.sub(r"[^\d,\.]", "", s)
    clean_s = s.replace(",", ".")
    try:
        digit = float(clean_s)
    except Exception:
        digit = np.nan
    finally:
        return digit
