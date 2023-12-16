import cv2
import numpy as np
import re
import pytesseract


def resize_image(img, size=(1024, 768)):
    """
    :param img: Resize image to given size
    :return: Resized image
    """
    return cv2.resize(img, size)


def find_meter_rectangle(img):
    """
    Find meter rectangle in image.

    :param: Image
    :return: Rectangle, i.e. cropped image
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 9
    )

    # Fill rectangular contours
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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


def morph_rect(rect, kernel_size=(3, 3)):
    """
    Preprocess rectangle (cropped image) using thresholding and morphological
    transformations.

    :param rect: Rectangle
    :param kernel_size: Kernel size passed to `cv2.morphologyEx(..., kernel_size)`
    :return: Transformed rectangle
    """
    gray = cv2.cvtColor(rect, cv2.COLOR_BGRA2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    morph_close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_size)
    return morph_close


def parse_rect(rect, custom_config=None):
    """
    Apply OCR using pytesseract to extract number from rectangle

    :param rect: Rectangle
    :param custom_config: Custom config passed to `pytesseract.image_to_string` function
    :return: Number
    """
    if custom_config is None:
        custom_config = r"--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789"
    result = pytesseract.image_to_string(rect, config=custom_config)
    cleaned_result = re.sub(r"[^\d,\.]", "", result)
    try:
        num = float(cleaned_result)
    except:
        num = np.nan
    return num
