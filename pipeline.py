from typing import Any
import numpy as np
from preprocessing import (
    deskew_image,
    resize_image,
    find_meter_rectangle,
    morph_rect,
    extract_digit,
)
from models import ImageModel


def pipeline(image_model: ImageModel, config: dict[str, Any]) -> float:
    """
    Pipeline to extract number (or digit) from image.

    :param img: Image
    :return: Number
    """
    img = np.array(image_model.image_data, dtype=np.uint8)
    img = deskew_image(img)
    img = resize_image(img)

    rect = find_meter_rectangle(img)
    rect_morph = morph_rect(rect)
    digit = extract_digit(rect_morph, custom_config=config["custom_config"])
    return np.round(digit, 2)
