from const import ROOT_DIR, DATA_PATH
from preprocessing import (
    resize_image,
    find_meter_rectangle,
    morph_rect,
    parse_rect,
    deskew_image,
)

from PIL import Image
import cv2
import pytesseract

import re
import numpy as np
import matplotlib.pyplot as plt


path = list(DATA_PATH.glob("*.png"))[1]
path = str(path)

paths = list((DATA_PATH / "raw").glob("*.png"))
# for path in paths[1:2]:
for path in paths:
    path = str(path)
    img = cv2.imread(path)
    img = deskew_image(img)
    img = resize_image(img)
    rect = find_meter_rectangle(img)

    rect_morph = morph_rect(rect)
    num = parse_rect(rect_morph)

    fig, ax = plt.subplots(2)
    ax[0].imshow(rect)
    ax[1].imshow(rect_morph)
    # plt.show()

    print(num, path)
