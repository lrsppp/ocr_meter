from preprocessing import (
    deskew_image,
    resize_image,
    find_meter_rectangle,
    morph_rect,
    parse_rect,
)


def read_image():
    pass


def pipeline(img):
    img = deskew_image(img)
    img = resize_image(img)

    rect = find_meter_rectangle(img)
    rect_morph = morph_rect(rect)
    return parse_rect(rect)
