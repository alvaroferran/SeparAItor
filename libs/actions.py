import cv2
import numpy as np


def get_most_frequent(preds):
    return max(set(preds), key=preds.count)


def detect_motion(mask, image_size):
    mask_threshold = 1.e-3
    img_array = mask.astype(np.uint8)
    white_list = cv2.findNonZero(img_array)
    if white_list is not None:
        num_white = len(white_list)
    else:
        num_white = 0
    percent_white = num_white / (image_size[0]*image_size[1])
    if percent_white > mask_threshold:
        return True
    else:
        return False


def sort_item(item):
    PMD = ["plastic", "cans", "cartons"]
    organic = ["eggs"]
    glass = ["glass"]
    classes = [PMD, organic, glass]
    for class_item in classes:
        if item in class_item:
            return classes.index(class_item)
    return None
