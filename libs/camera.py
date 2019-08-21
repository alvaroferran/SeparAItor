import time
import cv2
import numpy as np


def warmup_camera(video, seconds):
    time_start = time.time()
    while True:
        _, frame = video.read()
        time_end = time.time()
        time_elapsed = time_end - time_start
        if time_elapsed > seconds:
            break


def detect_motion(mask, image_size):
    mask_threshold = 1.e-2
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
