#!/usr/bin/env python3

import os
import cv2
import numpy as np
from libs.image import prepare_image, get_background_mask, get_foreground


image_size = (256, 256)
save_dir = "images"

try:
    # Input video stream
    vid = cv2.VideoCapture(0)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")

    # First frame for background
    _, frame = vid.read()
    background = cv2.resize(frame, image_size, interpolation=cv2.INTER_AREA)

    img_counter = 0
    while True:
        return_value, frame = vid.read()
        # Resize
        image = cv2.resize(frame, image_size, interpolation=cv2.INTER_AREA)
        back_mask = get_background_mask(image, background)
        foreground_white = get_foreground(image, back_mask)

        # Show stream
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.imshow("image", image)
        cv2.imshow("Foreground", foreground_white)
        cv2.moveWindow("Foreground", 400, 0)

        # Read keyboard
        key = cv2.waitKey(1) & 0xFF

        # Take picture
        if key == ord('s'):
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            img_name = "img" + str(img_counter) + ".jpg"
            img_path = os.path.join(save_dir, img_name)
            cv2.imwrite(img_path, foreground_white)
            print("Saved ", img_path)
            img_counter += 1

        # Refresh background
        if key == ord('r'):
            _, frame = vid.read()
            background = cv2.resize(frame, image_size,
                                    interpolation=cv2.INTER_AREA)

        # Quit
        if key == ord('q'):
            break

finally:
    vid.release()
    cv2.destroyAllWindows()
