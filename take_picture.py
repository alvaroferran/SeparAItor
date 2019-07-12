#!/usr/bin/env python3

import os
import cv2
import numpy as np


image_size = (256, 256)
save_dir = "images"

try:
    # Input video stream
    vid = cv2.VideoCapture(0)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    img_counter = 0
    while True:
        return_value, frame = vid.read()
        # Resize
        image = cv2.resize(frame, image_size, interpolation=cv2.INTER_AREA)

        # Show stream
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.imshow("image", image)

        # Take picture
        if cv2.waitKey(1) & 0xFF == ord('s'):
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            img_name = "img" + str(img_counter) + ".jpg"
            img_path = os.path.join(save_dir, img_name)
            cv2.imwrite(img_path, image)
            print("Saved ", img_path)
            img_counter += 1

        # Quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    vid.release()
    cv2.destroyAllWindows()
