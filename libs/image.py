import cv2
import numpy as np


def prepare_image(image):
    # Rescale by 255, just like during training
    image = np.array(image).astype('float32')/255
    # Add extra dimension accounting for minibatch during training
    image = np.expand_dims(image, axis=0)
    return image


def get_background_mask(image, background):
    diff = cv2.absdiff(background, image)
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff = cv2.GaussianBlur(diff, (21, 21), 0)
    return cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]


def get_foreground(image, back_mask):
    foreground_black = cv2.bitwise_and(image, image, mask=back_mask)
    foreground_white = np.zeros_like(foreground_black, np.uint8)
    foreground_white.fill(255)
    non_black_pixels = foreground_black > 1
    foreground_white[non_black_pixels] = foreground_black[non_black_pixels]
    return foreground_white


# def draw_text(image, text):
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     pos = (50, 50)
#     size = 1
#     color = (255, 255, 255)
#     thickness = 3
#     cv2.putText(image, text, pos, font, size, color, thickness, cv2.LINE_AA)
#     return image
