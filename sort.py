#!/usr/bin/env python3

import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from time import sleep


def prepare_image(image):
    # Rescale by 255, just like during training
    image = np.array(image).astype('float32')/255
    # Add extra dimension accounting for minibatch during training
    image = np.expand_dims(image, axis=0)
    return image


def get_most_frequent(preds):
    return max(set(preds), key=preds.count)


def detect_motion(mask):
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


def draw_text(image, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    pos = (50, 50)
    size = 1
    color = (255, 255, 255)
    thickness = 3
    cv2.putText(image, text, pos, font, size, color, thickness, cv2.LINE_AA)
    return image


save_dir = os.path.join("CNN", "save")
model_path = os.path.join(save_dir, "Fold0-0.9812.hdf5")
image_size = (256, 256)
mask_threshold = 1.e-3
stabilization_iterations = 3
prediction_iterations = 5

# Get labels
train_dir = os.path.join("CNN", os.path.join("dataset", "train"))
labels = os.listdir(train_dir)
labels.sort()
print(labels)

# Load trained model
model = load_model(model_path)
model.summary()

try:
    # Input video stream
    vid = cv2.VideoCapture(0)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")

    subtractor = cv2.createBackgroundSubtractorMOG2(history=10,
                                                    varThreshold=50)

    # # First frame for background
    _, frame = vid.read()
    background = cv2.resize(frame, image_size, interpolation=cv2.INTER_AREA)

    motion_list = [True] * stabilization_iterations
    image_stable = False
    moved_prev = False

    while True:
        return_value, frame = vid.read()

        image = cv2.resize(frame, image_size, interpolation=cv2.INTER_AREA)
        motion_mask = subtractor.apply(image)

        # Background subtraction
        diff = cv2.absdiff(background, image)
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        diff = cv2.GaussianBlur(diff, (21, 21), 0)
        back_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
        foreground_black = cv2.bitwise_and(image, image, mask=back_mask)

        foreground_white = np.zeros_like(foreground_black, np.uint8)
        foreground_white.fill(255)
        non_black_pixels = foreground_black > 1
        foreground_white[non_black_pixels] = foreground_black[non_black_pixels]

        # Show stream
        cv2.imshow("Camera Feed", image)
        cv2.moveWindow("Camera Feed", 0, 0)
        cv2.imshow("Movement", motion_mask)
        cv2.moveWindow("Movement", 0, 380)
        cv2.imshow("Background", back_mask)
        cv2.moveWindow("Background", 400, 380)
        cv2.imshow("Foreground", foreground_white)
        cv2.moveWindow("Foreground", 400, 0)

        # Check if object has moved in the last x frames
        motion_detected = detect_motion(motion_mask)
        motion_list.pop(0)
        motion_list.append(motion_detected)
        if True not in motion_list:
            image_stable = True
        else:
            image_stable = False
            moved_prev = True

        # Check for trigger falling edge (wait for object to stabilize)
        if moved_prev and image_stable:
            predictions = []
            # preprocessed_image = prepare_image(image)
            preprocessed_image = prepare_image(foreground_white)
            for i in range(prediction_iterations):
                preds = model.predict(preprocessed_image)
                preds = np.argmax(preds[0], axis=0)
                predictions.append(preds)
            prediction = get_most_frequent(predictions)
            prediction = labels[prediction]
            if prediction != "empty":
                print(prediction)
            moved_prev = False
            _, frame = vid.read()
            background = cv2.resize(frame, image_size,
                                    interpolation=cv2.INTER_AREA)

        # Quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


finally:
    vid.release()
    cv2.destroyAllWindows()
