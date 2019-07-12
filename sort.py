#!/usr/bin/env python3

import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model


def prepare_image(image):
    # Rescale by 255, just like during training
    image = np.array(image).astype('float32')/255
    # Add extra dimension accounting for minibatch during training
    image = np.expand_dims(image, axis=0)
    return image


def get_most_frequent(preds):
    return max(set(preds), key=preds.count)


save_dir = os.path.join("CNN", "save")
saved_model = os.path.join(save_dir, "Fold1-0.9417.hdf5")
image_size = (256, 256)

# Get labels
train_dir = os.path.join("CNN", os.path.join("dataset", "train"))
labels = os.listdir(train_dir)
labels.sort()
print(labels)

# Load trained model
model = load_model(saved_model)
model.summary()

try:
    # Input video stream
    vid = cv2.VideoCapture(0)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # # Output stream to save the video output
    # if config["save_video"]:
    #     video_name = config["output_name"] + ".mp4"
    #     video_fps = config["output_fps"]
    #     video_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #    out = cv2.VideoWriter(video_name, video_fourcc, video_fps, video_size)

    while True:
        return_value, frame = vid.read()
        # Resize
        image = cv2.resize(frame, image_size, interpolation=cv2.INTER_AREA)

        # Show stream
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.imshow("image", image)

        # Predict
        if cv2.waitKey(1) & 0xFF == ord('a'):
            predictions = []
            image = prepare_image(image)
            for i in range(5):
                preds = model.predict(image)
                # print(preds)
                preds = np.argmax(preds[0], axis=0)
                predictions.append(preds)
            prediction = get_most_frequent(predictions)
            print(labels[prediction])

        # Quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # # Save frame to output video
        # if config["save_video"]:
        #     out.write(image)

finally:
    vid.release()
    cv2.destroyAllWindows()
