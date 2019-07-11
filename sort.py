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


save_dir = os.path.join("CNN", "save")
saved_model = os.path.join(save_dir, "Fold 0-0.95.hdf5")
image_size = (256, 256)

# Get labels
train_dir = os.path.join("CNN", os.path.join("dataset", "train"))
labels = os.listdir(train_dir)

# Load trained model
model = load_model(saved_model)
model.summary()

print(labels)
labels.sort()
print(labels)

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
        # image = Image.fromarray(frame)
        image = cv2.resize(frame, image_size, interpolation=cv2.INTER_AREA)
        # image = np.asarray(image)


        # Show yolo stream
        # if config["show_stream"]:
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.imshow("image", image)

        if cv2.waitKey(1) & 0xFF == ord('a'):
            image = prepare_image(image)
            # predicted = model.predict_classes(image)
            predictions = model.predict(image)
            predicted = np.argmax(predictions[0], axis=0)
            print(predictions)
            print(labels[predicted])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # # Save frame to output video
        # if config["save_video"]:
        #     out.write(image)

finally:
    vid.release()
    cv2.destroyAllWindows()
    # cv2.destroyWindows("image")
