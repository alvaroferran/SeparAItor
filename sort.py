#!/usr/bin/env python3
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from libs.serial_comms import connect_serial, send_data
from libs.image import prepare_image, get_foreground
from libs.actions import get_most_frequent, sort_item
from libs.camera import warmup_camera, detect_motion
from libs.information import Information


try:
    # Configuration
    save_dir = os.path.join("CNN", "save")
    model_path = os.path.join(save_dir, "Fold0-0.9948.hdf5")
    image_size = (256, 256)
    stabilization_iterations = 10
    prediction_iterations = 3

    # Get labels
    train_dir = os.path.join("CNN", os.path.join("dataset", "train"))
    labels = os.listdir(train_dir)
    labels.sort()

    # Connect to base
    print("Waiting for base to connect...", end=" ")
    bt = connect_serial('/dev/rfcomm0', 19200)
    print("Done")

    # Load trained model
    model = load_model(model_path)
    model.summary()
    print("Model loaded")

    # Input video stream
    vid = cv2.VideoCapture(0)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")

    print("Waiting for camera to warm up...", end=" ")
    warmup_camera(vid, 2)
    print("Done")

    # Motion detection background subtractor
    motion_subtractor = cv2.createBackgroundSubtractorMOG2(history=10,
                                                           varThreshold=300)

    # First frame for background
    _, frame = vid.read()
    background = cv2.resize(frame, image_size, interpolation=cv2.INTER_AREA)

    # Initial state
    s = {
        "waiting_object": 0,
        "waiting_base": 1
        }
    state = s["waiting_object"]

    motion_list = [False] * stabilization_iterations
    image_stable = False
    moved_prev = False
    # Stop the motion detector from triggering at beginning
    motion_mask = motion_subtractor.apply(background)

    # Create information screen
    information = Information()

    while True:
        # Get foreground image of object for the prediction
        return_value, frame = vid.read()
        image = cv2.resize(frame, image_size, interpolation=cv2.INTER_AREA)
        motion_mask = motion_subtractor.apply(image)
        foreground_image = get_foreground(image, background)

        # Show stream
        cv2.imshow("Camera Feed", image)
        cv2.moveWindow("Camera Feed", 0, 0)
        cv2.imshow("Movement", motion_mask)
        cv2.moveWindow("Movement", 0, 380)
        cv2.imshow("Foreground", foreground_image)
        cv2.moveWindow("Foreground", 400, 0)
        cv2.imshow("Information", information.image)
        cv2.moveWindow("Information", 400, 380)

        # Check if object has moved in the last few frames
        motion_detected = detect_motion(motion_mask, image_size)
        motion_list.pop(0)
        motion_list.append(motion_detected)
        if True in motion_list:
            moved_prev = True
            image_stable = False
        else:
            image_stable = True

        # Check for new object and predict
        if state == s["waiting_object"]:
            # Check for trigger falling edge (wait for image to stabilize)
            if image_stable and moved_prev:
                preprocessed_image = prepare_image(foreground_image)
                predictions = []
                # Predict image class
                for i in range(prediction_iterations):
                    preds = model.predict(preprocessed_image)
                    preds = np.argmax(preds[0], axis=0)
                    predictions.append(preds)
                prediction = get_most_frequent(predictions)
                prediction = labels[prediction]
                sorted_class = sort_item(prediction)
                if sorted_class is not None:
                    # Go to corresponding bin
                    send_data(bt, sorted_class)
                    information.update(prediction)
                    state = s["waiting_base"]
                else:
                    print(f"No bin specified for class {prediction}")

        # Wait for base to report readiness
        elif state == s["waiting_base"]:
            bt.reset_input_buffer()
            in_data = bt.readline().decode()
            if len(in_data) > 0:
                # print(repr(in_data))
                if in_data[:1] == "c":
                    bt.write("d".encode())
                    bt.flush()
                    state = s["waiting_object"]
                    information.update()
                    # Refresh background image
                    _, frame = vid.read()
                    background = cv2.resize(frame, image_size,
                                            interpolation=cv2.INTER_AREA)
                    # Reset motion data
                    motion_list = [False] * stabilization_iterations
                    moved_prev = False

        # Quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    vid.release()
    cv2.destroyAllWindows()
    bt.close()
