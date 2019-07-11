#!/usr/bin/env python3

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model


checkpoint = "Fold 0-0.95.hdf5"
folder = "dataset/test/"
print_image_info = True
image_size = 256

# Find the number of files to convert
numImagesTotal = 0
for subdir, dirs, images in os.walk(folder):
    for img in images:
        numImagesTotal += 1

# Load trained model
model = load_model(os.path.join("save", checkpoint))

# Predict classes
test_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = test_datagen.flow_from_directory(
    folder,
    target_size=(image_size, image_size),
    batch_size=1,
    class_mode=None,
    shuffle=False)  # Keeps validation and prediction in the same order

nb_samples = validation_generator.samples
nb_classes = len(validation_generator.class_indices)
real_class_list = validation_generator.classes

inv_class_dictionary = validation_generator.class_indices
class_dictionary = {value: key for key, value in inv_class_dictionary.items()}

predictions = model.predict_generator(validation_generator, steps=nb_samples)
predicted_class_list = np.argmax(predictions, axis=-1)

# Print information about each file
if print_image_info:
    for i, pred_class in enumerate(predicted_class_list):
        file_name = validation_generator.filenames[i]
        probability = max(predictions[i])
        print(f"File: {file_name} \tActual: ",
              f"{class_dictionary[real_class_list[i]]}\tPredicted: ",
              f"{class_dictionary[pred_class]} \tProbability: {probability}")

# Get percentage of correctly predicted
nb_correctly_predicted = np.sum(real_class_list == predicted_class_list)
correct_percentage = nb_correctly_predicted / nb_samples
print(f"Correctly predicted {nb_correctly_predicted}/{nb_samples}",
      f"({correct_percentage*100.0:.4f}%)")
