from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from libs.plot import plot_results
from libs.kfold_dataset import refresh_k_fold_dataset
from libs.model import create_model
import os
import h5py
import shutil
import numpy as np
import datetime


target_size = (256, 256)
source_dir = "dataset_256"
dest_dir = "dataset"
epochs = 80
k_folds = 10
learning_rate = 1e-5
batch_size = 32
cores_cpu = 12
training_directory = os.path.join(dest_dir, "train")
validation_directory = os.path.join(dest_dir, "dev")
save_folder = "save"
log_folder = "log_dir"
plot_folder = "imgs"


# Create folder to save plots
dt = datetime.datetime.now()
subdir = f"{dt.year}-{dt.month}-{dt.day}_{dt.hour}-{dt.minute}"
plot_subdir = os.path.join(plot_folder, subdir)
if not os.path.isdir(plot_folder):
    os.mkdir(plot_folder)
if not os.path.isdir(plot_subdir):
    os.mkdir(plot_subdir)


validation_scores = []
for k in range(k_folds):

    # Clear old tensoboard logs
    log_train = os.path.join(log_folder, "train")
    log_validation = os.path.join(log_folder, "validation")
    if os.path.isdir(log_train):
        shutil.rmtree(log_train)
    if os.path.isdir(log_validation):
        shutil.rmtree(log_validation)

    # Rebuild dataset directories
    refresh_k_fold_dataset(source_dir, dest_dir, k_folds)

    # Create train and dev datasets
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=360,
        brightness_range=[1.5, 0.5],
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.5,
        horizontal_flip=True,
        vertical_flip=True)

    train_generator = train_datagen.flow_from_directory(
        training_directory,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical')

    val_datagen = ImageDataGenerator(rescale=1./255)

    validation_generator = val_datagen.flow_from_directory(
        validation_directory,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical')

    nb_train_samples = train_generator.n
    nb_validation_samples = validation_generator.n
    nb_classes = train_generator.num_classes

    # Get model
    model = create_model(target_size, learning_rate, nb_classes)

    # Create checkpoint and TensorBoard callbacks
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    file_name = "Fold " + str(k) + "-{val_accuracy:.2f}.hdf5"
    file_path = os.path.join(save_folder, file_name)
    checkpoint = ModelCheckpoint(file_path, monitor='val_accuracy', verbose=1,
                                 save_best_only=True, mode='max')

    if not os.path.isdir(log_folder):
        os.mkdir(log_folder)
    tensor_board = TensorBoard(
                log_dir=log_folder, histogram_freq=1, write_graph=True,
                write_images=True)

    callbacks_list = [tensor_board, checkpoint]

    # Train
    history = model.fit_generator(
        train_generator,
        epochs=epochs,
        steps_per_epoch=nb_train_samples // batch_size,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=callbacks_list,
        workers=cores_cpu)

    # Save loss and accuracy plots
    plot_results(history, k, plot_subdir)

    # Get validation
    validation_score = model.evaluate_generator(validation_generator)
    validation_scores.append(validation_score)

validation_average = np.average(validation_scores, axis=0)
print(f"Average loss and accuracy: {validation_average}")
