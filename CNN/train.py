import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from libs.plot import plot_results, create_plot_directory
from libs.kfold_dataset import refresh_k_fold_dataset, create_dataset_structure
from libs.model import create_model
import os
import h5py
import shutil
import numpy as np
import yaml
import multiprocessing


def train_model(k, plot_dir, output):

    with open("config.yml", 'r') as ymlfile:
        params = yaml.load(ymlfile)
    source_dir = params["source_dir"]
    dest_dir = params["dest_dir"]
    save_dir = params["save_dir"]
    log_dir = params["log_dir"]
    target_size = params["target_size"]
    epochs = params["epochs"]
    nb_folds = params["nb_folds"]
    batch_size = params["batch_size"]
    learning_rate = params["learning_rate"]
    cores_cpu = params["cores_cpu"]
    enable_checkpoint = params["enable_checkpoint"]
    enable_tensorboard = params["enable_tensorboard"]
    enable_plots = params["enable_plots"]
    enable_dynamic_allocation = params["enable_dynamic_allocation"]

    # Dynamically allocate GPU memory
    if enable_dynamic_allocation:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # Clear old tensoboard logs
    log_train = os.path.join(log_dir, "train")
    log_validation = os.path.join(log_dir, "validation")
    if os.path.isdir(log_train):
        shutil.rmtree(log_train)
    if os.path.isdir(log_validation):
        shutil.rmtree(log_validation)

    # Rebuild dataset directories
    refresh_k_fold_dataset(source_dir, dest_dir, nb_folds, k)

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

    training_directory = os.path.join(dest_dir, "train")
    train_generator = train_datagen.flow_from_directory(
        training_directory,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical')

    val_datagen = ImageDataGenerator(rescale=1./255)

    validation_directory = os.path.join(dest_dir, "dev")
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
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    file_name = "Fold" + str(k) + ".hdf5"
    file_path = os.path.join(save_dir, file_name)
    checkpoint = ModelCheckpoint(file_path, monitor='val_accuracy', verbose=1,
                                 save_best_only=True, mode='max')

    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    tensor_board = TensorBoard(log_dir=log_dir, histogram_freq=1,
                               write_graph=True, write_images=True)

    callbacks_list = []
    if enable_checkpoint:
        callbacks_list.append(checkpoint)
    if enable_tensorboard:
        callbacks_list.append(tensor_board)

    # Train
    history = model.fit_generator(
        train_generator,
        epochs=epochs,
        steps_per_epoch=nb_train_samples // batch_size,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=callbacks_list,
        workers=cores_cpu)

    # Add accuracy value to final file name to only have one file per fold
    accuracy = max(history.history['val_accuracy'])
    file_name_new = f"Fold{k}-{accuracy:.4}.hdf5"
    file_path_new = os.path.join(save_dir, file_name_new)
    os.rename(file_path, file_path_new)

    # Save loss and accuracy plots
    if enable_plots:
        plot_results(history, k, plot_dir)

    # Get validation
    output.put(model.evaluate_generator(validation_generator))


if __name__ == '__main__':

    with open("config.yml", 'r') as ymlfile:
        params = yaml.load(ymlfile)
    source_dir = params["source_dir"]
    dest_dir = params["dest_dir"]
    plot_dir = params["plot_dir"]
    nb_folds = params["nb_folds"]
    enable_plots = params["enable_plots"]

    create_dataset_structure(source_dir, dest_dir)

    plot_sub_dir = ""
    if enable_plots:
        plot_sub_dir = create_plot_directory(plot_dir)

    jobs = []
    queue = multiprocessing.Queue()
    for k in range(nb_folds):
        p = multiprocessing.Process(target=train_model, args=(k, plot_sub_dir,
                                    queue))
        jobs.append(p)
        p.start()
        p.join()

    validation_scores = [queue.get() for p in jobs]
    validation_average = np.average(validation_scores, axis=0)
    print(f"Average loss and accuracy: {validation_average}")
