from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,\
     Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from libs.plot import plot_results
import os
import h5py
import shutil


target_size = (128, 128)
training_directory = "dataset/train/"
validation_directory = "dataset/dev/"
save_folder = "save"
log_folder = "log_dir"
epochs = 300
learning_rate = 1e-4
batch_size = 16
cores_cpu = 12


# Clear old tensoboard logs
log_train = os.path.join(log_folder, "train")
log_validation = os.path.join(log_folder, "validation")
if os.path.isdir(log_train):
    shutil.rmtree(log_train)
if os.path.isdir(log_validation):
    shutil.rmtree(log_validation)


# Create datasets
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=360,
    brightness_range=[1.5, 0.5],
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range = 0.2,
    zoom_range = 0.5,
    horizontal_flip = True,
    vertical_flip = True)

train_generator = train_datagen.flow_from_directory(
    training_directory,
    target_size = target_size,
    batch_size = batch_size,
    class_mode = 'categorical')

val_datagen = ImageDataGenerator(rescale = 1./255)

validation_generator = val_datagen.flow_from_directory(
    validation_directory,
    target_size = target_size,
    batch_size = batch_size,
    class_mode = 'categorical')

nb_train_samples = train_generator.n
nb_validation_samples = validation_generator.n
nb_classes = train_generator.num_classes


# Define and compile model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape = (*target_size, 3),
                 activation = 'relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation = 'softmax'))

model.summary()
model.compile(loss = 'categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=learning_rate),
              metrics = ['accuracy'])


# Save checkpoint
if not os.path.isdir(save_folder):
    os.mkdir(save_folder)
file_name="checkpoint-{epoch:02d}-{val_accuracy:.2f}.hdf5"
file_path = os.path.join(save_folder, file_name)
checkpoint = ModelCheckpoint(file_path, monitor='val_accuracy', verbose=1,
                             save_best_only=True, mode='max')
# TensorBoard
if not os.path.isdir(log_folder):
    os.mkdir(log_folder)
tensor_board = TensorBoard(
               log_dir=log_folder, histogram_freq=1, write_graph=True,
               write_images=True)

callbacks_list = [tensor_board]


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
plot_results(history)
