from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.applications import VGG19


def create_model(target_size, learning_rate, nb_classes):
    conv_base = VGG19(weights='imagenet', include_top=False,
                      input_shape=(*target_size, 3))
    conv_base.trainable = False
    model = Sequential()
    model.add(conv_base)
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(nb_classes, activation='softmax'))

    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=learning_rate),
                  metrics=['accuracy'])
    return model
