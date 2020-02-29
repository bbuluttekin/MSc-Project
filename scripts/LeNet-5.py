import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D,
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense


# dimensions of our images.
img_width, img_height = 32, 32

train_data_dir = '/content/chest_xray/train/'
validation_data_dir = '/content/chest_xray/val/'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 16


input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))

model.add(MaxPooling2D(6, pool_size=14, strides=2))

model.add(Conv2D(16, (10, 10), activation='relu'))

model.add(MaxPooling2D(16, pool_size=5, strides=2))

model.add(Flatten())

model.add(Dense(84, activation='relu'))

model.add(Dense(2, activation="sigmoid"))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save_weights('model_1.h5')
