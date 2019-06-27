import os
import glob
from pathlib import Path
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from matplotlib import pyplot
from matplotlib.pyplot import figure

data_dir = Path("/content/chest_xray")

train_dir = data_dir / "train"

test_dir = data_dir / "test"

val_dir = data_dir / "val"

train_dir_normal = train_dir / "NORMAL"
train_dir_pne = train_dir / "PNEUMONIA"

test_dir_normal = test_dir / "NORMAL"
test_dir_pne = test_dir / "PNEUMONIA"

normal_val = val_dir / "NORMAL"
val_dir_pne = val_dir / "PNEUMONIA"

img = load_img(str(list(normal_val.glob('*.jpeg'))[1]))
img_array = img_to_array(img)

s = expand_dims(img_array, 0)

datagen = ImageDataGenerator(horizontal_flip=True)


def display_augmented_img(img, transformation):
    figure(num=None, figsize=(14, 12), dpi=80, facecolor='w', edgecolor='k')
    datagen = transformation
    s = expand_dims(img, 0)
    it = datagen.flow(s, batch_size=1)
    pyplot.subplot(1, 2, 1)
    pyplot.imshow(img.astype('uint8'))
    pyplot.subplot(1, 2, 2)
    pyplot.imshow(it.next()[0].astype('uint8'))
    pyplot.show()


if __name__ == "__main__":
    display_augmented_img(img_array, ImageDataGenerator(horizontal_flip=True))
    display_augmented_img(img_array, ImageDataGenerator(rotation_range=90))
    display_augmented_img(img_array, ImageDataGenerator(rotation_range=270))
    display_augmented_img(img_array, ImageDataGenerator(vertical_flip=True))
