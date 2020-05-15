import os
from pathlib import Path

import numpy as np
import tensorflow as tf


class GetDataset:
    def __init__(self, image_dims, batch_size=64, seed=123):
        self.seed = seed
        self.batch_size = batch_size
        self.width, self.height = image_dims
        self.autotune = tf.data.experimental.AUTOTUNE

    def get_class_names(self, path_dir, exclude_file='.DS_Store'):
        classes = [f.name for f in path_dir.glob(
            '*') if f.name != exclude_file]
        return np.array(classes)

    def get_array_label(self, file_path, class_names):
        parts = tf.strings.split(file_path, os.path.sep)
        return parts[-2] == class_names

    def get_binary_label(self, file_path, class_names):
        if len(class_names) != 2:
            raise ValueError("Number of classes can not be diffrent than 2.")
        parts = tf.strings.split(file_path, os.path.sep)
        label = parts[-2] == class_names[1]
        return label

    def decode_image(self, img, rgb=True):
        if rgb:
            img = tf.image.decode_jpeg(img, channels=3)
        else:
            img = tf.image.decode_jpeg(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return tf.image.resize(img, [self.width, self.height])

    def process_path(self, file_path):
        label = self.get_array_label(
            file_path,
            class_names=["NORMAL", "PNEUMONIA"]
        )
        img = tf.io.read_file(file_path)
        img = self.decode_image(img)
        return img, label

    def aug_process_path(self, file_path, transformation):
        label = self.get_array_label(
            file_path,
            class_names=["NORMAL", "PNEUMONIA"]
        )
        img = tf.io.read_file(file_path)
        img = self.decode_image(img)
        img = transformation(img)
        return img, label

    def make_train_data(
        self,
        path_dir,
        cache=True,
        shuffle_buffer_size=2000
    ):
        list_path = tf.data.Dataset.list_files(str(path_dir / "*/*.jpeg"))
        labelled_data = list_path.map(
            self.process_path,
            num_parallel_calls=self.autotune)
        if cache:
            if isinstance(cache, str):
                labelled_data = labelled_data.cache(cache)
            else:
                labelled_data = labelled_data.cache()

        labelled_data = labelled_data.shuffle(buffer_size=shuffle_buffer_size)
        labelled_data = labelled_data.batch(self.batch_size)
        labelled_data = labelled_data.prefetch(buffer_size=self.autotune)
        return labelled_data

    def image_flip_left_right(self, image):
        return tf.image.flip_left_right(image)

    def image_adjust_saturation(self, image, factor=3):
        return tf.image.adjust_saturation(image, saturation_factor=factor)

    def image_adjust_brightness(self, image, delta=0.4):
        return tf.image.adjust_brightness(image, delta=delta)

    def image_rot90(self, image):
        return tf.image.rot90(image)

    def image_central_crop(self, image, fraction=0.9):
        return tf.image.central_crop(image, central_fraction=fraction)

    def image_flip_up_down(self, image):
        return tf.image.flip_up_down(image)

    def make_augmented_dataset(
        self,
        path_dir,
        transformations,
        shuffle_buffer_size=2000
    ):
        list_path = tf.data.Dataset.list_files(str(path_dir / "*/*.jpeg"))
        labelled_data = list_path.map(
            self.process_path,
            num_parallel_calls=self.autotune)
        for i in transformations:
            augmented = list_path.map(
                lambda x: self.aug_process_path(x, i),
                num_parallel_calls=self.autotune)
            labelled_data = labelled_data.concatenate(augmented)
        labelled_data = labelled_data.cache()
        labelled_data = labelled_data.shuffle(buffer_size=shuffle_buffer_size)
        labelled_data = labelled_data.batch(self.batch_size)
        labelled_data = labelled_data.prefetch(buffer_size=self.autotune)
        return labelled_data

    def make_test_dataset(self, path_dir):
        raise NotImplementedError
