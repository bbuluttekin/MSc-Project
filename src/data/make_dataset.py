import tensorflow as tf
from pathlib import Path
import numpy as np


class GetDataset:
    def __init__(self, path_dir, batch_size=64, seed=123):
        self.seed = seed
        self.batch_size = batch_size
        self.path_dir = Path(path_dir)
        self.autotune = tf.data.experimental.AUTOTUNE

    def get_class_names(self, exclude_file='.DS_Store'):
        classes = [f.name for f in self.path_dir.glob(
            '*') if f.name != exclude_file]
        return np.array(classes)

    def make_train_data(self, path_dir):
        raise NotImplementedError

    def make_augmented_dataset(self, path_dir):
        raise NotImplementedError

    def make_test_dataset(self, path_dir):
        raise NotImplementedError
