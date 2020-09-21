from src.data.make_dataset import GetDataset
import tensorflow as tf
from pathlib import Path


def test_return_type():
    dt = GetDataset((224, 224))
    train = dt.make_train_data(Path("tests/sample_data"))
    assert type(train) == tf.python.data.ops.dataset_ops.PrefetchDataset


def test_batch_size():
    dt = GetDataset((224, 224))
    train = dt.make_train_data(Path("tests/sample_data"))
    for d, l in train:
        assert d.numpy().shape[0] == 8


def test_image_size():
    dt = GetDataset((224, 224))
    train = dt.make_train_data(Path("tests/sample_data"))
    for d, l in train:
        assert d.numpy().shape[1] == 224
        assert d.numpy().shape[2] == 224


def test_label_shape():
    dt = GetDataset((224, 224))
    train = dt.make_train_data(Path("tests/sample_data"))
    for d, l in train:
        assert l.numpy().shape[0] == 8


def test_true_values():
    dt = GetDataset((224, 224))
    train = dt.make_train_data(Path("tests/sample_data"))
    for d, l in train:
        assert l.numpy().sum() == 5


def test_balanced_type():
    dt = GetDataset((224, 224))
    train = dt.make_balaced_dataset(
        majority_class="tests/sample_data/PNEUMONIA",
        minority_class="tests/sample_data/NORMAL",
        transformations=[
            dt.image_flip_left_right,
            dt.image_central_crop,
            dt.image_adjust_saturation
        ])
    assert type(train) == tf.python.data.ops.dataset_ops.PrefetchDataset


def test_balanced_batch_size():
    dt = GetDataset((224, 224))
    train = dt.make_balaced_dataset(
        majority_class="tests/sample_data/PNEUMONIA",
        minority_class="tests/sample_data/NORMAL",
        transformations=[
            dt.image_flip_left_right,
            dt.image_central_crop,
            dt.image_adjust_saturation
        ])
    for d, l in train:
        assert d.numpy().shape[0] == 10


def test_balanced_image_size():
    dt = GetDataset((224, 224))
    train = dt.make_balaced_dataset(
        majority_class="tests/sample_data/PNEUMONIA",
        minority_class="tests/sample_data/NORMAL",
        transformations=[
            dt.image_flip_left_right,
            dt.image_central_crop,
            dt.image_adjust_saturation
        ])
    for d, l in train:
        assert d.numpy().shape[1] == 224
        assert d.numpy().shape[2] == 224


def test_balanced_label_shape():
    dt = GetDataset((224, 224))
    train = dt.make_balaced_dataset(
        majority_class="tests/sample_data/PNEUMONIA",
        minority_class="tests/sample_data/NORMAL",
        transformations=[
            dt.image_flip_left_right,
            dt.image_central_crop,
            dt.image_adjust_saturation
        ])
    for d, l in train:
        assert l.numpy().shape[0] == 10


def test_balanced_class_size():
    dt = GetDataset((224, 224))
    train = dt.make_balaced_dataset(
        majority_class="tests/sample_data/PNEUMONIA",
        minority_class="tests/sample_data/NORMAL",
        transformations=[
            dt.image_flip_left_right,
            dt.image_central_crop,
            dt.image_adjust_saturation
        ])
    for d, l in train:
        assert l.numpy().sum() == 5
