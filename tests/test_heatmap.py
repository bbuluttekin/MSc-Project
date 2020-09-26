from pathlib import Path

import numpy as np
import tensorflow as tf
from src.data.make_heatmap import GradCAM

paths = [str(f) for f in Path("tests/sample_data").glob("*/*")]
model = tf.keras.models.load_model("models/final_model.h5")
print(type(model))


def test_model_type():
    assert type(model) == tf.python.keras.engine.sequential.Sequential


def test_image_paths():
    assert type(paths) == list


def test_img_type():
    gcam = GradCAM(paths[0], (224, 224), model,
                   last_convolution_layer="conv2d_73")
    assert type(gcam.img) == np.ndarray


def test_image_shape():
    gcam = GradCAM(paths[0], (224, 224), model,
                   last_convolution_layer="conv2d_73")
    assert gcam.img.shape == (1, 224, 224, 3)


def test_image_size():
    gcam = GradCAM(paths[0], (224, 224), model,
                   last_convolution_layer="conv2d_73")
    assert gcam.img.shape[1] == 224
    assert gcam.img.shape[2] == 224


def test_conv_model_type():
    gcam = GradCAM(paths[0], (224, 224), model,
                   last_convolution_layer="conv2d_73")
    assert type(gcam.conv_model) == tf.python.keras.engine.functional.Functional


def test_pred_model_type():
    gcam = GradCAM(paths[0], (224, 224), model,
                   last_convolution_layer="conv2d_73")
    assert type(gcam.conv_model) == tf.python.keras.engine.functional.Functional


def test_conv_output_type():
    gcam = GradCAM(paths[0], (224, 224), model,
                   last_convolution_layer="conv2d_73")
    assert type(gcam.conv_model(gcam.img)
                ) == tf.python.framework.ops.EagerTensor


def test_conv_output_shape():
    gcam = GradCAM(paths[0], (224, 224), model,
                   last_convolution_layer="conv2d_73")
    assert gcam.conv_model(gcam.img).shape == [1, 14, 14, 320]


def test_pred_model_output_type():
    gcam = GradCAM(paths[0], (224, 224), model,
                   last_convolution_layer="conv2d_73")
    assert type(gcam.pred_model(gcam.conv_model(gcam.img))
                ) == tf.python.framework.ops.EagerTensor


def test_pred_output_shape():
    gcam = GradCAM(paths[0], (224, 224), model,
                   last_convolution_layer="conv2d_73")
    assert gcam.pred_model(gcam.conv_model(gcam.img)).shape == [1, 1]


def test_heatmap_type():
    gcam = GradCAM(paths[1], (224, 224), model,
                   last_convolution_layer="conv2d_73")
    assert type(gcam.heatmap) == np.ndarray


def test_heatmap_shape():
    gcam = GradCAM(paths[1], (224, 224), model,
                   last_convolution_layer="conv2d_73")
    assert gcam.heatmap.shape == (14, 14)
