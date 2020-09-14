import tensorflow as tf
import numpy as np


class GradCAM:

    def __init__(self,
                 img_path,
                 size,
                 model,
                 last_convolution_layer):
        self.img_path = img_path
        self.size = size
        self.conv_name = last_convolution_layer
        self.model = model

    def img_to_array(self, img_path, size):
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
        array = tf.keras.preprocessing.image.img_to_array(img)
        array = np.expand_dims(array, axis=0)
        return array

    def get_conv_model(self, model, last_convolution_layer):
        output_layer = model.get_layer(last_convolution_layer)
        output_model = tf.keras.Model(model.inputs, output_layer.output)
        return output_model

    def get_pred_model(self, model, last_convolution_layer):
        pred_layers = []
        found_conv = False
        for layer in model.layers:
            if layer.name == last_convolution_layer:
                found_conv = True
                continue
            if found_conv:
                pred_layers.append(layer.name)
        output_layer = model.get_layer(last_convolution_layer)
        model_input = tf.keras.Input(shape=output_layer.output.shape[1:])
        x = model_input
        for l in pred_layers:
            x = model.get_layer(l)(x)
        pred_model = tf.keras.Model(model_input, x)
        return pred_model

    def get_headmap(self):
        conv_model = self.get_conv_model(
            self.model,
            self.conv_name
        )
        pred_model = self.get_pred_model(
            self.model,
            self.conv_name
        )
        img = self.img_to_array(self.img_path)

        with tf.GradientTape() as tape:
            conv_output = conv_model(img)
            tape.watch(conv_output)
            preds = pred_model(conv_model)
