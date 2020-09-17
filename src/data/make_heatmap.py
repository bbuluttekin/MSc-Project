import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class GradCAM:

    def __init__(self,
                 img_path,
                 size,
                 model,
                 last_convolution_layer,
                 image_name="gcam_sample.jpg"):
        self.img_path = img_path
        self.size = size
        self.conv_name = last_convolution_layer
        self.model = model
        self.image_name = image_name

    @property
    def img(self):
        img = tf.keras.preprocessing.image.load_img(
            self.img_path, target_size=self.size)
        array = tf.keras.preprocessing.image.img_to_array(img)
        array = array / 255
        array = np.expand_dims(array, axis=0)
        return array

    @property
    def conv_model(self):
        output_layer = self.model.get_layer(self.conv_name)
        output_model = tf.keras.Model(self.model.inputs, output_layer.output)
        return output_model

    @property
    def pred_model(self):
        pred_layers = []
        found_conv = False
        for layer in self.model.layers:
            if layer.name == self.conv_name:
                found_conv = True
                continue
            if found_conv:
                pred_layers.append(layer.name)
        output_layer = self.model.get_layer(self.conv_name)
        model_input = tf.keras.Input(shape=output_layer.output.shape[1:])
        x = model_input
        for l in pred_layers:
            x = self.model.get_layer(l)(x)
        pred_model = tf.keras.Model(model_input, x)
        return pred_model

    @property
    def heatmap(self):
        with tf.GradientTape() as tape:
            conv_output = self.conv_model(self.img)
            tape.watch(conv_output)
            preds = self.pred_model(conv_output)[0]
        grads = tape.gradient(preds, conv_output)
        p_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
        conv_output = conv_output.numpy()[0]
        for i in range(p_grads.shape[-1]):
            conv_output[:, :, i] *= p_grads[i]
        heatmap = np.mean(conv_output, axis=-1)
        heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
        return heatmap

    def plot_gradcam(self):
        img = tf.keras.preprocessing.image.load_img(self.img_path)
        img = tf.keras.preprocessing.image.img_to_array(img)
        heatmap = np.uint8(255 * self.heatmap)
        jet = cm.get_cmap("jet")
        # Select RGB values
        jet_c = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_c[heatmap]
        # Turn into image
        jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
        # Resize to original image
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
        simp_img = jet_heatmap * 0.4 + img
        simp_img = tf.keras.preprocessing.image.array_to_img(
            simp_img)
        simp_img.save(self.image_name)
        plt.imshow(simp_img)
