import tensorflow as tf


class CloudTrainer:
    def __init__(self, experiment_name):
        self.name = experiment_name
        self.main_path = "drive/My Drive/models"
        self.model_path = f"{self.main_path}/{self.name}"
        self.log_path = f"{self.main_path}/{self.name}/logs"
        self.callbacks = [
            tf.keras.callbacks.ModelCheckpoint(self.model_path),
            tf.keras.callbacks.TensorBoard(
                log_dir=self.log_path
            )
        ]

    def train_keras_seq_model(self, model):
        self.model = model

    def train_keras_functional_model(self, model):
        self.model = model
