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
        # TPU detection
        try:
            self.tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            print('Running on TPU ', tpu.master())
        except ValueError:
            self.tpu = None
            self.gpu = tf.config.experimental.list_logical_devices("GPU")
        # Set the training strategy
        if self.tpu:
            tf.config.experimental_connect_to_cluster(self.tpu)
            tf.tpu.experimental.initialize_tpu_system(self.tpu)
            self.strategy = tf.distribute.experimental.TPUStrategy(self.tpu)
        elif len(self.gpu) > 0:
            self.strategy = tf.distribute.MirroredStrategy(self.gpu)
            print('Running on ', len(self.gpu), ' GPU(s) ')
        else:
            self.strategy = tf.distribute.get_strategy()
            print('Running on CPU')

    def add_keras_seq_model(self, model):
        self.model = model

    def add_keras_functional_model(self, model):
        self.model = model

    def compile(self, optimizer, loss, metrics):
        if self.model:
            with self.strategy.scope():
                self.model.compile(
                    optimizer=optimizer,
                    loss=loss,
                    metrics=metrics
                )  # Add args
        else:
            raise ValueError("Define model needs to be added")
