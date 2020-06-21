import json
import os

import tensorflow as tf


class CloudTrainer:
    def __init__(
        self,
        experiment_name,
        environment="default",
        env_path=None,
        gpu_strategy="auto",
        save_freq=5
    ):
        self.name = experiment_name
        if environment == "default":
            if env_path:
                self.main_path = env_path
            else:
                self.main_path = "drive/My Drive/models"
            self.model_path = f"{self.main_path}/{self.name}"
            self.log_path = f"{self.main_path}/{self.name}/logs"
            self.ckpt = self.model_path + "/model-{epoch}.ckpt"
        elif environment == "s3":
            raise NotImplementedError
        self.callbacks = [
            tf.keras.callbacks.TensorBoard(
                log_dir=self.log_path
            )
        ]
        if save_freq:
            if type(save_freq) != int:
                raise TypeError("Save frequency must be an integer")
            self.callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(self.ckpt,
                                                   save_weights_only=True,
                                                   period=save_freq)
            )
        else:
            self.callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(self.ckpt,
                                                   save_weights_only=True)
            )
        # TPU detection
        try:
            self.tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            print('Running on TPU ', self.tpu.master())
        except ValueError:
            self.tpu = None
            self.gpu = tf.config.experimental.list_logical_devices("GPU")
        # Set the training strategy
        if self.tpu:
            tf.config.experimental_connect_to_cluster(self.tpu)
            tf.tpu.experimental.initialize_tpu_system(self.tpu)
            self.strategy = tf.distribute.experimental.TPUStrategy(self.tpu)
            # Batch size reminder
            print(
                f"Recomenden batch size: {16 * self.strategy.num_replicas_in_sync}")
        elif len(self.gpu) > 0:
            if gpu_strategy == "auto":
                self.strategy = tf.distribute.get_strategy()
            elif gpu_strategy == "mirrored":
                self.strategy = tf.distribute.MirroredStrategy(self.gpu)
            else:
                raise ValueError(
                    "Only auto and mirrored strategies available for GPU(s)"
                )
            print('Running on ', len(self.gpu), ' GPU(s) ')
        else:
            self.strategy = tf.distribute.get_strategy()
            print('Running on CPU')

    def add_callbacks(self, callbacks):
        for callback in callbacks:
            self.callbacks.append(callback)

    def add_keras_seq_model(self, model):
        self.model_pointer = model

    def add_keras_functional_model(self, model):
        self.model_pointer = model

    def compile(self, optimizer, loss, metrics, **kwargs):
        if self.model_pointer:
            with self.strategy.scope():
                self.model = self.model_pointer()
                self.model.compile(
                    optimizer=optimizer,
                    loss=loss,
                    metrics=metrics,
                    **kwargs
                )  # Add args
        else:
            raise ValueError("Define model needs to be added")

    def fit(
        self,
        x=None,
        y=None,
        batch_size=None,
        epochs=1,
        verbose=1,
        validation_split=0.0,
        validation_data=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_batch_size=None,
        validation_freq=1,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False
    ):
        if not os.path.isdir(self.model_path):
            self.history = self.model.fit(
                x=x,
                y=y,
                batch_size=batch_size,
                epochs=epochs,
                verbose=verbose,
                callbacks=self.callbacks,
                validation_split=validation_split,
                validation_data=validation_data,
                shuffle=shuffle,
                class_weight=class_weight,
                sample_weight=sample_weight,
                initial_epoch=initial_epoch,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                validation_batch_size=validation_batch_size,
                validation_freq=validation_freq,
                max_queue_size=max_queue_size,
                workers=workers,
                use_multiprocessing=use_multiprocessing
            )
            with open(f"{self.model_path}/history.json", "w") as f:
                json.dump(self.history.history, f)
            self.model.save(f"{self.model_path}/final_model.h5")
        else:
            # Find the saved model from last epoch
            latest_ckpt = tf.train.latest_checkpoint(self.model_path)
            print(f"Found a checkpoint for the experiment: {latest_ckpt}")
            init_epoch = int(str(latest_ckpt).split("-")[1].split(".")[0])
            # Load the last model weights to model
            self.model.load_weights(latest_ckpt)
            # Set the initial_epoch to last epoch
            # fit the model
            self.history = self.model.fit(
                x=x,
                y=y,
                batch_size=batch_size,
                epochs=epochs,
                verbose=verbose,
                callbacks=self.callbacks,
                validation_split=validation_split,
                validation_data=validation_data,
                shuffle=shuffle,
                class_weight=class_weight,
                sample_weight=sample_weight,
                initial_epoch=init_epoch,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                validation_batch_size=validation_batch_size,
                validation_freq=validation_freq,
                max_queue_size=max_queue_size,
                workers=workers,
                use_multiprocessing=use_multiprocessing
            )
            with open(f"{self.model_path}/history.json", "w") as f:
                json.dump(self.history.history, f)
            self.model.save(f"{self.model_path}/final_model.h5")
