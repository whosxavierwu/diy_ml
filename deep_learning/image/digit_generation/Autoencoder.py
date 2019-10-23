# -*- coding: utf8 -*-
# Created by: wuzewei
# Created by: 2019/10/23
import numpy as np
from matplotlib import pyplot as plt
import os
import pickle

import tensorflow as tf

# error with tf.keras, but ok with keras, why???
# Inputs to eager execution function cannot be Keras symbolic tensors

# from tensorflow.keras import datasets as K_datasets
# from tensorflow.keras import models as K_models
# from tensorflow.keras import layers as K_layers
# from tensorflow.keras import utils as K_utils
# from tensorflow.keras import optimizers as K_optimizers
# from tensorflow.keras import backend as K_backend
from keras import datasets as K_datasets
from keras import models as K_models
from keras import layers as K_layers
from keras import utils as K_utils
from keras import optimizers as K_optimizers
from keras import backend as K_backend

from keras.callbacks import ModelCheckpoint

from callbacks import CustomCallback, step_decay_schedule


class Autoencoder:
    def __init__(self,
                 input_dim, z_dim,
                 encoder_conv_settings, decoder_conv_settings,
                 is_variational=True, use_batch_norm=False, use_dropout=False):
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.encoder_conv_settings = encoder_conv_settings  # filters, kernel size, strides
        self.decoder_conv_settings = decoder_conv_settings  # filters, kernel size, strides
        self.is_variational = is_variational
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self._build()
        return

    def _build(self):
        # Encoder
        encoder_input = K_layers.Input(shape=self.input_dim, name='encoder_input')
        x = encoder_input
        for i, encoder_conv_setting in enumerate(self.encoder_conv_settings):
            # Conv layers
            conv_layer = K_layers.Conv2D(
                filters=encoder_conv_setting[0],
                kernel_size=encoder_conv_setting[1],
                strides=encoder_conv_setting[2],
                padding='same',
                name='encoder_conv_' + str(i)
            )
            x = conv_layer(x)
            # "BAD": batch norm & activation & dropout
            if self.use_batch_norm:
                x = K_layers.BatchNormalization()(x)
            x = K_layers.LeakyReLU()(x)
            if self.use_dropout:
                x = K_layers.Dropout(0.25)(x)
        # Flatten layer
        shape_before_flatten = K_backend.int_shape(x)[1:]
        x = K_layers.Flatten()(x)

        if self.is_variational:
            self.mu = K_layers.Dense(self.z_dim, name='mu')(x)
            self.log_var = K_layers.Dense(self.z_dim, name='log_var')(x)

            def __sampling(args):
                mu, log_var = args
                epsilon = K_backend.random_normal(shape=K_backend.shape(mu), mean=0, stddev=1)
                return mu + K_backend.exp(log_var / 2) * epsilon  # mu + sigma * epsilon

            encoder_output = K_layers.Lambda(__sampling, name='encoder_output')([self.mu, self.log_var])
        else:
            encoder_output = K_layers.Dense(self.z_dim, name='encoder_output')(x)

        self.encoder = K_models.Model(encoder_input, encoder_output)

        # Decoder
        decoder_input = K_layers.Input(shape=(self.z_dim,), name='decoder_input')
        x = K_layers.Dense(np.prod(shape_before_flatten))(decoder_input)

        x = K_layers.Reshape(shape_before_flatten)(x)

        for i, decoder_conv_setting in enumerate(self.decoder_conv_settings):
            conv_t_layer = K_layers.Conv2DTranspose(
                filters=decoder_conv_setting[0],
                kernel_size=decoder_conv_setting[1],
                strides=decoder_conv_setting[2],
                padding='same',
                name='decoder_conv_t_' + str(i)
            )
            x = conv_t_layer(x)
            if i < len(self.decoder_conv_settings) - 1:
                x = K_layers.LeakyReLU()(x)
            else:
                x = K_layers.Activation('sigmoid')(x)
        decoder_output = x

        self.decoder = K_models.Model(decoder_input, decoder_output)

        # Full Autoencoder
        self.model = K_models.Model(encoder_input, self.decoder(encoder_output))
        return

    def compile(self, learning_rate, r_loss_factor=1000):
        self.learning_rate = learning_rate

        def __r_loss(y_true, y_pred):
            return K_backend.mean(K_backend.square(y_true - y_pred), axis=[1, 2, 3])

        def __kl_loss(y_true, y_pred):
            # todo what's KL divergence?
            return -0.5 * K_backend.sum(
                1 + self.log_var - K_backend.square(self.mu) - K_backend.exp(self.log_var),
                axis=1
            )

        def __loss(y_true, y_pred):
            if self.is_variational:
                return r_loss_factor * __r_loss(y_true, y_pred) + __kl_loss(y_true, y_pred)
            else:
                return __r_loss(y_true, y_pred)

        optimizer = K_optimizers.Adam(lr=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss=__loss,
            metrics=[__r_loss, __kl_loss]
        )
        return

    def train(self,
              X_train,
              batch_size, epochs,
              run_folder,
              print_every_n_batches=100, initial_epoch=0,
              lr_decay=1
              ):
        custom_callback = CustomCallback(run_folder, print_every_n_batches, initial_epoch, self)
        lr_schedule = step_decay_schedule(self.learning_rate, lr_decay, step_size=1)

        checkpoint_filepath1 = os.path.join(run_folder, 'weights/weights-{epoch:03d}-{loss:.2f}.h5')
        checkpoint_filepath2 = os.path.join(run_folder, 'weights/weights.h5')
        checkpoint1 = ModelCheckpoint(checkpoint_filepath1, save_weights_only=True, verbose=1)
        checkpoint2 = ModelCheckpoint(checkpoint_filepath2, save_weights_only=True, verbose=1)
        callbacks_list = [checkpoint1, checkpoint2, custom_callback, lr_schedule]
        self.model.fit(
            X_train, X_train,
            batch_size=batch_size,
            shuffle=True,
            epochs=epochs,
            initial_epoch=initial_epoch,
            callbacks=callbacks_list
        )
        return

    def save(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
            os.makedirs(os.path.join(folder, 'viz'))
            os.makedirs(os.path.join(folder, 'weights'))
            os.makedirs(os.path.join(folder, 'images'))
        with open(os.path.join(folder, 'params.pkl'), 'wb') as fout:
            pickle.dump([
                self.input_dim,
                self.z_dim,
                self.encoder_conv_settings,
                self.decoder_conv_settings,
                self.is_variational,
                self.use_batch_norm,
                self.use_dropout,
            ], fout)
        self.plot_model(folder)
        return

    def load(self, filepath):
        self.model.load_weights(filepath)
        return

    def plot_model(self, run_folder):
        K_utils.plot_model(self.model, os.path.join(run_folder, 'viz/model.png'),
                           show_shapes=True, show_layer_names=True)
        K_utils.plot_model(self.encoder, os.path.join(run_folder, 'viz/encoder.png'),
                           show_shapes=True, show_layer_names=True)
        K_utils.plot_model(self.decoder, os.path.join(run_folder, 'viz/decoder.png'),
                           show_shapes=True, show_layer_names=True)
        return


if __name__ == '__main__':
    ae = Autoencoder(
        input_dim=[28, 28, 1],
        encoder_conv_settings=[
            [32, 3, 1],
            [64, 3, 2],
            [64, 3, 2],
            [64, 3, 1],
        ],
        decoder_conv_settings=[
            [64, 3, 1],
            [64, 3, 2],
            [32, 3, 2],
            [1, 3, 1],
        ],
        z_dim=2
    )
    ae.compile(0.0005, 1000)


