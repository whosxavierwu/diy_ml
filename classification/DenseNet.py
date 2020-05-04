# -*- coding: utf8 -*-
# Created by: wuzewei
# Created at: 2020/4/6 0006
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from DataLoader import DataLoader

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)


class BottleNeck(layers.Layer):
    def __init__(self, growth_rate, drop_rate):
        super().__init__()
        self.listLayers = [
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(4 * growth_rate, kernel_size=1, strides=1, padding='same'),

            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(growth_rate, kernel_size=3, strides=1, padding='same'),

            layers.Dropout(drop_rate),
        ]
        return

    def call(self, X, **kwargs):
        Y = X
        for layer in self.listLayers:
            Y = layer(Y)
        Y = layers.concatenate([X, Y], axis=-1)
        return Y


class DenseBlock(layers.Layer):
    def __init__(self, num_layers, growth_rate, drop_rate=0.5):
        super().__init__()
        self.listLayers = [
            BottleNeck(growth_rate, drop_rate)
            for _ in range(num_layers)
        ]
        return

    def call(self, X, **kwargs):
        for layer in self.listLayers:
            X = layer(X)
        return X


class TransitionLayer(layers.Layer):
    def __init__(self, out_channels):
        super().__init__()
        self.listLayers = [
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(out_channels, kernel_size=1, strides=1, padding='same'),
            layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
        ]
        return

    def call(self, X, **kwargs):
        for layer in self.listLayers:
            X = layer(X)
        return X


class DenseNet(keras.Model):
    def __init__(self, num_init_features, growth_rate, block_layers, compression_rate, drop_rate):
        super().__init__()
        self.listLayers = []
        self.listLayers.extend([
            layers.Conv2D(num_init_features, kernel_size=7, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
        ])

        num_channels = num_init_features
        for i in range(len(block_layers)):
            self.listLayers.append(DenseBlock(block_layers[i], growth_rate, drop_rate))
            if i < len(block_layers) - 1:
                num_channels = (num_channels + growth_rate * block_layers[i]) * compression_rate
                self.listLayers.append(TransitionLayer(out_channels=int(num_channels)))

        self.listLayers.extend([
            layers.GlobalAveragePooling2D(),
            layers.Dense(10, activation='softmax'),
        ])
        return

    def call(self, X, **kwargs):
        for layer in self.listLayers:
            X = layer(X)
        return X


if __name__ == '__main__':
    net = DenseNet(
        num_init_features=64,
        growth_rate=32,
        block_layers=[4, 4, 4, 4],
        compression_rate=0.5,
        drop_rate=0.5
    )

    # check the output_shape of each layer
    input_shape = (1, 96, 96, 1)
    net.build(input_shape=input_shape)
    print(net.summary())
    X = tf.random.uniform(input_shape)
    for blk in net.layers:
        X = blk(X)
        print(blk.name, '\t', X.shape)

    net.compile(
        optimizer=keras.optimizers.Adam(1e-7),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    batch_size = 16
    dataLoader = DataLoader()

    weight_filename = 'tmp/densenet_weights.h5'

    epochs = 1
    num_iter = dataLoader.num_train // batch_size
    for ep in range(epochs):
        for n in range(num_iter):
            X_batch, y_batch = dataLoader.get_batch_train(batch_size)
            net.fit(X_batch, y_batch)
            if n % 20 == 0:
                net.save_weights(weight_filename)

    net.load_weights(weight_filename)
    X_test, y_test = dataLoader.get_batch_test(2000)
    net.evaluate(X_test, y_test, verbose=2)
