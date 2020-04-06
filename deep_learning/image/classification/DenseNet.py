# -*- coding: utf8 -*-
# Created by: wuzewei
# Created at: 2020/4/6 0006
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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


class DataLoader():
    def __init__(self):
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (self.X_train, self.y_train), (self.X_test, self.y_test) = fashion_mnist.load_data()
        self.X_train = np.expand_dims(self.X_train.astype(np.float32) / 255.0, axis=-1)
        self.X_test = np.expand_dims(self.X_test.astype(np.float32) / 255.0, axis=-1)
        self.y_train = self.y_train.astype(np.int32)
        self.y_test = self.y_test.astype(np.int32)
        self.num_train, self.num_test = self.X_train.shape[0], self.X_test.shape[0]

    def get_batch_train(self, batch_size):
        index = np.random.randint(0, self.num_train, batch_size)
        # need to resize images to (224,224)
        resized_images = tf.image.resize_with_pad(self.X_train[index], 224, 224, )
        return resized_images.numpy(), self.y_train[index]

    def get_batch_test(self, batch_size):
        index = np.random.randint(0, self.num_test, batch_size)
        # need to resize images to (224,224)
        resized_images = tf.image.resize_with_pad(self.X_test[index], 224, 224, )
        return resized_images.numpy(), self.y_test[index]


if __name__ == '__main__':
    blk = DenseBlock(2, 10)
    X = tf.random.uniform((4, 8, 8, 3))
    Y = blk(X)
    print(Y.shape)
    blk = TransitionLayer(10)
    print(blk(Y).shape)  # TensorShape([4, 4, 4, 10])

    net = DenseNet(
        num_init_features=64,
        growth_rate=32,
        block_layers=[4, 4, 4, 4],
        compression_rate=0.5,
        drop_rate=0.5
    )

    input_shape = (1, 96, 96, 1)
    X = tf.random.uniform(input_shape)
    for blk in net.layers:
        X = blk(X)
        print(blk.name, '\t', X.shape)

    net.build(input_shape=input_shape)
    print(net.summary())

    batch_size = 16
    dataLoader = DataLoader()
    x_batch, y_batch = dataLoader.get_batch_train(batch_size)
    print("x_batch shape:", x_batch.shape, "y_batch shape:", y_batch.shape)

    net.compile(
        optimizer=keras.optimizers.Adam(1e-7),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    epochs = 1
    num_iter = dataLoader.num_train // batch_size
    for ep in range(epochs):
        for n in range(num_iter):
            X_batch, y_batch = dataLoader.get_batch_train(batch_size)
            net.fit(X_batch, y_batch)
            if n % 20 == 0:
                net.save_weights('densenet_weights.h5')

    net.load_weights('densenet_weights.h5')
    X_test, y_test = dataLoader.get_batch_test(2000)
    net.evaluate(X_test, y_test, verbose=2)
