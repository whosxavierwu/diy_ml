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


class Residual(keras.Model):
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(
            num_channels,
            kernel_size=3,
            strides=strides,
            padding='same'
        )
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()

        self.conv2 = layers.Conv2D(
            num_channels,
            kernel_size=3,
            padding='same'
        )
        self.bn2 = layers.BatchNormalization()

        self.conv3 = None
        if use_1x1conv:
            self.conv3 = layers.Conv2D(
                num_channels,
                kernel_size=1,
                strides=strides
            )

        self.relu2 = layers.ReLU()
        return

    def call(self, X, **kwargs):
        Y = self.relu1(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return self.relu2(Y + X)


class ResnetBlock(layers.Layer):
    def __init__(self, num_channels, num_residuals, first_block=False, **kwargs):
        super().__init__(**kwargs)
        self.listLayers = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                self.listLayers.append(Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                self.listLayers.append(Residual(num_channels))
        return

    def call(self, X, **kwargs):
        for layer in self.listLayers:
            X = layer(X)
        return X


class ResNet(keras.Model):
    def __init__(self, num_blocks, **kwargs):
        super().__init__(**kwargs)
        self.listLayers = [
            layers.Conv2D(64, kernel_size=7, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool2D(pool_size=3, strides=2, padding='same'),

            ResnetBlock(64, num_blocks[0], first_block=True),
            ResnetBlock(128, num_blocks[1]),
            ResnetBlock(256, num_blocks[2]),
            ResnetBlock(512, num_blocks[3]),

            layers.GlobalAvgPool2D(),
            layers.Dense(10, activation='softmax')
        ]
        return

    def call(self, X, **kwargs):
        for layer in self.listLayers:
            X = layer(X)
        return X


if __name__ == '__main__':
    blk = Residual(3)
    # tensorflow input shpe     (n_images, x_shape, y_shape, channels).
    # mxnet.gluon.nn.conv_layers    (batch_size, in_channels, height, width)
    X = tf.random.uniform((4, 6, 6, 3))
    print(blk(X).shape)  # TensorShape([4, 6, 6, 3])

    blk = Residual(6, use_1x1conv=True, strides=2)
    print(blk(X).shape)
    # TensorShape([4, 3, 3, 6])

    net = ResNet([2, 2, 2, 2])

    # # the net above is same as:
    # num_blocks = [2, 2, 2, 2]
    # net = keras.Sequential([
    #     layers.Conv2D(64, kernel_size=7, strides=2, padding='same'),
    #     layers.BatchNormalization(),
    #     layers.Activation('relu'),
    #     layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
    #
    #     ResnetBlock(64, num_blocks[0], first_block=True),
    #     ResnetBlock(128, num_blocks[1]),
    #     ResnetBlock(256, num_blocks[2]),
    #     ResnetBlock(512, num_blocks[3]),
    #
    #     layers.GlobalAvgPool2D(),
    #     layers.Dense(10, activation='softmax')
    # ])

    input_shape = (1, 224, 224, 1)
    X = tf.random.uniform(input_shape)
    for blk in net.layers:
        X = blk(X)
        print(blk.name, '\t', X.shape)

    net.build(input_shape=input_shape)
    print(net.summary())

    batch_size = 64
    dataLoader = DataLoader()
    x_batch, y_batch = dataLoader.get_batch_train(batch_size)
    print("x_batch shape:", x_batch.shape, "y_batch shape:", y_batch.shape)

    net.compile(
        optimizer=keras.optimizers.Adam(1e-7),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    weight_filename = 'tmp/resnet_weights.h5'

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
