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


class Inception(layers.Layer):
    def __init__(self, c1, c2, c3, c4):
        super().__init__()
        # route 1: 1x1 conv
        self.p1_1 = keras.layers.Conv2D(c1, kernel_size=1, padding='same', activation='relu')

        # route 2: 1x1 conv & 3x3 conv
        self.p2_1 = keras.layers.Conv2D(c2[0], kernel_size=1, padding='same', activation='relu')
        self.p2_2 = keras.layers.Conv2D(c2[1], kernel_size=3, padding='same', activation='relu')

        # route 3: 1x1 conv & 5x5 conv
        self.p3_1 = keras.layers.Conv2D(c3[0], kernel_size=1, padding='same', activation='relu')
        self.p3_2 = keras.layers.Conv2D(c3[1], kernel_size=5, padding='same', activation='relu')

        # route 4: 3x3 pool & 1x1 conv
        self.p4_1 = keras.layers.MaxPool2D(pool_size=3, strides=1, padding='same')
        self.p4_2 = keras.layers.Conv2D(c4, kernel_size=1, padding='same', activation='relu')
        return

    def call(self, x, **kwargs):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        return tf.concat([p1, p2, p3, p4], axis=-1)


if __name__ == '__main__':
    blk_1 = keras.Sequential([
        layers.Conv2D(64, kernel_size=7, strides=2, padding='same', activation='relu'),
        layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
    ])

    blk_2 = keras.Sequential([
        layers.Conv2D(64, kernel_size=1, padding='same', activation='relu'),
        layers.Conv2D(64 * 3, kernel_size=3, padding='same', activation='relu'),
        layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
    ])

    blk_3 = keras.Sequential([
        Inception(64, (96, 128), (16, 32), 32),
        Inception(128, (128, 192), (32, 96), 64),
        layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
    ])

    blk_4 = keras.Sequential([
        Inception(192, (96, 208), (16, 48), 64),
        Inception(160, (112, 224), (24, 64), 64),
        Inception(128, (128, 256), (24, 64), 64),
        Inception(112, (144, 288), (32, 64), 64),
        Inception(256, (160, 320), (32, 128), 128),
        layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
    ])

    blk_5 = keras.Sequential([
        Inception(256, (160, 320), (32, 128), 128),
        Inception(384, (192, 384), (48, 128), 128),
        layers.GlobalAveragePooling2D(),
    ])

    net = keras.Sequential([
        blk_1, blk_2, blk_3, blk_4, blk_5,
        layers.Dense(10),
    ])

    input_shape = (1, 96, 96, 1)
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

    weight_filename = 'tmp/googlenet_weights.h5'
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
