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


def vgg_block(num_convs, num_channels):
    blk = keras.Sequential()
    for _ in range(num_convs):
        blk.add(
            keras.layers.Conv2D(
                num_channels, kernel_size=3,
                padding='same', activation='relu'
            )
        )
    blk.add(keras.layers.MaxPool2D(pool_size=2, strides=2))
    return blk


def vgg(conv_arch):
    net = keras.Sequential()
    for (num_convs, num_channels) in conv_arch:
        net.add(vgg_block(num_convs, num_channels))
    net.add(keras.Sequential([
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='sigmoid')
    ]))
    return net


if __name__ == '__main__':
    batch_size = 128
    dataLoader = DataLoader()
    x_batch, y_batch = dataLoader.get_batch_train(batch_size)
    print("x_batch shape:", x_batch.shape, "y_batch shape:", y_batch.shape)
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

    net = vgg(conv_arch)

    X = tf.random.uniform([1, 224, 224, 1])
    for blk in net.layers:
        X = blk(X)
        print(blk.name, X.shape)

    ratio = 4
    small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
    net = vgg(small_conv_arch)

    net.compile(
        optimizer=keras.optimizers.SGD(
            learning_rate=0.05, momentum=0.0, nesterov=False
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    weight_filename = 'tmp/vgg_weights.h5'
    epochs = 5
    num_iter = dataLoader.num_train // batch_size
    for ep in range(epochs):
        for n in range(num_iter):
            X_batch, y_batch = dataLoader.get_batch_train(batch_size)
            net.fit(X_batch, y_batch)
            if n % 20 == 0:
                net.save_weights(weight_filename)

    net.load_weights(weight_filename)

    x_test, y_test = dataLoader.get_batch_test(2000)
    net.evaluate(x_test, y_test, verbose=2)
