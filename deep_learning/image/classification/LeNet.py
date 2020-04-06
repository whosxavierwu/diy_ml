# -*- coding: utf8 -*-
# Created by: wuzewei
# Created at: 2020/4/6 0006
import numpy as np

import tensorflow as tf
from tensorflow import keras

from DataLoader import DataLoader

if __name__ == '__main__':
    # 卷积层块里的基本单位是卷积层后接最大池化层：卷积层用来识别图像里的空间模式，
    # 如线条和物体局部，
    # 之后的最大池化层则用来降低卷积层对位置的敏感性。
    # 卷积层块由两个这样的基本单位重复堆叠构成。

    net = keras.Sequential([
        # conv 1
        keras.layers.Conv2D(
            filters=6, kernel_size=5, activation='sigmoid',
            input_shape=[28, 28, 1]
        ),
        keras.layers.MaxPool2D(pool_size=2, strides=2),
        # conv 2
        keras.layers.Conv2D(
            filters=16, kernel_size=5, activation='sigmoid'
        ),
        keras.layers.MaxPool2D(pool_size=2, strides=2),
        keras.layers.Flatten(),
        keras.layers.Dense(120, activation='sigmoid'),
        keras.layers.Dense(84, activation='sigmoid'),
        keras.layers.Dense(10, activation='sigmoid'),
    ])

    X = tf.random.uniform([1, 28, 28, 1])
    for layer in net.layers:
        X = layer(X)
        print(layer.name, 'output shape\t', X.shape)

    net.compile(
        optimizer=keras.optimizers.SGD(
            learning_rate=0.9, momentum=0.0, nesterov=False
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # fashion_mnist = keras.datasets.fashion_mnist
    # (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    #
    # X_train = tf.reshape(X_train, list(X_train.shape) + [1])
    # X_test = tf.reshape(X_test, list(X_test.shape) + [1])
    # print(X_train.shape, X_test.shape)
    # net.fit(X_train, y_train, epochs=20, validation_split=0.1)
    # net.evaluate(X_test, y_test, verbose=2)

    dataLoader = DataLoader()
    batch_size = 128
    weight_filename = 'tmp/lenet_weights.h5'
    epochs = 10
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
