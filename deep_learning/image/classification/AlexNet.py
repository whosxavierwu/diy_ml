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

if __name__ == '__main__':
    net = keras.Sequential([
        # conv 1
        layers.Conv2D(96, kernel_size=11, strides=4, activation='relu'),
        layers.MaxPool2D(pool_size=3, strides=2),
        # conv 2
        layers.Conv2D(256, kernel_size=5, padding='same', activation='relu'),
        layers.MaxPool2D(pool_size=3, strides=2),
        # conv 3 4 5
        layers.Conv2D(384, kernel_size=3, padding='same', activation='relu'),
        layers.Conv2D(384, kernel_size=3, padding='same', activation='relu'),
        layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'),
        layers.MaxPool2D(pool_size=3, strides=2),
        # dense
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='sigmoid'),
    ])

    # print(net.summary())

    X = tf.random.uniform((1, 224, 224, 1))
    for layer in net.layers:
        X = layer(X)
        print(layer.name, 'output shape\t', X.shape)

    batch_size = 128
    dataLoader = DataLoader()
    x_batch, y_batch = dataLoader.get_batch_train(batch_size)
    print("x_batch shape:", x_batch.shape, "y_batch shape:", y_batch.shape)

    net.compile(
        optimizer=keras.optimizers.SGD(
            learning_rate=0.01, momentum=0.0, nesterov=False
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    weight_filename = 'tmp/alexnet_weights.h5'
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
