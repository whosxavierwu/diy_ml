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


def nin_block(num_channels, kernel_size, strides, padding):
    blk = keras.Sequential([
        layers.Conv2D(num_channels, kernel_size=kernel_size,
                      strides=strides, padding=padding, activation='relu'),
        layers.Conv2D(num_channels, kernel_size=1, activation='relu'),
        layers.Conv2D(num_channels, kernel_size=1, activation='relu'),
    ])
    return blk


if __name__ == '__main__':
    net = keras.Sequential([
        nin_block(96, kernel_size=11, strides=4, padding='valid'),
        layers.MaxPool2D(pool_size=3, strides=2),

        nin_block(256, kernel_size=5, strides=1, padding='same'),
        layers.MaxPool2D(pool_size=3, strides=2),

        nin_block(384, kernel_size=3, strides=1, padding='same'),
        layers.MaxPool2D(pool_size=3, strides=2),

        layers.Dropout(0.5),

        # 使用全局平均池化层对每个通道中所有元素求平均并直接用于分类。
        # 这里的全局平均池化层即窗口形状等于输入空间维形状的平均池化层。
        # NiN的这个设计的好处是可以显著减小模型参数尺寸，从而缓解过拟合。
        # 然而，该设计有时会造成获得有效模型的训练时间的增加。
        nin_block(10, kernel_size=3, strides=1, padding='same'),
        layers.GlobalAveragePooling2D(),

        layers.Flatten(),
    ])

    input_shape = (1, 224, 224, 1)
    X = tf.random.uniform(input_shape)
    for blk in net.layers:
        X = blk(X)
        print(blk.name, '\t', X.shape)

    net.build(input_shape=input_shape)
    print(net.summary())

    batch_size = 128
    dataLoader = DataLoader()
    x_batch, y_batch = dataLoader.get_batch_train(batch_size)
    print("x_batch shape:", x_batch.shape, "y_batch shape:", y_batch.shape)

    net.compile(
        optimizer=keras.optimizers.Adam(1e-7),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    weight_filename = 'tmp/nin_weights.h5'
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
