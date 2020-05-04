# -*- coding: utf8 -*-
# Created by: wuzewei
# Created at: 2020/3/23 0023
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras as K
from tensorflow.keras import layers as L


def xyplot(x_vals, y_vals, name):
    # %config InlineBackend.figure_format = 'svg'
    plt.rcParams['figure.figsize'] = (5, 2.5)
    plt.plot(x_vals.numpy(), y_vals.numpy())
    plt.xlabel('x')
    plt.ylabel(name + '(x)')
    return


def plot_activation_function(func, func_name):
    x = tf.Variable(tf.range(-8, 8, 0.1), dtype=tf.float32)
    y = func(x)
    xyplot(x, y, func_name)
    plt.show()

    with tf.GradientTape() as tape:
        tape.watch(x)
        y = func(x)
    dy_dx = tape.gradient(y, x)
    xyplot(x, dy_dx, 'grad of ' + func_name)
    plt.show()


if __name__ == '__main__':
    # plot_activation_function(tf.nn.relu, 'relu')
    # plot_activation_function(tf.nn.sigmoid, 'sigmoid')
    # plot_activation_function(tf.nn.tanh, 'tanh')

    (X_train, y_train), (X_test, y_test) = K.datasets.fashion_mnist.load_data()
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    model = K.Sequential([
        L.Flatten(input_shape=(28,28)),
        L.Dense(256, activation='relu'),
        L.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=K.optimizers.SGD(lr=0.5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print(model.summary())
    model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=256,
        validation_data=(X_test, y_test),
        validation_freq=1
    )

