# -*- coding: utf8 -*-
# Created by: wuzewei
# Created at: 2020/3/23 0023
import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels):
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.reshape((28, 28)))
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


def softmax(logits, axis=-1):
    return tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis, keepdims=True)


def net(X):
    logits = tf.matmul(tf.reshape(X, (-1, W.shape[0])), W) + b
    return softmax(logits)


def cross_entropy(y_pred, y_true):
    y_true = tf.cast(tf.reshape(y_true, (-1, 1)), dtype=tf.int32)
    y_true = tf.one_hot(y_true, depth=y_pred.shape[-1])
    y_true = tf.cast(tf.reshape(y_true, (-1, y_pred.shape[-1])), dtype=tf.int32)
    # why +1e-8 ???
    return -tf.math.log(tf.boolean_mask(y_pred, mask=y_true) + 1e-8)


def accuracy(y_pred, y_true):
    return np.mean(tf.argmax(y_pred, axis=1) == y_true)


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        y = tf.cast(y, dtype=tf.int64)
        acc_sum += np.sum(tf.cast(tf.argmax(net(X), axis=1), dtype=tf.int64) == y)
        n += y.shape[0]
    return acc_sum / n


def train(net, train_iter, test_iter, loss, num_of_epochs, batch_size,
          params=None, lr=None, trainer=None):
    num_of_epochs = 5
    lr = 0.1
    for ep in range(num_of_epochs):
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        n = 0
        for X, y in train_iter:
            with tf.GradientTape() as tape:
                y_pred = net(X)
                loss1 = tf.reduce_sum(loss(y_pred, y))
            grads = tape.gradient(loss1, params)
            if trainer is None:
                for i, param in enumerate(params):
                    param.assign_sub(lr * grads[i] / batch_size)
            else:
                trainer.apply_gradients(
                    zip([grad / batch_size for grad in grads], params)
                )
            y = tf.cast(y, dtype=tf.float32)
            train_loss_sum += loss1.numpy()
            train_acc_sum += tf.reduce_sum(
                tf.cast(
                    tf.argmax(y_pred, axis=1) == tf.cast(y, dtype=tf.int64),
                    dtype=tf.int64
                )
            ).numpy()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'% (ep + 1, train_loss_sum / n, train_acc_sum / n, test_acc))
    return


if __name__ == '__main__':
    from tensorflow.keras.datasets import fashion_mnist
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    # data preprocess
    X_train = tf.cast(X_train, tf.float32) / 255.0
    X_test = tf.cast(X_test, tf.float32) / 255.0
    # model
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=X_train.shape[1:3]),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(
        optimizer=keras.optimizers.SGD(0.1),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print(model.summary())
    # training
    model.fit(X_train, y_train, epochs=10, batch_size=256)
    # test
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(test_loss, test_acc)

