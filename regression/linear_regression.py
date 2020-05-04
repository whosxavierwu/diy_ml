# -*- coding: utf8 -*-
# Created by: wuzewei
# Created at: 2020/3/22 0022
import tensorflow as tf
from tensorflow import data as tfdata
from tensorflow import initializers as init
from tensorflow import losses
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers


if __name__ == '__main__':
    # generate sample data
    num_of_features = 10
    num_of_samples = 10000
    batch_size = 16
    w_true = tf.random.normal([num_of_features, 1], mean=0, stddev=1)
    b_true = tf.random.uniform([1, 1], minval=-10, maxval=10)
    eps = tf.random.normal([num_of_samples, 1], mean=0, stddev=0.01)
    X = tf.random.normal([num_of_samples, num_of_features], mean=0, stddev=1)
    y = tf.matmul(X, w_true) + b_true + eps
    dataset = tfdata.Dataset.from_tensor_slices((X, y))
    # data preprocessing
    dataset = dataset.shuffle(buffer_size=num_of_samples).batch(batch_size)
    # model
    model = keras.Sequential()
    model.add(layers.Dense(1, kernel_initializer=init.RandomNormal(mean=0, stddev=0.01)))
    # loss
    loss = losses.MeanSquaredError()
    # optimizer
    opt = optimizers.SGD(learning_rate=0.03)
    # training
    num_epochs = 5
    for ep in range(num_epochs):
        for X_cur, y_cur in dataset:
            with tf.GradientTape() as tape:
                batch_loss = loss(model(X_cur, training=True), y_cur)
            grads = tape.gradient(batch_loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
        epoch_loss = loss(model(X), y)
        print("epoch #%d: %.6f" % (ep, tf.reduce_mean(epoch_loss)))

