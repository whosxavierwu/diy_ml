# -*- coding: utf8 -*-
# Created by: wuzewei
# Created at: 2020/4/6 0006
import numpy as np

import tensorflow as tf
from tensorflow import keras


class DataLoader:
    def __init__(self):
        fashion_mnist = keras.datasets.fashion_mnist
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
    pass

