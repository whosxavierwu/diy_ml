# -*- coding: utf8 -*-
# Created by: wuzewei
# Created by: 2019/10/28
import tensorflow as tf
from keras import datasets as K_datasets
from keras.preprocessing.image import ImageDataGenerator


def load_mnist():
    (X_train, y_train), (X_test, y_test) = K_datasets.mnist.load_data()
    X_train = X_train.astype('float32') / 255.
    X_train = X_train.reshape(X_train.shape + (1,))
    X_test = X_test.astype('float32') / 255.
    X_test = X_test.reshape(X_test.shape + (1,))
    return X_train, y_train, X_test, y_test


def load_CelebA(data_folder):
    data_gen = ImageDataGenerator(rescale=1./255)
    data_flow = data_gen.flow_from_directory(
        data_folder,
        target_size=(128, 128),
        batch_size=32,
        shuffle=True,
        class_mode='input',
        subset='training'
    )
    return data_flow


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_mnist()
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

