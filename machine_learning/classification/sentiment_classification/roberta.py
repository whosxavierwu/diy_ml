# -*- coding: utf-8 -*-

import codecs
import os

import keras_radam as Radam
import pandas as pd
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import *
from keras.models import Model, load_model
from keras_bert import load_trained_model_from_checkpoint, Tokenizer, get_custom_objects


# roberta_zh_large_model from: https://github.com/brightmart/roberta_zh
CONFIG_PATH = 'tmp/roeberta_zh_L-24_H-1024_A-16/bert_config_large.json'
CHECKPOINT_PATH = 'tmp/roeberta_zh_L-24_H-1024_A-16/roberta_zh_large_model.ckpt'
DICT_PATH = 'tmp/roeberta_zh_L-24_H-1024_A-16/vocab.txt'

# CONFIG_PATH = 'tmp/roberta_zh_L-6-H-768_A-12/bert_config.json'
# CHECKPOINT_PATH = 'tmp/roberta_zh_L-6-H-768_A-12/bert_model.ckpt'
# DICT_PATH = 'tmp/roberta_zh_L-6-H-768_A-12/vocab.txt'

CONFIG = {
    'max_len': 500,
    'batch_size': 1,
    'epochs': 100,
    'use_multiprocessing': True,
    'model_dir': os.path.join('tmp/model-bert'),
    'use_gpu': False,
    # 'trainable_layers': 12,
}


def split_train_test(data, X_name, y_name, train_size=0.85, test_size=None):
    """
    对数据切分成训练集和测试集
    :param data:
    :param X_name:特征列
    :param y_name:标签列
    :param train_size:训练比例
    :return:
    """
    train_data = []
    test_data = []
    if (not train_size) and test_size:
        train_size = 1 - test_size
    for i in range(data.shape[0]):
        if i % 100 < train_size * 100:
            train_data.append([str(data.loc[i][X_name]), data.loc[i][y_name]])
        else:
            test_data.append([str(data.loc[i][X_name]), data.loc[i][y_name]])
    return np.array(train_data), np.array(test_data)


def seq_padding(X, padding=0):
    ML = max([len(x) for x in X])
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))])
        if len(x) < ML
        else x
        for x in X
    ])


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类字符
            else:
                R.append('[UNK]')  # 其他字符
        return R


class BertClassify:
    def __init__(self, is_train=True):
        if is_train:
            self.bert_model = load_trained_model_from_checkpoint(
                config_file=CONFIG_PATH,
                checkpoint_file=CHECKPOINT_PATH
            )
            # for layer in self.bert_model.layers[: -CONFIG['trainable_layers']]:
            #     layer.trainable = True
            for layer in self.bert_model.layers:
                layer.trainable = True
        self.model = None
        token_dict = {}
        with codecs.open(DICT_PATH, 'r', 'utf8') as fin:
            for line in fin:
                token_dict[line.strip()] = len(token_dict)
        self.tokenizer = OurTokenizer(token_dict=token_dict)
        self.__build()
        return

    def __build(self):
        input_layer_1 = Input(shape=(None,))
        input_layer_2 = Input(shape=(None,))

        bert_output_layer = self.bert_model([input_layer_1, input_layer_2])
        lambda_layer = Lambda(lambda x: x[:, 0])(bert_output_layer)
        dropout_layer = Dropout(0.5)(lambda_layer)
        output_layer = Dense(3, activation='softmax')(dropout_layer)

        self.model = Model([input_layer_1, input_layer_2], output_layer)

        self.model.compile(
            # loss='binary_crossentropy',
            loss='categorical_crossentropy',
            optimizer=Radam.RAdam(1e-5),  # 用足够小的学习率
            metrics=['accuracy']
        )
        self.model.summary()
        return

    def __preprocess(self, data):
        idxs = list(range(len(data)))
        np.random.shuffle(idxs)
        X1, X2, Y = [], [], []
        for i in idxs:
            d = data[i]
            x1, x2 = self.tokenizer.encode(first=d[0][:CONFIG['max_len']])
            # y = d[1]
            if d[1] == -1:
                y = [1, 0, 0]
            elif d[1] == 1:
                y = [0, 0, 1]
            else:
                y = [0, 1, 0]

            X1.append(x1)
            X2.append(x2)
            Y.append(y)
        X1 = seq_padding(X1)
        X2 = seq_padding(X2)
        Y = seq_padding(Y)
        return [X1, X2], Y

    def train(self, train_data, valid_data):
        """
        训练
        :param train_data:
        :param valid_data:
        :return:
        """
        save = ModelCheckpoint(
            os.path.join(CONFIG['model_dir'], 'bert.h5'),
            monitor='val_acc',
            verbose=True,
            save_best_only=True,
            mode='auto'
        )
        early_stopping = EarlyStopping(
            monitor='val_acc',
            min_delta=0,
            patience=8,
            verbose=True,
            mode='auto'
        )
        callbacks = [save, early_stopping]

        X_train, Y_train = self.__preprocess(train_data)
        X_valid, Y_valid = self.__preprocess(valid_data)

        self.model.fit(
            x=X_train,
            y=Y_train,
            batch_size=CONFIG['batch_size'],
            epochs=CONFIG['epochs'],
            callbacks=callbacks,
            validation_data=(X_valid, Y_valid),
            use_multiprocessing=CONFIG['use_multiprocessing']
        )
        return

    def predict(self, test_data):
        """
        预测
        :param test_data:
        :return:
        """
        X1 = []
        X2 = []
        for s in test_data:
            x1, x2 = self.tokenizer.encode(first=s[:CONFIG['max_len']])
            X1.append(x1)
            X2.append(x2)
        X1 = seq_padding(X1)
        X2 = seq_padding(X2)
        predict_results = self.model.predict([X1, X2])
        return predict_results

    def load(self, model_dir):
        """
        load the pre-trained model
        """
        model_path = os.path.join(model_dir, 'bert.h5')
        try:
            graph = tf.Graph()
            with graph.as_default():
                session = tf.Session()
                with session.as_default():
                    self.reply = load_model(
                        str(model_path),
                        custom_objects=get_custom_objects(),
                        compile=False
                    )
                    with open(os.path.join(model_dir, 'label_map_bert.txt'), 'r') as f:
                        self.label_map = eval(f.read())
                    self.graph = graph
                    self.session = session
        except Exception as ex:
            print('load error')
        return self


if __name__ == "__main__":
    # 数据集加载 划分
    origin_file = "data/sentiment_corpus_20191108.txt"
    data = pd.read_csv(origin_file, encoding='utf-8', sep='\t', names=['label', 'text_a'])
    data['label'] = data['label'].apply(lambda label: {'positive': 1, 'neutral': 0, 'negative': -1}[label])
    train_data, valid_data = split_train_test(data, 'text_a', 'label', train_size=0.8)
    # 建模 训练

    tf.debugging.set_log_device_placement(False)

    if CONFIG['use_gpu']:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        print("Num GPUs Available: ", len(gpus))
        if gpus:
            try:
                for gpu in gpus:
                    # Currently, memory growth needs to be the same across GPUs
                    tf.config.experimental.set_memory_growth(gpu, True)
                # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
                )
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
                model = BertClassify(is_train=True)
                model.train(train_data, valid_data)
            except RuntimeError as e:
                print(e)
    else:
        with tf.device('/CPU:0'):
            model = BertClassify(is_train=True)
            model.train(train_data, valid_data)
    print('DONE!')



