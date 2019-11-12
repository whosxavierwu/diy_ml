# -*- coding: utf-8 -*-

import codecs
import os

import keras_radam as Radam
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import *
from keras.models import Model, load_model
from keras_bert import load_trained_model_from_checkpoint, Tokenizer, get_custom_objects

# roberta_zh_large_model from: https://github.com/brightmart/roberta_zh
CONFIG_PATH = 'tmp/roeberta_zh_L-24_H-1024_A-16/bert_config_large.json'
CHECKPOINT_PATH = 'tmp/roeberta_zh_L-24_H-1024_A-16/roberta_zh_large_model.ckpt'
DICT_PATH = 'tmp/roeberta_zh_L-24_H-1024_A-16/vocab.txt'

CONFIG = {
    'max_len': 1024,  # todo maxlen 需要调大一些 或者提前预处理把数据剪掉一些
    'batch_size': 6,
    'epochs': 32,
    'use_multiprocessing': True,
    'model_dir': os.path.join('model_files/bert'),
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


class DataGenerator:
    def __init__(self, data, tokenizer, batch_size=CONFIG['batch_size']):
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
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
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []


class BertClassify:
    def __init__(self, is_train=True):
        if is_train:
            self.bert_model = load_trained_model_from_checkpoint(
                config_file=CONFIG_PATH,
                checkpoint_file=CHECKPOINT_PATH
            )
            # for layer in self.bert_model.layers[: -CONFIG['trainable_layers']]:
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

        train_D = DataGenerator(train_data, self.tokenizer)
        valid_D = DataGenerator(valid_data, self.tokenizer)
        self.model.fit_generator(
            train_D.__iter__(),
            steps_per_epoch=len(train_D),
            epochs=CONFIG['epochs'],
            callbacks=callbacks,
            validation_data=valid_D.__iter__(),
            use_multiprocessing=CONFIG['use_multiprocessing'],
            validation_steps=len(valid_D)
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
    model = BertClassify(is_train=True)
    model.train(train_data, valid_data)
    print('DONE!')

