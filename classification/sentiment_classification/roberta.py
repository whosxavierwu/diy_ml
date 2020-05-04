# -*- coding: utf-8 -*-

import codecs
import os

from sklearn import model_selection
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
    'batch_size': 16,
    'epochs': 100,
    'use_multiprocessing': True,
    'model_dir': os.path.join('tmp/model-bert'),
    'use_gpu': False,
    'trainable_layers': 12,  # train only final few layers
}


class MyTokenizer(Tokenizer):
    def __init__(self):
        token_dict = {}
        with codecs.open(DICT_PATH, 'r', 'utf8') as fin:
            for line in fin:
                token_dict[line.strip()] = len(token_dict)
        super().__init__(token_dict)

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


def seq_padding(X, padding=0):
    ML = max([len(x) for x in X])
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))])
        if len(x) < ML
        else x
        for x in X
    ])


def preprocess_data(df_data):
    tokenizer = MyTokenizer()
    is_training = 'label_id' in df_data.columns
    X_token_ids, X_segment_ids, Y = [], [], []
    for _, r in df_data.iterrows():
        token_ids, segment_ids = tokenizer.encode(first=r['content'][:CONFIG['max_len']])
        X_token_ids.append(token_ids)
        X_segment_ids.append(segment_ids)
        if is_training:
            if r['label_id'] == -1:
                y = [1, 0, 0]
            elif r['label_id'] == 1:
                y = [0, 0, 1]
            else:
                y = [0, 1, 0]
            Y.append(y)
    X_token_ids = seq_padding(X_token_ids)
    X_segment_ids = seq_padding(X_segment_ids)
    Y = seq_padding(Y) if is_training else None
    return [X_token_ids, X_segment_ids], Y


class BertClassify:
    def __init__(self, is_train=True):
        if is_train:
            self.bert_model = load_trained_model_from_checkpoint(
                config_file=CONFIG_PATH,
                checkpoint_file=CHECKPOINT_PATH
            )
            for layer in self.bert_model.layers[-CONFIG['trainable_layers']: ]:
                layer.trainable = True
        self.model = None
        self.tokenizer = MyTokenizer()
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
            loss='categorical_crossentropy',
            optimizer=Radam.RAdam(1e-5),
            metrics=['accuracy']
        )
        print(self.model.summary())
        return

    def train(self, X_train, Y_train, X_valid, Y_valid):
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

    def predict(self, X_test):
        Y_test_pred = self.model.predict(X_test)
        return Y_test_pred

    def load(self, model_dir):
        # is not tested
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
    ### Load data
    origin_file = "data/sentiment_corpus_20191108.txt"
    df_data = pd.read_csv(origin_file, encoding='utf-8', sep='\t', names=['label', 'content'])

    label2id = {'negative': -1, 'neutral': 0, 'positive': 1}
    df_data['content_id'] = range(len(df_data))
    df_data['label_id'] = df_data['label'].apply(lambda x: label2id[x])

    ### Split data
    df_train, df_val = model_selection.train_test_split(
        df_data,
        test_size=0.2,
        # random_state=42,
        # stratify=df_data['label_id']
        shuffle=True
    )

    ### Preprocess data
    X_train, y_train = preprocess_data(df_train)
    X_valid, y_valid = preprocess_data(df_val)

    with tf.device('/GPU:0' if CONFIG['use_gpu'] else '/CPU:0'):
        ### Modeling
        model = BertClassify(is_train=True)
        ### Training
        model.train(X_train, y_train, X_valid, y_valid)
        print('DONE training!')
        ### Predict
        test_file = "data/real_senti_demo_nolabel.txt"
        df_test = pd.read_csv(test_file, encoding='utf-8', sep='\t', names=['content'])
        df_test['content_id'] = range(len(df_test))
        X_test, _ = preprocess_data(df_test)
        Y_test_pred = model.predict(X_test)
        y_test_pred_str = []
        for y_test_pred in Y_test_pred:
            if y_test_pred[0] == 1:
                y_test_pred_str.append('negative')
            elif y_test_pred[2] == 1:
                y_test_pred_str.append('positive')
            else:
                y_test_pred_str.append('neutral')
        df_test['label_pred'] = y_test_pred_str
        df_test.sort_values(['content_id'], ascending=True)[['label_pred', 'content']]\
            .to_csv('data/test_pred.csv', sep='\t', encoding='utf8')

