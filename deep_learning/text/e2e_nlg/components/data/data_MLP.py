# -*- coding: utf8 -*-
# Created by: wuzewei
# Created by: 2019/8/1
import pandas as pd
import re

# Regular expressions used to tokenize strings.
_WORD_SPLIT = re.compile(r"([.,!?\"':;)(])")
_FRAC_NUM_PAT = re.compile(r'(.?)(\d+\.\d+)')
_ZERO_PAT = re.compile(r'.?\d+\.00')

PAD_TOKEN = '<blank>'
BOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
UNK_TOKEN = '<unk>'

# we dont need to use the value of fields "name" and "near"
NAME_TOKEN = '<name>'
NEAR_TOKEN = '<near>'

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

START_VOCAB = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

MR_FIELDS = ["name", "familyFriendly", "eatType", "food", "priceRange", "near", "area", "customer rating"]
MR_KEYMAP = dict((key, idx) for idx, key in enumerate(MR_FIELDS))
MR_KEY_NUM = len(MR_FIELDS)

class MLPData:
    def __init__(self, config_dict):
        self.config_dict = config_dict

    def setup(self):
        train_filename = self.config_dict.get('train_filename', None)
        dev_filename = self.config_dict.get('dev_filename', None)
        test_filename = self.config_dict.get('test_filename', None)
        df_train = None
        df_dev = None
        df_test = None
        if train_filename is not None: df_train = self.read_data(train_filename)
        if dev_filename is not None: df_dev = self.read_data(dev_filename)
        if test_filename is not None: df_test = self.read_data(test_filename)
        return df_train, df_dev, df_test

    def process_mr(self, mr):
        d = {}
        for kv in mr.split(", "):
            k, v = kv.replace(']', '').split('[')
            d[k] = v
        return d

    def process_ref(self, r):
        # todo ths goes wrong with token
        sentence = r['ref'].replace(r['name'], NAME_TOKEN).replace(r['near'], NEAR_TOKEN)
        tokens = []
        for fragment in sentence.strip().split(" "):
            fragment_tokens = _WORD_SPLIT.split(fragment)
            tokens.extend(fragment_tokens)
        tokens = [token.strip() for token in tokens if token.strip() != '']
        return tokens

    def read_data(self, filename):
        df = pd.read_csv(filename, encoding='utf8', sep=',')
        df['mr_dict'] = df['mr'].apply(self.process_mr)
        for field in MR_FIELDS:
            df[field] = df['mr_dict'].apply(lambda d: d.get(field, ''))
        cols = MR_FIELDS
        if 'ref' in df.columns:
            df['tokens'] = df.apply(lambda r: self.process_ref(r), axis=1)
            cols.append('tokens')
        return df[cols]


if __name__ == '__main__':
    data = MLPData({
        'train_filename': '../e2e-dataset/trainset.csv',
        'dev_filename': '../e2e-dataset/devset.csv',
        'test_filename': '../e2e-dataset/testset.csv',
    })
    df_train, df_dev, df_test = data.setup()
    print(df_train.shape, df_dev.shape, df_test.shape)
    print(df_train.head())
    print(df_dev.head())
    print(df_test.head())

