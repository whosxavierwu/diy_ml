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


class BaseData(object):
    def __init__(self, config_dict):
        self.config_dict = config_dict
        self.lexicalization = {
            'train': None,
            'dev': None,
            'test': None
        }

    def setup(self):
        """
        read train/dev/test data
        setup vocab
        :return:
        """
        train_filename = self.config_dict.get('train_filename', None)
        if train_filename is not None:
            train_x_raw = []
            train_y_raw = []
            train_lex = []
            df_train = pd.read_csv(train_filename, encoding='utf8', sep=',')

            def vectorize_mr(mr):
                mr_vec = [PAD_ID] * MR_KEY_NUM
                for kv in mr.split(", "):
                    k, v = kv.replace(']', '').split('[')
                    mr_vec[MR_KEYMAP[k]] = v
                return mr_vec

            df_train['mr_vec'] = df_train['mr'].apply(vectorize_mr)

            def tokenize_snt(r):
                # should replace name & near here
                tokens = []
                snt = r['ref']
                snt = snt.replace(r['mr_vec'][MR_KEYMAP['name']], '')
                snt = snt.replace(r['mr_vec'][MR_KEYMAP['near']], '')
                for fragment in snt.strip().split(" "):
                    fragment_tokens = _WORD_SPLIT.split(fragment)
                    tokens.extend(fragment_tokens)
                tokens = [token.strip() for token in tokens if token.strip() != '']
                return tokens
            df_train['snt_tokens'] = df_train.apply(lambda r: tokenize_snt(r), axis=1)

            # curr_src, curr_lex = self.process_e2e_mr(curr_mr)  # list of attribute values
            # curr_text = self.tokenize(curr_snt, curr_lex)  # list of tokens

            self.lexicalization['train'] = train_lex




if __name__ == '__main__':
    pass

