# -*- coding: utf8 -*-
# Created by: wuzewei
# Created by: 2019/8/1
import pandas as pd
import re
import logging
import csv

from components.constants import *
from components.data.vocabulary import *

# Regular expressions used to tokenize strings.
_WORD_SPLIT = re.compile(r"([.,!?\"':;)(])")
_FRAC_NUM_PAT = re.compile(r'(.?)(\d+\.\d+)')
_ZERO_PAT = re.compile(r'.?\d+\.00')


class BaseData(object):
    def __init__(self, config_dict):
        self.config_dict = config_dict
        self.max_src_len = self.config_dict['max_src_len']
        self.max_tgt_len = self.config_dict['max_tgt_len']
        self.vocab = None
        self.fnames = {}
        self.lexicalizations = {
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
        logger.info("Setting up data...")

        train_x_raw = train_y_raw = None
        dev_x_raw = dev_y_raw = None
        test_x_raw = None

        train_filename = self.config_dict.get('train_filename', None)
        dev_filename = self.config_dict.get('dev_filename', None)
        test_filename = self.config_dict.get('test_filename', None)
        vocab_path = "{}.vocab".format(train_filename)

        # train
        if train_filename is not None:
            logger.debug("Reading train data...")
            train_x_raw, train_y_raw, train_lex = self.read_csv_train(train_filename, group_ref=True)
            self.lexicalizations['train'] = train_lex
        # dev
        if dev_filename is not None:
            logger.debug("Reading dev data...")
            dev_x_raw, dev_y_raw, dev_lex = self.read_csv_train(dev_filename, group_ref=True)
            self.lexicalizations['dev'] = dev_lex
        # test
        if test_filename is not None:
            logger.debug("Reading test data...")
            test_x_raw, test_lex = self.read_csv_test(test_filename)
            self.lexicalizations['test'] = test_lex

        # setup vocabulary
        self.setup_vocab(vocab_path, train_x_raw, train_y_raw)
        # tokenize data
        if train_x_raw is not None:
            self.train = self.data_to_token_ids_train(train_x_raw, train_y_raw)
            self.fnames['train'] = train_filename
        if dev_x_raw is not None:
            self.dev = self.data_to_token_ids_train(dev_x_raw, dev_y_raw)
            self.fnames['dev'] = dev_filename
        if test_x_raw is not None:
            self.test = self.data_to_token_ids_test(test_x_raw)
            self.fnames['test'] = test_filename

    def read_csv_train(self, train_filename, group_ref=False):
        raw_data_x = []
        raw_data_y = []
        lexicalizations = []

        orig = []
        with open(train_filename, 'r', encoding='utf8') as fin:
            reader = csv.reader(fin, delimiter=',', quotechar='"')
            header = next(reader)
            assert header == ['mr', 'ref']

            first_row = next(reader)
            curr_mr = first_row[0]
            curr_snt = first_row[1]
            orig.append((curr_mr, curr_snt))
            curr_src, curr_lex = self.process_mr(curr_mr)
            curr_text = self.tokenize(curr_snt, curr_lex)

            raw_data_x.append(curr_src)
            raw_data_y.append(curr_text)
            lexicalizations.append(curr_lex)

            for row in list(reader):
                mr = row[0]
                text = row[1]
                orig.append((mr, text))
                this_src, this_lex = self.process_mr(mr)
                this_text = self.tokenize(text, this_lex)
                raw_data_x.append(this_src)
                raw_data_y.append(this_text)
                if this_src != curr_src:
                    lexicalizations.append(this_lex)
                    curr_src = this_src
        if group_ref:
            self.gen_multi_ref_dev(orig, filename="{}.multi-ref".format(train_filename))
        return raw_data_x, raw_data_y, lexicalizations

    def read_csv_test(self, test_filename):
        raw_data_x = []
        lexicalizations = []
        with codecs.open(test_filename, 'r', encoding='utf8') as fin:
            reader = csv.reader(test_filename, delimiter=',', quotechar='"')
            header = next(reader)
            for row in list(reader):
                mr = row[0]
                this_src, this_lex = self.process_mr(mr)
                raw_data_x.append(this_src)
                lexicalizations.append(this_lex)
        return raw_data_x, lexicalizations

    def tokenize_normalize(self, snt, lex=None):
        words = []
        if lex:
            for l, t in zip(lex, (NAME_TOKEN, NEAR_TOKEN)):
                snt = snt.replace(l, t) if (l is not None) else snt
        for fragment in snt.strip().split(' '):
            # deal with price
            match = _FRAC_NUM_PAT.match(fragment)
            if match:
                fragment_tokens = []
                pound = match.group(1)
                price = match.group(2)
                price = re.sub(r'.00', '', price)
                if not pound.isdigit():
                    fragment_tokens.append(pound)
                fragment_tokens.append(price)
            # deal with zeros
            match = _ZERO_PAT.match(fragment)
            if match:
                fragment_tokens = [re.sub('.00', '', fragment)]
            else:
                fragment_tokens = _WORD_SPLIT.split(fragment)
            words.extend(fragment_tokens)
        tokens = [w for w in words if w]
        return tokens

    def tokenize(self, snt, lex=None):
        words = []
        if lex:
            for l, t in zip(lex, (NAME_TOKEN, NEAR_TOKEN)):
                snt = snt.replace(l,t) if (l is not None) else snt
        for fragment in snt.strip().split(' '):
            fragment_tokens = _WORD_SPLIT.split(fragment)
            words.extend(fragment_tokens)
        tokens = [w for w in words if w]
        return tokens

    def setup_vocab(self, vocab_path, train_x_raw, train_y_raw):
        raise NotImplementedError()

    def process_mr(self, mr):
        raise NotImplementedError()

    def data_to_token_ids_train(self, train_x_raw, train_y_raw):
        raise NotImplementedError()

    def data_to_token_ids_test(self, test_x_raw):
        raise NotImplementedError()

    def prepare_training_data(self, xy_ids, batch_size):
        raise NotImplementedError()

    def gen_multi_ref_dev(self, dev_xy, filename):
        logger.debug("Generating a multi-ref file...")
        multi_ref_src_fn = "{}.src".format(filename)
        with codecs.open(filename, 'w', encoding='utf8') as fout, \
                codecs.open(multi_ref_src_fn, 'w', encoding='utf8') as fout_src:
            curr_mr, curr_txt = dev_xy[0]
            fout.write("{}\n".format(curr_txt))
            fout_src.write("{}\n".format(curr_mr))
            for mr, txt in dev_xy[1:]:
                if mr == curr_mr:
                    fout.write(txt + '\n')
                else:
                    fout.write('\n')
                    fout.write(txt + '\n')
                    curr_mr = mr
                    fout_src.write(mr + '\n')


if __name__ == '__main__':
    pass

