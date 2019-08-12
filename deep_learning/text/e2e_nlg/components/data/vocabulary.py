# -*- coding: utf8 -*-
# Created by: wuzewei
# Created by: 2019/8/12
import logging
import os
import codecs
from components.constants import *

logger = logging.getLogger('experiment')


class VocabularyBase(object):
    def __init__(self, lower=True):
        self.id2tok = {}
        self.tok2id = {}
        self.lower = lower

    def load(self, vocab_path):
        logger.debug("Loading vocabulary from {}".format(vocab_path))
        token_list = []
        with codecs.open(vocab_path, encoding='utf8') as fin:
            for line in fin:
                token_list.append(line.strip())
        for idx, tok in enumerate(token_list):
            self.id2tok[idx] = tok
            self.tok2id[tok] = idx

    def process_raw_data(self, raw_data):
        token_set = set()
        for x in raw_data:
            for tok in x:
                tok = tok.lower() if self.lower else tok
                token_set.add(tok)
        return list(token_set)

    def get_word(self, key, default=UNK_ID):
        key = key.lower() if self.lower else key
        val = self.tok2id.get(key, default)
        return val

    def get_label(self, idx):
        return self.id2tok[idx]

    def __len__(self):
        return self.size()

    def size(self):
        return len(self.id2tok)


class VocabularyShared(VocabularyBase):
    def __init__(self, vocab_path, raw_data_src=None, raw_data_tgt=None, lower=True):
        super().__init__(lower=lower)
        if not os.path.exists(os.path.abspath(vocab_path)):
            assert (raw_data_src is not None) and (raw_data_tgt is not None), \
                "You need to process train data before creating vocabulary!"
            self.create_vocabulary(
                raw_data_src=raw_data_src,
                raw_data_tgt=raw_data_tgt,
                vocab_path=vocab_path
            )
        else:
            self.load(vocab_path)

    def create_vocabulary(self, raw_data_src, raw_data_tgt, vocab_path):
        logger.info("Creating vocabulary...")
        token_list = START_VOCAB
        token_list.extend(self.process_raw_data(raw_data_src))
        token_list.extend(self.process_raw_data(raw_data_tgt))
        with codecs.open(vocab_path, 'w', encoding='utf8') as fout:
            for w in token_list:
                fout.write("{}\n".format(w))
        for idx, tok in enumerate(token_list):
            self.id2tok[idx] = tok
            self.tok2id[tok] = idx
        logger.info('Created vocabulary of size: {}'.format(self.size()))


if __name__ == '__main__':
    pass

