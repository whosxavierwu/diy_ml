# -*- coding: utf8 -*-
# Created by: wuzewei
# Created by: 2019/8/8
import logging
import numpy as np
import copy

from components.data import BaseData
from components.data.vocabulary import *
from components.data.common import *

logger = logging.getLogger("experiment")


class MLPData(BaseData):
    def process_mr(self, mr):
        items = mr.split(", ")
        mr_data = [PAD_ID] * MR_KEY_NUM
        lex = [None, None]
        for idx, item in enumerate(items):
            key, raw_val = item.replace("]", "").split("[")
            key_idx = MR_KEYMAP[key]
            if key == 'name':
                mr_val = NAME_TOKEN
                lex[0] = raw_val
            elif key == 'near':
                mr_val = NEAR_TOKEN
                lex[1] = raw_val
            else:
                mr_val = raw_val
            mr_data[key_idx] = mr_val
        return mr_data, lex

    def data_to_token_ids_train(self, train_x_raw, train_y_raw):
        data_split_x = data_split_y = []
        skipped_cnt = 0
        for idx, x in enumerate(train_x_raw):
            src_ids = [self.vocab.get_word(tok) for tok in x]
            src_len = len(src_ids)
            y = train_y_raw[idx]
            tgt_ids = [self.vocab.get_word(tok) for tok in y]
            tgt_len = len(tgt_ids)

            if src_len > self.max_src_len or tgt_len >= self.max_tgt_len:
                logger.info("Skipped long snt: %d (src) / %d (tgt)" % (src_len, tgt_len))
                skipped_cnt += 1
                continue
            data_split_x.append(src_ids)
            data_split_y.append(tgt_ids)
        logger.debug("Skipped %d long sentences" % skipped_cnt)
        return (data_split_x, data_split_y)

    def data_to_token_ids_test(self, test_x_raw):
        data_split_x = []
        for idx, x in enumerate(test_x_raw):
            src_ids = [self.vocab.get_word(tok) for tok in x]
            src_len = len(src_ids)
            if src_len > self.max_src_len:
                logger.debug("Skipped long snt: %d" % idx)
                continue
            data_split_x.append(src_ids)
        return (data_split_x, None)

    def index_data(self, data_size, mode="no_shuffling"):
        if mode == "random":
            indices = np.random.choice(np.arange(data_size), data_size, replace=False)
        elif mode == "no_shuffling":
            indices = np.arange(data_size)
        else:
            raise NotImplementedError()
        return indices

    def prepare_training_data(self, xy_ids, batch_size):
        sorted_data = sorted(zip(*xy_ids), key=lambda p: len(p[0]), reverse=True)
        data_size = len(sorted_data)
        num_batches = data_size
        data_indices = self.index_data(data_size, mode="no_shuffling")
        batch_pairs = []
        for bi in range(num_batches):
            batch_x = batch_y = []
            curr_batch_indices = data_indices[bi * batch_size: (bi + 1) * batch_size]
            for idx in curr_batch_indices:
                x_ids, y_ids = sorted_data[idx]
                x_ids_copy = copy.deepcopy(x_ids)
                x_ids_copy.append(EOS_ID)
                batch_x.append(x_ids_copy)
                y_ids_copy = copy.deepcopy(y_ids)
                y_ids_copy.append(EOS_ID)
                batch_y.append(y_ids_copy)
            batch_x_lens = [len(s) for s in batch_x]
            batch_y_lens = [len(s) for s in batch_y]
            max_src_len = max(batch_x_lens)
            max_tgt_len = max(batch_y_lens)
            batch_x_padded = [pad_snt(x, max_src_len) for x in batch_x]
            batch_y_padded = [pad_snt(y, max_tgt_len) for y in batch_y]
            batch_pairs.append((batch_x_padded, batch_y_padded))
        return batch_pairs

    def setup_vocab(self, vocab_path, train_x_raw, train_y_raw):
        self.vocab = VocabularyShared(vocab_path, train_x_raw, train_y_raw)


component = MLPData

if __name__ == '__main__':
    pass

