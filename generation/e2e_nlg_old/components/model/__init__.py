# -*- coding: utf8 -*-
# Created by: wuzewei
# Created by: 2019/8/1
import torch
from torch import nn as nn

from components.data.common import *


class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config
        self.use_cuda = torch.cuda.is_available()


class Seq2SeqModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self._src_vocab_size = None
        self._tgt_vocab_size = None
        self._max_src_len = None
        self._max_tgt_len = None

    def set_src_vocab_size(self, vocab_size):
        self._src_vocab_size = vocab_size

    def set_tgt_vocab_size(self, vocab_size):
        self._tgt_vocab_size = vocab_size

    def set_max_src_len(self, length):
        self._max_src_len = length

    def set_max_tgt_len(self, length):
        self._max_tgt_len = length


class E2ESeq2SeqModel(Seq2SeqModel):
    def set_flags(self):
        self.teacher_forcing_ratio = self.config.get("teacher_forcing_ratio", 1.0)

    def set_data_dependent_param(self, data):
        self.set_src_vocab_size(len(data.vocab))
        self.set_tgt_vocab_size(len(data.vocab))
        self.set_max_src_len(data.max_src_len)
        self.set_max_tgt_len(data.max_tgt_len)

    def set_embeddings(self):
        self.embedding_dim = self.config["embedding_dim"]
        self.embedding_mat =nn.Embedding(self._src_vocab_size, self.embedding_dim, padding_idx=PAD_ID)
        embedding_drop_prob = self.config.get("embedding_dropout", 0.0)
        self.embedding_dropout_layer = nn.Dropout(embedding_drop_prob)

    def set_encoder(self):
        raise NotImplementedError()

    def set_decoder(self):
        raise NotImplementedError()

    def setup(self, data):
        self.set_flags()
        self.set_data_dependent_param(data)
        self.set_embeddings()
        self.set_encoder()
        self.set_decoder()

    def embedding_lookup(self, ids):
        return self.embedding_mat(ids)


if __name__ == '__main__':
    pass

