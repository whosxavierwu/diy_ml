# -*- coding:utf-8 -*-
# Created by: wuzewei
# Created on: 2019/8/24 0024

import torch
import torch.nn as nn


class AttnBahd(nn.Module):
    def __init__(self, enc_dim, dec_dim, num_directions, attn_dim=None):
        super().__init__()
        self.num_directions = num_directions
        self.h_dim = enc_dim
        self.s_dim = dec_dim
        self.a_dim = self.s_dim if attn_dim is None else attn_dim

        self.U = nn.Linear(self.h_dim * self.num_directions, self.a_dim)
        self.W = nn.Linear(self.s_dim, self.a_dim)
        self.v = nn.Linear(self.a_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, prev_h_batch, enc_outputs):
        src_seq_len, batch_size, enc_dim = enc_outputs.size()
        uh = self.U(enc_outputs.view(-1, self.h_dim)).view(src_seq_len, batch_size, self.a_dim)
        wq = self.W(prev_h_batch.view(-1, self.s_dim)).unsqueeze(0)
        wq3d = wq.expand_as(uh)
        wquh = self.tanh(wq3d + uh)
        attn_unnorm_scores = self.v(wquh.view(-1, self.a_dim)).view(batch_size, src_seq_len)
        attn_weights = self.softmax(attn_unnorm_scores)
        return attn_weights

