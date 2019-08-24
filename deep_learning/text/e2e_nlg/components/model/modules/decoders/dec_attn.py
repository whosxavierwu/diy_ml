# -*- coding:utf-8 -*-
# Created by: wuzewei
# Created on: 2019/8/24 0024

import torch
import torch.nn as nn
from components.model.modules.decoders import DecoderRNNAttnBase
from components.model.modules.attention.attn_bahd import AttnBahd


class DecoderRNNAttnBahd(DecoderRNNAttnBase):
    def __init__(self, rnn_config, output_size, prev_y_dim, enc_dim, enc_num_directions):
        super().__init__()

        self.rnn = nn.GRU(
            input_size=rnn_config['input_size'],
            hidden_size=rnn_config['hidden_size'],
            dropout=rnn_config['dropout'],
            bidirectional=rnn_config.get('bidirectional', False)
        )

        dec_dim = rnn_config['hidden_size']
        self.attn_module =AttnBahd(enc_dim, dec_dim, enc_num_directions)
        # ?
        self.W_combine = nn.Linear(prev_y_dim + enc_dim * enc_num_directions, dec_dim)
        self.W_out = nn.Linear(dec_dim, output_size)
        self.log_softmax = nn.LogSoftmax()

    def combine_context_run_rnn_step(self, prev_y_batch, prev_h_batch, context):
        """
        todo how does it work?
        :param prev_y_batch:
        :param prev_h_batch:
        :param context:
        :return:
        """
        y_ctx = torch.cat((prev_y_batch, context.squeeze(1)), 1)
        rnn_input = self.W_combine(y_ctx)
        output, decoder_hidden = self.rnn(rnn_input.unsqueeze(0), prev_h_batch)
        return output, decoder_hidden

    def compute_output(self, rnn_output):
        unnormalized_logits = self.W_out(rnn_output[0])  # BxTV
        logits = self.log_softmax(unnormalized_logits)  # BxTV
        return logits

    def forward(self, prev_y_batch, prev_h_batch, encoder_outputs_batch):
        attn_weights = self.attn_module(prev_h_batch, encoder_outputs_batch)
        # bmm?
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs_batch.transpose(0, 1))
        dec_rnn_output, dec_hidden = self.combine_context_run_rnn_step(
            prev_y_batch,
            prev_h_batch,
            context
        )
        dec_output = self.compute_output(dec_rnn_output)
        return dec_output, dec_hidden, attn_weights

