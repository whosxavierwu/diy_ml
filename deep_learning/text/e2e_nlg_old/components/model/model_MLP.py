# -*- coding: utf8 -*-
# Created by: wuzewei
# Created by: 2019/8/1
import logging
import numpy as np
import random

import torch
from torch.autograd import Variable

from components.data.common import cuda_if_gpu

from components.model import *
from components.model.modules.encoders.enc_mlp import *
from components.model.modules.decoders.dec_attn import *

logger = logging.getLogger('experiment')


class MLPModel(E2ESeq2SeqModel):
    def __init__(self, config):
        super().__init__(config)

    def fix_seed(self, seed):
        logger.debug("Fixing seed: %d" % seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    def set_encoder(self):
        self.encoder = EncoderMLP(self.config["encoder_params"])

    def set_decoder(self):
        decoder_rnn_params = self.config["decoder_params"]
        self.decoder = DecoderRNNAttnBahd(rnn_config=self.config["decoder_params"],
                                          output_size=self._tgt_vocab_size,
                                          prev_y_dim=self.embedding_dim,
                                          enc_dim=self.encoder.hidden_size,
                                          enc_num_directions=1)

    def forward(self, datum):
        """
        Run the model on one data instance
        :param datum:
        :return:
        """
        # size is SLxB, TLxB
        batch_x_var, batch_y_var = datum
        # Embedding lookup
        encoder_input_embedded = self.embedding_lookup(batch_x_var)  # SLxBxE
        encoder_input_embedded = self.embedding_dropout_layer(encoder_input_embedded)
        # Encode embedded input, shape: SLxBxH, 1xBxH
        encoder_outputs, encoder_hidden = self.encoder(encoder_input_embedded)
        # decoding
        logits = self.decode(batch_y_var, encoder_hidden, encoder_outputs)
        return logits

    def decode(self, dec_input_var, encoder_hidden, encoder_outputs):
        """
        decoding using one of two policies
        1. teacher forcing: use gold standard label as output of prev step
        2. dynamic decoding: use prev prediction as output of the prev step
         See: http://minds.jacobs-university.de/sites/default/files/uploads/papers/ESNTutorialRev.pdf
        and the official PyTorch tutorial: http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
        :param dec_input_var:
        :param encoder_hidden:
        :param encoder_outputs:
        :return:
        """
        dec_len = dec_input_var.size()[0]
        batch_size = dec_input_var.size()[1]
        dec_hidden = encoder_hidden
        dec_input = cuda_if_gpu(Variable(torch.LongTensor([BOS_ID] * batch_size)))
        predicted_logits = cuda_if_gpu(Variable(torch.zeros(dec_len, batch_size, self._tgt_vocab_size)))

        use_teacher_force = random.random() < self.teacher_forcing_ratio
        for di in range(dec_len):
            prev_y = self.embedding_mat(dec_input)
            # shape: BxTV, 1xBxdec_dim, BxSL
            dec_output, dec_hidden, attn_weights = self.decoder(prev_y, dec_hidden, encoder_outputs)
            predicted_logits[di] = dec_output
            if use_teacher_force:
                # Decoding policy 1: feeding the ground truth label as a target
                dec_input = dec_input_var[di]
            else:
                # Decoding policy 2: feeding the previous prediction as a target
                topval, topidx = dec_output.data.topk(1)  # ???
                dec_input = cuda_if_gpu(Variable(torch.LongTensor(topidx.squeeze().cpu().numpy())))
        return predicted_logits

    def predict(self, input_var):
        """
        todo how does it work?
        :param input_var:
        :return:
        """
        encoder_input_embedded = self.embedding_mat(input_var)
        encoder_outputs, encoder_hidden = self.encoder(encoder_input_embedded)

        dec_ids, attn_w = [], []
        curr_token_id = BOS_ID
        curr_dec_idx = 0
        dec_input_var = cuda_if_gpu(Variable(torch.LongTensor([curr_token_id])))
        dec_hidden = encoder_hidden[:1]

        while (curr_token_id != EOS_ID and curr_dec_idx <= self._max_tgt_len):
            prev_y = self.embedding_mat(dec_input_var)
            decoder_output, dec_hidden, decoder_attention = self.decoder(prev_y, dec_hidden, encoder_outputs)
            attn_w.append(decoder_attention.data)
            topval, topidx = decoder_output.data.topk(1)
            curr_token_id = topidx[0][0]
            dec_ids.append(curr_token_id)
            dec_input_var = cuda_if_gpu(Variable(torch.LongTensor([curr_token_id])))
            curr_dec_idx += 1
        return dec_ids, attn_w


component = MLPModel

if __name__ == '__main__':
    pass

