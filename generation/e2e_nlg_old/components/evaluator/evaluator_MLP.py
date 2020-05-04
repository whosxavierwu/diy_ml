# -*- coding: utf8 -*-
# Created by: wuzewei
# Created by: 2019/8/8
import logging

from components.data import common
from components.constants import *

logger = logging.getLogger("experiment")


class MLPEvaluator:
    def __init__(self, config):
        self.config = config

    def evaluate_model(self, model, dev_data):
        decoded_ids = []
        decoded_attn_weights = []

        curr_x_ids = dev_data[0]
        out_ids, attn_weights = self.predict_one(model, curr_x_ids)
        decoded_ids.append(out_ids)
        decoded_attn_weights.append(attn_weights)

        for snt_ids in dev_data[1:]:
            if snt_ids != curr_x_ids:
                out_ids, attn_weights = self.predict_one(model, snt_ids)
                decoded_ids.append(out_ids)
                decoded_attn_weights.append(attn_weights)
                curr_x_ids = snt_ids
        return decoded_ids, decoded_attn_weights

    def lexicalize_predictions(self, all_tokids, data_lexs, id2word):
        all_tokens = []
        for idx, snt_ids in enumerate(all_tokids):
            this_snt_toks = []
            this_snt_lex = data_lexs[idx]
            for t in snt_ids[:-1]:
                tok = id2word[t.item()]
                if tok == NAME_TOKEN:
                    if this_snt_lex[0] is not None:
                        this_snt_toks.append(this_snt_lex[0])
                elif tok == NEAR_TOKEN:
                    if this_snt_lex[1] is not None:
                        this_snt_toks.append(this_snt_lex[1])
                else:
                    this_snt_toks.append(tok)
            all_tokens.append(this_snt_toks)
        return all_tokens

    def predict_one(self, model, src_snt_ids):
        input_var = common.ids2var(src_snt_ids, -1, 1, addEOS=True)
        output_ids, attn_weights = model.predict(input_var)
        return output_ids, attn_weights


component = MLPEvaluator

if __name__ == '__main__':
    pass

