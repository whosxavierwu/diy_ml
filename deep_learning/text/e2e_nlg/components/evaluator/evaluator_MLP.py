# -*- coding: utf8 -*-
# Created by: wuzewei
# Created by: 2019/8/8
import logging

logger = logging.getLogger("experiment")


class MLPEvaluator:
    def __init__(self, config):
        self.config = config

    def evaluate_model(self, model, o):
        pass

    def lexicalize_predictions(self, ids, lex, word):
        pass


component = MLPEvaluator

if __name__ == '__main__':
    pass

