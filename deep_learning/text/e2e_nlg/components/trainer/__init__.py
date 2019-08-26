# -*- coding: utf8 -*-
# Created by: wuzewei
# Created by: 2019/8/1
import time
import os
import logging

import torch

from components.constants import *
from components.utils import visualize
from components.utils import serialization
from components.evaluator.evaluator_MLP import MLPEvaluator

logger = logging.getLogger("experiment")


class BaseTrainer(object):
    def  __init__(self, config):
        self.score_file_header = None
        self.config = config
        # init params
        self.n_epochs = self.config['n_epochs']
        self.batch_size = self.config['batch_size']
        self.lr = self.config['learning_rate']
        self.model_dir = self.config['model_dir']
        self.evaluate_prediction = self.config['evaluate_prediction']
        self.save_model = self.config['save_model_each_epoch']
        self.use_cuda = torch.cuda.is_available()
        self.train_losses = []
        self.dev_losses = []
        if self.evaluate_prediction:
            self.nist_scores = []
            self.bleu_scores = []
            self.cider_scores = []
            self.rouge_scores = []
            self.meteor_scores = []

    def start_training(self, model, data):
        start_time = time.time()
        logger.info("Start training...")
        logger.debug(visualize.torch_summarize(model))

        evaluator = MLPEvaluator(self.config)

        logger.debug("Preparing training data")

        train_batches = data.prepare_training_data(data.train, self.batch_size)
        dev_batches = data.prepare_training_data(data.dev, self.batch_size)

        id2word = data.vocab.id2tok
        dev_lex = data.lexicalizations['dev']
        dev_multi_ref_fn = "%s.multi-ref" % data.fnames['dev']

        self.set_optimizer(model, self.config['optimizer'])
        self.set_train_criterion(len(id2word), PAD_ID)

        if self.use_cuda:
            model = model.cuda()

        for epoch_idx in range(1, self.n_epochs + 1):
            epoch_start_time = time.time()
            pred_fn = os.path.join(self.model_dir, 'predictions.epoch%d' % epoch_idx)
            train_loss = self.train_epoch(epoch_idx, model, train_batches)
            dev_loss = self.compute_val_loss(model, dev_batches)
            predicted_ids, attention_weights = evaluator.evaluate_model(model, data.dev[0])
            predicted_tokens = evaluator.lexicalize_predictions(predicted_ids, dev_lex, id2word)
            serialization.save_predictions_txt(predicted_tokens, pred_fn)
            self.record_loss(train_loss, dev_loss)
            if self.evaluate_prediction:
                self.run_external_eval(dev_multi_ref_fn, pred_fn)
            if self.save_model:
                serialization.save_model(model, os.path.join(self.model_dir, 'weights.epoch%d' % epoch_idx))
            logger.info("Epoch %d/%d: time=%s" % (epoch_idx, self.n_epochs, time.time()-epoch_start_time))
        self.plot_lcurve()
        if self.evaluate_prediction:
            score_fname = os.path.join(self.model_dir, 'scores.csv')
            scores = self.get_scores_to_save()
            serialization.save_scores(scores, self.score_file_header, score_fname)
            self.plot_training_results()
        logger.info("End training time=%s" % (time.time()-start_time))

    def set_optimizer(self, model, param):
        pass

    def set_train_criterion(self, param, PAD_ID):
        pass

    def train_epoch(self, epoch_idx, model, train_batches):
        pass

    def compute_val_loss(self, model, dev_batches):
        pass

    def record_loss(self, train_loss, dev_loss):
        pass

    def run_external_eval(self, dev_multi_ref_fn, pred_fn):
        pass

    def plot_lcurve(self):
        pass

    def get_scores_to_save(self):
        pass

    def plot_training_results(self):
        pass


if __name__ == '__main__':
    pass

