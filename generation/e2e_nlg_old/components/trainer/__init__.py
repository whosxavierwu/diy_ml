# -*- coding: utf8 -*-
# Created by: wuzewei
# Created by: 2019/8/1
import time
import os
import logging
import numpy as np

import torch

from components.constants import *
from components.utils import visualize
from components.utils import serialization
from components.utils import timing
from components.evaluator.evaluator_MLP import MLPEvaluator

logger = logging.getLogger("experiment")


class BaseTrainer(object):
    def  __init__(self, config):
        self.score_file_header = ['bleu', 'nist', 'cider', 'rouge', 'meteor', 'train_loss', 'dev_loss']
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

        logger.debug("Preparing training data")
        train_batches = data.prepare_training_data(data.train, self.batch_size)
        dev_batches = data.prepare_training_data(data.dev, self.batch_size)

        id2word = data.vocab.id2tok
        dev_lex = data.lexicalizations['dev']
        dev_multi_ref_fn = "%s.multi-ref" % data.fnames['dev']

        self.set_optimizer(model, self.config['optimizer'])
        self.set_train_criterion(len(id2word), PAD_ID)
        evaluator = MLPEvaluator(self.config)

        if self.use_cuda: model = model.cuda()

        for epoch_idx in range(1, self.n_epochs + 1):
            epoch_start_time = time.time()
            pred_fn = os.path.join(self.model_dir, 'predictions.epoch%d' % epoch_idx)
            train_loss = self.train_epoch(model, train_batches)
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

    def set_optimizer(self, model, opt_name):
        opt_name = opt_name.lower()
        logger.debug("Setting %s as optimizer" % opt_name)
        if opt_name == 'sgd':
            self.optimizer = torch.optim.SGD(params=model.parameters(), lr=self.lr)
        elif opt_name == 'adam':
            self.optimizer = torch.optim.Adam(params=model.parameters(), lr=self.lr)
        elif opt_name == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(params=model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError()

    def set_train_criterion(self, *args, **kwargs):
        raise NotImplementedError()

    def train_epoch(self, model, train_batches):
        np.random.shuffle(train_batches)
        running_losses = []
        epoch_losses = []
        num_train_batches = len(train_batches)
        bar = timing.create_progress_bar('train_loss')
        for pair_idx in bar(range(num_train_batches)):
            self.optimizer.zero_grad()  # why not random?
            loss_var = self.train_step(model, train_batches[pair_idx])
            loss_data = loss_var.data.item()
            loss_var.backward()
            self.optimizer.step()
            running_losses = ([loss_data] + running_losses)[:20]
            bar.dynamic_messages['train_loss'] = np.mean(running_losses)
            epoch_losses.append(loss_data)
        epoch_loss_avg = np.mean(epoch_losses)
        return epoch_loss_avg

    def compute_val_loss(self, model, dev_batches):
        total_loss = 0
        running_losses = []
        num_dev_batches = len(dev_batches)
        bar = timing.create_progress_bar('dev_loss')
        for batch_idx in bar(range(num_dev_batches)):
            loss_var = self.train_step(model, dev_batches[batch_idx])
            loss_data = loss_var.data.item()
            running_losses= ([loss_data] + running_losses)[:20]
            bar.dynamic_messages['dev_loss'] = np.mean(running_losses)
            total_loss += loss_data
        total_loss_avg = total_loss / num_dev_batches
        return total_loss_avg

    def record_loss(self, train_loss, dev_loss):
        self.train_losses.append(train_loss)
        self.dev_losses.append(dev_loss)
        logger.info("tloss=%0.5f, dloss=%0.5f" % (train_loss, dev_loss))

    def run_external_eval(self, ref_fn, pred_fn):
        # todo measure_scores.py not found!
        pass

    def plot_lcurve(self):
        pass

    def get_scores_to_save(self):
        return list(zip(self.bleu_scores,
                          self.nist_scores,
                          self.cider_scores,
                          self.rouge_scores,
                          self.meteor_scores,
                          self.train_losses,
                          self.dev_losses))

    def plot_training_results(self):
        pass

    def train_step(self, model, param):
        pass


if __name__ == '__main__':
    pass

