# -*- coding: utf8 -*-
# Created by: wuzewei
# Created by: 2019/8/1
import logging
import os
import numpy as np

import torch
import torch.nn as nn

from components.trainer import BaseTrainer
from components.data.common import *
from components.utils import visualize

logger = logging.getLogger("experiment")


class MLPTrainer(BaseTrainer):
    def set_train_criterion(self, vocab_size, pad_id):
        weight = torch.ones(vocab_size)
        weight[pad_id] = 0
        # self.criterion = nn.NLLLoss(weight, size_average=True)
        self.criterion = nn.NLLLoss(weight, size_average=True)

        if self.use_cuda:
            self.criterion = self.criterion.cuda()

    def train_step(self, model, datum):
        datum = [cuda_if_gpu(torch.autograd.Variable(torch.LongTensor(t)).transpose(0, 1))
                 for t in datum]  # [SL x B, TL x B]

        logits = model.forward(datum)  # TL x B x TV
        loss_var = self.calc_loss(logits, datum)  # have to compute log_logits, since using NLL loss
        return loss_var

    def calc_loss(self, logits, datum):
        batch_y_var = datum[1]
        vocab_size = logits.size()[-1]
        logits = logits.contiguous().view(-1, vocab_size)
        targets = batch_y_var.contiguous().view(-1, 1).squeeze(1)
        loss = self.criterion(logits, targets)
        return loss

    def plot_lcurve(self):
        fig_fname = os.path.join(self.model_dir, "lcurve.pdf")
        title = self.config['modeltype']
        visualize.plot_lcurve(self.train_losses, self.dev_losses, img_title=title, save_path=fig_fname, show=False)

    def plot_training_results(self):
        losses = np.asarray([self.train_losses, self.dev_losses]).transpose()
        visualize.plot_train_progress(scores=(losses,
                                    self.bleu_scores,
                                    self.nist_scores,
                                    self.cider_scores,
                                    self.rouge_scores,
                                    self.meteor_scores),
                            names=self.get_plot_names(),
                            img_title=self.config['modeltype'],
                            save_path=os.path.join(self.model_dir, "lcurve_scores.pdf"),
                            show=False)

    def get_plot_names(self):
        return [['TrainLoss', 'DevLoss'], 'BLEU', 'NIST', 'CIDEr', 'ROUGE_L', 'METEOR']


component = MLPTrainer

if __name__ == '__main__':
    pass

