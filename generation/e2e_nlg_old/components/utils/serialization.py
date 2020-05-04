# -*- coding: utf8 -*-
# Created by: wuzewei
# Created by: 2019/8/8
import os
import codecs
import csv
from datetime import datetime as dt
import logging

import torch

logger = logging.getLogger('experiment')

def make_model_dir(config_dict):
    """
    create a directory
    :param config_dict:
    :return:
    """
    mode = config_dict['mode']
    if mode == 'train':
        all_experiments_dir = os.path.abspath(config_dict['experiments_dir'])
        # the name of model module
        model_type = config_dict['model-module'].split('.')[-1]
        config_dict['trainer_params']['model_type'] = model_type
        # hyper parameters
        seed = config_dict.get('random_seed', 1)
        embedding_dim = config_dict['model_params']['embedding_dim']
        hidden_size = config_dict['model_params']['encoder_params']['hidden_size']
        dropout = config_dict['model_params']['encoder_params']['dropout']
        batch_size = config_dict['trainer_params']['batch_size']
        lr = config_dict['trainer_params']['learning_rate']
        hp_name = 'seed{}-emb{}-hid{}-drop{}-bs{}-lr{}'.format(
            seed, embedding_dim, hidden_size, dropout, batch_size, lr
        )
        # timestamp
        timestamp = dt.now().strftime("%Y%m%d%H%M%S")
        # model name
        model_name = '_'.join([model_type, hp_name, timestamp])
        model_dir = os.path.join(all_experiments_dir, model_name)
    elif mode == 'test':
        model_filename = config_dict['model_filename']
        model_dir = os.path.split(model_filename)[0]
    else:
        raise NotImplementedError()
    config_dict['trainer_params']['model_dir'] = model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return model_dir


def save_predictions_txt(predictions, predictions_file):
    logger.info('Saving predictions to a txt file: %s' % predictions_file)
    with codecs.open(predictions_file, 'w', encoding='utf8') as fout:
        if isinstance(predictions, str):
            fout.write(predictions)
        elif isinstance(predictions, list):
            fout.write('%s\n'%(' '.join(s) if isinstance(s, list) else s) for s in predictions)
        else:
            raise NotImplementedError()


def save_model(model, model_fn):
    logger.info("Saving model to %s" % model_fn)
    torch.save(model.state_dict(), open(model_fn, 'wb'))


def save_scores(scores, header, fname):
    with open(fname, 'w') as csv_out:
        csv_writer = csv.writer(csv_out, delimiter=',')
        csv_writer.writerow(header)
        for ep_socres in scores:
            csv_writer.writerow(ep_socres)
    logger.info("Scores saved to %s" % fname)


if __name__ == '__main__':
    pass


