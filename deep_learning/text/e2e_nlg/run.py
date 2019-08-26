# -*- coding: utf8 -*-
# Created by: wuzewei
# Created by: 2019/8/1
# Use this file to run train or test process
import sys
import os
from datetime import datetime as dt
import yaml
import importlib
import codecs
import json

import torch

from components.utils import serialization
from components.utils import log

def main(config_dict):
    # specify modules from config dict
    data_module = config_dict['data-module']
    model_module = config_dict['model-module']
    trainer_module = config_dict['trainer-module']
    evaluator_module = config_dict['evaluator-module']
    mode = config_dict['mode']

    # import modules
    DataClass = importlib.import_module(data_module).component
    ModelClass = importlib.import_module(model_module).component
    TrainerClass = importlib.import_module(trainer_module).component
    EvaluatorClass = importlib.import_module(evaluator_module).component

    # set a logger
    model_dir = serialization.make_model_dir(config_dict)
    log_filename = os.path.join(model_dir, "{}.log".format(dt.today().strftime("%Y%m%d")))
    logger = log.set_logger(config_dict['log_level'], log_filename)

    # setup data
    data = DataClass(config_dict['data_params'])
    data.setup()

    # setup model
    model = ModelClass(config_dict['model_params'])
    model.fix_seed(config_dict['random_seed'])
    model.setup(data)

    if mode == 'train':
        trainer_params = config_dict['trainer_params']
        trainer = TrainerClass(trainer_params)
        trainer.start_training(model, data)
        config_json_filename = os.path.join(model_dir, 'config_{}.json'.format(dt.today().strftime("%Y%m%d")))
        json.dump(config_dict, codecs.open(config_json_filename, 'w', encoding='utf8'))
    elif mode == 'test':
        evaluator = EvaluatorClass(config_dict)
        # load model
        model_filename = config_dict['model_filename']
        model.load_state_dict(torch.load(open(model_filename, 'rb')))
        id2word = data.vocab.id2tok
        if 'dev' in data.filenames:
            logger.info("Predicting on dev data")
            predicted_ids, attention_weights = evaluator.evaluate_model(model, data.dev[0])
            data_lexicalizations = data.lexicalizations['dev']
            predictions_file = "%s.devset.predictions.txt" % model_filename
            predicted_sentences = evaluator.lexicalize_predictions(predicted_ids, dev_lex, id2word)
            serialization.save_predictions_txt(predicted_sentences, predictions_file)
        if 'test' in data.filenames:
            logger.info("Predicting on test data")
            predicted_ids, attention_weights = evaluator.evaluate_model(model, data.dev[0])
            data_lexicalizations = data.lexicalizations['test']
            predictions_file = "%s.testset.predictions.txt" % model_filename
            predicted_sentences = evaluator.lexicalize_predictions(predicted_ids, dev_lex, id2word)
            serialization.save_predictions_txt(predicted_sentences, predictions_file)
    else:
        logger.error("Check the 'mode' field in the config file!")
    logger.info('DONE')


if __name__ == '__main__':
    assert len(sys.argv) == 2, 'Specify extractly ONE argument -- config file'
    config_filename = sys.argv[1]
    main(yaml.load(open(config_filename), Loader=yaml.FullLoader))

