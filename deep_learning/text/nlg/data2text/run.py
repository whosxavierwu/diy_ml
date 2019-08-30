# -*- coding: utf8 -*-
# Created by: wuzewei
# Created by: 2019/8/27
import sys
import importlib
import yaml
import codecs

from components.utils import log


def run_experiments(configs):
    """
    Given a config dict, start running experiments
    :type configs: dict
    """
    logger = log.set_logger(configs['log_level'], configs['log_filename'])

    # here we suppose every module file should have a 'component' pointing to the class
    DataClass = importlib.import_module(configs['data-module']).component
    ModelClass = importlib.import_module(configs['model-module']).component
    data = DataClass()
    model = ModelClass()

    data.setup(configs['data_params'])
    model.setup(configs['model_params'])

    mode = configs['mode']
    if mode == 'train':
        TrainerClass = importlib.import_module(configs['trainer-module']).component
        EvaluatorClass = importlib.import_module(configs['evaluator-module']).component
        trainer = TrainerClass()
        evaluator = EvaluatorClass()
        trainer.setup(configs['train_params'])
        evaluator.setup(configs['evaluator_params'])
        trainer.start_training(data, model, evaluator)
    elif mode == 'test':
        prediction = model.predict(data)
        prediction.save()
    else:
        logger.error("Check the 'mode' field in the config file!")
    logger.info('DONE')


if __name__ == '__main__':
    assert len(sys.argv == 2), 'Specify exactly ONE argument -- config file'
    conf_fn = sys.argv[1]
    run_experiments(yaml.load(codecs.open(conf_fn, encoding='utf8'), Loader=yaml.FullLoader))

