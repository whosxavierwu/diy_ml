# -*- coding: utf8 -*-
# Created by: wuzewei
# Created by: 2019/8/27
import sys
import importlib
import yaml
import codecs


def run_experiments(configs):
    """
    Given a config dict, start running experiments
    :type configs: dict
    """
    mode = configs['mode']

    # here we suppose every module file should have a 'component' pointing to the class
    DataClass = importlib.import_module(configs['data-module']).component
    ModelClass = importlib.import_module(configs['model-module']).component
    TrainerClass = importlib.import_module(configs['trainer-module']).component
    EvaluatorClass = importlib.import_module(configs['evaluator-module']).component



if __name__ == '__main__':
    assert len(sys.argv == 2), 'Specify exactly ONE argument -- config file'
    conf_fn = sys.argv[1]
    run_experiments(yaml.load(codecs.open(conf_fn, encoding='utf8'), Loader=yaml.FullLoader))

