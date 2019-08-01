# -*- coding: utf8 -*-
# Created by: wuzewei
# Created by: 2019/8/1
# Use this file to run train or test process
import sys
import yaml
import importlib

def main(config_dict):
    # read modules
    data_module = config_dict['data-module']
    model_module = config_dict['model-module']
    trainer_module = config_dict['trainer-module']
    mode = config_dict['mode']

    DataClass = importlib.import_module(data_module).component
    ModelClass = importlib.import_module(model_module).component
    TrainerClass = importlib.import_module(trainer_module).component

    data = DataClass(config_dict['data_params'])
    data.setup()

    model = ModelClass(config_dict['model_params'])
    model.setup(data)

    if mode == 'train':
        # build model
        # train model
        pass
    elif mode == 'dev':
        # read model
        # predict on dev set
        # get statistic
        pass
    elif mode == 'test':
        # read model
        # predict on test set
        pass


if __name__ == '__main__':
    assert len(sys.argv) == 2, 'Specify extractly ONE argument -- config file'
    config_filename = sys.argv[1]
    main(yaml.load(open(config_filename), Loader=yaml.FullLoader))

