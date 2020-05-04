# -*- coding: utf8 -*-
# Created by: wuzewei
# Created by: 2019/8/29
from components.data import BaseData
from components.model import BaseModel
from components.evaluator import BaseEvaluator

class BaseTrainer(object):
    def __init__(self):
        pass

    def start_training(self, data: BaseData, model: BaseModel, evaluator: BaseEvaluator):
        raise NotImplementedError()


if __name__ == '__main__':
    pass

