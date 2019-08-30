# -*- coding: utf8 -*-
# Created by: wuzewei
# Created by: 2019/8/27
from components.data import BaseData

class BaseModel(object):
    def __init__(self):
        pass

    def setup(self, configs: dict):
        raise NotImplementedError()

    def predict(self, data: BaseData):
        raise NotImplementedError()


if __name__ == '__main__':
    pass

