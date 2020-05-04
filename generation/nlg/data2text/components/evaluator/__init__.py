# -*- coding: utf8 -*-
# Created by: wuzewei
# Created by: 2019/8/27

class BaseEvaluator(object):
    def __init__(self):
        pass

    def setup(self, configs: dict):
        raise NotImplementedError()


if __name__ == '__main__':
    pass

