# -*- coding: utf8 -*-
# Created by: wuzewei
# Created by: 2019/8/27


class BaseRecord(object):
    def __init__(self):
        self._mr = {}
        self._ref = ""

    @property
    def mr(self): return self._mr

    @property
    def ref(self): return self._ref

    @mr.setter
    def mr(self, value): self._mr = value

    @ref.setter
    def ref(self, value): self._ref = value


class BaseData(object):
    def __init__(self):
        # it should be list of BaseRecords
        self.train = None
        self.dev = None
        self.test = None

    def read_data(self):
        raise NotImplementedError()

    def get_train_set(self):
        raise NotImplementedError()

    def get_test_set(self):
        raise NotImplementedError()

    def split(self):
        raise NotImplementedError()


if __name__ == '__main__':
    pass

