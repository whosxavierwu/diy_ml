# -*- coding: utf8 -*-
# Created by: wuzewei
# Created by: 2019/8/27
from typing import Dict, Any, List


class BaseRecord(object):
    _ref: str
    _mr: Dict[Any, Any]

    def __init__(self, mr: dict = None, ref: str = ""):
        self._mr = {} if mr is None else mr
        self._ref = ref

    @property
    def mr(self): return self._mr

    @property
    def ref(self): return self._ref

    @mr.setter
    def mr(self, value): self._mr = value

    @ref.setter
    def ref(self, value): self._ref = value


class BaseData(object):
    records: List[BaseRecord]

    def __init__(self):
        self.records = []

    def setup(self, configs: dict):
        raise NotImplementedError()

    def save(self):
        raise NotImplementedError()


if __name__ == '__main__':
    pass

