# -*- coding: utf8 -*-
# Created by: wuzewei
# Created by: 2019/8/27
from typing import Dict, Any, List


class BaseRecord(object):
    _ref: str
    _mr: Dict[str, str]

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
    records: List[BaseRecord]

    def __init__(self):
        self.records = []

    def read_from_file(self, filename: str):
        raise NotImplementedError()


if __name__ == '__main__':
    pass

