# -*- coding: utf8 -*-
# Created by: wuzewei
# Created by: 2019/8/30
import pandas as pd

from components.data import BaseRecord, BaseData


class E2EData(BaseData):
    def __init__(self):
        super().__init__()

    def setup(self, configs: dict):
        train_filename = configs['train_filename']
        df = pd.read_csv(train_filename, sep=',')
        for _, r in df.iterrows():
            # todo working here
            pass

    def save(self):
        pass


if __name__ == '__main__':
    pass

