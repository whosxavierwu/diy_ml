# -*- coding: utf8 -*-
# Created by: wuzewei
# Created by: 2019/8/30
import pandas as pd

from components.data import BaseRecord, BaseData


class E2EData(BaseData):
    def __init__(self):
        super().__init__()

    def setup(self, configs: dict):
        """
        prepare everything we need after
        :param configs:
        :return:
        """
        train_filename = configs['train_filename']
        df = pd.read_csv(train_filename, sep=',')  # mr, ref
        for _, r in df.iterrows():
            record = BaseRecord()
            record.ref = r['ref']
            for kv in r['mr'].split(','):
                parts = kv.replace(']', '').split('[')
                feat_name = parts[0].strip()
                feat_value = parts[1].strip()
                record.mr[feat_name] = feat_value
            self.records.append(record)
        print('Data loaded: ', len(self.records))

    def save(self):
        pass


if __name__ == '__main__':
    D = E2EData()
    D.setup({'train_filename': '../../dataset/e2e-dataset/trainset.csv'})

