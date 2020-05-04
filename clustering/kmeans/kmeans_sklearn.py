#!/usr/bin/env python
# -*- coding:utf8 -*-
# Created at: 2020/5/4
# Created by: whosxavierwu@gmail.com
import os
import sys
import codecs
import itertools
from datetime import datetime
from scipy import sparse
from sklearn import metrics
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from collections import namedtuple

MAP2REDUCE = namedtuple("MAP2REDUCE", ["token1", "token2", "count"])


def count_tokens(infile):
    token2count = {}
    print('calculating token2count...')
    print(datetime.now())
    with codecs.open(infile, encoding='utf8') as fin:
        for line in fin:
            tokens = [t for t in line.strip().split(' ') if t != '']
            tokens_set = set(tokens)
            for token in tokens_set:
                if token not in token2count:
                    token2count[token] = [0, 0] # count, doc_count
                token2count[token][1] += 1
            for token in tokens:
                token2count[token][0] += 1
    df_count = pd.DataFrame([
        [token, v[0], v[1]]
        for token, v in token2count.items()
    ], columns=['token', 'count', 'doc_count'])
    df_count = df_count.sort_values(['count'], ascending=False)
    df_count['token_id'] = range(len(df_count))
    df_count.to_csv('tokenCount.tsv', encoding='utf8', sep='\t', index=False)


class TokenPairCount(object):
    def __init__(self, conf):
        self.token2id = {}
        token_count_filename = "autocar_statistics/common/dict/tokenCount.tsv"
        with codecs.open(token_count_filename, encoding='utf8') as fin:
            line = fin.readline()  # first line in header
            for line in fin:
                parts = line.strip().split("\t")
                token = parts[0]
                token_id = int(parts[-1])
                self.token2id[token] = token_id

    def map(self):
        def dumps_output(tuple1):
            sep = "\t"
            return sep.join([str(v) for v in tuple1])

        for line in sys.stdin:
            token_ids = [self.token2id[tok] for tok in set(line.strip().split(" ")) if tok in self.token2id]
            for token_id_pair in itertools.permutations(token_ids, 2):
                tuple1 = MAP2REDUCE(token1=token_id_pair[0], token2=token_id_pair[1], count=1)
                print(dumps_output(tuple1))

    def reduce(self):
        token_pair_dict = {}
        for line in sys.stdin:
            parts = line.strip().split("\t")
            token_id_1 = int(parts[0])
            token_id_2 = int(parts[1])
            if token_id_1 not in token_pair_dict:
                token_pair_dict[token_id_1] = {}
            if token_id_2 not in token_pair_dict[token_id_1]:
                token_pair_dict[token_id_1][token_id_2] = 0
            token_pair_dict[token_id_1][token_id_2] += 1
        for token_id_1, vec_dict in token_pair_dict.items():
            vec_str = " ".join([
                "{}:{}".format(token_id_2, cnt)
                for token_id_2, cnt in vec_dict.items()
            ])
            print("{}\t{}".format(token_id_1, vec_str))


if __name__ == '__main__':
    df_tokenCount = pd.read_csv('tokenCount.tsv', encoding='utf8', sep='\t')
    N = len(df_tokenCount)
    print('N=%d' % N)

    npz_filename = 'X_csr.npz'
    if npz_filename in os.listdir('.'):
        X = sparse.load_npz(npz_filename)
        print('whole sparse matrix loaded from %s' % npz_filename)
    else:
        print('generating sparse matrix...')
        df = pd.read_csv("car_article_baijiahao_token_count.tmp", encoding='utf8', sep='\t', names=['token_id', 'vec'])
        X = sparse.dok_matrix((N, N), dtype=np.float32)
        for _, r in df.iterrows():
            t1 = int(r['token_id'])
            tok_cnt_list = r['vec'].split(' ')
            m = len(tok_cnt_list)
            for tok_cnt in tok_cnt_list:
                t2 = int(tok_cnt.split(':')[0])
                cnt = float(tok_cnt.split(':')[1])
                X[t1, t2] = cnt / m
        X = X.tocsr()
        sparse.save_npz(npz_filename, X)
        print('whole sparse matrix saved in %s' % npz_filename)

    n = 60000
    print('n=%d' % n)
    X = X[:n, :n]
    print('clustering...')
    for k in [10, 50, 100, 300, 500, 700, 1000]:
        print('#'*30)
        print('k=%d'%k)
        t0 = datetime.now()
        model = KMeans(k, n_jobs=-1).fit(X)
        print('done', datetime.now() - t0)
        df_pred = df_tokenCount[['token_id', 'token']].sort_values(['token_id'], ascending=True).head(n)
        df_pred['label'] = model.labels_
        out_filename = 'result_n%d_k%d.tsv' % (n, k)
        df_pred.to_csv(out_filename, encoding='utf8', sep='\t', index=False)
        print("result saved in %s" % out_filename)
        print("silhouette_score: {}".format(metrics.silhouette_score(X, df_pred['label'])))

