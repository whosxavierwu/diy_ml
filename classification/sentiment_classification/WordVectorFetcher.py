# -*- coding:utf-8 -*-
# Created by: wuzewei
# Created on: 2019/3/25 0025

from gensim.models import KeyedVectors
import numpy as np
from pyhanlp import HanLP
import pandas as pd
# import jieba


class WordVectorFetcher:
    def __init__(self, filename):
        self.wv_filename = filename
        self.wv = None

    def init(self):
        self.wv = KeyedVectors.load_word2vec_format(self.wv_filename)

    def get_word_vector(self, word):
        if word not in self.wv:
            return np.zeros(self.wv.vector_size)
        else:
            return self.wv[word]

    def get_sentence_vector(self, sentence):
        words = [item.word for item in HanLP.segment(sentence)]
        # words = [w for w in jieba.cut(sentence) if w.strip()!='']
        cnt = 0
        vec_fin = np.zeros(self.wv.vector_size)
        for w in words:
            if w in self.wv:
                vec_fin += self.get_word_vector(w)
                cnt += 1
        if cnt > 0:
            vec_fin = vec_fin / cnt
        return vec_fin

    def get_sentence_similarity(self, s1, s2):
        v1 = self.get_sentence_vector(s1)
        v2 = self.get_sentence_vector(s2)
        return self.wv.cosine_similarities(v1, [v2])[0]
        # return self.wv.wmdistance(s1, s2)


if __name__ == '__main__':
    # fn = 'SGNS/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5'
    fn = 'SGNS/merge_sgns_bigram_char300.txt'
    fetcher = WordVectorFetcher(fn)
    fetcher.init()
    # wv1 = fetcher.get_sentence_vector(u'今天天气算不错的了')
    # wv2 = fetcher.get_sentence_vector(u'今天没下雨')
    print(fetcher.get_sentence_similarity(u'今天天气算不错的了', u'今天在北京没下雨'))
    print(fetcher.get_sentence_similarity(u'车头大面积进气格栅用镀铬材质进行装饰后年轻化效果显著', u'同时，在车头两侧，还有LED光源的头灯进行加持，夜间点亮后辨识度也很高'))
    print(fetcher.get_sentence_similarity(u'方向盘低速灵活高速平稳，就算18寸的大脚跑高速120都稳稳得一点都不飘', u'在路上不放音乐听发动机声音很平顺，高速过弯车身倾斜也很小，高速120会有风噪声'))
    df_label = pd.read_csv('iv_label.tsv', encoding='utf-8', sep='\t')
    df_label['label'] = df_label['first_level_label_name'] + df_label['second_level_label_name']
    for t in df_label['label'].drop_duplicates():
        print(t, fetcher.get_sentence_similarity(t, u'新车的内饰整体科技感较强，中间大尺寸触控屏搭载互联网汽车智能系统2.0-基于AliOS的斑马智行解决方案，可实现主副驾双区智能语音控制和IoT手机远程车控功能，为车主带来全场景下的智能出行体验'))
        # print(fetcher.get_sentence_vector(s)[:3])

