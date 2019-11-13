# -*- coding: utf8 -*-
# Created by: wuzewei
# Created by: 2019/11/13
import pandas as pd
import numpy as np
import xgboost
from sklearn import model_selection
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import codecs
import json

from pyhanlp import *
import gensim
from gensim.models import KeyedVectors, TfidfModel
from gensim.similarities import SparseMatrixSimilarity

from WordVectorFetcher import WordVectorFetcher

# NLPTokenizer = JClass("com.hankcs.hanlp.tokenizer.NLPTokenizer")
def seg(doc):
    tokens = []
#     for item in NLPTokenizer.segment(doc):
    for item in HanLP.segment(doc):
        word = item.word
        tag = item.nature.toString()
        # http://www.hankcs.com/nlp/part-of-speech-tagging.html#h2-8
        if tag[0] not in ['b','m','p','q','u','x']:
            tokens.append(word)
#         tokens.append(word)
    return tokens


class TfidfWordVectorCombiner:
    def __init__(self):
        self.dictionary = None
        self.tfidf_model = None
        self.fetcher = WordVectorFetcher('tmp/sgns.sogou.word.bz2')
        #         self.fetcher = WordVectorFetcher('tmp/sgns.zhihu.bigram-char.bz2')
        print('Loading word vector file...')
        self.fetcher.init()
        print('Done')

    def fit(self, df_train):
        corpus_train = list(df_train['content'].apply(seg))
        self.dictionary = gensim.corpora.Dictionary(corpus_train)
        with codecs.open('tmp/tfidf_vocab.csv', 'w', encoding='utf8') as fout:
            for id, tok in self.dictionary.id2token.items():
                fout.write("{}\t{}\n".format(id, tok))
        corpus_train_bow = [self.dictionary.doc2bow(tokens) for tokens in corpus_train]
        self.tfidf_model = TfidfModel(corpus_train_bow)
        return

    def transform(self, df):
        corpus = list(df['content'].apply(seg))
        corpus_bow = [self.dictionary.doc2bow(tokens) for tokens in corpus]
        tfidf_corpus = [t for t in self.tfidf_model[corpus_bow]]
        arr = []
        for tfidf_doc in tfidf_corpus:
            vec = np.zeros_like(self.fetcher.get_word_vector(u""))
            for token_id, token_tfidf in tfidf_doc:
                token = self.dictionary[token_id]
                vec += token_tfidf * self.fetcher.get_word_vector(token)
            arr.append(vec.reshape((1, len(vec))))
        X = np.concatenate(arr)
        return X

    def fit_transform(self, df_train):
        corpus_train = list(df_train['content'].apply(seg))
        self.dictionary = gensim.corpora.Dictionary(corpus_train)
        with codecs.open('tmp/tfidf_vocab.csv', 'w', encoding='utf8') as fout:
            for id, tok in self.dictionary.id2token.items():
                fout.write("{}\t{}\n".format(id, tok))
        corpus_train_bow = [self.dictionary.doc2bow(tokens) for tokens in corpus_train]
        self.tfidf_model = TfidfModel(corpus_train_bow)

        tfidf_corpus = [t for t in self.tfidf_model[corpus_train_bow]]
        arr = []
        for tfidf_doc in tfidf_corpus:
            vec = np.zeros_like(self.fetcher.get_word_vector(u""))
            for token_id, token_tfidf in tfidf_doc:
                token = self.dictionary[token_id]
                vec += token_tfidf * self.fetcher.get_word_vector(token)
            arr.append(vec.reshape((1, len(vec))))
        X = np.concatenate(arr)
        return X


if __name__ == '__main__':
    ### load data
    df_data = pd.read_csv('data/sentiment_corpus_20191108.txt', encoding='utf8', sep='\t', names=['label', 'content'])
    label2id = {'negative': -1, 'neutral': 0, 'positive': 1}
    df_data['content_id'] = range(len(df_data))
    df_data['label_id'] = df_data['label'].apply(lambda x: label2id[x])
    print(df_data.shape)

    combiner = TfidfWordVectorCombiner()

    xgb = xgboost.XGBClassifier(n_estimators=500, n_jobs=-1)
    rf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
    lr = LogisticRegression(n_jobs=-1, solver='lbfgs', multi_class='auto')
    svc = SVC(gamma='scale', kernel='rbf')

    model = svc

    y_data = df_data['label'].values
    X_data = combiner.fit_transform(df_data)
    print('training...')
    model.fit(X_data, y_data)
    print('done')

    df_test = pd.read_csv('data/real_senti_demo_nolabel.txt', encoding='utf8', sep='\t', names=['content'])
    X_test = combiner.transform(df_test)
    y_test_pred = model.predict(X_test)
    df_test['label'] = y_test_pred
    df_test[['label', 'content']].to_csv('data/submission.csv', encoding='utf8', sep='\t', index=False)

