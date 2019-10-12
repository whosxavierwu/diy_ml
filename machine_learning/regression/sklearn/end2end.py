# -*- coding: utf8 -*-
# Created by: wuzewei
# Created by: 2019/10/12
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

from pandas.plotting import scatter_matrix

from sklearn import model_selection
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from sklearn import metrics

import joblib


if __name__ == '__main__':
    # todo dataset
    df_housing = pd.read_csv('../datasets/housing/housing.csv', encoding='utf8', sep=',')
    print(df_housing.shape)



