import skflow
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import scipy.fftpack as fft
from matplotlib import rc
from math import pi as pi
from sklearn import datasets, metrics
import jholiday
import datetime
from sklearn import tree
import skflow
import tensorflow as tf


font = {'family': 'Osaka'}
rc('font', **font)

def imput_data():
    # 観光宿泊数データ（学習用）
    target = pd.read_csv('~/PycharmProjects/1stBigDataAnalyticsContest/target/target_train.csv', header=0,
                         parse_dates='date', index_col='date')
    # 為替データ（学習用）
    exchange = pd.read_csv('~/PycharmProjects/1stBigDataAnalyticsContest/exchange_train_KRW.csv', parse_dates='date'
                           , index_col='date')

    gtrends = pd.read_csv("~/PycharmProjects/1stBigDataAnalyticsContest/gtrends_train.csv", header=0)

    geo = pd.read_csv('~/PycharmProjects/1stBigDataAnalyticsContest/geo_location/geo_location_train.csv', header=0, parse_dates='date', index_col='date')

    # 為替データ（テスト用）
    exchange_test = pd.read_csv('~/PycharmProjects/1stBigDataAnalyticsContest/exchange_test_KRW.csv',
                                parse_dates='date', index_col='date')

    gtrends_test = pd.read_csv("~/PycharmProjects/1stBigDataAnalyticsContest/gtrends_test.csv", header=0)

    geo_test = pd.read_csv('~/PycharmProjects/1stBigDataAnalyticsContest/geo_location/geo_location_test.csv', header=0,
                           parse_dates='date', index_col='date')
    # 応募用データ
    submit = pd.read_csv('~/PycharmProjects/1stBigDataAnalyticsContest/sample_submit.csv', header=-1)

    return target, exchange, gtrends, geo, exchange_test, gtrends_test, geo_test, submit






def make_data(exchange, gtrends, geo, exchange_test, gtrends_test, geo_test, col):

    exchange = exchange.resample('D',fill_method="ffill")
    exchange_test = exchange_test.resample('D',fill_method="ffill")


    Holiday_train = []
    for j in range(0,len(exchange)):
        h = is_holiday(exchange.index[j])
        h = tf_transformer(h)
        Holiday_train.append(h)
    Holiday_test = []
    for j in range(0,len(exchange_test)):
        h = is_holiday(exchange_test.index[j])
        h = tf_transformer(h)
        Holiday_test.append(h)

    X_train = []
    for j in range(0,len(exchange)):
        x = (exchange["CAD_JPY"][j],exchange["CNY_JPY"][j],exchange["EUR_JPY"][j],exchange["GBP_JPY"][j],exchange["USD_JPY"][j],
             exchange["KRW_JPY"][j],gtrends[col][j],geo[col][j],Holiday_train[j],exchange.index.month[j],exchange.index.weekday[j])
        x = list(x)
        X_train.append(x)

    X_test = []
    for j in range(0,len(exchange_test)):
        x = (exchange_test["CAD_JPY"][j],exchange_test["CNY_JPY"][j],exchange_test["EUR_JPY"][j],exchange_test["GBP_JPY"][j],
             exchange_test["USD_JPY"][j],exchange_test["KRW_JPY"][j],gtrends_test[col][j],geo_test[col][j],Holiday_test[j],exchange_test.index.month[j],exchange.index.weekday[j])
        x = list(x)
        X_test.append(x)
    return X_train, X_test

def tf_transformer(TF):
    if TF == "True":
        TF = 1
    elif TF == "False":
        TF = 0
    return TF


def is_saturday_or_sunday(date):
    if 4 < date.weekday() < 7:
        return True
    return False

def is_holiday(date):
    if is_saturday_or_sunday(date) or \
            jholiday.holiday_name(date.year, date.month, date.day):
       return True
    return False

if __name__ == '__main__':
    # データの取り込み
    target, exchange, gtrends, geo, exchange_test, gtrends_test, geo_test, submit = imput_data()

    # 予測地点の市区町村コードをリスト化
    col_list = ['01202', '04100', '13102', '14382', '14384', '16201', '17201', '22205', '24203', '26100', '32203',
                '34100', '42201', '47207']
    # 予測対象のサフィックスをリスト化（全観光客数、訪日外国人）
    suff_list = ['_total', '_inbound']


    i = 1
    for suff in suff_list:

        for col in col_list:

            X_train, X_test = make_data(exchange, gtrends, geo, exchange_test, gtrends_test, geo_test, col)
            target_col = col + suff
            classifier = skflow.TensorFlowDNNRegressor(hidden_units=[1000,4000,10000,4000,1000], n_classes=1)
            s = np.array(target[target_col].values)
            X_train = np.array(X_train)
            X_test = np.array(X_test)

            print(X_test)
            print(X_train)
            print(s)
            print(len(X_train))

            classifier.fit(X_train, s)

            forecast = classifier.predict(X_test)
            print(len(X_test))
            print(forecast)
            print(len(forecast))
            submit[i] = forecast
            i += 1

    submit.to_csv('my_submit4.csv', index=False, header=False)
