# -*-coding:utf-8-*-
from sqlite3 import Date
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import scipy.fftpack as fft
from matplotlib import rc
from math import pi as pi
import tensorflow as tf
from pandas.core.datetools import day
from scipy.interpolate import interp1d
from datetime import datetime


if __name__ == '__main__':
    gtrend_weekly = pd.read_csv("~/PycharmProjects/1stBigDataAnalyticsContest/googletrends_data/googletrends.csv", index_col=None)
    gtrend_weekly = gtrend_weekly.set_index('Date')
    gtrend_weekly.index = pd.to_datetime(gtrend_weekly.index)
    print(gtrend_weekly)

    gtrend_daily = gtrend_weekly.resample('D', fill_method='ffill')
    print(gtrend_daily)
    gtrends_train = gtrend_daily.ix[datetime(2014,6,1):datetime(2015,5,31),:]
    gtrends_test = gtrend_daily.ix[datetime(2015,6,1):datetime(2015,11,30),:]
    gtrends_train.columns = ['01202', '04100', '13102', '14382', '14384', '16201', '17201', '22205', '24203', '26100', '32203',
                    '34100', '42201', '47207']
    gtrends_test.columns = ['01202', '04100', '13102', '14382', '14384', '16201', '17201', '22205', '24203', '26100', '32203',
                    '34100', '42201', '47207']
    gtrends_train.to_csv('gtrends_train.csv')
    gtrends_test.to_csv('gtrends_test.csv')


