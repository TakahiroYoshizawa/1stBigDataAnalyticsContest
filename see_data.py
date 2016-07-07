import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import scipy.fftpack as fft
from matplotlib import rc
from math import pi as pi
font = {'family':'Osaka'}
rc('font',**font)

def imput_data():
    # 観光宿泊数データ（学習用）
    target = pd.read_csv('~/PycharmProjects/1stBigDataAnalyticsContest/target/target_train.csv', header=0, parse_dates='date', index_col='date')
    # 為替データ（学習用）
    exchange = pd.read_csv('~/PycharmProjects/1stBigDataAnalyticsContest/exchange_train_KRW.csv', parse_dates='date', index_col='date')
    # ロケーション付SNSデータ（学習用）
    geo = pd.read_csv('~/PycharmProjects/1stBigDataAnalyticsContest/geo_location/geo_location_train.csv', header=0, parse_dates='date', index_col='date')
    # 為替データ（テスト用）
    exchange_test = pd.read_csv('~/PycharmProjects/1stBigDataAnalyticsContest/exchange_test_KRW.csv', parse_dates='date', index_col='date')
    # ロケーション付SNSデータ（テスト用）
    geo_test = pd.read_csv('~/PycharmProjects/1stBigDataAnalyticsContest/geo_location/geo_location_test.csv', header=0, parse_dates='date', index_col='date')
    # 応募用データ
    submit = pd.read_csv('~/PycharmProjects/1stBigDataAnalyticsContest/sample_submit.csv', header=-1)

    return target, exchange, geo, exchange_test, geo_test, submit

def forcast(target, exchange, geo, exchange_test, geo_test, submit):
    i = 1
    for suff in suff_list:
        # 予測地点の市区町村毎にループ
        for col in col_list:
            # 宿泊者数のカラム名を指定
            target_col = col + suff

            #target.index = range(0,365)
            #x = list(target.index)

            plt.figure()
            plt.plot(target.index,target[target_col])
            plt.title(target_col)
            plt.show()
            """
            x = sm.add_constant(x)

            for g in range(0,len(target)):
                if target[target_col][g] == 0:
                    target[target_col][g] = 1

            y = target[target_col].apply(np.log)
            print(target[target_col])
            model = sm.OLS(y,x)
            results = model.fit()
            #print(results.summary)
            intercept, x1 = results.params
            pred = target.index * x1 + intercept
            Y = y - pred
            #print(Y)
            L = len(Y)
            #print(L)

            fftY = fft.fft(Y)
            freqs = fft.fftfreq(L)
            power = np.abs(fftY)
            phase = [np.arctan2(float(c.imag), float(c.real)) for c in fftY]

            wave = newwave(L,intercept,x1,power,freqs,phase)
            submit[i] = wave
            i += 1
            """
    return submit


def newwave(L,intercept,x1,power,freqs,phase):
    wave = []
    for t in range(366,549):
        average = 0
        for po,fr,ph in zip(power,freqs,phase):
            average += po*np.cos((2*pi*fr) * t + ph)/183
            #average = int(average)
        wave.append(average)
    wave = wave + intercept + [s * x1 for s in range(366,549)]
    wave = np.exp(wave)

    return wave

if __name__ == '__main__':
    target, exchange, geo, exchange_test, geo_test, submit = imput_data()
    # 予測地点の市区町村コードをリスト化

    plt.figure()
    plt.plot(exchange.index,exchange["KRW_JPY"])
    plt.legend(loc="best")
    plt.show()

    col_list = ['01202', '04100', '13102', '14382', '14384', '16201', '17201', '22205', '24203', '26100', '32203', '34100', '42201', '47207']
    # 予測対象のサフィックスをリスト化（全観光客数、訪日外国人）
    suff_list = ['_total', '_inbound']
    submit = forcast(target, exchange, geo, exchange_test, geo_test, submit)
#     submit.to_csv('my_submit.csv', index=False, header=False)
