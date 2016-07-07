import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import scipy.fftpack as fft
from matplotlib import rc
from math import pi as pi

font = {'family': 'Osaka'}
rc('font', **font)


def imput_data():
    # 観光宿泊数データ（学習用）
    target = pd.read_csv('~/PycharmProjects/1stBigDataAnalyticsContest/target/target_train.csv', header=0,
                         parse_dates='date', index_col='date')
    # 為替データ（学習用）
    exchange = pd.read_csv('~/PycharmProjects/1stBigDataAnalyticsContest/exchange_train_new.csv', parse_dates='date',
                           index_col='date')
    # ロケーション付SNSデータ（学習用）
    geo = pd.read_csv('~/PycharmProjects/1stBigDataAnalyticsContest/geo_location/geo_location_train.csv', header=0,
                      parse_dates='date', index_col='date')
    # 気象データ
    weather = pd.read_csv('~/PycharmProjects/1stBigDataAnalyticsContest/weather/weather_train.csv', header=0,
                      parse_dates='date', index_col='date')

    sensor = pd.read_csv('~/PycharmProjects/1stBigDataAnalyticsContest/sensor/sensor_train.csv', header=0,
                      parse_dates='date', index_col='date')

    gtrends = pd.read_csv("~/PycharmProjects/1stBigDataAnalyticsContest/gtrends_train.csv", header=0)



    # 為替データ（テスト用）
    exchange_test = pd.read_csv('~/PycharmProjects/1stBigDataAnalyticsContest/exchange_test_new.csv',
                                parse_dates='date', index_col='date')
    # ロケーション付SNSデータ（テスト用）
    geo_test = pd.read_csv('~/PycharmProjects/1stBigDataAnalyticsContest/geo_location/geo_location_test.csv', header=0,
                           parse_dates='date', index_col='date')

    weather_test = pd.read_csv('~/PycharmProjects/1stBigDataAnalyticsContest/weather/weather_test.csv', header=0,
                      parse_dates='date', index_col='date')

    sensor_test = pd.read_csv('~/PycharmProjects/1stBigDataAnalyticsContest/sensor/sensor_test.csv', header=0,
                      parse_dates='date', index_col='date')

    gtrends_test = pd.read_csv("~/PycharmProjects/1stBigDataAnalyticsContest/gtrends_test.csv", header=0)


    # 応募用データ
    submit = pd.read_csv('~/PycharmProjects/1stBigDataAnalyticsContest/sample_submit.csv', header=-1)



    return target, exchange, geo, weather, sensor, gtrends, exchange_test, geo_test, weather_test, sensor_test, gtrends_test, submit


def total_forcast(target, exchange, geo, weather, sensor, gtrends, exchange_test, geo_test, weather_test, sensor_test, gtrends_test, submit, i):
    for col in col_list:

        # 宿泊者数のカラム名を指定
        target_col = col + suff

        target.index = range(0, 365)
        X = sm.add_constant(exchange, prepend=False)
        X.index = range(0,365)
        #print(X)
        # log取る前の加工
        for g in range(0, len(target)):
            if target[target_col][g] == 0:
                target[target_col][g] = 1

        y = target[target_col]#.apply(np.log)
        #print(target[target_col])
        model = sm.OLS(y, X)
        results = model.fit()
        print(results.summary())
        pred = results.predict(X)
        Y = y - pred
        #plt.figure()
        #plt.plot(Y)
        #plt.show()
        L = len(Y)

        #Y = pd.rolling_mean(Y,7)
        fftY = fft.fft(Y)
        freqs = fft.fftfreq(L)
        power = np.abs(fftY)
        phase = [np.arctan2(float(c.imag), float(c.real)) for c in fftY]

        X_test = sm.add_constant(exchange_test[col], prepend=False)
        X_test.index = range(0,183)


        wave = newwave(L, results, power, freqs, phase, X_test)
        submit[i] = wave
        i += 1


    return submit, i

def newwave(L, results, power, freqs, phase, X_test):
    wave = []
    for t in range(366, 549):
        average = 0
        for po, fr, ph in zip(power, freqs, phase):
            average += po * np.cos((2 * pi * fr) * t + ph) / (L/2)
        wave.append(average)
    #print(wave)
    #print(X_test)
    pred = results.predict(X_test)

    wave = wave + pred

    for g in range(0, len(wave)):
        if wave[g] <= 0:
            wave[g] = 0




    return wave

def inbound_forcast(target, exchange, geo, weather, sensor, gtrends, exchange_test, geo_test, weather_test, sensor_test, gtrends_test, submit, i):
    for col in col_list:

        # 宿泊者数のカラム名を指定
        target_col = col + suff

        target.index = range(0, 365)
        X = sm.add_constant(exchange[col], prepend=False)
        X.index = range(0,365)
        #print(X)
        # log取る前の加工
        for g in range(0, len(target)):
            if target[target_col][g] == 0:
                target[target_col][g] = 1

        y = target[target_col]#.apply(np.log)
        #print(target[target_col])
        model = sm.OLS(y, X)
        results = model.fit()
        print(results.summary())
        pred = results.predict(X)
        Y = y - pred
        #plt.figure()
        #plt.plot(Y)
        #plt.show()
        L = len(Y)

        #Y = pd.rolling_mean(Y,7)
        fftY = fft.fft(Y)
        freqs = fft.fftfreq(L)
        power = np.abs(fftY)
        phase = [np.arctan2(float(c.imag), float(c.real)) for c in fftY]

        X_test = sm.add_constant(exchange_test[col], prepend=False)
        X_test.index = range(0,183)

        wave = newwave(L, results, power, freqs, phase, X_test)
        submit[i] = wave
        i += 1

    return submit, i

def newwave_i(L, results ,pred, power, freqs, phase, X_test):
    wave = []
    for t in range(366, 549):
        average = 0
        for po, fr, ph in zip(power, freqs, phase):
            average += po * np.cos((2 * pi * fr) * t + ph) / (L/2)
        wave.append(average)
    #print(wave)
    #print(X_test)
    pred = results.predict(X_test)

    wave = wave + pred
    for g in range(0, len(wave)):
        if wave[g] <= 0:
            wave[g] = 0
    return wave


if __name__ == '__main__':
    # データの取り込み
    target, exchange, geo, weather, sensor, gtrends, exchange_test, geo_test, weather_test, sensor_test, gtrends_test, submit = imput_data()

    # 予測地点の市区町村コードをリスト化
    col_list = ['01202', '04100', '13102', '14382', '14384', '16201', '17201', '22205', '24203', '26100', '32203',
                '34100', '42201', '47207']
    # 予測対象のサフィックスをリスト化（全観光客数、訪日外国人）
    suff_list = ['_total', '_inbound']
    gtrend = gtrends.set_index('Date')
    gtrend_test = gtrends_test.set_index('Date')
    exchange = exchange.resample('D',fill_method="ffill")
    exchange_test = exchange_test.resample('D',fill_method="ffill")


    #print(gtrend)

    i = 1
    for suff in suff_list:

        if suff == '_total':
            submit, i = total_forcast(target, exchange, geo, weather, sensor, gtrends, exchange_test, geo_test, weather_test, sensor_test, gtrends_test, submit, i)

        elif suff == '_inbound':
            submit, i = inbound_forcast(target, exchange, geo, weather, sensor, gtrends, exchange_test, geo_test, weather_test, sensor_test, gtrends_test, submit, i)

    submit.to_csv('my_submit5.csv', index=False, header=False)
