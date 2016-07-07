#-*-coding:utf-8-*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import statsmodels.api as sm
import scipy.fftpack as fft
from matplotlib import rc
from math import pi as pi
font = {'family':'Osaka'}
rc('font',**font)

"""
def make_df(code):

    df_by_code = data[data["コード"]==code]
    df_by_code = df_by_code.set_index("日付")
    df_by_code.index=pd.to_datetime(df_by_code.index)

#学習データ
    df_by_code2 = df_by_code[(df_by_code.index.year >= 2007) & (df_by_code.index.year <= 2012)]
    #移動平均
    df_by_code3 = pd.rolling_mean(df_by_code2["売上"],5)

    df_dropna = df_by_code3.dropna().reset_index()
    df_dropna.columns = ["date","sales"]

#検証データ
    awave = df_by_code[(df_by_code.index.year >= 2007) & (df_by_code.index.year <= 2013)]
    #移動平均
    awave = pd.rolling_mean(awave,5)
    awave = awave.drop([list(awave.index)[i] for i in range(4)])


    fftY, power, freqs, pred, phase, Y, intercept, x1 = transform_sales(df_dropna)

    Y = pd.rolling_mean(df_by_code2["売上"],5)

    return fftY, power, freqs, pred, phase, Y, intercept, x1, awave

def transform_sales(df):
    #ログをとる
    df["sales_logged"] = df["sales"].apply(np.log)
    #傾きをなくす
    x = list(df.index)
    x = sm.add_constant(x)
    y = df["sales_logged"]
    model = sm.OLS(y,x)
    results = model.fit()
    intercept, x1 = results.params
    #残差をとる
    pred = df.index*x1 + intercept
    Y = df["sales_logged"] - pred
    df2 = df.set_index("date")

    fftY, freqs, power, phase = ftransform(Y)

    return fftY, power, freqs, pred, phase, Y, intercept, x1

def ftransform(Y):
    fftY = fft.fft(Y)
    #fftY[20:-20] = 0
    freqs = fft.fftfreq(len(Y))
    power = np.abs(fftY)
    phase = [np.arctan2(float(c.imag), float(c.real)) for c in fftY]

    return fftY, freqs, power, phase



def newwave(fftY,freqs,power,phase):
    wave = []
    for t in range(0,MN):
        average = 0
        for po,fr,ph in zip(power,freqs,phase):
            average += po*np.cos((2*pi*fr) * t + ph)/len(Y)
        wave.append(average)

    iwave = fft.ifft(fftY)
    return wave,iwave

def newwave2(fftY,freqs,power,phase):
    pi_frArray = []
    phArray = []
    poArray = []
    #leny = len(Y)

    for po,fr,ph in zip(power,freqs,phase):
        pi_frArray.append(2*pi*fr)
        phArray.append(ph)
        poArray.append(po)

            #average += po*np.cos((2*pi*fr) * t + ph)/len(Y)

    wave = []
    for t in range(0,MN):
        average = 0
        for cnt in range(0,len(pi_frArray)):
        #for po,fr,ph in zip(power,freqs,phase):
            average += poArray[cnt]*np.cos((pi_frArray[cnt]) * t + phArray[cnt])/len(Y)
            wave.append(average)

    iwave = fft.ifft(fftY)
    return wave,iwave


def inverse(wave,iwave,intercept,x1,predict):
    wave = wave + intercept + [i * x1 for i in range(0,MN)]
    wave = np.exp(wave)
    #adjust = [0, 0, 0, 0]
    #wave = array.insert(0,adjust)

    iwave += predict
    iwave = np.exp(iwave)

    return wave,iwave
"""

if __name__ == "__main__":

    target_train = pd.read_csv("~/PycharmProjects/1stBigDataAnalyticsContest/target/target_train.csv",parse_dates=0,header=None)
    plt.figure()
    plt.plot(target_train)
    plt.show()

"""

    fftY, power, freqs, pred, phase, Y, intercept, x1, awave = make_df(5335)
    print(len(Y))
    print(np.var(Y))
    global MN
    MN = len(Y)+361
    wave, iwave = newwave(fftY,freqs,power,phase)
    wave, iwave = inverse(wave,iwave,intercept,x1,pred)

    reg = wave - awave["売上"]
    MAPE = sum(abs(reg[MN-366:MN])/awave["売上"][MN-366:MN])/365*100
    print(MAPE)
    plt.figure()
    plt.suptitle("Compare Real and Forecast")
    plt.plot(awave["売上"][0:MN], c="r", label="Real")
    plt.plot(wave[0:MN], c="g", label="Forecast")
    plt.legend(loc="best")
    plt.figure()
    plt.suptitle("Regiduals")
    plt.plot(reg[MN-371:MN], label="Regiduals")
    plt.axhline(0,c="g")
    plt.show()
"""
