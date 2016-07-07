# ライブラリの読み込み
import pandas as pd
import statsmodels.api as sm
import scipy.fftpack as fft
import numpy as np
from math import pi as pi

# データの読み込み
# 観光宿泊数データ（学習用）
target = pd.read_csv('~/PycharmProjects/1stBigDataAnalyticsContest/target/target_train.csv', header=0, parse_dates='date', index_col='date')
# 為替データ（学習用）
exchange = pd.read_csv('~/PycharmProjects/1stBigDataAnalyticsContest/exchange_train_new.csv', parse_dates='date', index_col='date')
# ロケーション付SNSデータ（学習用）
geo = pd.read_csv('~/PycharmProjects/1stBigDataAnalyticsContest/geo_location/geo_location_train.csv', header=0, parse_dates='date', index_col='date')
# 為替データ（テスト用）
exchange_test = pd.read_csv('~/PycharmProjects/1stBigDataAnalyticsContest/exchange_test_new.csv', parse_dates='date', index_col='date')
# ロケーション付SNSデータ（テスト用）
geo_test = pd.read_csv('~/PycharmProjects/1stBigDataAnalyticsContest/geo_location/geo_location_test.csv', header=0, parse_dates='date', index_col='date')
# 応募用データ
submit = pd.read_csv('~/PycharmProjects/1stBigDataAnalyticsContest/sample_submit.csv', header=-1)

gtrends = pd.read_csv("~/PycharmProjects/1stBigDataAnalyticsContest/gtrends_train.csv", header=0)

gtrends_test = pd.read_csv("~/PycharmProjects/1stBigDataAnalyticsContest/gtrends_test.csv", header=0)


# 予測地点の市区町村コードをリスト化
col_list = ['01202', '04100', '13102', '14382', '14384', '16201', '17201',
'22205', '24203', '26100', '32203', '34100', '42201', '47207']
# 予測対象のサフィックスをリスト化（全観光客数、訪日外国人）
suff_list = ['_total', '_inbound']
# 変数の初期化
i = 1

# 重回帰モデルを仮定
# 予測対象毎にループ
for suff in suff_list:
    # 予測地点の市区町村毎にループ
    for col in col_list:
        # 宿泊者数のカラム名を指定
        target_col = col + suff
        target.index = range(0, 365)
        # 為替データとロケーション付SNSデータを説明変数に指定
        exchange['geo'] = geo[col]
        X = sm.add_constant(gtrends[col], prepend=False)
        X.index = range(0,365)
        print(X)
        # 宿泊者数を目的変数に指定
        y = target[target_col]
        # 重回帰分析
        model = sm.OLS(y, X)
        results = model.fit()
        print(results.summary())
        # テストデータによる予測
        exchange_test['geo'] = geo_test[col]
        X_test = sm.add_constant(gtrends_test[col], prepend=False)
        X_test.index = range(0,183)
        # 応募データへ追加
        submit[i] = results.predict(X_test)
        i += 1

# 応募データの書き出し
submit.to_csv('my_submit1.csv', index=False, header=False)

# END
