# ライブラリの読み込み
import pandas as pd
import statsmodels.api as sm

# データの読み込み
# 観光宿泊数データ（学習用）
target = pd.read_csv('target_train.csv', header=0, parse_dates='date', index_col='date')
# 為替データ（学習用）
exchange = pd.read_csv('exchange_train_new.csv', parse_dates='date', index_col='date')
# ロケーション付SNSデータ（学習用）
geo = pd.read_csv('geo_location_train.csv', header=0, parse_dates='date', index_col='date')
# 為替データ（テスト用）
exchange_test = pd.read_csv('exchange_test_new.csv', parse_dates='date', index_col='date')
# ロケーション付SNSデータ（テスト用）
geo_test = pd.read_csv('geo_location_test.csv', header=0, parse_dates='date', index_col='date')
# 応募用データ
submit = pd.read_csv('sample_submit.csv', header=-1)

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
        # 為替データとロケーション付SNSデータを説明変数に指定
        exchange['geo'] = geo[col]
        X = sm.add_constant(exchange, prepend=False)
        # 宿泊者数を目的変数に指定
        y = target[target_col]
        # 重回帰分析
        model = sm.OLS(y, X)
        results = model.fit()
        # テストデータによる予測
        exchange_test['geo'] = geo_test[col]
        X_test = sm.add_constant(exchange_test, prepend=False)
        # 応募データへ追加
        submit[i] = results.predict(X_test)
        i += 1

# 応募データの書き出し
submit.to_csv('my_submit.csv', index=False, header=False)

# END
