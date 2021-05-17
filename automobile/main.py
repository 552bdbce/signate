# ライブラリのimport
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_squared_error as MSE


# データの読み込み
data = pd.read_csv('./data/train.tsv', sep='\t')

# idの削除
data = data.drop(columns=['id'])
data = data[data.horsepower != '?']

# 欠損値を含む行の削除
data = data.dropna()

# 目的変数と説明変数の準備
X = data[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin']]
y = data['mpg']

# 学習用データと評価用データの分割
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=32)

# モデルの箱を準備
lr = LR()

# モデルを学習
lr.fit(X_train, y_train)

# モデルから予測結果を求める
y_pred_train = lr.predict(X_train)
y_pred_test = lr.predict(X_test)

# 学習データのRMSEの算出
mse_train = MSE(y_train, y_pred_train)
rmse_train = np.sqrt(mse_train)

# 評価データのRMSEの算出
mse_test = MSE(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)

# 学習および評価データに対するRMSEを表示
print(rmse_train)
print(rmse_test)

# 散布図の描画
plt.figure(figsize=(5, 5))
plt.scatter(y_test, y_pred_test)

# y_test及びy_pred_testの最小値・最大値を求める
test_max = np.max(y_test)
test_min = np.min(y_test)
pred_max = np.max(y_pred_test)
pred_min = np.min(y_pred_test)

# それぞれの値を比較し、最終的な最小値・最大値を求める
max_value = np.maximum(test_max, pred_max)
min_value = np.minimum(test_min, pred_min)

# x軸およびy軸の値域を指定する
plt.xlim([min_value, max_value])
plt.ylim([min_value, max_value])

# 対角線を引く
plt.plot([min_value, max_value], [min_value, max_value])

# x軸とy軸に名前を付ける
plt.xlabel('true')
plt.ylabel('pred')

# 可視化結果を表示する為に必要な関数
plt.show()