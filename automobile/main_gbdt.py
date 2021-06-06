import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_squared_error as MSE
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV


# -----------------------------------
# 学習データ、テストデータの読み込み
# -----------------------------------
# 学習データ、テストデータの読み込み
train = pd.read_csv('./data/train.tsv', sep='\t')
test = pd.read_csv('./data/test.tsv', sep='\t')

train = train.drop(columns=['id'])
# test = test.drop(columns=['id'])

# train = train[train.horsepower != '?']
train = train.replace('?', np.NaN)
test = test.replace('?', np.NaN)
# train = train.dropna()

import missingno as msno
msno.matrix(df=train, figsize=(20,14), color=(0.5,0,0))
plt.show()


def kesson_table(df):
    null_val = df.isnull().sum()
    percent = 100 * df.isnull().sum() / len(df)
    kesson_table = pd.concat([null_val, percent], axis=1)
    kesson_table_ren_columns = kesson_table.rename(
        columns={0: '欠損数', 1: '%'})
    return kesson_table_ren_columns

print("訓練データの欠損情報")
kesson = kesson_table(train)

# 学習データを特徴量と目的変数に分ける
train_x = train.drop(['mpg'], axis=1)
train_y = train['mpg']

# テストデータは特徴量のみなので、そのままでよい
test_x = test.copy()

# 学習用データと評価用データの分割
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=32)

# -----------------------------------
# 特徴量作成
# -----------------------------------
train_x = train_x.drop(columns=['car name'])
train_x['horsepower'] = train_x['horsepower'].astype(float)
test_x = test_x.drop(columns=['id', 'car name'])
test_x['horsepower'] = test_x['horsepower'].astype(float)

# dtrain = xgb.DMatrix(train_x, label=train_y)
# dtest = xgb.DMatrix(test_x.values)


# それぞれのカテゴリ変数にlabel encodingを適用する
# for c in ['Sex', 'Embarked']:
#     # 学習データに基づいてどう変換するかを定める
#     le = LabelEncoder()
#     le.fit(train_x[c].fillna('NA'))

    # # 学習データ、テストデータを変換する
    # train_x[c] = le.transform(train_x[c].fillna('NA'))
    # test_x[c] = le.transform(test_x[c].fillna('NA'))

# -----------------------------------
# モデル作成
# -----------------------------------
# xgboostモデルの作成
# reg = xgb.XGBRegressor()
#
# # ハイパーパラメータ探索
# reg_cv = GridSearchCV(reg, {'max_depth': [2,4,6], 'n_estimators': [50,100,200]}, verbose=1)
# reg_cv.fit(train_x, train_y)

# モデルの作成および学習データを与えての学習
model = xgb.XGBRegressor()
model.fit(train_x, train_y)

# テストデータの予測値を確率で出力する
pred = model.predict(test_x)
# 提出用ファイルの作成
submission = pd.DataFrame({'id': test['id'], 'mpg': pred})
submission.to_csv('./submission_automobile.csv', index=False)
print('csv output')
