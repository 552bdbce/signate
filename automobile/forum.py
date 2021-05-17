import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from scipy.stats import skew


def carName(val, splitNm):
    list = val.split(" ")
    if len(list) > splitNm:
        ret = list[splitNm]
    else:
        ret = "Other"
    return ret


train = pd.read_csv('./data/train.tsv', sep='\t', header='infer', encoding='utf8')
test = pd.read_csv('./data/test.tsv', sep='\t', header='infer', encoding='utf8')

train = train[train.id != 41]
train = train[train.id != 62]
train = train[train.id != 353]

test['mpg'] = 'test'

mergedDf = pd.concat([train, test], sort=True)
mpg = mergedDf['mpg'].values

mergedDf['brand'] = mergedDf['car name'].apply(carName, splitNm=0)
mergedDf['brand'].replace("chevroelt", "chevrolet", inplace=True)
mergedDf['brand'].replace("mercedes", "mercedes-benz", inplace=True)
mergedDf['brand'].replace("toyouta", "toyota", inplace=True)
mergedDf['brand'].replace("vokswagen", "volkswagen", inplace=True)
mergedDf['brand'].replace("vw", "volkswagen", inplace=True)
mergedDf['brand'].replace("maxda", "mazda", inplace=True)

mergedDf.drop("car name", axis=1, inplace=True)

mergedDf['cylinders'] = mergedDf['cylinders'].astype(str)
mergedDf['origin'] = mergedDf['origin'].astype(str)

mergedDf.replace("?", np.nan, inplace=True)
mergedDf['horsepower'] = mergedDf['horsepower'].astype(float)
mergedDf["horsepower"] = mergedDf.groupby(["brand", 'cylinders'])["horsepower"].transform(lambda x: x.fillna(x.mean()))

numeric_feats = mergedDf.dtypes[mergedDf.dtypes != 'object'].index
skwed_feats = mergedDf[numeric_feats].apply(lambda x: skew(x.dropna()))
skwed_feats = skwed_feats[skwed_feats > 0.75].index
mergedDf = pd.get_dummies(data=mergedDf, dummy_na=True)
mergedDf = mergedDf.fillna(mergedDf.mean())
mergedDf[skwed_feats] = np.log1p(mergedDf[skwed_feats])

mergedDf['mpg'] = mpg

train = mergedDf[mergedDf.mpg != 'test']
test = mergedDf[mergedDf.mpg == 'test']
test.drop('mpg', axis=1, inplace=True)

Y = train["mpg"]
X = train.drop(['id', 'mpg'], axis=1)

model = LGBMRegressor(bagging_seed=1, feature_fraction=0.4, feature_fraction_seed=24, learning_rate=0.1, max_bin=21,
                      min_data_in_leaf=28, min_sum_hessian_in_leaf=5, n_estimators=85, num_leaves=8,
                      objective='regression')
model.fit(X, Y)

test["mpg"] = model.predict(test.drop('id', axis=1).copy())
test = test[["id", "mpg"]]
test.to_csv("sample_submit.csv", index=False, header=False, encoding='cp932')
