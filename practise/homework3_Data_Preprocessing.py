# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 21:37:58 2021

@author: s93yo
"""

# 作業三：健康檢查資料前處理
import pandas as pd
import numpy as np

# In[] 匯入資料
health_data = pd.read_csv("HealthCheck.csv")


# In[] 切分「自變數、應變數」
X = health_data.iloc[:, :-1].values

Y = health_data.iloc[:, 3].values


# In[] 缺失資料補足
# 查看有無缺失值
health_data.isnull().sum()

# 引入 SimpleImputer 套件
from sklearn.impute import SimpleImputer
# 用平均值來填補缺失值 
imputer = SimpleImputer(missing_values= np.nan, strategy = "mean")
# 填補欄位 2、欄位 3 的數值
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])


# In[] 類別資料數位化
from sklearn.preprocessing import LabelEncoder

# 將 Y 類別資料數位化
labelEncoder = LabelEncoder()
Y_label = labelEncoder.fit_transform(Y).astype("float64")

# 將 X 類別資料數位化
ary_dummies = pd.get_dummies(X[:, 0]).values
X_label = np.concatenate((ary_dummies, X[:, 1:3]), axis = 1).astype("float64")


# In[] 切分「訓練集、測試集」
from sklearn.model_selection import train_test_split
# 訓練集佔比 0.8，測試集佔比 0.2
X_train, X_test, Y_train, Y_test = train_test_split(X_label, Y_label, test_size = 0.2, random_state = 0)


# In[] 特徵縮放
from sklearn.preprocessing import StandardScaler

# 產生 1 個 StandardScaler 物件，並用 X_train 產生縮放計算模型
sc_X = StandardScaler().fit(X_train)
# 用縮放後的值，替換 X_train 內的原值
X_train_sc = sc_X.transform(X_train)
# 用縮放後的值，替換 X_test 內的原值
X_test_sc = sc_X.transform(X_test)


# In[] 印出資料前處理結果
print("資料前處理「一般版」")
print("自變數訓練集：\n", X_train_sc)
print("應變數訓練集：\n", Y_train)
print("自變數測試集：\n", X_test_sc)
print("應變數測試集：\n", Y_test)




# In[] 健康檢查資料前處理「快樂版」
import HappyML.preprocessor as pp

# In[] 匯入資料
happy_health_data = pp.dataset(file = "HealthCheck.csv")


# In[] 切分「自變數、應變數」
happy_X, happy_Y = pp.decomposition(happy_health_data, x_columns = [i for i in range(3)], y_columns = [3])


# In[] 檢查有無缺失值並填補
happy_X = pp.missing_data(happy_X, strategy = "mean")


# In[] 類別資料數位化
happy_Y_laber, Y_mapping = pp.label_encoder(happy_Y, mapping = True)

happy_X_laber = pp.onehot_encoder(happy_X, columns = [0])


# In[] 切分「訓練集、測試集」
happy_X_train, happy_X_test, happy_Y_train, happy_Y_test = pp.split_train_test(happy_X_laber, happy_Y_laber, train_size= 0.8, random_state= 0)


# In[] 特徵縮放
happy_X_train_sc, happy_X_test_sc = pp.feature_scaling(happy_X_train, transform_arys = (happy_X_train, happy_X_test))

# In[] 印出「快樂版」結果
# 跟一般版空一格
print("")
print("資料前處理「快樂版」")
print("自變數訓練集：\n", happy_X_train_sc)
print("應變數訓練集：\n", happy_Y_train)
print("自變數測試集：\n", happy_X_test_sc)
print("應變數測試集：\n", happy_Y_test)
