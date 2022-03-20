# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 23:38:01 2019

@author: User
"""
### 資料前處理「一般版」範例
import numpy as np
import pandas as pd

# In[] 讀入資料 csv 檔
dataset = pd.read_csv("CarEvaluation.csv")


# In[] 切分自變數、應變數
# 機器學習底層函式庫大多只接受傳入 NDArray，所以需將 DataFrame 轉換為 NDArray
X = dataset.iloc[:, :-1].values
# 如要跳過特定幾欄，可用串列生成式的寫法 (快樂版也是用此原理操作)
X = dataset.iloc[:, [i for i in range(4)]].values

Y = dataset.iloc[:, 4].values


# In[] 檢查有無缺失值
# 回傳 DateFrame 格式 (不建議使用)
dataset.isnull()
# 回傳每個欄位的缺失值總數 (Series 格式)
dataset.isnull().sum()
# 回傳每個欄位是否有缺失值 (Series 格式。不加總，會比 sum() 快一些) 
dataset.isnull().any()
# 回傳缺失值個數數字
sum(dataset.isnull().sum())


# In[] 填補缺失值
# 引入 SimpleImputer 套件
from sklearn.impute import SimpleImputer
# 指定缺失值填入策略 (這邊用平均值)
imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
# 針對有缺失的「欄1、欄2、欄3」，計算各欄平均值
imputer = imputer.fit(X[:, 1:4])
# 將所有缺失值，轉化為各欄平均值
X[:, 1:4] = imputer.transform(X[:, 1:4])


# In[] 類別資料數位化 LabelEncoder (Y 結果用)
# 引入 LabelEncoder (標籤編碼器)
from sklearn.preprocessing import LabelEncoder
# 由 LabelEncoder 類別產生一個物件，用變數 labelEncoder 保存起來
labelEncoder = LabelEncoder()
# astype("float64") 為了與 X 陣列的浮點數運算時，型態相同而做轉換 (非必要，但沒轉換會跑出警告訊息)
Y = labelEncoder.fit_transform(Y).astype("float64")

# 類別資料數位化 One-Hot Encoder (獨熱編碼器，X 類別資料用。一個標籤，一個欄位)
# 傳入第一欄，並針對這一欄做 "One-Hot Encoder" 後以 DataFrame 傳回，再轉換為 NDArray
ary_dummies = pd.get_dummies(X[:, 0]).values
# 合併 ary_dummies 及 X 第 0 欄以外的欄位 (將 X 的第一欄用 ary_dummies 取代)
X = np.concatenate((ary_dummies, X[:, 1:4]), axis = 1).astype("float64")


# In[] 切分訓練集、測試集 
# 引入「train_test_split」物件
from sklearn.model_selection import train_test_split
# test_size = 設定測試集佔比，random_state = 整數「亂數種子」(相同數字，每次切分的結果相同)，要每次不同的話設 = None
X_trian, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# In[] 特徵縮放
# 引入 「StandardScaler」物件
from sklearn.preprocessing import StandardScaler
# 產生 1 個 StandardScaler 物件，並產生縮放計算模型
sc_X = StandardScaler().fit(X_trian)
# 用縮放後的值，替換 X_train 內的原值
X_trian = sc_X.transform(X_trian)
# 用縮放後的值，替換 X_test 內的原值
X_test = sc_X.transform(X_test)








