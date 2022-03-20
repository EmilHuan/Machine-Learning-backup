# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 23:38:01 2019

@author: User
"""
### 資料前處裡「快樂版」範例
# 已預先引入前處理必要套件，不必自行引入多個套件
import HappyML.preprocessor as pp

# In[]  讀入資料 csv 檔
dataset = pp.dataset(file = "CarEvaluation.csv")

# In[] 切分自變數、應變數
# 回傳值為 DataFrame (能保留欄列名稱)
# 可使用串列生成式指定欄位 (一般版其實也可以，但寫起來比較複雜)
X, Y = pp.decomposition(dataset, x_columns = [i for i in range(4)], y_columns = [4])


# In[] 檢查有無缺失值
dataset.isnull().sum()

dataset.isnull().any()

sum(dataset.isnull().sum())

# In[] 填補缺失值
X = pp.missing_data(X, strategy = "mean")

# In[] 類別資料數位化
# Y 類別資料數位化 LabelEncoder
Y, Y_mapping = pp.label_encoder(Y, mapping = True)

# X 類別資料數位化 One-Hot Encoder
X = pp.onehot_encoder(X, columns = [0])


# In[] 切分訓練集、測試集 
X_train, X_test, Y_train, Y_test = pp.split_train_test(X, Y, train_size=0.8, random_state=0)


# In[] 特徵縮放
X_train, X_test = pp.feature_scaling(X_train, transform_arys=(X_train, X_test))











