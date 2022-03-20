# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 18:30:09 2021

@author: s93yo
"""

# In[] 健康檢查資料前處理「快樂版」
import HappyML.preprocessor as pp


# In[] 匯入資料
salary_data = pp.dataset(file = "Salary_Data.csv")


# In[] 切分「自變數、應變數」
X, Y = pp.decomposition(salary_data, x_columns = [0], y_columns = [1])


# In[] 切分「訓練集、測試集」
X_train, X_test, Y_train, Y_test = pp.split_train_test(X, Y, train_size= 2/3)


# In[] 特徵縮放
# X_train, X_test = pp.feature_scaling(X_train, transform_arys = (X_train, X_test))
# Y_train, Y_test = pp.feature_scaling(fit_ary=Y_train, transform_arys=(Y_train, Y_test))



# In[] 簡單線性迴歸
from sklearn.linear_model import LinearRegression
# 建立一個 LinearRegression 物件
regressor = LinearRegression()
# 訓練「簡單線性迴歸」模型
regressor.fit(X_train, Y_train)
# 利用模型預測答案
Y_pred = regressor.predict(X_test)
# 計算迴歸模型的 R-square
R_Score = regressor.score(X_test, Y_test)


# In[] 簡單線性迴歸 「快樂版」
# 引入 SimpleRegressor 類別
from HappyML.regression import SimpleRegressor

# 產生 SimpleRegressor 類別的物
regressor = SimpleRegressor()
# 訓練 + 預測
Y_pred = regressor.fit(X_train, Y_train).predict(X_test)
# 印出 R2，以評估模型好壞
print("R-Squared Score:", regressor.r_score(X_test, Y_test))


# In[] 簡單線性迴歸模型繪圖
from HappyML import model_drawer as md

sample_data = (X_train, Y_train)
model_data = (X_train, regressor.predict(X_train))

md.sample_model(sample_data = sample_data, model_data = model_data,
                title = "訓練集樣本點 vs. 預測模型", font = "Microsoft JhengHei")

md.sample_model(sample_data = (X_test, Y_test), model_data = (X_test, Y_pred),
                title = "測試集樣本點 vs. 預測模型", font = "Microsoft JhengHei")








