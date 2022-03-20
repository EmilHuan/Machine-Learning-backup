# -*- coding: utf-8 -*-
"""
Created on Sat May  8 22:36:12 2021

@author: s93yo
"""
# 作業六：發電機失效時間 鄭桓安
# In[] 資料前處理
import HappyML.preprocessor as pp

# Load Dataset
dataset = pp.dataset(file = "Device_Failure.csv")

# Decomposition of Variables (切分自變數、應變數)
X, Y = pp.decomposition(dataset, x_columns = [0], y_columns = [1])

# Training / Testing Set (切分訓練集、測試集)
X_train, X_test, Y_train, Y_test = pp.split_train_test(x_ary = X, y_ary = Y, train_size = 0.8) 


# In[] 多項式線性迴歸
from HappyML.regression import PolynomialRegressor

reg_poly = PolynomialRegressor()

# 計算最佳 degree
best_degree = reg_poly.best_degree(x_train = X_train, y_train = Y_train, x_test = X_test, y_test = Y_test, verbose = False)

# 用最佳 degree 去 fit 原始全部資料
Y_poly = reg_poly.fit(x_train = X, y_train = Y).predict(x_test = X)

# 計算 RMSE
from HappyML.performance import rmse
rmse_poly = rmse(Y, Y_poly)


# In[] 請使用者輸入年份並計算故障時數
input_year = eval(input("請輸入設備已使用年份："))

import pandas as pd
# 將使用者輸入年份轉為 data.frame
user_input_year = pd.DataFrame([[input_year]], columns = ["Years"])

# 用模型預測「總失效時間」
Y_poly_user = reg_poly.fit(x_train = X, y_train = Y).predict(x_test = user_input_year)

# 將 Y_poly_user 的數值取出 (原本為 data.frame 形式)
Y_poly_user_hour = Y_poly_user.iat[0, 0]

# 計算「平均每年失效時間」
fail_mean = (Y_poly_user_hour / input_year)

# 印出使用者輸入年份的預測結果
print("您的設備預計總失效時間 = {:.4f} 小時".format(Y_poly_user_hour))
print("平均每年失效時間 = {:.4f} 小時/年".format(fail_mean))


# In[] 模型視覺化
import HappyML.model_drawer as md

md.sample_model(sample_data = (X, Y), model_data = (X, Y_poly))

# 印出最佳 degree 及 RMSE 
print("Degree = {}  RMSE = {:.4f}".format(best_degree, rmse_poly))





