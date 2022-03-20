# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 19:39:34 2021

@author: s93yo
"""

# In[] Preprocessing
import HappyML.preprocessor as pp

# Load Dataset
dataset = pp.dataset(file = "Position_Salaries.csv")

# Decomposition of Variables (切分自變數、應變數)
X, Y = pp.decomposition(dataset, x_columns = [1], y_columns = [2])

# Training / Testing Set (切分訓練集、測試集)
X_train, X_test, Y_train, Y_test = pp.split_train_test(x_ary = X, y_ary = Y, train_size = 0.8)


# In[] Linear Regression as comparison
from HappyML.regression import SimpleRegressor
import HappyML.model_drawer as md

reg_simple = SimpleRegressor()
Y_simple = reg_simple.fit(x_train = X, y_train = Y).predict(x_test = X)

# md.sample_model(sample_data = (X, Y), model_data = (X, Y_simple))
print("R-Squared of Simple Regression:", reg_simple.r_score(x_test = X, y_test = Y))


# In[] Polynomial Regression (標準版)
# from sklearn.preprocessing import PolynomialFeatures
# from HappyML.performance import rmse
# import pandas as pd

# # 以次方倍增加特徵
# deg = 6
# poly_reg = PolynomialFeatures(degree = deg)
# X_poly = pd.DataFrame(poly_reg.fit_transform(X))

# # 因為自變數資料太少 (只有 10 筆)，所以模型直接用 X_poly，不切分訓練集
# # 快樂版的簡單線性迴歸本質是 OLS，只是沒有反向淘汰法
# regressor = SimpleRegressor()
# regressor.fit(X_poly, Y)
# Y_predict = regressor.predict(x_test=X_poly)

# md.sample_model(sample_data=(X, Y), model_data=(X, Y_predict))
# print("Degree={}  RMSE={:.4f}".format(deg, rmse(Y, Y_predict)))


# In[] Polynomial Regression with HappyML's Class (快樂版)
from HappyML.regression import PolynomialRegressor

reg_poly = PolynomialRegressor()
reg_poly.best_degree(x_train=X_train, y_train=Y_train, x_test=X_test, y_test=Y_test, verbose = True)
Y_poly = reg_poly.fit(x_train=X, y_train=Y).predict(x_test=X)

md.sample_model(sample_data=(X, Y), model_data=(X, Y_poly))


# In[] 5+ year, 要求 180000 月薪，合理嗎？
import math
import pandas as pd

work_years = float(input("請輸入對方的年資："))
request_salary = float(input("請輸入對方要求的月薪："))
salary_pred = reg_poly.predict(pd.DataFrame([[work_years]])).iloc[0, 0]
salary_lower = reg_poly.predict(pd.DataFrame([[math.floor(work_years)]])).iloc[0, 0]
salary_upper = reg_poly.predict(pd.DataFrame([[math.ceil(work_years)]])).iloc[0, 0]
print("合理的月薪推測在 {:.2f} 左右".format(salary_pred))
print("範圍落在 {:.2f}～{:.2f} 之間".format(salary_lower, salary_upper))


# In[] Comparing the two models (比較兩個模型)
from HappyML.performance import rmse

rmse_linear = rmse(Y, Y_simple)
rmse_poly = rmse(Y, Y_poly)

if rmse_linear < rmse_poly:
    print("RMSE Linear:{:.4f} < RMSE Polynomial:{:.4f}...Linear smaller, WIN!!".format(rmse_linear, rmse_poly))
else:
    print("RMSE Linear:{:.4f} > RMSE Polynomial:{:.4f}...Polynomial smaller, WIN!!".format(rmse_linear, rmse_poly))




