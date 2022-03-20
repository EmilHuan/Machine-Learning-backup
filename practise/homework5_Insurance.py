# -*- coding: utf-8 -*-
"""
Created on Tue May  4 23:14:53 2021

@author: s93yo
"""
# 作業五：保險資料集 鄭桓安
# In[] Preprocessing
import HappyML.preprocessor as pp

# Load Dataset
dataset = pp.dataset(file = "Insurance.csv")

# 切分自變數、應變數
X, Y = pp.decomposition(dataset, x_columns = [0, 1, 2, 3, 4, 5], y_columns = [6])


# 檢查 & 填補缺失值 （經測試無缺失值）
#X.isnull().sum()
# X = pp.missing_data(X, strategy = "mean")

# 類別資料數位化 + 移除 Dummy Variables Trap
X = pp.onehot_encoder(X, columns = [1, 4, 5], remove_trap = True)

# 切分訓練集、測試集
X_train, X_test, Y_train, Y_test = pp.split_train_test(X, Y, train_size = 0.75, random_state = None)


# 特徵縮放 (檢驗前提用)
X_train, X_test = pp.feature_scaling(fit_ary = X_train, transform_arys = (X_train, X_test))
Y_train, Y_test = pp.feature_scaling(fit_ary = Y_train, transform_arys = (Y_train, Y_test))


# In[] 建立多元線性迴歸
# 反向淘汰法 「快樂版」
from HappyML.regression import MultipleRegressor

# 加上多元線性迴歸需要的常數項 (要在特徵縮放後加入)
X_train = pp.add_constant(X_train)
X_test = pp.add_constant(X_test)

regressor = MultipleRegressor()
select_features = regressor.backward_elimination(x_train = X_train, y_train = Y_train, verbose = False)

Y_predict = regressor.fit(x_train=X_train.iloc[:, select_features], y_train = Y_train).predict(x_test = X_test.iloc[:, select_features])

# 印出反向淘汰後留下的特徵欄位數字
print("*** FINAL FEATURES: {}".format(select_features))
# 印出模型的 Adjusted R2)
print("Goodness of Model (Adjusted R-Squared Score):", regressor.r_score())

# In[] 檢查選中的 "自變數" 是否符合五大線性迴歸前提
from HappyML.criteria import AssumptionChecker

# Y_predict = Y_predict.to_frame()


# X_train, X_test = pp.feature_scaling(fit_ary = X_train, transform_arys = (X_train, X_test))

# Y_train, Y_test, Y_predict = pp.feature_scaling(fit_ary = Y_train, transform_arys = (Y_train, Y_test, Y_predict))

# 檢驗線性迴歸成立前提
checker = AssumptionChecker(X_train, X_test, Y_train, Y_test, Y_predict)

# 設定 殘差等分散性 繪圖殘差顯示範圍
checker.y_lim = (-4, 4)
# 設定 「自變數無共線性」 檢驗顯示相關矩陣圖
checker.heatmap = True
# 一次檢查五大前提
checker.check_all()




