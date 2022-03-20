# -*- coding: utf-8 -*-
"""
Created on Wed May  5 21:58:54 2021

@author: s93yo
"""
# 作業七：乳癌資料處理＆預測 鄭桓安
# In[] 資料前處理
import pandas as pd
from sklearn.datasets import load_breast_cancer # 乳癌資料套件
import HappyML.preprocessor as pp

# 載入乳癌資料
dataset = load_breast_cancer()

# 切分資料集，X, Y 做成 DataFrame 形式
X = pd.DataFrame(dataset.data, columns = dataset.feature_names)
Y = pd.DataFrame(dataset.target, columns = ["isBreastCancer"])

# Feature Selection 「快樂版」 特徵選擇 (在切分訓練集、測試集前)
from HappyML.preprocessor import KBestSelector

selector = KBestSelector()

X = selector.fit(x_ary = X, y_ary = Y, auto = True, verbose = True, sort = True).transform(x_ary = X)


# Split Training & Testing set (在切分訓練集、測試集)
X_train, X_test, Y_train, Y_test = pp.split_train_test(X, Y, train_size = 0.75)

# Feature Scaling (特徵縮放)
X_train, X_test = pp.feature_scaling(fit_ary = X_train, transform_arys = (X_train, X_test))


# In[] Logistic Regression with HappyML's Class 「快樂版」 邏輯迴歸
from HappyML.regression import LogisticRegressor
from HappyML.performance import ClassificationPerformance

# Training & Predict (邏輯迴歸預測)
regressor1 = LogisticRegressor()
Y_predict = regressor1.fit(X_train, Y_train).predict(X_test)

# Performance (5 種指標一併計算)
pfm = ClassificationPerformance(Y_test, Y_predict)
# 顯示模型評估結果
print("Confusion Matrix:\n", pfm.confusion_matrix())
print("Accuracy: {:.2%}".format(pfm.accuracy()))
print("Recall: {:.2%}".format(pfm.recall()))
print("Precision: {:.2%}".format(pfm.precision()))
print("F1-score: {:.2%}".format(pfm.f_score()))


# In[] 只選兩個特徵的模型
# 載入乳癌資料
#dataset = load_breast_cancer()

# 切分資料集，X, Y 做成 DataFrame 形式
X = pd.DataFrame(dataset.data, columns = dataset.feature_names)
Y = pd.DataFrame(dataset.target, columns = ["isBreastCancer"])

# Feature Selection 「快樂版」 特徵選擇 (在切分訓練集、測試集前)
selector = KBestSelector(best_k = 2)

X = selector.fit(x_ary = X, y_ary = Y, auto = True, verbose = False, sort = True).transform(x_ary = X)


# Split Training & Testing set
X_train, X_test, Y_train, Y_test = pp.split_train_test(X, Y, train_size = 0.75)

# Feature Scaling
X_train, X_test = pp.feature_scaling(fit_ary = X_train, transform_arys = (X_train, X_test))


# Training & Predict (邏輯迴歸預測)
regressor2 = LogisticRegressor()
Y_predict = regressor2.fit(X_train, Y_train).predict(X_test)


# In[] Visualize the Result 結果視覺化
import HappyML.model_drawer as md

md.classify_result(x = X_train, y = Y_train, classifier = regressor2.regressor, 
                   title = "訓練集樣本點 vs. 模型", font = "Microsoft JhengHei")
md.classify_result(x = X_test, y = Y_test, classifier = regressor2.regressor, 
                   title = "測試集樣本點 vs. 模型", font = "Microsoft JhengHei")














