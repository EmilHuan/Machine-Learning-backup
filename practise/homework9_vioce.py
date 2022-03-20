# -*- coding: utf-8 -*-
"""
Created on Fri May 21 22:13:54 2021

@author: s93yo
"""

# 作業九：辨別「男聲」或「女聲」 鄭桓安
# In[] 資料前處理
import HappyML.preprocessor as pp

dataset = pp.dataset("Voice.csv")

X, Y = pp.decomposition(dataset, x_columns = [i for i in range(20)], y_columns = [20])

print("未調整超參數之前")

# Feature Selection
from HappyML.preprocessor import KBestSelector
selector = KBestSelector(best_k = "auto")
X = selector.fit(x_ary = X, y_ary = Y, verbose = True, sort = True).transform(x_ary = X)

# 應變數做 label encoder
Y, Y_mapping = pp.label_encoder(Y, mapping = True)

# 切分訓練集、測試集
X_train, X_test, Y_train, Y_test = pp.split_train_test(x_ary = X, y_ary = Y)

# 特徵縮放
X_train, X_test = pp.feature_scaling(fit_ary = X_train, transform_arys=(X_train, X_test))



# In[] 使用 SVM 預設超參數
from HappyML.classification import SVM

classifier = SVM()

Y_pred_svm_default = classifier.fit(X_train, Y_train).predict(X_test)


# K-fold 交叉驗證模式效能
from HappyML.performance import KFoldClassificationPerformance

K = 10
kfp1 = KFoldClassificationPerformance(x_ary = X, y_ary = Y, classifier = classifier.classifier, k_fold = K)

print("----- SVM Classification -----")
print("{} Folds Mean Accuracy: {}".format(K, kfp1.accuracy()))
print("{} Folds Mean Recall: {}".format(K, kfp1.recall()))
print("{} Folds Mean Precision: {}".format(K, kfp1.precision()))
print("{} Folds Mean F1-Score: {}".format(K, kfp1.f_score()))



# In[] 手動調整超參數
import numpy as np

# 設定超參數範圍
C_range = np.logspace(3, 6, 4) # [1000, 10000, 100000, 1000000]
Gamma_range = np.logspace(-4, -1, 4) # [0.0001, 0.0001, 0.001, 0.1]
Coef0_range = np.logspace(0, 3, 4) # [1, 10, 100, 1000]


# 超參數各自合併為一個字典
Linear_dict = dict(kernel = ["linear"], C = C_range, coef0 = Coef0_range)
RBF_dict = dict(kernel = ["rbf"], C = C_range, gamma = Gamma_range)
Sigmoid_dict = dict(kernel = ["sigmoid"], C = C_range, gamma = Gamma_range, coef0 = Coef0_range)
# 所有字典合併為串列
params_list = [Linear_dict, RBF_dict, Sigmoid_dict]


# GridSearch with HappyML (快樂版 網格搜尋)
from HappyML.performance import GridSearch

validator = GridSearch(estimator = classifier.classifier, parameters = params_list, verbose = False)
validator.fit(x_ary = X, y_ary = Y)

print()
print("已調整超參數之後")
print("Best Parameters: {}  Best Score: {}".format(validator.best_parameters, validator.best_score))
classifier.classifier = validator.best_estimator

# Train & Predict
Y_pred_svm_adj = classifier.fit(X_train, Y_train).predict(X_test)


# # K-fold 交叉驗證模式效能
from HappyML.performance import KFoldClassificationPerformance

K = 10
kfp2 = KFoldClassificationPerformance(x_ary = X, y_ary = Y, classifier = classifier.classifier, k_fold = K)

print("----- SVM Classification -----")
print("{} Folds Mean Accuracy: {}".format(K, kfp2.accuracy()))
print("{} Folds Mean Recall: {}".format(K, kfp2.recall()))
print("{} Folds Mean Precision: {}".format(K, kfp2.precision()))
print("{} Folds Mean F1-Score: {}".format(K, kfp2.f_score()))


# 網格搜尋調整超參數結果
# Best Parameters: {'C': 1000000.0, 'gamma': 0.0001, 'kernel': 'rbf'}
# ----- SVM Classification -----
# 10 Folds Mean Accuracy: 0.955807011939464
# 10 Folds Mean Recall: 0.9557837751771355
# 10 Folds Mean Precision: 0.957667986024005
# 10 Folds Mean F1-Score: 0.9557316897211662


