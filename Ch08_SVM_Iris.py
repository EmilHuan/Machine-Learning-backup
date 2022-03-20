# -*- coding: utf-8 -*-
"""
Created on Wed May 12 17:59:42 2021

@author: s93yo
"""
# 支援向量機 課堂練習
# In[] 資料前處理
from sklearn.datasets import load_iris
# 載入「鳶尾花（Iris）」資料集
dataset = load_iris()

import pandas as pd
X = pd.DataFrame(dataset.data, columns = dataset.feature_names)
Y = pd.DataFrame(dataset.target, columns = ["Iris_Type"])
# 將 target 名稱放在一個 list
Y_name = dataset.target_names.tolist()

# Load HappyML
from HappyML.preprocessor import KBestSelector
import HappyML.preprocessor as pp

# Feature Selection
selector = KBestSelector(best_k = 2)
X = selector.fit(x_ary = X, y_ary = Y, verbose = False, sort = True).transform(x_ary = X)

# Split Training / TEsting Set
X_train, X_test, Y_train, Y_test = pp.split_train_test(x_ary=X, y_ary=Y)

# Feature Scaling
X_train, X_test = pp.feature_scaling(fit_ary=X_train, transform_arys=(X_train, X_test))


# In[] 跟單純貝氏做比較
from HappyML.classification import NaiveBayesClassifier

clr_bayes = NaiveBayesClassifier()
Y_pred_bayes = clr_bayes.fit(X_train, Y_train).predict(X_test)

# Performance
from HappyML.performance import KFoldClassificationPerformance

K = 10
kfp = KFoldClassificationPerformance(x_ary = X, y_ary = Y, classifier = clr_bayes.classifier, k_fold=K)

print("----- Naive Bayes Classification -----")
print("{} Folds Mean Accuracy: {}".format(K, kfp.accuracy()))
print("{} Folds Mean Recall: {}".format(K, kfp.recall()))
print("{} Folds Mean Precision: {}".format(K, kfp.precision()))
print("{} Folds Mean F1-Score: {}".format(K, kfp.f_score()))

# Visualization
import HappyML.model_drawer as md

md.classify_result(x=X_train, y=Y_train, classifier=clr_bayes.classifier, fg_color=("orange", "blue", "white"), bg_color=("red", "green", "black"), title="訓練集 vs. 貝氏模型", font="Microsoft JhengHei")
md.classify_result(x=X_test, y=Y_test, classifier=clr_bayes.classifier, fg_color=("orange", "blue", "white"), bg_color=("red", "green", "black"), title="測試集 vs. 貝氏模型", font="Microsoft JhengHei")


# In[] 支援向量機 「標準版」
# from sklearn.svm import SVC
# import time

# classifier = SVC()


# In[] SVM with HappyML & Gridsearch

# Parameters --------------------
import numpy as np

# Ranges of Hyper Parameters
C_range = np.logspace(3, 6, 4) # Create [1000, 10000, 100000, 1000000]
Gamma_range = np.logspace(-4, -1, 4) # Create [0.001, 0.01, 0.1, 1]
Coef0_range = np.logspace(0, 3, 4) # Create [1, 10, 100, 1000]

# Combination of Hyper Parameters
Linear_dict = dict(kernel=["linear"], C=C_range, coef0=Coef0_range)
RBF_dict = dict(kernel=["rbf"], C=C_range, gamma=Gamma_range)
Sigmoid_dict = dict(kernel=["sigmoid"], C=C_range, gamma=Gamma_range, coef0=Coef0_range)

# Collect all Combinations for Grid Search
params_list = [Linear_dict, RBF_dict, Sigmoid_dict]

# GridSearch --------------------
from HappyML.classification import SVM
classifier = SVM()

# # GridSearch without HappyML (標準版 網格搜尋)
# from sklearn.model_selection import GridSearchCV

# grid_search = GridSearchCV(estimator=classifier.classifier, param_grid=params_list, verbose=10, cv=10)
# grid_search.fit(X, Y.values.ravel())

# print("Best Parameters: {}  Best Score: {}".format(grid_search.best_params_, grid_search.best_score_))
# classifier.classifier = grid_search.best_estimator_

# GridSearch with HappyML (快樂版 網格搜尋)
from HappyML.performance import GridSearch

validator = GridSearch(estimator = classifier.classifier, parameters = params_list, verbose = True)
validator.fit(x_ary = X, y_ary = Y)

print("Best Parameters: {}  Best Score: {}".format(validator.best_parameters, validator.best_score))
classifier.classifier = validator.best_estimator

# Train & Predict
Y_pred_svm = classifier.fit(X_train, Y_train).predict(X_test)



# In[] Performance (Note: Never try kernel = "linear" of SVC() for K-fold, it will hang!!)

# Performance k-fold 10 次驗證結果
K = 10
kfp = KFoldClassificationPerformance(x_ary=X, y_ary=Y, classifier=classifier.classifier, k_fold=K)

print("----- SVM Classification -----")
print("{} Folds Mean Accuracy: {}".format(K, kfp.accuracy()))
print("{} Folds Mean Recall: {}".format(K, kfp.recall()))
print("{} Folds Mean Precision: {}".format(K, kfp.precision()))
print("{} Folds Mean F1-Score: {}".format(K, kfp.f_score()))

# Visualization 支援向量結果視覺化
import HappyML.model_drawer as md

md.classify_result(x=X_train, y=Y_train, classifier=classifier.classifier, fg_color=("orange", "blue", "white"), bg_color=("red", "green", "black"), title="訓練集 vs. SVM 模型", font="Microsoft JhengHei")
md.classify_result(x=X_test, y=Y_test, classifier=classifier.classifier, fg_color=("orange", "blue", "white"), bg_color=("red", "green", "black"), title="測試集 vs. SVM 模型", font="Microsoft JhengHei")












