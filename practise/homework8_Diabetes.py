# -*- coding: utf-8 -*-
"""
Created on Mon May 10 14:36:09 2021

@author: s93yo
"""
# 作業八：糖尿病資料庫 鄭桓安
# In[] 資料前處理
import HappyML.preprocessor as pp
# 載入糖尿病資料
dataset = pp.dataset(file = "Diabetes.csv")

# X, Y decomposition
X, Y = pp.decomposition(dataset, x_columns=[i for i in range(8)], y_columns=[8])


# Feature Selection 特徵篩選 (自動選取)
from HappyML.preprocessor import KBestSelector
selector = KBestSelector()
X = selector.fit(x_ary = X, y_ary = Y, auto = True, verbose = True, sort = True).transform(x_ary = X)


# Split Training / TEsting Set
X_train, X_test, Y_train, Y_test = pp.split_train_test(x_ary = X, y_ary = Y)

# Feature Scaling
X_train, X_test = pp.feature_scaling(fit_ary = X_train, transform_arys=(X_train, X_test))


# In[] Training & Testing using HappyML's Class 快樂版
from HappyML.classification import NaiveBayesClassifier

classifier = NaiveBayesClassifier()
Y_pred = classifier.fit(X_train, Y_train).predict(X_test)

# In[] K-Fold Validation　10　次交叉驗證檢驗模型 （單純貝氏分類）
from HappyML.performance import KFoldClassificationPerformance

K = 10
kfp1 = KFoldClassificationPerformance(x_ary = X, y_ary = Y, classifier = classifier.classifier, k_fold = K, verbose = False)

print("{} Folds Mean Accuracy: {}".format(K, kfp1.accuracy()))
print("{} Folds Mean Recall: {}".format(K, kfp1.recall()))
print("{} Folds Mean Precision: {}".format(K, kfp1.precision()))
print("{} Folds Mean F1-Score: {}".format(K, kfp1.f_score()))

# In[] 檢查模型是否符合「單純貝氏」分類器的前提 （自變數獨立）
from HappyML.criteria import AssumptionChecker

# 輸入檢查數據
checker = AssumptionChecker(x_train = X_train, x_test = X_test, y_train = Y_train, y_test = Y_test, y_pred = Y_pred)
# 輸出檢查結果 （設定不繪圖）
checker.features_correlation(heatmap = False)



# In[] 只選兩個特徵的模型
# 載入糖尿病資料
dataset = pp.dataset(file = "Diabetes.csv")

# X, Y decomposition
X, Y = pp.decomposition(dataset, x_columns=[i for i in range(8)], y_columns=[8])


# Feature Selection 特徵選擇 (設定只選兩個特徵)
from HappyML.preprocessor import KBestSelector
selector = KBestSelector(best_k = 2)
X = selector.fit(x_ary = X, y_ary = Y, auto = True, verbose = False, sort = True).transform(x_ary = X)


# Split Training / TEsting Set
X_train, X_test, Y_train, Y_test = pp.split_train_test(x_ary = X, y_ary = Y)

# Feature Scaling
X_train, X_test = pp.feature_scaling(fit_ary = X_train, transform_arys=(X_train, X_test))

# 單純貝氏分類 「快樂版」
from HappyML.classification import NaiveBayesClassifier

classifier = NaiveBayesClassifier()
Y_pred = classifier.fit(X_train, Y_train).predict(X_test)


# In[] 模型結果視覺化
import HappyML.model_drawer as md

md.classify_result(x = X_train, y = Y_train, classifier = classifier, title="訓練集 vs. 模型", font = "Microsoft JhengHei")

md.classify_result(x = X_test, y = Y_test, classifier = classifier, title="測試集 vs. 模型", font = "Microsoft JhengHei")



# In[] 使用邏輯迴歸
# 載入糖尿病資料
dataset = pp.dataset(file = "Diabetes.csv")

# X, Y decomposition
X, Y = pp.decomposition(dataset, x_columns=[i for i in range(8)], y_columns=[8])


# Feature Selection
from HappyML.preprocessor import KBestSelector
selector = KBestSelector()
X = selector.fit(x_ary = X, y_ary = Y, auto = True, verbose = False, sort = True).transform(x_ary = X)


# Split Training / TEsting Set
X_train, X_test, Y_train, Y_test = pp.split_train_test(x_ary = X, y_ary = Y)

# Feature Scaling
X_train, X_test = pp.feature_scaling(fit_ary = X_train, transform_arys=(X_train, X_test))


# In[] 邏輯迴歸 「快樂版」
from HappyML.regression import LogisticRegressor

# Training & Predict (邏輯迴歸預測)
regressor = LogisticRegressor()
Y_predict = regressor.fit(X_train, Y_train).predict(X_test)

# K-Fold Validation　10　次交叉驗證檢驗模型 （邏輯迴歸）
K = 10
kfp2 = KFoldClassificationPerformance(x_ary = X, y_ary = Y, classifier = regressor.regressor, k_fold = K, verbose = False)


# In[] 顯示兩種模型評估結果
print()
print("K-Fold Validation 結果比較：")
print("單純貝氏分類模型 10　次交叉驗證檢驗評估結果")
print("{} Folds Mean Accuracy: {:.2%}".format(K, kfp1.accuracy()))
print("{} Folds Mean Recall: {:.2%}".format(K, kfp1.recall()))
print("{} Folds Mean Precision: {:.2%}".format(K, kfp1.precision()))
print("{} Folds Mean F1-Score: {:.2%}".format(K, kfp1.f_score()))

print()
print("邏輯迴歸模型 10　次交叉驗證檢驗評估結果")
print("{} Folds Mean Accuracy: {:.2%}".format(K, kfp2.accuracy()))
print("{} Folds Mean Recall: {:.2%}".format(K, kfp2.recall()))
print("{} Folds Mean Precision: {:.2%}".format(K, kfp2.precision()))
print("{} Folds Mean F1-Score: {:.2%}".format(K, kfp2.f_score()))

# 比較結果說明
print()
print("結果說明：四種評估數值邏輯迴歸模型皆大於單純貝氏，使用邏輯迴歸模型較好")





