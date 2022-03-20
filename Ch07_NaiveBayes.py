# -*- coding: utf-8 -*-
"""
Created on Wed May  5 20:23:45 2021

@author: s93yo
"""
#　「貝氏模型」　課堂練習
# In[] Data Preprocessing
import HappyML.preprocessor as pp

# Load Data
dataset = pp.dataset(file = "Social_Network_Ads.csv")

# X, Y decomposition
X, Y = pp.decomposition(dataset, x_columns = [1, 2, 3], y_columns = [4])

# One-Hot Encoder
X = pp.onehot_encoder(ary = X, columns = [0], remove_trap = True)

# Feature Selection 特徵選擇
from HappyML.preprocessor import KBestSelector
selector = KBestSelector()
X = selector.fit(x_ary = X, y_ary = Y, auto = True, verbose = True, sort = True).transform(x_ary = X)

# Split Training / TEsting Set
X_train, X_test, Y_train, Y_test = pp.split_train_test(x_ary = X, y_ary = Y)

# Feature Scaling
X_train, X_test = pp.feature_scaling(fit_ary = X_train, transform_arys = (X_train, X_test))


# # In[] 貝氏模型 「標準版」
# from sklearn.naive_bayes import GaussianNB

# classifier = GaussianNB()
# classifier.fit(X_train, Y_train.values.ravel())

# Y_pred = classifier.predict(X_test)


# In[] Training & Testing using HappyML's Class 快樂版」
from HappyML.classification import NaiveBayesClassifier

classifier = NaiveBayesClassifier()
Y_pred = classifier.fit(X_train, Y_train).predict(X_test)


# # In[] Performance 貝氏模型效能評估 (評估方式跟邏輯迴歸相同)
# from HappyML.performance import ClassificationPerformance

# pfm = ClassificationPerformance(Y_test, Y_pred)

# print("Confusion Matrix:\n", pfm.confusion_matrix())
# print("Accuracy: {:.2%}".format(pfm.accuracy()))
# print("Recall: {:.2%}".format(pfm.recall()))
# print("Precision: {:.2%}".format(pfm.precision()))
# print("F1-score: {:.2%}".format(pfm.f_score()))


# In[] K-Fold Validation　10　次交叉驗證檢驗模型
from HappyML.performance import KFoldClassificationPerformance

K = 10
kfp = KFoldClassificationPerformance(x_ary=X, y_ary=Y, classifier=classifier.classifier, k_fold=K, verbose=False)

print("{} Folds Mean Accuracy: {}".format(K, kfp.accuracy()))
print("{} Folds Mean Recall: {}".format(K, kfp.recall()))
print("{} Folds Mean Precision: {}".format(K, kfp.precision()))
print("{} Folds Mean F1-Score: {}".format(K, kfp.f_score()))




















