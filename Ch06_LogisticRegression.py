# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 20:51:47 2021

@author: s93yo
"""
# 邏輯迴歸 (Logistic Regression) 課堂練習
# In[] Preprocessing
import HappyML.preprocessor as pp

# Load Dataset
dataset = pp.dataset(file="Social_Network_Ads.csv")

# X, Y Decomposition
X, Y = pp.decomposition(dataset, x_columns = [1, 2, 3], y_columns = [4])

# Categorical Data Encoding & Remove Dummy Variable Trap
X = pp.onehot_encoder(X, columns = [0], remove_trap = True)


# Feature Selection 「快樂版」 特徵選擇 (在切分訓練集、測試集前)
from HappyML.preprocessor import KBestSelector
selector = KBestSelector()
X = selector.fit(x_ary = X, y_ary = Y, auto = True, verbose = True, sort = True).transform(x_ary = X)


# Split Training & Testing set
X_train, X_test, Y_train, Y_test = pp.split_train_test(X, Y)

# Feature Scaling
X_train, X_test = pp.feature_scaling(fit_ary = X_train, transform_arys = (X_train, X_test))


# # In[] Logistic Regression 「一般版」
# from sklearn.linear_model import LogisticRegression
# import time

# # Model Creation
# classifier = LogisticRegression(solver="lbfgs", random_state=int(time.time()))


# # 在模型訓練、測試之間做特徵選擇，提高模型整體準確度 （也有可能降低）
# # Features Selection （一般版）
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2

# kbest = SelectKBest(score_func=chi2, k = 2)
# kbest = kbest.fit(X, Y)
# print("The p-values of Feature Importance: {}".format(kbest.pvalues_))

# X_train = kbest.transform(X_train)
# X_test = kbest.transform(X_test)


# # Training
# classifier.fit(X_train, Y_train.values.ravel())

# # Testing
# Y_pred_logistic = classifier.predict(X_test)

# # In[] 評估模型好壞 （一般版）
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
# from sklearn.metrics import fbeta_score

# print("Confusion Matrix:\n", confusion_matrix(y_true = Y_test, y_pred = Y_pred_logistic))
# print("Accuracy: {:.2%}".format(accuracy_score(Y_test, Y_pred_logistic)))
# print("Recall: {:.2%}".format(recall_score(Y_test, Y_pred_logistic)))
# print("Precision: {:.2%}".format(precision_score(Y_test, Y_pred_logistic)))
# print("F1-score: {:.2%}".format(fbeta_score(Y_test, Y_pred_logistic, beta=1)))


# In[] Logistic Regression with HappyML's Class 「快樂版」 邏輯迴歸
from HappyML.regression import LogisticRegressor
from HappyML.performance import ClassificationPerformance

# Training & Predict (邏輯迴歸預測)
regressor = LogisticRegressor()
Y_predict = regressor.fit(X_train, Y_train).predict(X_test)

# Performance (5 種指標一併計算)
pfm = ClassificationPerformance(Y_test, Y_predict)

print("Confusion Matrix:\n", pfm.confusion_matrix())
print("Accuracy: {:.2%}".format(pfm.accuracy()))
print("Recall: {:.2%}".format(pfm.recall()))
print("Precision: {:.2%}".format(pfm.precision()))
print("F1-score: {:.2%}".format(pfm.f_score()))


# In[] Visualize the Result
import HappyML.model_drawer as md

md.classify_result(x=X_train, y=Y_train, classifier=regressor.regressor, 
                   title="訓練集樣本點 vs. 模型", font="Microsoft JhengHei")
md.classify_result(x=X_test, y=Y_test, classifier=regressor.regressor, 
                   title="測試集樣本點 vs. 模型", font="Microsoft JhengHei")





