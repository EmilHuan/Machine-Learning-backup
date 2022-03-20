# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 11:00:57 2021

@author: s93yo
"""
# 作業十三：挽留銀行客戶 (神經網路)
# In[] 資料前處理
import HappyML.preprocessor as pp

dataset = pp.dataset("Churn_Modelling.csv")

X, Y = pp.decomposition(dataset, x_columns = [i for i in range(3, 13)], y_columns = [13])

# 檢查有無缺失資料 (檢查後確定無缺失資料)
# dataset.isnull().sum()

# Dummy Variables
X = pp.onehot_encoder(X, columns = [1, 2], remove_trap= True)
#Y = pp.label_encoder(ary = Y)

# 特徵選擇 KBest
from HappyML.preprocessor import KBestSelector
selector = KBestSelector(best_k = "auto")
X = selector.fit(x_ary = X, y_ary = Y, verbose = True, sort = True).transform(x_ary = X)


# # 特徵縮放 Feature Scaling (optional) PCA 用
# X = pp.feature_scaling(fit_ary = X, transform_arys = X)

# # Feature Selection (PCA)
# from HappyML.preprocessor import PCASelector

# selector = PCASelector(best_k = 8)
# X = selector.fit(x_ary = X, verbose = True, plot = True).transform(x_ary = X)

# 切分測試集、訓練集
X_train, X_test, Y_train, Y_test = pp.split_train_test(x_ary = X, y_ary = Y)

# 特徵縮放 Feature Scaling (optional)
X_train, X_test = pp.feature_scaling(fit_ary = X_train, transform_arys = (X_train, X_test))


# In[] Neural Networks (keras)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Initialize the whole Neural Networks 初始化整個神經網路
classifier = Sequential()

# Add the Input & First Hidden Layer 加入輸入層、第一隱藏層
classifier.add(Dense(input_dim = X_train.shape[1], units = 150, kernel_initializer = "random_normal", activation = "relu"))

# Add the Second Hidden Layer 加入第二隱藏層
classifier.add(Dense(units = 10, kernel_initializer = "random_normal", activation = "relu"))

# Add the Output Layer 加入輸出層
classifier.add(Dense(units = 1, kernel_initializer = "random_normal", activation = "sigmoid"))


# Compile the whole Neural Networks 編譯整個神經網路 (文字轉數字)
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# Fit
classifier.fit(x = X_train, y = Y_train, batch_size = 10, epochs = 100)

# Predict
import pandas as pd
Y_pred_flot = classifier.predict(x = X_test)
Y_pred = (Y_pred_flot > 0.5).astype(int)
Y_pred = pd.DataFrame(Y_pred, index = Y_test.index, columns = Y_test.columns)


# In[] Performance
from HappyML.performance import ClassificationPerformance

pfm = ClassificationPerformance(Y_test, Y_pred)

print("Confusion Matrix:\n", pfm.confusion_matrix())
print("Accuracy: {:.2%}".format(pfm.accuracy()))
print("Recall: {:.2%}".format(pfm.recall()))
print("Precision: {:.2%}".format(pfm.precision()))
print("F1-score: {:.2%}".format(pfm.f_score()))









