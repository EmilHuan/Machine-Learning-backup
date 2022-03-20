# -*- coding: utf-8 -*-
"""
Created on Wed May 19 19:14:09 2021

@author: s93yo
"""
# 「隨機森林」課堂練習
# In[] Preprocessing #1: Without PCA and Boundary Visualization
# 不使用 PCA 前處理
# import HappyML.preprocessor as pp

# # Load Data, also can be loaded by sklearn.datasets.load_wine()
# dataset = pp.dataset(file="Wine.csv")

# # 切分自變數、應變數
# X, Y = pp.decomposition(dataset, x_columns = [i for i in range(13)], y_columns = [13]) 

# # By KBestSelector
# from HappyML.preprocessor import KBestSelector
# selector = KBestSelector(best_k = 2)
# X = selector.fit(x_ary = X, y_ary = Y, verbose = True, sort = True).transform(x_ary = X)

# # Split Training / Testing Set
# X_train, X_test, Y_train, Y_test = pp.split_train_test(x_ary = X, y_ary = Y)

# # # Feature Scaling
# X_train, X_test = pp.feature_scaling(fit_ary=X_train, transform_arys=(X_train, X_test))

# In[] Preprocessing #3: With PCA, and Boundary Visualization
# 使用 PCA 前處理
import HappyML.preprocessor as pp

# Load Data, also can be loaded by sklearn.datasets.load_wine()
dataset = pp.dataset(file="Wine.csv")

# Decomposition
X, Y = pp.decomposition(dataset, x_columns=[i for i in range(13)], y_columns=[13])

# Feature Scaling (最大方差法一定要做)
X = pp.feature_scaling(fit_ary=X, transform_arys=X)

# # PCA without HappyML's Class 「一般版」 PCA
# from sklearn.decomposition import PCA
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

# # 設定 None : 每個特徵的貢獻量
# pca = PCA(n_components = None)
# pca.fit(X)
# info_covered = pca.explained_variance_ratio_
# cumulated_sum = np.cumsum(info_covered)
# plt.plot(cumulated_sum, color="blue")

# pca = PCA(n_components= 2 )
# X_columns = ["PCA_{}".format(i+1) for i in range(2)]
# X = pd.DataFrame(pca.fit_transform(X), index=X.index, columns=X_columns)

# # Split Training / Testing Set
# X_train, X_test, Y_train, Y_test = pp.split_train_test(x_ary=X, y_ary=Y)


# PCA with HappyML's Class 「快樂版」 PCA
from HappyML.preprocessor import PCASelector

selector = PCASelector(best_k = 2)
X = selector.fit(x_ary = X, verbose = True, plot = True).transform(X)

# Split Training / Testing Set
X_train, X_test, Y_train, Y_test = pp.split_train_test(x_ary=X, y_ary=Y)



# In[] Random Forest 隨機森林
# # Without HappyML's Class 一般版 隨機森林 
# from sklearn.ensemble import RandomForestClassifier
# import time

# classifier = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=int(time.time()))
# classifier.fit(X_train, Y_train.values.ravel())
# Y_pred = classifier.predict(X_test)

# With HappyML's Class 快樂版 隨機森林  
from HappyML.classification import RandomForest

classifier = RandomForest(n_estimators = 10, criterion = "entropy")
Y_pred = classifier.fit(X_train, Y_train).predict(X_test)


# In[] Performance 效能測試
from HappyML.performance import KFoldClassificationPerformance

K = 10
kfp = KFoldClassificationPerformance(x_ary = X, y_ary = Y, classifier = classifier.classifier, k_fold = K)

print("----- Random Forest Classification -----")
print("{} Folds Mean Accuracy: {}".format(K, kfp.accuracy()))
print("{} Folds Mean Recall: {}".format(K, kfp.recall()))
print("{} Folds Mean Precision: {}".format(K, kfp.precision()))
print("{} Folds Mean F1-Score: {}".format(K, kfp.f_score()))


# In[] Tree Visualization
GRAPHVIZ_INSTALL = "C:/Program Files (x86)/Graphviz/bin"

import HappyML.model_drawer as md
from IPython.display import Image, display


clfr = classifier.classifier.estimators_[0]
graph = md.tree_drawer(classifier=clfr, feature_names=X_test.columns, target_names="123", graphviz_bin=GRAPHVIZ_INSTALL)
display(Image(graph.create_png()))


# In[] Boundary Visualization
import HappyML.model_drawer as md

md.classify_result(x=X_train, y=Y_train, classifier=classifier.classifier,
                    fg_color=("orange", "blue", "white"), bg_color=("red", "green", "black"),
                    title="訓練集 vs. 隨機森林模型", font="DFKai-sb")
md.classify_result(x=X_test, y=Y_test, classifier=classifier.classifier,
                    fg_color=("orange", "blue", "white"), bg_color=("red", "green", "black"),
                    title="測試集 vs. 隨機森林模型", font="DFKai-sb")






