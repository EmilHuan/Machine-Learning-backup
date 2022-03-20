# -*- coding: utf-8 -*-
"""
Created on Wed May 26 10:39:23 2021

@author: s93yo
"""
# 作業十一：動物園分類 鄭桓安
# In[] 資料前處理
import HappyML.preprocessor as pp
# 特徵資料
dataset = pp.dataset("Zoo_Data.csv")
# 分類名稱資料
dataset_classname  = pp.dataset("Zoo_Class_Name.csv")
# 取出 class 名稱及數字
class_names = [row['Class_Type'] for index, row in dataset_classname.iterrows()]

# Decomposition
X, Y = pp.decomposition(dataset, x_columns = [i for i in range(1, 17)], y_columns = [17])

# One-Hot Encoder
X = pp.onehot_encoder(ary=X, columns = [i for i in range(17)], remove_trap = True)

# By KBestSelector
from HappyML.preprocessor import KBestSelector
selector = KBestSelector(best_k = "auto")
X = selector.fit(x_ary = X, y_ary = Y, verbose = True, sort = True).transform(x_ary = X)

# Split Training / Testing Set
X_train, X_test, Y_train, Y_test = pp.split_train_test(x_ary = X, y_ary = Y)

# In[] 隨機森林 with KBestSelector
from HappyML.classification import RandomForest

classifier = RandomForest(n_estimators = 10, criterion = "entropy")
Y_pred = classifier.fit(X_train, Y_train).predict(X_test)

# In[] Performance (隨機森林 with KBestSelector)
from HappyML.performance import KFoldClassificationPerformance

K = 10
kfp1 = KFoldClassificationPerformance(x_ary = X, y_ary = Y, classifier = classifier.classifier, k_fold = K)

print("----- Random Forest Classification with KBest -----")
print("{} Folds Mean Accuracy: {}".format(K, kfp1.accuracy()))
print("{} Folds Mean Recall: {}".format(K, kfp1.recall()))
print("{} Folds Mean Precision: {}".format(K, kfp1.precision()))
print("{} Folds Mean F1-Score: {}".format(K, kfp1.f_score()))


# In[] PCA select feature
# Decomposition
X_PCA, Y_PCA = pp.decomposition(dataset, x_columns = [i for i in range(1, 17)], y_columns = [17])

# One-Hot Encoder
X_PCA = pp.onehot_encoder(ary=X_PCA, columns = [i for i in range(17)], remove_trap = True)

# PCA with HappyML's Class
from HappyML.preprocessor import PCASelector

selector = PCASelector(best_k = "auto")
X_PCA = selector.fit(x_ary = X_PCA, verbose = True, plot = True).transform(X_PCA)

# Split Training / Testing Set
X_PCA_train, X_PCA_test, Y_PCA_train, Y_PCA_test = pp.split_train_test(x_ary = X_PCA, y_ary = Y_PCA)

# In[] 隨機森林 with PCA select feature
from HappyML.classification import RandomForest

classifier_PCA = RandomForest(n_estimators = 10, criterion = "entropy")
Y_PCA_pred = classifier_PCA.fit(X_PCA_train, Y_PCA_train).predict(X_PCA_test)

# In[] Performance (隨機森林 with PCA select feature)
from HappyML.performance import KFoldClassificationPerformance

K = 10
kfp2 = KFoldClassificationPerformance(x_ary = X_PCA, y_ary = Y_PCA, classifier = classifier_PCA.classifier, k_fold = K)

print()
print("----- Random Forest Classification with PCA -----")
print("{} Folds Mean Accuracy: {}".format(K, kfp2.accuracy()))
print("{} Folds Mean Recall: {}".format(K, kfp2.recall()))
print("{} Folds Mean Precision: {}".format(K, kfp2.precision()))
print("{} Folds Mean F1-Score: {}".format(K, kfp2.f_score()))


# In[] Visualization 隨機森林 with KBestSelector 視覺化並輸出圖片
print()
print("K-fold 數值 KBest > PCA，使用 KBest 模型繪製隨機森林其中一棵樹")

GRAPHVIZ_INSTALL = "C:/Program Files (x86)/Graphviz/bin"

# 決策數繪圖 「快樂版」
import HappyML.model_drawer as md
from IPython.display import Image, display

# 將 Y 名稱串列命名為 class_names
cls_name = class_names

# 繪製隨機森林 with KBestSelector 第 1 顆樹
clfr = classifier.classifier.estimators_[0]
# 生成點陣圖
graph = md.tree_drawer(classifier = clfr, feature_names=X_test.columns, target_names=cls_name, graphviz_bin=GRAPHVIZ_INSTALL)
# 點陣圖轉換為 PNG
graph_final = display(Image(graph.create_png()))
# 輸出圖片到當前資料夾
#graph.write_png("動物園分類_RandomForest.png")



