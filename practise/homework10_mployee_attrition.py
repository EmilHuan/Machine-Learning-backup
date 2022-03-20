# -*- coding: utf-8 -*-
"""
Created on Sat May 22 14:04:23 2021

@author: s93yo
"""
# 作業十：員工離職原因調查 鄭桓安
# In[] 資料前處理
import HappyML.preprocessor as pp

dataset = pp.dataset("HR-Employee-Attrition.csv")

X, Y = pp.decomposition(dataset, x_columns = [i for i in range(35) if i != 1], y_columns = [1])

# Dummy Variables
X = pp.onehot_encoder(X, columns=[1, 3, 6, 10, 14, 16, 20, 21], remove_trap = True)
Y, Y_mapping = pp.label_encoder(Y, mapping = True)

# 特徵選擇
from HappyML.preprocessor import KBestSelector

selector = KBestSelector(best_k = "auto")
X = selector.fit(x_ary = X, y_ary = Y, verbose = True, sort = True).transform(x_ary = X)

# 切分訓練集、測試集
X_train, X_test, Y_train, Y_test = pp.split_train_test(x_ary = X, y_ary = Y) 

# 特徵縮放
X_train, X_test = pp.feature_scaling(fit_ary = X_train, transform_arys = (X_train, X_test))


# In[] 決策樹「快樂版」
from HappyML.classification import DecisionTree

classifier = DecisionTree()
Y_pred = classifier.fit(X_train, Y_train).predict(X_test)


# In[] 決策樹 K-fold 效能檢測
from HappyML.performance import KFoldClassificationPerformance

K = 10
kfp = KFoldClassificationPerformance(x_ary = X, y_ary = Y, classifier = classifier.classifier, k_fold = K)

print("----- Decision Tree Classification -----")
print("{} Folds Mean Accuracy: {}".format(K, kfp.accuracy()))
print("{} Folds Mean Recall: {}".format(K, kfp.recall()))
print("{} Folds Mean Precision: {}".format(K, kfp.precision()))
print("{} Folds Mean F1-Score: {}".format(K, kfp.f_score()))


# In[] Visualization 決策樹視覺化並輸出圖片
GRAPHVIZ_INSTALL = "C:/Program Files (x86)/Graphviz/bin"

# 決策數繪圖 「快樂版」
import HappyML.model_drawer as md
from IPython.display import Image, display

# 將 Y 應變數文字取出為串列
cls_name = [Y_mapping[key] for key in sorted(Y_mapping.keys())]

# 生成點陣圖
graph = md.tree_drawer(classifier = classifier.classifier, feature_names = X_test.columns, target_names = cls_name, graphviz_bin = GRAPHVIZ_INSTALL)
# 點陣圖轉換為 PNG
graph_final = display(Image(graph.create_png()))
# 輸出圖片到當前資料夾
# graph.write_png("員工離職原因_decisionTree.png")

print("員工留／離職最主要的一個原因 ： OverTime (是否經常性加班)")



# In[] Visualization 決策樹繪圖 「標準版」
GRAPHVIZ_INSTALL = "C:/Program Files (x86)/Graphviz/bin"

# 決策數繪圖 「標準版」
from sklearn import tree
import pydotplus
from IPython.display import Image, display
import os

# 保險用，避免出現錯誤訊息 （沒有這行還是可以跑）
os.environ["PATH"] += os.pathsep + GRAPHVIZ_INSTALL


dot_data = tree.export_graphviz(classifier.classifier, filled = True, feature_names = X_test.columns, class_names = cls_name, rounded = True, special_characters = True)

pydot_graph = pydotplus.graph_from_dot_data(dot_data)
pydot_graph.set_size('"1000000,1000000!"')
pydot_graph.write_png('resized_tree_100萬.png')






