# -*- coding: utf-8 -*-
"""
Created on Wed May 12 21:01:07 2021

@author: s93yo
"""
# 決策樹 課堂練習
# In[] Preprocessing
import HappyML.preprocessor as pp

# Load Data
dataset = pp.dataset(file="Mushrooms.csv")

# Decomposition
X, Y = pp.decomposition(dataset, x_columns=[i for i in range(1, 23)], y_columns=[0])

# Dummy Variables
X = pp.onehot_encoder(X, columns=[i for i in range(22)], remove_trap=True)
Y, Y_mapping = pp.label_encoder(Y, mapping=True)

# Feature Selection
from HappyML.preprocessor import KBestSelector
selector = KBestSelector(best_k="auto")
X = selector.fit(x_ary=X, y_ary=Y, verbose=True, sort=True).transform(x_ary=X)

# Split Training / TEsting Set
X_train, X_test, Y_train, Y_test = pp.split_train_test(x_ary=X, y_ary=Y)

# In[] Decision Tree

# Without HappyML's Class
# from sklearn.tree import DecisionTreeClassifier
# import time

# classifier = DecisionTreeClassifier(criterion="entropy", random_state=int(time.time()))
# classifier.fit(X_train, Y_train)
# Y_pred = classifier.predict(X_test)

# With HappyML's Class
from HappyML.classification import DecisionTree

classifier = DecisionTree()
Y_pred = classifier.fit(X_train, Y_train).predict(X_test)

# In[] Performance
from HappyML.performance import KFoldClassificationPerformance

K = 10
kfp = KFoldClassificationPerformance(x_ary=X, y_ary=Y, classifier=classifier.classifier, k_fold=K)

print("----- Decision Tree Classification -----")
print("{} Folds Mean Accuracy: {}".format(K, kfp.accuracy()))
print("{} Folds Mean Recall: {}".format(K, kfp.recall()))
print("{} Folds Mean Precision: {}".format(K, kfp.precision()))
print("{} Folds Mean F1-Score: {}".format(K, kfp.f_score()))


# In[] Visualization
GRAPHVIZ_INSTALL = "C:/Program Files (x86)/Graphviz/bin"

# # 決策數繪圖 「標準版」
# from sklearn import tree
# import pydotplus
# from IPython.display import Image, display
# import os

# # 保險用，避免出現錯誤訊息 （沒有這行還是可以跑）
# os.environ["PATH"] += os.pathsep + GRAPHVIZ_INSTALL

# cls_name = [Y_mapping[key] for key in sorted(Y_mapping.keys())]
# dot_data = tree.export_graphviz(classifier.classifier, filled=True, feature_names=X_test.columns, class_names=cls_name, rounded=True, special_characters=True)
# graph = pydotplus.graph_from_dot_data(dot_data)
# display(Image(graph.create_png()))


# 決策數繪圖 「快樂版」
import HappyML.model_drawer as md
from IPython.display import Image, display

cls_name = [Y_mapping[key] for key in sorted(Y_mapping.keys())]
graph = md.tree_drawer(classifier=classifier.classifier, feature_names=X_test.columns, target_names=cls_name, graphviz_bin=GRAPHVIZ_INSTALL)
display(Image(graph.create_png()))

















