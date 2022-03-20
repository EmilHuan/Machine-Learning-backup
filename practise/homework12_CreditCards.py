# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 11:00:57 2021

@author: s93yo
"""
# 作業十二：信用卡客戶集群分析
# In[] 資料前處理
import HappyML.preprocessor as pp

dataset = pp.dataset("CreditCards.csv")

X = pp.decomposition(dataset, x_columns = [i for i in range(1, 18)])

# 檢查有無缺失資料
dataset.isnull().sum()

# 填補缺失資料
X = pp.missing_data(X, strategy = "mean")

# 特徵縮放
X = pp.feature_scaling(fit_ary = X, transform_arys = X)

# Feature Selection (PCA)
from HappyML.preprocessor import PCASelector

selector = PCASelector(best_k = 2)
X = selector.fit(x_ary = X, verbose = False, plot = False).transform(x_ary = X)


# In[] K-mean 集群
from HappyML.clustering import KMeansCluster

cluster = KMeansCluster()

Y_pred = cluster.fit(x_ary = X, verbose = False, plot = False).predict(x_ary = X, y_column = "Customer Type")

# 將 Y_pred 結果合併到 dataset
dataset = pp.combine(dataset, Y_pred)

# In[] Visualization (With HappyML's Class)
import HappyML.model_drawer as md

md.cluster_drawer(x = X, y = Y_pred, centroids = cluster.centroids, title = "Customers Segmentation")


# In[] 附加集群結果的資料集做資料前處理
X_clu, Y_clu = pp.decomposition(dataset, x_columns = [i for i in range(1, 18)], y_columns = [18])

# 填補缺失值
X_clu = pp.missing_data(X_clu, strategy = "mean")

# Dummy Variables
Y_clu, Y_clu_mapping = pp.label_encoder(Y_clu, mapping = True)

# Feature Selection (KBest)
from HappyML.preprocessor import KBestSelector

selector_KB = KBestSelector(best_k = "auto")
X_clu = selector_KB.fit(x_ary = X_clu, y_ary = Y_clu, verbose = False, sort = False).transform(x_ary = X_clu)

# Feature scaling
X_clu = pp.feature_scaling(fit_ary = X_clu, transform_arys = X_clu)

# Split Training / Testing Set
X_clu_train, X_clu_test, Y_clu_train, Y_clu_test = pp.split_train_test(x_ary = X_clu, y_ary = Y_clu)

# In[] 使用「決策樹」執行「監督式學習」
from HappyML.classification import DecisionTree

classifier = DecisionTree()
Y_pred = classifier.fit(X_clu_train, Y_clu_train).predict(X_clu_test)

# K-fold 交叉驗證 10 次
from HappyML.performance import KFoldClassificationPerformance

K = 10
kfp = KFoldClassificationPerformance(x_ary = X_clu, y_ary = Y_clu, classifier = classifier.classifier, k_fold = K)

print("----- Decision Tree Classification -----")
print("Accuracy: {:.2%}".format(kfp.accuracy()))
print("Recall: {:.2%}".format(kfp.recall()))
print("Precision: {:.2%}".format(kfp.precision()))
print("F1-score: {:.2%}".format(kfp.f_score()))



# 決策數繪圖 「快樂版」
import HappyML.model_drawer as md
from IPython.display import Image, display

GRAPHVIZ_INSTALL = "C:/Program Files (x86)/Graphviz/bin"

# 將 Y 名稱串列命名為 class_names
cls_name_int = [Y_clu_mapping[key] for key in sorted(Y_clu_mapping.keys())]
# 生成一個新串列，將原本串列裡面的數字轉換為文字 (才能用於 target_names)
cls_name_str = [str(i) for i in cls_name_int]

# 生成點陣圖
graph = md.tree_drawer(classifier = classifier.classifier, feature_names = X_clu_test.columns, target_names = cls_name_str, graphviz_bin = GRAPHVIZ_INSTALL)
# 點陣圖轉換為 PNG
graph_final = display(Image(graph.create_png()))
# 輸出圖片到當前資料夾
graph.write_png("信用卡客戶集群_decisionTree.png")

print("最重要的決定因素為 PURCHASES_FREQUENCY (購物頻率)")



