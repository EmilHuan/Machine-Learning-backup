# -*- coding: utf-8 -*-
"""
Created on Wed May 26 18:14:51 2021

@author: s93yo
"""
# K-Means Clustering 課堂練習
# In[] 資料前處理
import HappyML.preprocessor as pp

dataset = pp.dataset("Mall_Customers.csv")

X = pp.decomposition(dataset, x_columns=[1, 2, 3, 4])

# One-Hot Encoding
X = pp.onehot_encoder(ary = X, columns = [0], remove_trap = True)

# Feature Scaling (for PCA Feature Selection)
X = pp.feature_scaling(fit_ary = X, transform_arys = X)

# Feature Selection (PCA)
from HappyML.preprocessor import PCASelector

selector = PCASelector()
X = selector.fit(x_ary = X, verbose = True, plot = True).transform(x_ary = X)


# In[] K-Means Clustering 「標準版」 （假設知道 K = 4）ㄋ
# from sklearn.cluster import KMeans
# import time

# # 產生物件本身
# kmeans = KMeans(n_clusters = 4, init = "k-means++", random_state = int(time.time()))
# # 訓練 & 預測
# Y_pred = kmeans.fit_predict(X)

# In[] K-Means Clustering with Visual Clusters, which is also = 4 (標準版)
# from sklearn.cluster import KMeans
# import time

# # Find Best K
# # 儲存 1~10 的 WCSS
# wcss = []

# for i in range(1, 11):
#     # 擬合這次 K = i 的結果
#     kmeans = KMeans(n_clusters=i, init="k-means++", random_state=int(time.time()))
#     kmeans.fit(X)
#     # 將這次的 WCSS 存起來
#     wcss.append(kmeans.inertia_)

# # Draw WCSS for each K
# import matplotlib.pyplot as plt
# # 繪製所有的 WCSS 值
# plt.plot(range(1, 11), wcss)
# plt.title("The Best K")
# plt.xlabel("# of Clusters")
# plt.ylabel("WCSS")
# plt.show()

# # Clustering with Visual K, which is also = 4
# kmeans = KMeans(n_clusters = 4, init = "k-means++", random_state = int(time.time()))
# Y_pred = kmeans.fit_predict(X)


# In[] K-Means Clustering (With HappyML's Class) 「快樂版」
from HappyML.clustering import KMeansCluster

cluster = KMeansCluster()
Y_pred = cluster.fit(x_ary = X, verbose = True, plot = True).predict(x_ary = X, y_column = "Customer Type")

# Optional, Attach the Y_pred to Dataset & Save as .CSV file
# 把 Y_pred 結果附加到 dataset
dataset = pp.combine(dataset, Y_pred)
# 匯出有結果的 dataset 為 csv file
dataset.to_csv("Mall_Customers_answers.csv")

# In[] Visualization (With HappyML's Class) 「快樂版」
import HappyML.model_drawer as md

md.cluster_drawer(x = X, y = Y_pred, centroids = cluster.centroids, title = "Customers Segmentation")


















