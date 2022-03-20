# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 11:01:08 2021

@author: s93yo
"""

# 作業四：台灣學童身高、體重評估 鄭桓安
# In[] 詢問使用者輸入數值：
user_gender = eval(input("請輸入您的性別（1.男 2.女）：")) -1
user_age = eval(input("請輸入您的年齡（6-15）："))
user_height = eval(input("請輸入您的身高（cm）："))
user_weight = eval(input("請輸入您的體重（kg）："))


# In[] 資料前處理 (快樂版)
import pandas as pd
import HappyML.preprocessor as pp
from HappyML import model_drawer as md # 模型視覺化

# 匯入資料
Weight_data = pp.dataset("Student_Weight.csv")
Height_data = pp.dataset("Student_Height.csv")

# 切分自變數、應變數
Weight_X, Weight_Y = pp.decomposition(Weight_data, x_columns = [1], y_columns = [3, 4])

Height_X, Height_Y = pp.decomposition(Height_data, x_columns = [1], y_columns = [3, 4])

# 檢查 & 填補缺失值
Weight_X = pp.missing_data(Weight_X, strategy = "mean")
Height_X = pp.missing_data(Height_X, strategy = "mean")

# 切分訓練集 75%, 測試集 25%
Weight_X_trian, Weight_X_test, Weight_Y_train, Weight_Y_test = pp.split_train_test(Weight_X, Weight_Y, train_size = 0.75, random_state = 0)

Height_X_trian, Height_X_test, Height_Y_train, Height_Y_test = pp.split_train_test(Height_X, Height_Y, train_size = 0.75, random_state = 0)


# In[] 簡單線性迴歸
from HappyML.regression import SimpleRegressor
# 建立 4 個 LinearRegression 物件
regressor = [[SimpleRegressor(), SimpleRegressor()], [SimpleRegressor(), SimpleRegressor()]]

# 執行 4 個簡單線性迴歸
# regressor[0][0]:預測男性身高
Height_Y_pred_male = regressor[0][0].fit(Height_X_trian, Height_Y_train.iloc[:,0].to_frame()).predict(Height_X_test)
        
# regressor[1][0]:預測男性體重
Wdight_Y_pred_male = regressor[1][0].fit(Weight_X_trian, Weight_Y_train.iloc[:,0].to_frame()).predict(Weight_X_test)


# regressor[0][1]:預測女性身高
Height_Y_pred_fmale = regressor[0][1].fit(Height_X_trian, Height_Y_train.iloc[:,1].to_frame()).predict(Height_X_test)
    
# regressor[1][1]:預測女性體重
Wdight_Y_pred_fmale = regressor[1][1].fit(Weight_X_trian, Weight_Y_train.iloc[:,1].to_frame()).predict(Weight_X_test)


# In[] 計算使用者性別的平均身高及體重，並將結果四捨五入到小數點下二位
height_avg = regressor[0][user_gender].predict(x_test=pd.DataFrame([[user_age]])).iloc[0, 0]

plot_height_avg = round(height_avg, 2)

wdight_avg = regressor[1][user_gender].predict(x_test=pd.DataFrame([[user_age]])).iloc[0, 0]

plot_wdight_avg = round(wdight_avg, 2)


# In[] 模型視覺化
# 先看使用者性別，再決定要顯示男或女
if  (user_gender + 1) == 1:
    Gander = "男"
else:
    Gander = "女"

# 使用者身高落點繪圖
sample_data = (user_age, user_height)

model_data = (Height_X_trian, regressor[0][user_gender].predict(Height_X_trian))

md.sample_model(sample_data = sample_data, model_data = model_data,
                title = "身高落點分佈", xlabel = "年齡", ylabel = "身高",
                font = "Microsoft JhengHei")

print("{} 歲{}生平均身高為 {} 公分，您的身高為 {} 公分".format(user_age, Gander, plot_height_avg, user_height))


# 使用者體重落點繪圖
sample_data = (user_age, user_weight)

model_data = (Weight_X_trian, regressor[1][user_gender].predict(Weight_X_trian))

md.sample_model(sample_data = sample_data, model_data = model_data,
                title = "體重落點分佈", xlabel = "年齡", ylabel = "體重",
                font = "Microsoft JhengHei")

print("{} 歲{}生平均體重為 {} 公斤，您的體重為 {} 公斤".format(user_age, Gander, plot_wdight_avg, user_weight))

