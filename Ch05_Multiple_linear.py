# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 19:57:01 2021

@author: s93yo
"""

# In[] Pre-processing
import HappyML.preprocessor as pp

# Dataset Loading
startups_data = pp.dataset("50_Startups.csv")

# Independent/Dependent Variables Decomposition
X, Y = pp.decomposition(startups_data, [0, 1, 2, 3], [4])

# Apply One Hot Encoder to Column[3] & Remove Dummy Variable Trap
# X = pp.onehot_encoder(X, columns=[3])
# X = pp.remove_columns(X, [3]) # 去除相關性欄位
X = pp.onehot_encoder(X, columns = [3], remove_trap = True) # remove_trap 去除相關性欄位

# Split Training vs. Testing Set
X_train, X_test, Y_train, Y_test = pp.split_train_test(X, Y, train_size = 0.8)

# Feature Scaling (optional)
# X_train, X_test = pp.feature_scaling(fit_ary = X_train, transform_arys = (X_train, X_test))
# Y_train, Y_test = pp.feature_scaling(fit_ary = Y_train, transform_arys = (Y_train, Y_test))

# In[] 簡單線性迴歸 (作為跟多元線性迴歸的比較)
from HappyML.regression import SimpleRegressor

regressor = SimpleRegressor()
Y_pred_simple = regressor.fit(X_train, Y_train).predict(X_test)
print("R-Squared Score:", regressor.r_score(X_test, Y_test))



# In[] 多元線性迴歸
# 加上常數項 「一般版」
import statsmodels.tools.tools as smtools
# X_train 加上一個 = 1 的常數項
X_train = smtools.add_constant(X_train)
X_test = smtools.add_constant(X_test)
# 加上常數項「快樂版」
# X_train = pp.add_constant(X_train)

# In[] 多元線性迴歸模型見計算
import statsmodels.api as sm

features = [0, 1, 2, 3, 4, 5]
X_opt = X_train.iloc[:, features]

regressor_OLS = sm.OLS(exog = X_opt, endog = Y_train).fit()
regressor_OLS.summary()


# In[] 反向淘汰法 「一般版」
# 反向淘汰法最終結果
# features = [0, 1, 3]
# X_opt = X_train.iloc[:, features]

# regressor_OLS = sm.OLS(exog = X_opt, endog = Y_train).fit()
# regressor_OLS.summary()

# In[] 測試集加上常數像去做預測
X_test = smtools.add_constant(X_test)
# X_test = pp.add_constant(X_test) 標準版

# Use selected features to predict (測試集預測)
X_opttest = X_test.iloc[:, features]
Y_predmulti = regressor_OLS.predict(X_opttest)

# In[] 反向淘汰法 「快樂版」
from HappyML.regression import MultipleRegressor

regressor = MultipleRegressor()
select_features = regressor.backward_elimination(x_train = X_train, y_train = Y_train, verbose = True)

Y_predict = regressor.fit(x_train=X_train.iloc[:, select_features], y_train=Y_train).predict(x_test=X_test.iloc[:, select_features])

print("Goodness of Model (Adjusted R-Squared Score):", regressor.r_score())

# In[] 使用「均方差」（Mean Squared Error, MSE）比較兩個模型的好壞。MSE 小者勝
# 均方差定義 = SUM(Yi-Yh)^2/n
# 亦有人將「均方差」再開根號，得到「均方差根」（Root of MSE, RMSE）後，才來比較
from HappyML.performance import rmse

rmse_linear = rmse(Y_test, Y_pred_simple)
rmse_multi = rmse(Y_test, Y_predict)

if rmse_linear < rmse_multi:
    print("RMSE Linear:{:.4f} < RMSE Multi-Linear:{:.4f}...Linear smaller, WIN!!".format(rmse_linear, rmse_multi))
else:
    print("RMSE Linear:{:.4f} > RMSE Multi-Linear:{:.4f}...Multi-Linear smaller, WIN!!".format(rmse_linear, rmse_multi))





