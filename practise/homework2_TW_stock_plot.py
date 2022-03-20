# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 11:39:29 2021

@author: s93yo
"""
### "作業二：台灣股票線圖繪製" 鄭桓安
# 導入資料處立套件
import pandas as pd
# 導入繪圖套件
import matplotlib.pyplot as plt
# 導入「抓取股票資料」套件
import pandas_datareader as data
# 導入 "RemoteDataError" (股票代碼錯誤例外處理)
from pandas_datareader._utils import RemoteDataError
# 導入 「萬能日期解析器」套件
import dateutil.parser as par
# 導入 "ParserError" (日期格式錯誤例外處理)
from dateutil.parser import ParserError

# 主程式用 while True 包覆，讓使用者可以重複輸入
while True:
    # 例外處理
    try:
        # 使用者輸入股票、起始及截止日期
        stock_iput = input("請輸入台灣股票名稱、或代號：")
        date_start_input = input("請輸入查詢起始日期：")
        date_end_input = input("請輸入查詢截止日期：")
        
        # 讀入台灣股票代號 csv file
        TaiwanStockID = pd.read_csv("TaiwanStockID.csv")
        
        # 使用者輸入股票稱為代號 or 名稱
        if stock_iput.isdigit():
            # 輸入股票為代號 (stock_ID)，用台灣股票代號 csv 資料找出股票名稱 (stock_name)
            stock_ID = stock_iput
            condition_ID = TaiwanStockID["StockID"] == eval(stock_iput)
            stock_name = TaiwanStockID[condition_ID].iloc[0]["StockName"]
        else:
            # 輸入股票為名稱 (stock_name)，用台灣股票代號 csv 資料找出股票代號 (stock_ID)
            stock_name = stock_iput
            condition_name = TaiwanStockID["StockName"] == stock_iput
            stock_ID = TaiwanStockID[condition_name].iloc[0]["StockID"]
        
        # 將 stock_ID 變為 "股票代號.TW" 格式
        stock_ID = ".".join([str(stock_ID), "TW"])
        
        
        # 將使用者輸入的日期放進「萬能日期解析器」轉換
        date_start = (par.parse(date_start_input)).date()
        date_end = (par.parse(date_end_input)).date()   
        
        
        # 依照使用者輸入內容抓取股票資料
        stock_data = data.DataReader(stock_ID, "yahoo",date_start, date_end)

    except IndexError:
        print("股票代碼或名稱錯誤！請重新輸入")
    except RemoteDataError:
        print("股票代碼錯誤！請重新輸入")
    except ParserError:
        print("日期格式錯誤！請重新輸入")
    except ValueError:
        print("錯誤！起始日期必須早於截止日期，請重新輸入")
    else:
        # 單獨取出股票資料 "Close" 欄位
        close_price = stock_data["Close"]
        
        ## 繪製股票資訊折線圖
        # 把預設字型改為 "微軟正黑體"
        plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
        plt.rcParams["axes.unicode_minus"] = False
        # 圖標題
        plt.title(" ".join([stock_ID, stock_name, str(date_start), "~", str(date_end), "收盤價"]))
        # 繪製收盤價折線圖
        close_price.plot(label = "收盤價")
        # 設定 Y 軸名稱
        plt.ylabel("指數")
        # 繪製 20 日均線
        close_price.rolling(window = 20).mean().plot(label = "20MA")
        # 繪製 60 日均線
        close_price.rolling(window = 60).mean().plot(label = "60MA")
        # 繪製圖例
        plt.legend(loc = "best")
        plt.show()
        
        # 詢問使用者是否要繼續查詢其它股票
        next_stock = input("要繼續查詢下一個股票嗎 ？(繼續請輸入 y，結束請直接按 Enter)：")
        if next_stock.lower() == "y":
            print()
        else:
            print("感謝您的使用！")
            # 終止 while True
            break
        