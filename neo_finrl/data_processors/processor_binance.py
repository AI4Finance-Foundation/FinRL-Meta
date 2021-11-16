import pandas as pd
import requests
import json
from datetime import datetime,timedelta
from talib.abstract import MACD, RSI, CCI, DX
import numpy as np
from neo_finrl.data_processors.basic_processor import BasicProcessor

class BinanceProcessor():
    # def __init__(self):
    #     self.url = "https://api.binance.com/api/v3/klines"

    def __init__(self, data_source: str, **kwargs):
        BasicProcessor.__init__(self, data_source, **kwargs)
        self.url = "https://api.binance.com/api/v3/klines"
    
    #main functions
    def download_data(self, ticker_list, start_date, end_date, 
                      time_interval):
        startTime = datetime.strptime(start_date, '%Y-%m-%d')
        endTime = datetime.strptime(end_date, '%Y-%m-%d')
        
        self.start_time = self.stringify_dates(startTime)
        self.end_time = self.stringify_dates(endTime)
        self.interval = time_interval
        self.limit = 1440
        
        final_df = pd.DataFrame()
        for i in ticker_list:
            hist_data = self.dataframe_with_limit(symbol=i)
            df = hist_data.iloc[:-1]
            df = df.dropna()
            df['tic'] = i
            final_df = final_df.append(df)
        
        return final_df
    
    def clean_data(self, df):
        df = df.dropna()
        
        return df


    # helper functions
    def stringify_dates(self, date:datetime):
        return str(int(date.timestamp()*1000))

    def get_binance_bars(self, last_datetime, symbol):
        req_params = {"symbol": symbol, 'interval': self.interval,
                      'startTime': last_datetime, 'endTime': self.end_time, 'limit': self.limit}
        # For debugging purposes, uncomment these lines and if they throw an error
        # then you may have an error in req_params
        # r = requests.get(self.url, params=req_params)
        # print(r.text) 
        df = pd.DataFrame(json.loads(requests.get(self.url, params=req_params).text))
        if (len(df.index) == 0):
            return None
        
        df = df.iloc[:,0:6]
        df.columns = ['datetime','open','high','low','close','volume']

        df.open = df.open.astype("float")
        df.high = df.high.astype("float")
        df.low = df.low.astype("float")
        df.close = df.close.astype("float")
        df.volume = df.volume.astype("float")

        # No stock split and dividend announcement, hence close is same as adjusted close
        df['adj_close'] = df['close']
        df['datetime'] = [datetime.fromtimestamp(
            x / 1000.0) for x in df.datetime
        ]
        df.index = [x for x in range(len(df))]
        return df
    
    def dataframe_with_limit(self, symbol):
        df_list = []
        last_datetime = self.start_time
        while True:
            new_df = self.get_binance_bars(last_datetime, symbol)
            if new_df is None:
                break
            df_list.append(new_df)
            last_datetime = max(new_df.datetime) + timedelta(days=1)
            last_datetime = self.stringify_dates(last_datetime)
            
        final_df = pd.concat(df_list)
        date_value = [x.strftime('%Y-%m-%d %H:%M:%S') for x in final_df['datetime']]
        final_df.insert(0,'time',date_value)
        final_df.drop('datetime',inplace=True,axis=1)
        
        return final_df