import pandas as pd
import requests
import json
from datetime import datetime,timedelta
from talib.abstract import MACD, RSI, CCI, DX
import numpy as np

class BinanceProcessor():
    def __init__(self):
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
        
        df_list =[]
        for i in ticker_list:
            hist_data = self.dataframe_with_limit(symbol=i)
            df = hist_data.iloc[:-1]
            df = df.dropna()
            df['tic'] = i
            df_list.append(df)
        final_df = pd.concat(df_list, axis=0, ignore_index=True)
        
        return final_df
    
    def clean_data(self, df):
        df = df.dropna()
        
        return df
    
    def add_technical_indicator(self, df, tech_indicator_list):
        print('Adding self-defined technical indicators is NOT supported yet.')
        print('Use default: MACD, RSI, CCI, DX.')
        self.tech_indicator_list = ['open', 'high', 'low', 'close', 'volume', 
                                         'macd', 'macd_signal', 'macd_hist', 
                                         'rsi', 'cci', 'dx']
        df_list = []
        for i in df.tic.unique():
            tic_df = df.loc[df.tic == i].copy()
            tic_df['macd'], tic_df['macd_signal'], tic_df['macd_hist'] = MACD(tic_df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
            tic_df['rsi'] = RSI(tic_df['close'], timeperiod=14)
            tic_df['cci'] = CCI(tic_df['high'], tic_df['low'], tic_df['close'], timeperiod=14)
            tic_df['dx'] = DX(tic_df['high'], tic_df['low'], tic_df['close'], timeperiod=14)
            df_list.append(tic_df)
        final_df = pd.concat(df_list, axis=0, ignore_index=True)
        
        return final_df
    
    def add_turbulence(self, df):
        print('Turbulence not supported yet. Return original DataFrame.')
        
        return df
    
    def add_vix(self, df):
        print('VIX is not applicable for cryptocurrencies. Return original DataFrame')
        
        return df
    
    def df_to_array(self, df, tech_indicator_list, if_vix):
        unique_ticker = df.tic.unique()
        if_first_time = True
        for tic in unique_ticker:
            if if_first_time:
                price_array = df[df.tic==tic][['close']].values
                #price_ary = df[df.tic==tic]['close'].values
                tech_array = df[df.tic==tic][tech_indicator_list].values
                if_first_time = False
            else:
                price_array = np.hstack([price_array, df[df.tic==tic][['close']].values])
                tech_array = np.hstack([tech_array, df[df.tic==tic][self.tech_indicator_list].values])
                
        assert price_array.shape[0] == tech_array.shape[0]
        
        return price_array, tech_array, np.array([])
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