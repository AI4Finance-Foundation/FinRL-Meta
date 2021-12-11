from datetime import datetime, timedelta
from typing import List
import numpy as np
import pandas as pd
import requests
# from talib.abstract import CCI, DX, MACD, RSI
# from basic_processor import BasicProcessor
from finrl_meta.data_processors.basic_processor import BasicProcessor


class BinanceProcessor(BasicProcessor):
    def __init__(self, data_source: str, **kwargs):
        BasicProcessor.__init__(self, data_source, **kwargs)
        self.url = "https://api.binance.com/api/v3/klines"
    
    #main functions
    def download_data(self, ticker_list: List[str], start_date: str, end_date: str, time_interval: str) -> pd.DataFrame:
        startTime = datetime.strptime(start_date, '%Y-%m-%d')
        endTime = datetime.strptime(end_date, '%Y-%m-%d')
        
        self.start_time = self.stringify_dates(startTime)
        self.end_time = self.stringify_dates(endTime)
        self.interval = time_interval
        self.limit = 1440
        
        final_df = pd.DataFrame()
        for i in ticker_list:
            hist_data = self.dataframe_with_limit(symbol=i)
            df = hist_data.iloc[:-1].dropna()
            df['tic'] = i
            final_df = final_df.append(df)
        
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
        final_df = pd.DataFrame()
        for i in df.tic.unique():
            tic_df = df[df.tic==i] 
            tic_df['macd'], tic_df['macd_signal'], tic_df['macd_hist'] = MACD(tic_df['close'], fastperiod=12, 
                                                                                slowperiod=26, signalperiod=9)
            tic_df['rsi'] = RSI(tic_df['close'], timeperiod=14)
            tic_df['cci'] = CCI(tic_df['high'], tic_df['low'], tic_df['close'], timeperiod=14)
            tic_df['dx'] = DX(tic_df['high'], tic_df['low'], tic_df['close'], timeperiod=14)
            final_df = final_df.append(tic_df)
        
        return final_df
    

    def add_turbulence(self, df):
        print('Turbulence not supported yet. Return original DataFrame.')
        
        return df
    

    def add_vix(self, df):
        print('VIX is not applicable for cryptocurrencies. Return original DataFrame')
        
        return df
    

    def df_to_array(self, df, tech_indicator_list, if_vix):
        unique_ticker = df.tic.unique()
        price_array = np.column_stack([df[df.tic==tic].close for tic in unique_ticker])
        tech_array = np.hstack([df.loc[(df.tic==tic), self.tech_indicator_list] for tic in unique_ticker])       
        assert price_array.shape[0] == tech_array.shape[0]
        return price_array, tech_array, np.array([])
    

    # helper functions
    def stringify_dates(self, date:datetime):
        return str(int(date.timestamp()*1000))


    def get_binance_bars(self, last_datetime, symbol):
        '''
        klines api returns data in the following order:
        open_time, open_price, high_price, low_price, close_price, 
        volume, close_time, quote_asset_volume, n_trades, 
        taker_buy_base_asset_volume, taker_buy_quote_asset_volume, 
        ignore
        '''
        req_params = {"symbol": symbol, 'interval': self.interval,
                      'startTime': last_datetime, 'endTime': self.end_time, 
                      'limit': self.limit}
        # For debugging purposes, uncomment these lines and if they throw an error
        # then you may have an error in req_params
        # r = requests.get(self.url, params=req_params)
        # print(r.text) 
        df = pd.DataFrame(requests.get(self.url, params=req_params).json())
        
        if df.empty:
            return None
        
        df = df.iloc[:, 0:6]
        df.columns = ['datetime','open','high','low','close','volume']

        df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)

        # No stock split and dividend announcement, hence adjusted close is the same as close
        df['adj_close'] = df['close']
        df['datetime'] = df.datetime.apply(lambda x: datetime.fromtimestamp(x/1000.0))
        df.reset_index(drop=True, inplace=True)

        return df
    

    def dataframe_with_limit(self, symbol):
        final_df = pd.DataFrame()
        last_datetime = self.start_time
        while True:
            new_df = self.get_binance_bars(last_datetime, symbol)
            if new_df is None:
                break
            final_df = final_df.append(new_df)
            last_datetime = max(new_df.datetime) + timedelta(days=1)
            last_datetime = self.stringify_dates(last_datetime)
            
        date_value = final_df['datetime'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        final_df.insert(0, 'time', date_value)
        final_df.drop('datetime', inplace=True, axis=1)
        return final_df
