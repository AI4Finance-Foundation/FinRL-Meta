import calendar
from datetime import datetime
from typing import List

import ccxt
import numpy as np
import pandas as pd

# from basic_processor import _Base
from finrl_meta.data_processors._base import _Base


class Ccxt(_Base):
    def __init__(self, data_source: str, start_date: str, end_date: str, time_interval: str, **kwargs):
        super().__init__(data_source, start_date, end_date, time_interval, **kwargs)
        self.binance = ccxt.binance()

    def download_data(self, ticker_list: List[str]):

        crypto_column = pd.MultiIndex.from_product([ticker_list, ['open', 'high', 'low', 'close', 'volume']])
        first_time = True
        for ticker in ticker_list:
            start_dt = datetime.strptime(self.start_date, "%Y%m%d %H:%M:%S")
            end_dt = datetime.strptime(self.end_date, "%Y%m%d %H:%M:%S")
            start_timestamp = calendar.timegm(start_dt.utctimetuple())
            end_timestamp = calendar.timegm(end_dt.utctimetuple())
            if self.time_interval == '1Min':
                date_list = [datetime.utcfromtimestamp(float(time)) \
                             for time in range(start_timestamp, end_timestamp, 60 * 720)]
            else:
                date_list = [datetime.utcfromtimestamp(float(time)) \
                             for time in range(start_timestamp, end_timestamp, 60 * 1440)]
            df = self.ohlcv(date_list, ticker, self.time_interval)
            if first_time:
                dataset = pd.DataFrame(columns=crypto_column, index=df['time'].values)
                first_time = False
            temp_col = pd.MultiIndex.from_product([[ticker], ['open', 'high', 'low', 'close', 'volume']])
            dataset[temp_col] = df[['open', 'high', 'low', 'close', 'volume']].values
        print('Actual end time: ' + str(df['time'].values[-1]))
        self.dataframe = dataset

    # def add_technical_indicators(self, df, pair_list, tech_indicator_list = [
    #     'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'dx_30',
    #     'close_30_sma', 'close_60_sma']):
    #     df = df.dropna()
    #     df = df.copy()
    #     column_list = [pair_list, ['open','high','low','close','volume']+(tech_indicator_list)]
    #     column = pd.MultiIndex.from_product(column_list)
    #     index_list = df.index
    #     dataset = pd.DataFrame(columns=column,index=index_list)
    #     for pair in pair_list:
    #         pair_column = pd.MultiIndex.from_product([[pair],['open','high','low','close','volume']])
    #         dataset[pair_column] = df[pair]
    #         temp_df = df[pair].reset_index().sort_values(by=['index'])
    #         temp_df = temp_df.rename(columns={'index':'date'})
    #         crypto_df = Sdf.retype(temp_df.copy())
    #         for indicator in tech_indicator_list:
    #             temp_indicator = crypto_df[indicator].values.tolist()
    #             dataset[(pair,indicator)] = temp_indicator
    #     print('Succesfully add technical indicators')
    #     return dataset

    def df_to_ary(self, pair_list, tech_indicator_list=None):
        if tech_indicator_list is None:
            tech_indicator_list = [
                'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'dx_30',
                'close_30_sma', 'close_60_sma']
        df = self.dataframe
        df = df.dropna()
        date_ary = df.index.values
        price_array = df[pd.MultiIndex.from_product([pair_list, ['close']])].values
        tech_array = df[pd.MultiIndex.from_product([pair_list, tech_indicator_list])].values
        return price_array, tech_array, date_ary

    def min_ohlcv(self, dt, pair, limit):
        since = calendar.timegm(dt.utctimetuple()) * 1000
        return self.binance.fetch_ohlcv(
            symbol=pair, timeframe='1m', since=since, limit=limit
        )

    def ohlcv(self, dt, pair, period='1d'):
        ohlcv = []
        limit = 1000
        if period == '1Min':
            limit = 720
        elif period == '1D':
            limit = 1
        elif period == '1H':
            limit = 24
        elif period == '5Min':
            limit = 288
        for i in dt:
            start_dt = i
            since = calendar.timegm(start_dt.utctimetuple()) * 1000
            if period == '1Min':
                ohlcv.extend(self.min_ohlcv(start_dt, pair, limit))
            else:
                ohlcv.extend(self.binance.fetch_ohlcv(symbol=pair, timeframe=period, since=since, limit=limit))
        df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        df['time'] = [datetime.fromtimestamp(float(time) / 1000) for time in df['time']]
        df['open'] = df['open'].astype(np.float64)
        df['high'] = df['high'].astype(np.float64)
        df['low'] = df['low'].astype(np.float64)
        df['close'] = df['close'].astype(np.float64)
        df['volume'] = df['volume'].astype(np.float64)
        return df
