import calendar

import ccxt
import pandas as pd

# from basic_processor import BasicProcessor
from finrl_meta.data_processors.basic_processor import BasicProcessor


class CCXTProcessor(BasicProcessor):
    def __init__(self, data_source: str, **kwargs):
        BasicProcessor.__init__(self, data_source, **kwargs)
        self.binance = ccxt.binance()

    def min_ohlcv(dt, pair, limit):
        since = calendar.timegm(dt.utctimetuple()) * 1000
        return self.binance.fetch_ohlcv(
            symbol=pair, timeframe='1m', since=since, limit=limit
        )

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

    def df_to_ary(self, pair_list, tech_indicator_list=[
        'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'dx_30',
        'close_30_sma', 'close_60_sma']):
        df = self.dataframe
        df = df.dropna()
        date_ary = df.index.values
        price_array = df[pd.MultiIndex.from_product([pair_list, ['close']])].values
        tech_array = df[pd.MultiIndex.from_product([pair_list, tech_indicator_list])].values
        return price_array, tech_array, date_ary
